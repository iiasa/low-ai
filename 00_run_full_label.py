"""
00_run_full_label.py

Full-data norms labeling with checkpoint saves every CHECKPOINT_SIZE comments per sector.
- Temporal uniform distribution: each checkpoint batch samples equally across years with remaining items
- Without replacement: tracks labeled comment bodies; removes them from pool after each checkpoint
- Saturated years (all items labeled) are automatically skipped in subsequent checkpoints
- Resume support: if norms_labels_full.json exists, picks up where it left off
- Progress tracker written to 00_tracker_full.txt after every checkpoint

Usage:
    python 00_run_full_label.py

Output:
    paper4data/norms_labels_full.json   -- growing accumulated labels (same format as norms_labels.json)
    00_tracker_full.txt                 -- progress, throughput, ETA
"""

import json
import os
import sys
import time
import asyncio
import random
import importlib.util
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# ---------------------------------------------------------------------------
# Import from 00_vLLM_hierarchical.py using importlib (filename starts with 0)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_mod_path = str(_HERE / "00_vLLM_hierarchical.py")
_spec = importlib.util.spec_from_file_location("vllm_hierarchical", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_cache          = _mod.load_cache
load_api_config     = _mod.load_api_config
check_vllm_health   = _mod.check_vllm_health
label_one_item_norms = _mod.label_one_item_norms
SECTOR_NAMES        = _mod.SECTOR_NAMES
VLLM_MODEL_KEY      = _mod.VLLM_MODEL_KEY
MAX_CONCURRENT      = _mod.MAX_CONCURRENT
CACHE_PATH          = _mod.CACHE_PATH

import aiohttp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_SIZE       = 1000                                          # per sector per checkpoint
EXISTING_LABELS_PATH  = "paper4data/norms_labels.json"                # initial 9k (excluded from re-labeling)
FULL_LABELS_PATH      = "paper4data/norms_labels_full.json"           # growing merged output
CHECKPOINT_DIR        = "paper4data/checkpoints"                      # one file per checkpoint
TRACKER_PATH          = "00_tracker_full.txt"
MIN_YEAR              = 2010
RANDOM_SEED           = 42


# ---------------------------------------------------------------------------
# Temporal uniform sampling (within a remaining pool)
# ---------------------------------------------------------------------------
def sample_temporal_uniform(items: List[Dict], n: int, min_year: int = MIN_YEAR) -> List[Dict]:
    """
    Sample up to n items distributed as uniformly as possible across years.
    Only considers items with year >= min_year.  Items without year data are
    appended last if quota not met.
    Saturated years (fewer items than quota) contribute all their items.
    """
    if not items:
        return []
    n = min(n, len(items))

    with_year_set = {id(it) for it in items if it.get("year") and int(it["year"]) >= min_year}
    with_year    = [it for it in items if id(it) in with_year_set]
    without_year = [it for it in items if id(it) not in with_year_set]

    if not with_year:
        return random.sample(items, n)

    by_year: Dict[int, List] = defaultdict(list)
    for it in with_year:
        by_year[int(it["year"])].append(it)

    years = sorted(by_year.keys())
    n_years = len(years)
    per_year = n // n_years if n_years else 0

    sampled: List[Dict] = []
    for y in years:
        take = min(per_year, len(by_year[y]))
        if take > 0:
            sampled.extend(random.sample(by_year[y], take))

    # Fill remaining slots (from rounding) proportionally from years with surplus
    remaining_slots = n - len(sampled)
    if remaining_slots > 0:
        sampled_ids = {id(x) for x in sampled}
        pool = [it for it in with_year if id(it) not in sampled_ids]
        if pool:
            add = min(remaining_slots, len(pool))
            sampled.extend(random.sample(pool, add))

    # Still short? pad with no-year items
    if len(sampled) < n and without_year:
        need = n - len(sampled)
        sampled.extend(random.sample(without_year, min(need, len(without_year))))

    return sampled


# ---------------------------------------------------------------------------
# Label a batch for one sector (custom version with offset comment_index)
# ---------------------------------------------------------------------------
async def label_batch_norms(
    batch: List[Dict],
    sector: str,
    base_url: str,
    model_name: str,
    max_concurrent: int,
    index_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Label each item with all norms + survey questions; return list of result dicts.
    comment_index = index_offset + local_i for globally unique indices per sector.
    """
    sem = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                label_one_item_norms(session, it, index_offset + i, base_url, model_name, sem, sector=sector)
            )
            for i, it in enumerate(batch)
        ]
        results = []
        with tqdm(total=len(tasks), desc=f"{sector}", unit="item", dynamic_ncols=True, leave=False) as pbar:
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                pbar.update(1)
    results.sort(key=lambda x: x["comment_index"])
    return results


# ---------------------------------------------------------------------------
# Tracker file
# ---------------------------------------------------------------------------
def update_tracker(
    tracker_path: str,
    all_results: Dict[str, List],
    cache_sizes: Dict[str, int],
    checkpoint_num: int,
    start_time: float,
    last_chkpt_start: float,
    last_chkpt_n: int,
    year_coverage: Dict[str, Dict[int, int]],
):
    now = datetime.now()
    elapsed = time.time() - start_time

    lines = [
        "=== NORMS LABELING TRACKER ===",
        f"Updated:    {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint: {checkpoint_num}",
        "",
        "OVERALL PROGRESS:",
    ]

    total_labeled = 0
    total_items   = 0
    for sector in SECTOR_NAMES:
        n_done  = len(all_results.get(sector, []))
        n_total = cache_sizes.get(sector, 0)
        pct     = n_done / n_total * 100 if n_total else 0
        lines.append(f"  {sector:12s}: {n_done:>9,} / {n_total:>9,}  ({pct:.2f}%)")
        total_labeled += n_done
        total_items   += n_total

    total_pct = total_labeled / total_items * 100 if total_items else 0
    lines += [
        f"  {'TOTAL':12s}: {total_labeled:>9,} / {total_items:>9,}  ({total_pct:.2f}%)",
        "",
    ]

    if elapsed > 10 and total_labeled > 0:
        rate_sec  = total_labeled / elapsed
        rate_min  = rate_sec * 60
        remaining = total_items - total_labeled
        eta_sec   = remaining / rate_sec
        eta_str   = str(timedelta(seconds=int(eta_sec)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        lines += [
            "THROUGHPUT:",
            f"  Rate:               {rate_min:.0f} comments/min",
            f"  Elapsed:            {elapsed_str}",
            f"  Estimated remaining:{eta_str}  ({eta_sec/3600:.1f} hrs  /  {eta_sec/86400:.1f} days)",
            "",
        ]

    chkpt_elapsed = time.time() - last_chkpt_start
    if chkpt_elapsed > 0 and last_chkpt_n > 0:
        chkpt_rate = last_chkpt_n / chkpt_elapsed * 60
        lines += [
            f"LAST CHECKPOINT: {last_chkpt_n} comments in {chkpt_elapsed:.0f}s  ({chkpt_rate:.0f}/min)",
            "",
        ]

    lines.append("YEAR COVERAGE (labeled per year, per sector):")
    for sector in SECTOR_NAMES:
        by_yr = year_coverage.get(sector, {})
        yr_str = "  ".join(f"{y}:{c}" for y, c in sorted(by_yr.items()))
        lines.append(f"  {sector}: {yr_str or '(no year data)'}")

    with open(tracker_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_year_coverage(all_results: Dict[str, List]) -> Dict[str, Dict[int, int]]:
    cov: Dict[str, Dict[int, int]] = {}
    for sector, recs in all_results.items():
        by_yr: Dict[int, int] = defaultdict(int)
        for r in recs:
            y = r.get("year")
            if y:
                by_yr[int(y)] += 1
        cov[sector] = dict(by_yr)
    return cov


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_full_label_checkpoints(
    cache_path: str    = CACHE_PATH,
    out_path: str      = FULL_LABELS_PATH,
    tracker_path: str  = TRACKER_PATH,
    checkpoint_size: int = CHECKPOINT_SIZE,
    model_key: str     = VLLM_MODEL_KEY,
    max_concurrent: int = MAX_CONCURRENT,
    seed: int          = RANDOM_SEED,
):
    random.seed(seed)

    # ── Load API ──────────────────────────────────────────────────────────
    api        = load_api_config(model_key)
    base_url   = api["base_url"]
    model_name = api["model_name"]
    print(f"vLLM: {base_url}  model={model_name}")
    if not check_vllm_health(base_url):
        print("WARNING: vLLM did not respond to health check.")
    else:
        print("vLLM server OK.")

    # ── Load cache ────────────────────────────────────────────────────────
    print(f"\nLoading full cache from {cache_path} ...")
    cache = load_cache(cache_path)
    cache_sizes = {s: len(cache.get(s, [])) for s in SECTOR_NAMES}
    total_cache = sum(cache_sizes.values())
    print(f"Cache loaded: {total_cache:,} total items")
    for s in SECTOR_NAMES:
        print(f"  {s}: {cache_sizes[s]:,}")

    # ── Track already-labeled bodies ─────────────────────────────────────
    labeled_bodies: Dict[str, Set[str]] = {s: set() for s in SECTOR_NAMES}
    all_results:    Dict[str, List]     = {s: [] for s in SECTOR_NAMES}
    sector_idx_offset: Dict[str, int]  = {s: 0 for s in SECTOR_NAMES}

    # Load existing 9k labels (exclude from re-labeling, but NOT merged into output)
    if os.path.exists(EXISTING_LABELS_PATH):
        print(f"\nLoading existing labels from {EXISTING_LABELS_PATH} ...")
        with open(EXISTING_LABELS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for sector, recs in existing.items():
            bodies = {r["comment"] for r in recs if r.get("comment")}
            labeled_bodies[sector].update(bodies)
        total_existing = sum(len(v) for v in existing.values())
        print(f"  Excluding {total_existing:,} already-labeled comments from sampling.")

    # Resume from previous full run if output file exists
    if os.path.exists(out_path):
        print(f"\nResuming from {out_path} ...")
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for sector, recs in saved.items():
            all_results[sector] = list(recs)
            for r in recs:
                body = r.get("comment", "")
                if body:
                    labeled_bodies[sector].add(body)
            sector_idx_offset[sector] = len(recs)
        total_resumed = sum(len(v) for v in all_results.values())
        print(f"  Resumed {total_resumed:,} records across all sectors.")

    # ── Build remaining pools ─────────────────────────────────────────────
    print("\nBuilding remaining pools (filtering already-labeled items)...")
    remaining: Dict[str, List] = {}
    for sector in SECTOR_NAMES:
        done     = labeled_bodies[sector]
        pool     = [it for it in cache.get(sector, []) if it.get("body") and it["body"] not in done]
        remaining[sector] = pool
        pct_done = (cache_sizes[sector] - len(pool)) / cache_sizes[sector] * 100 if cache_sizes[sector] else 0
        print(f"  {sector}: {len(pool):,} remaining  ({pct_done:.1f}% already done)")

    total_remaining = sum(len(v) for v in remaining.values())
    print(f"\nTotal remaining to label: {total_remaining:,}")
    if total_remaining == 0:
        print("All items already labeled. Nothing to do.")
        return

    # ── Checkpoint loop ───────────────────────────────────────────────────
    start_time     = time.time()
    checkpoint_num = 0
    last_chkpt_start = start_time
    last_chkpt_n   = 0

    # Track per-sector sizes before each checkpoint (for saving per-checkpoint slice)
    run_full_label_checkpoints._prev_sizes = {s: len(all_results[s]) for s in SECTOR_NAMES}

    print(f"\nStarting checkpoint loop (checkpoint_size={checkpoint_size}/sector)...\n")

    while any(remaining[s] for s in SECTOR_NAMES):
        checkpoint_num += 1
        chkpt_start = time.time()
        chkpt_n     = 0

        print(f"{'='*60}")
        print(f"CHECKPOINT {checkpoint_num}  ({datetime.now().strftime('%H:%M:%S')})")

        for sector in SECTOR_NAMES:
            if not remaining[sector]:
                print(f"  [{sector}] fully labeled - skipping")
                continue

            n_take = min(checkpoint_size, len(remaining[sector]))
            batch  = sample_temporal_uniform(remaining[sector], n_take, min_year=MIN_YEAR)

            # Year stats for this batch
            batch_year_counts: Dict[int, int] = defaultdict(int)
            for it in batch:
                if it.get("year"):
                    batch_year_counts[int(it["year"])] += 1
            yr_summary = "  ".join(f"{y}:{c}" for y, c in sorted(batch_year_counts.items()))
            print(f"  [{sector}] batch={len(batch)}  years: {yr_summary or 'no year'}")

            # Remove batch from remaining pool (without-replacement guarantee)
            batch_bodies = {it["body"] for it in batch}
            remaining[sector] = [it for it in remaining[sector] if it["body"] not in batch_bodies]

            # Label
            idx_off = sector_idx_offset[sector]
            results = asyncio.run(
                label_batch_norms(batch, sector, base_url, model_name, max_concurrent, index_offset=idx_off)
            )
            all_results[sector].extend(results)
            sector_idx_offset[sector] += len(results)
            chkpt_n += len(results)

            n_done  = len(all_results[sector])
            n_total = cache_sizes[sector]
            print(f"  [{sector}] done so far: {n_done:,}/{n_total:,} ({n_done/n_total*100:.2f}%)")

        # ── Save this checkpoint's data to checkpoints/ folder ────────
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        chkpt_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_num:04d}.json")
        chkpt_data: Dict[str, List] = {}
        # Each checkpoint file contains only this batch's results (same format as norms_labels.json)
        for sector in SECTOR_NAMES:
            n_this = chkpt_n  # approximate; track per sector for accuracy
        # Re-build per-sector checkpoint slice (items added this checkpoint)
        sector_sizes_before = getattr(run_full_label_checkpoints, "_prev_sizes", {s: 0 for s in SECTOR_NAMES})
        for sector in SECTOR_NAMES:
            prev_n = sector_sizes_before.get(sector, 0)
            chkpt_data[sector] = all_results[sector][prev_n:]
        run_full_label_checkpoints._prev_sizes = {s: len(all_results[s]) for s in SECTOR_NAMES}

        print(f"\n  Saving checkpoint {checkpoint_num} -> {chkpt_file}")
        with open(chkpt_file, "w", encoding="utf-8") as f:
            json.dump(chkpt_data, f, ensure_ascii=False, indent=0)

        # ── Save running merged output ─────────────────────────────────
        print(f"  Saving merged output -> {out_path}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=0)
        os.replace(tmp_path, out_path)  # atomic rename

        # ── Update tracker ────────────────────────────────────────────
        year_cov = compute_year_coverage(all_results)
        update_tracker(
            tracker_path, all_results, cache_sizes,
            checkpoint_num, start_time, chkpt_start, chkpt_n, year_cov,
        )
        last_chkpt_start = chkpt_start
        last_chkpt_n     = chkpt_n

        # Print tracker summary to stdout
        total_done = sum(len(v) for v in all_results.values())
        total_all  = sum(cache_sizes.values())
        elapsed    = time.time() - start_time
        rate_min   = total_done / elapsed * 60 if elapsed > 0 else 0
        remaining_n = total_all - total_done
        eta_hrs    = (remaining_n / (total_done / elapsed)) / 3600 if total_done > 0 else 0
        print(f"\n  TOTAL: {total_done:,}/{total_all:,} ({total_done/total_all*100:.2f}%)  "
              f"rate={rate_min:.0f}/min  ETA={eta_hrs:.1f}hrs")
        print(f"  Tracker updated: {tracker_path}\n")

    print("\n" + "="*60)
    print("ALL SECTORS FULLY LABELED.")
    # Final tracker update
    year_cov = compute_year_coverage(all_results)
    update_tracker(
        tracker_path, all_results, cache_sizes,
        checkpoint_num, start_time, time.time(), 0, year_cov,
    )
    print(f"Final output: {out_path}")
    print(f"Tracker:      {tracker_path}")


if __name__ == "__main__":
    run_full_label_checkpoints()
