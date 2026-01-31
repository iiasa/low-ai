"""
00_vLLM_hierarchical.py

1. Build sector_to_comments_cache from CSVs in paper4data/subreddit_filtered_by_regex/:
   - Only 3 sectors: transport, housing, food (aligned with sample_local_llm / notebook).
   - Sector from matched_keyword column (keyword_to_sector) or from regex keyword match on text.
   - body/title by type (comment vs submission); save id and body (text) per row.

2. Load cache and label each item with vLLM (disagreement yes/no) via concurrent requests.
   API config from local_LLM_api_from_vLLM.json (model key 5 = Qwen3-VL-4B on port 8006).
"""

import json
import os
import random
import re
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3 sectors: transport, housing, food (from reddit_Paper4_EVs.ipynb)
# keyword_to_sector: ignore _strong/_weak; map each keyword -> sector
# ---------------------------------------------------------------------------
SECTOR_KEYWORD_STRENGTH = {
    "transport_strong": [
        "electric vehicle", "evs", "bev", "battery electric", "battery-electric vehicle",
        "tesla model", "model 3", "model y", "chevy bolt", "nissan leaf",
        "ioniq 5", "mustang mach-e", "id.4", "rivian", "lucid air",
        "supercharger", "gigafactory", "zero emission vehicle", "zero-emission vehicle",
        "pure electric", "all-electric", "fully electric", "100% electric",
        "electric powertrain", "electric drivetrain", "electric motor vehicle",
        "level 2 charger", "dc fast charger", "public charger", "home charger",
        "charging network", "range anxiety", "mpge",
        "bike lane", "protected cycleway", "car-free", "low emission zone",
    ],
    "transport_weak": [
        "electric car", "electric truck", "electric suv", "plug-in hybrid",
        "phev", "charging station", "charge point", "kw charger",
        "battery swap", "solid-state battery", "gigacast",
        "tax credit", "zev mandate", "ev rebate", "phase-out ice",
        "e-bike", "micro-mobility", "last-mile delivery", "transit electrification",
        "tesla", "spacex launch price?", "elon says",
        "rail electrification", "hydrogen truck", "low carbon transport",
    ],
    "housing_strong": [
        "rooftop solar", "solar pv", "pv panel", "photovoltaics",
        "solar array", "net metering", "feed-in tariff", "solar inverter",
        "kwh generated", "solar roof", "sunrun", "sunpower",
        r"solar\s+panel(s)?", r"solar\s+pv", r"rooftop\s+solar",
        r"solar\s+power", r"photovoltaic(s)?",
    ],
    "housing_weak": [
        "solar panels", "solar power", "solar installer",
        "battery storage", "powerwall", "home battery", "smart thermostat",
        "energy audit", "energy efficiency upgrade", "led retrofit",
        "green home", "net-zero house", "zero-energy building",
        "solar tax credit", "pvgis", "renewable portfolio standard",
        "community solar", "virtual power plant", "rooftop rebate",
    ],
    "food_strong": [
        "vegan", "plant-based diet", "veganism", "veganuary", "vegetarian", "veg lifestyle",
        "carnivore diet", "meat lover", "steakhouse", "barbecue festival",
        "bacon double", "grass-fed beef", "factory farming",
        "meatless monday", "beyond meat", "impossible burger",
        "plant-based burger", "animal cruelty free",
    ],
    "food_weak": [
        "red meat", "beef consumption", "dairy free", "plant protein",
        "soy burger", "nutritional yeast", "seitan", "tofurky",
        "agricultural emissions", "methane footprint", "carbon hoofprint",
        "cow burps", "livestock emissions", "feedlot",
        "recipe vegan", "tofu scramble", "almond milk", "oat milk",
        "flexitarian", "climatetarian",
        "cultivated meat", "lab-grown meat", "precision fermentation",
    ],
}

# keyword -> sector (ignore _strong/_weak)
KEYWORD_TO_SECTOR: Dict[str, str] = {}
for kw in SECTOR_KEYWORD_STRENGTH["transport_strong"] + SECTOR_KEYWORD_STRENGTH["transport_weak"]:
    KEYWORD_TO_SECTOR[kw] = "transport"
for kw in SECTOR_KEYWORD_STRENGTH["housing_strong"] + SECTOR_KEYWORD_STRENGTH["housing_weak"]:
    KEYWORD_TO_SECTOR[kw] = "housing"
for kw in SECTOR_KEYWORD_STRENGTH["food_strong"] + SECTOR_KEYWORD_STRENGTH["food_weak"]:
    KEYWORD_TO_SECTOR[kw] = "food"

SECTOR_NAMES = ("transport", "housing", "food")

# ---------------------------------------------------------------------------
# Config: load from local_LLM_api_from_vLLM.json
# ---------------------------------------------------------------------------
CONFIG_PATH = "local_LLM_api_from_vLLM.json"
CACHE_PATH = os.path.join("paper4data", "sector_to_comments_cache.json")
# Fast-load sample for downstream API testing (10k per sector; use with --limit-total 100)
SAMPLE_CACHE_PATH = os.path.join("paper4data", "sector_to_comments_cache_10k_sample.json")
SAMPLE_SIZE_PER_SECTOR = 10_000
CSV_DIR = os.path.join("paper4data", "subreddit_filtered_by_regex")
CHAT_ENDPOINT = "/v1/chat/completions"
# RTX 5090 32GB + vLLM: high throughput, use high concurrency (80)
MAX_CONCURRENT = 80
REQUEST_TIMEOUT = 60

# Model key for vLLM (e.g. "5" = Qwen3-VL-4B on port 8006)
VLLM_MODEL_KEY = "5"


def load_api_config(model_key: str = None) -> Dict[str, str]:
    """Load base_url and model_name from local_LLM_api_from_vLLM.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    key = model_key or cfg.get("default_model_key", "1")
    models = cfg["available_models"]
    if key not in models:
        raise KeyError(f"model_key {key} not in {list(models.keys())}")
    m = models[key]
    return {"base_url": m["base_url"], "model_name": m["model_name"]}


# Chunk size for streaming CSV read (avoid OOM)
CSV_CHUNK_SIZE = 100_000

def sector_from_filename(filename: str) -> str:
    """e.g. electricvehicles_comments_regex.csv -> electricvehicles."""
    base = os.path.basename(filename)
    sector = re.sub(r"_(comments|submissions)_regex\.csv$", "", base, flags=re.I)
    return sector


def _is_submission_file(filename: str) -> bool:
    """True if filename is *_submissions_regex.csv."""
    return "submissions" in os.path.basename(filename).lower()


def _pick_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, Optional[str]]:
    """Return (id_col, body_col, title_col, type_col, matched_keyword_col); match by lower name."""
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id") or (df.columns[0] if len(df.columns) else None)
    body_col = cols_lower.get("body")
    title_col = cols_lower.get("title")
    type_col = cols_lower.get("type")
    matched_col = cols_lower.get("matched_keyword")
    return id_col, body_col, title_col, type_col, matched_col


def _sector_from_matched_keyword(matched: str) -> Optional[str]:
    """Map matched_keyword to one of transport, housing, food (ignore _strong/_weak)."""
    if not matched or not str(matched).strip():
        return None
    return KEYWORD_TO_SECTOR.get(str(matched).strip())


def build_cache_from_csvs(csv_dir: str = CSV_DIR, cache_path: str = CACHE_PATH) -> Dict[str, List[Dict[str, str]]]:
    """
    Load *_regex.csv; assign each row to one of 3 sectors (transport, housing, food).
    Sector from matched_keyword column only (keyword_to_sector). No regex search on text.
    Rows with no or empty matched_keyword are skipped and counted; count reported at end.
    Text = body for comments, title for submissions (filename or type column); fallback other if empty.
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")

    sector_items: Dict[str, List[Dict[str, str]]] = {s: [] for s in SECTOR_NAMES}
    skipped_no_matched_keyword = 0
    csv_files = sorted(csv_dir.glob("*_regex.csv"))

    for p in tqdm(csv_files, desc="CSVs"):
        is_submission = _is_submission_file(str(p))

        try:
            chunks = pd.read_csv(
                p,
                encoding="utf-8",
                on_bad_lines="skip",
                usecols=lambda c: c and c.strip().lower() in ("id", "body", "title", "type", "matched_keyword"),
                dtype=str,
                chunksize=CSV_CHUNK_SIZE,
            )
        except Exception:
            try:
                chunks = pd.read_csv(
                    p,
                    encoding="utf-8",
                    on_bad_lines="skip",
                    dtype=str,
                    chunksize=CSV_CHUNK_SIZE,
                )
            except Exception as e2:
                print(f"Skip {p.name}: {e2}")
                continue

        for chunk in chunks:
            id_col, body_col, title_col, type_col, matched_col = _pick_columns(chunk)
            if id_col is None:
                continue
            keep = [c for c in (id_col, body_col, title_col, type_col, matched_col) if c and c in chunk.columns]
            chunk = chunk[keep].copy()
            id_ser = chunk[id_col].fillna("").astype(str).str.strip()
            body_ser = chunk[body_col].fillna("").astype(str).str.strip() if body_col in chunk.columns else pd.Series("", index=chunk.index)
            title_ser = chunk[title_col].fillna("").astype(str).str.strip() if title_col in chunk.columns else pd.Series("", index=chunk.index)
            if type_col in chunk.columns:
                type_ser = chunk[type_col].fillna("").astype(str).str.strip().str.lower()
                is_sub = type_ser.str.startswith("sub")
                primary = title_ser.where(is_sub, body_ser)
                fallback = body_ser.where(is_sub, title_ser)
                text_ser = primary.where(primary != "", fallback)
            else:
                text_ser = title_ser.where(title_ser != "", body_ser) if is_submission else body_ser.where(body_ser != "", title_ser)
            has_matched_col = matched_col and matched_col in chunk.columns
            matched_ser = chunk[matched_col].fillna("").astype(str).str.strip() if has_matched_col else None

            mask = (id_ser != "") & (text_ser != "")
            ids = id_ser[mask].tolist()
            texts = text_ser[mask].tolist()
            matched_list = matched_ser[mask].tolist() if matched_ser is not None else [""] * len(ids)
            for rid, text, matched in zip(ids, texts, matched_list):
                if not matched or not str(matched).strip():
                    skipped_no_matched_keyword += 1
                    continue
                sector = _sector_from_matched_keyword(matched)
                if sector is None or sector not in SECTOR_NAMES:
                    continue
                sector_items[sector].append({"id": rid, "body": text})
            del chunk

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sector_items, f, ensure_ascii=False, indent=0)
    total = sum(len(v) for v in sector_items.values())
    print(f"Saved cache: {cache_path} ({total} items, sectors: {list(sector_items.keys())})")
    print(f"Skipped (no or empty matched_keyword): {skipped_no_matched_keyword}")
    return sector_items


def load_cache(cache_path: str = CACHE_PATH) -> Dict[str, List[Dict[str, str]]]:
    """Load sector -> list of {id, body} from JSON."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_from_cache(
    cache_path: str = CACHE_PATH,
    sample_path: str = SAMPLE_CACHE_PATH,
    n_per_sector: int = SAMPLE_SIZE_PER_SECTOR,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Sample up to n_per_sector (default 10k) comments per sector from cache; save for fast load.
    Use sample_path with --cache for downstream API testing (e.g. --limit-total 100).
    """
    cache = load_cache(cache_path)
    random.seed(seed)
    sample: Dict[str, List[Dict[str, str]]] = {}
    for sector in SECTOR_NAMES:
        items = cache.get(sector, [])
        if len(items) <= n_per_sector:
            sample[sector] = list(items)
        else:
            sample[sector] = random.sample(items, n_per_sector)
    os.makedirs(os.path.dirname(sample_path) or ".", exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=0)
    total = sum(len(v) for v in sample.values())
    print(f"Saved 10k sample: {sample_path} ({total} items, sectors: {list(sample.keys())})")
    return sample


# ---------------------------------------------------------------------------
# vLLM: disagreement labelling (concurrent)
# ---------------------------------------------------------------------------
DISAGREEMENT_SYSTEM = "You are an annotator. Answer only with 'yes' or 'no'."
DISAGREEMENT_USER = "Does the following comment or post show disagreement with something or someone? Answer only: yes or no.\n\n{text}"


async def call_vllm_disagreement(
    session: aiohttp.ClientSession,
    text: str,
    base_url: str,
    model_name: str,
) -> Tuple[str, str]:
    """Call vLLM /v1/chat/completions; return (normalized_label, raw_content)."""
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": DISAGREEMENT_SYSTEM},
            {"role": "user", "content": DISAGREEMENT_USER.format(text=text[:4000])},
        ],
        "temperature": 0.1,
        "max_tokens": 16,
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip().lower()
            # Normalize to yes/no
            if "yes" in content and "no" not in content[: content.find("yes")]:
                return "yes", content
            if "no" in content:
                return "no", content
            return "unknown", content
    except Exception as e:
        return "error", str(e)


async def label_sector_items(
    items: List[Dict[str, str]],
    base_url: str,
    model_name: str,
    max_concurrent: int = MAX_CONCURRENT,
) -> List[Dict[str, Any]]:
    """Label each item (id, body) with disagreement yes/no via concurrent vLLM calls."""
    sem = asyncio.Semaphore(max_concurrent)

    async def one(session: aiohttp.ClientSession, item: Dict[str, str]) -> Dict[str, Any]:
        async with sem:
            label, raw = await call_vllm_disagreement(
                session, item["body"], base_url, model_name
            )
        return {"id": item["id"], "body": item["body"], "disagreement": label, "raw": raw}

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(one(session, it)) for it in items]
        results = []
        with tqdm(total=len(tasks), desc="vLLM", unit="item", unit_scale=False, dynamic_ncols=True) as pbar:
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                pbar.update(1)
    return results


def check_vllm_health(base_url: str) -> bool:
    """Return True if vLLM server responds (GET /health or /v1/models)."""
    import urllib.request
    for path in ("/health", "/v1/models"):
        try:
            req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status in (200, 204):
                    return True
        except Exception:
            continue
    return False


def run_label_cache(
    cache_path: str = CACHE_PATH,
    model_key: str = VLLM_MODEL_KEY,
    max_concurrent: int = MAX_CONCURRENT,
    out_path: str = None,
    limit_per_sector: int = None,
    limit_total: int = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load cache, label each item with vLLM (disagreement), return and optionally save results.
    out_path: if set, save JSON with sector -> list of {id, body, disagreement, raw}.
    limit_per_sector: if set, only label first N items per sector (for testing).
    limit_total: if set, cap total items labelled across all sectors (takes from first sectors).
    """
    api = load_api_config(model_key)
    base_url = api["base_url"]
    model_name = api["model_name"]
    print(f"vLLM: {base_url} model={model_name}")
    if not check_vllm_health(base_url):
        print("WARNING: vLLM server did not respond to /health or /v1/models. Start the server (e.g. Docker on 8006) or requests may fail.")
    else:
        print("vLLM server OK.")

    cache = load_cache(cache_path)
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    total_labelled = 0

    for sector, items in tqdm(cache.items(), desc="Sectors", unit="sector", dynamic_ncols=True):
        if limit_total is not None and total_labelled >= limit_total:
            all_results[sector] = []
            continue
        if limit_per_sector is not None:
            items = items[: limit_per_sector]
        if limit_total is not None:
            remaining = limit_total - total_labelled
            items = items[: remaining]
        if not items:
            all_results[sector] = []
            continue
        total_labelled += len(items)
        results = asyncio.run(
            label_sector_items(items, base_url, model_name, max_concurrent=max_concurrent)
        )
        all_results[sector] = results

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Saved labels: {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Build sector cache from CSVs and/or label with vLLM (disagreement). "
        "Use --build-sample after cache for fast 10k-per-sector sample; then --label-only --cache paper4data/sector_to_comments_cache_10k_sample.json --limit-total 100 for API testing."
    )
    p.add_argument("--build-only", action="store_true", help="Only build cache from CSVs, then exit.")
    p.add_argument("--build-sample", action="store_true", help="Sample 10k per sector from cache and save to sector_to_comments_cache_10k_sample.json (fast load for API testing).")
    p.add_argument("--label-only", action="store_true", help="Only load cache and label; skip building.")
    p.add_argument("--cache", default=CACHE_PATH, help="Path to cache JSON (default: full cache; use sector_to_comments_cache_10k_sample.json for fast API test).")
    p.add_argument("--model-key", default=VLLM_MODEL_KEY, help="Key in local_LLM_api_from_vLLM.json (e.g. 5 for Qwen3-VL-4B)")
    p.add_argument("--out", default=None, help="Output JSON path for labels (default: paper4data/disagreement_labels.json)")
    p.add_argument("--limit", type=int, default=None, help="Max items per sector to label (for testing)")
    p.add_argument("--limit-total", type=int, default=None, help="Max total items to label across all sectors (e.g. 100 for quick API test)")
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent vLLM requests")
    args = p.parse_args()

    out_path = args.out or os.path.join("paper4data", "disagreement_labels.json")

    if not args.label_only:
        print("Building cache from CSVs...")
        build_cache_from_csvs(csv_dir=CSV_DIR, cache_path=args.cache)
    if args.build_only:
        print("Done (build-only).")
        exit(0)

    if args.build_sample:
        print("Building 10k-per-sector sample from cache...")
        build_sample_from_cache(cache_path=args.cache, sample_path=SAMPLE_CACHE_PATH, n_per_sector=SAMPLE_SIZE_PER_SECTOR)
        print("Done (build-sample). Use for API test: --label-only --cache paper4data/sector_to_comments_cache_10k_sample.json --limit-total 100")
        exit(0)

    print("Labelling with vLLM (disagreement yes/no)...")
    run_label_cache(
        cache_path=args.cache,
        model_key=args.model_key,
        max_concurrent=args.max_concurrent,
        out_path=out_path,
        limit_per_sector=args.limit,
        limit_total=args.limit_total,
    )
    print("Done.")
