"""
00_vLLM_hierarchical.py

1. Build sector_to_comments_cache from CSVs in paper4data/subreddit_filtered_by_regex/:
   - Only 3 sectors: transport, housing, food (aligned with sample_local_llm / notebook).
   - Sector from matched_keyword column (keyword_to_sector).
   - Extract year from created_utc (Unix timestamp) for temporal analysis.
   - body/title by type (comment vs submission); save id, body (text), and year per row.
   - Deduplicate by body per sector (keep first occurrence).

2. Load cache and label each item with vLLM via hierarchical norms questions (IPCC social drivers):
   - Multiple questions per comment: norm signal present (1.1_gate), author stance (1.1.1_stance), 
     descriptive/injunctive norms (1.2.1, 1.2.2), reference group (1.3.1), perceived reference stance (1.3.1b),
     second-order normative beliefs (1.3.3).
   - Sector-specific prompts (EVs vs solar vs veganism/diet).
   - Safety net: recheck "against" stance for "pro but lack of options" misclassification.
   - Second-pass stringent recheck for "against" labels (against/frustrated but still pro/unclear stance).
   - When limit_total is set, sample equally across years (2010+) for temporal analysis.
   - Preserve year in output for temporal visualization.
   - API config from local_LLM_api_from_vLLM.json (default model key 6 = Mistral-7B on port 8001; use --qwen for Qwen3-VL-4B on port 8006).
   - Output: sector -> list of { comment_index, comment, year?, answers: { "1.1_gate": "1", "1.1.1_stance": "pro", ... } }.
   - Use with 00_vLLM_visualize.py to build norms_hierarchical_dashboard.html and examples.
"""

import json
import os
import random
import re
import asyncio
import aiohttp
from collections import defaultdict
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

# Model key for vLLM (default "6" = Mistral-7B on port 8001; "5" = Qwen3-VL-4B on port 8006)
VLLM_MODEL_KEY = "6"


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


def _pick_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, Optional[str], Optional[str]]:
    """Return (id_col, body_col, title_col, type_col, matched_keyword_col, created_utc_col); match by lower name."""
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id") or (df.columns[0] if len(df.columns) else None)
    body_col = cols_lower.get("body")
    title_col = cols_lower.get("title")
    type_col = cols_lower.get("type")
    matched_col = cols_lower.get("matched_keyword")
    created_utc_col = cols_lower.get("created_utc")
    return id_col, body_col, title_col, type_col, matched_col, created_utc_col


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
                usecols=lambda c: c and c.strip().lower() in ("id", "body", "title", "type", "matched_keyword", "created_utc"),
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
            id_col, body_col, title_col, type_col, matched_col, created_utc_col = _pick_columns(chunk)
            if id_col is None:
                continue
            keep = [c for c in (id_col, body_col, title_col, type_col, matched_col, created_utc_col) if c and c in chunk.columns]
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
            # Extract year from created_utc (Unix timestamp)
            year_ser = None
            if created_utc_col and created_utc_col in chunk.columns:
                try:
                    created_utc_ser = pd.to_numeric(chunk[created_utc_col], errors="coerce")
                    year_ser = pd.to_datetime(created_utc_ser, unit="s", errors="coerce").dt.year
                except Exception:
                    year_ser = None

            mask = (id_ser != "") & (text_ser != "")
            ids = id_ser[mask].tolist()
            texts = text_ser[mask].tolist()
            matched_list = matched_ser[mask].tolist() if matched_ser is not None else [""] * len(ids)
            years_list = year_ser[mask].tolist() if year_ser is not None else [None] * len(ids)
            for rid, text, matched, year in zip(ids, texts, matched_list, years_list):
                if not matched or not str(matched).strip():
                    skipped_no_matched_keyword += 1
                    continue
                sector = _sector_from_matched_keyword(matched)
                if sector is None or sector not in SECTOR_NAMES:
                    continue
                item = {"id": rid, "body": text}
                if year is not None and not pd.isna(year):
                    item["year"] = int(year)
                sector_items[sector].append(item)
            del chunk

    # Dedupe per sector by body (keep first occurrence)
    for sector in SECTOR_NAMES:
        seen_bodies: set = set()
        unique: List[Dict[str, str]] = []
        for item in sector_items[sector]:
            body = (item.get("body") or "").strip()
            if body and body not in seen_bodies:
                seen_bodies.add(body)
                unique.append(item)
        sector_items[sector] = unique

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sector_items, f, ensure_ascii=False, indent=0)
    total = sum(len(v) for v in sector_items.values())
    print(f"Saved cache: {cache_path} ({total} items, sectors: {list(sector_items.keys())})")
    print(f"Skipped (no or empty matched_keyword): {skipped_no_matched_keyword}")
    return sector_items


def load_cache(cache_path: str = CACHE_PATH) -> Dict[str, List[Dict[str, str]]]:
    """Load sector -> list of {id, body, year?} from JSON."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_by_year_equal(items: List[Dict[str, Any]], n_total: int, min_year: int = 2010) -> List[Dict[str, Any]]:
    """
    Sample n_total items distributed equally across years (min_year onwards).
    If a year has fewer items than the per-year quota, take all available.
    Returns items with year >= min_year, distributed as evenly as possible.
    """
    # Filter to items with year >= min_year
    items_with_year = [it for it in items if it.get("year") and isinstance(it["year"], (int, float)) and int(it["year"]) >= min_year]
    items_no_year = [it for it in items if not (it.get("year") and isinstance(it["year"], (int, float)) and int(it["year"]) >= min_year)]
    
    if not items_with_year:
        # No year data: fallback to random sample
        return random.sample(items, min(n_total, len(items))) if items else []
    
    # Group by year
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for it in items_with_year:
        year = int(it["year"])
        by_year[year].append(it)
    
    years = sorted(by_year.keys())
    if not years:
        return random.sample(items, min(n_total, len(items))) if items else []
    
    # Calculate per-year quota (whole number)
    n_years = len(years)
    per_year_quota = n_total // n_years  # Integer division
    
    sampled = []
    for year in years:
        year_items = by_year[year]
        n_take = min(per_year_quota, len(year_items))
        if n_take > 0:
            sampled.extend(random.sample(year_items, n_take))
    
    # If we have leftover slots (n_total % n_years), fill from years with remaining items
    remaining = n_total - len(sampled)
    if remaining > 0:
        available = []
        for year in years:
            year_items = by_year[year]
            already_taken = {id(it.get("id", "")) for it in sampled}
            remaining_items = [it for it in year_items if id(it.get("id", "")) not in already_taken]
            available.extend(remaining_items)
        if available:
            n_add = min(remaining, len(available))
            sampled.extend(random.sample(available, n_add))
    
    # If still not enough, add items without year data
    if len(sampled) < n_total and items_no_year:
        remaining = n_total - len(sampled)
        n_add = min(remaining, len(items_no_year))
        sampled.extend(random.sample(items_no_year, n_add))
    
    return sampled


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
        # Dedupe by body (keep first) so sample has no duplicate comments
        seen_bodies: set = set()
        unique: List[Dict[str, str]] = []
        for item in items:
            body = (item.get("body") or "").strip()
            if body and body not in seen_bodies:
                seen_bodies.add(body)
                unique.append(item)
        items = unique
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


# ---------------------------------------------------------------------------
# Norms labelling (IPCC social drivers) — questions and prompts for dashboard
# Output keys match norms_hierarchical_dashboard: 1.1_gate, 1.1.1_stance, etc.
# Schema loaded from 00_vllm_ipcc_social_norms_schema.json
# ---------------------------------------------------------------------------
def load_norms_schema(schema_path: str = "00_vllm_ipcc_social_norms_schema.json") -> Dict[str, Any]:
    """Load norms schema from JSON file. Returns dict with keys: norms_system, sector_topic, stance_against_recheck_template, stance_against_strict_recheck_template, stance_against_strict_recheck_options, norms_questions."""
    schema_file = Path(__file__).parent / schema_path
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema


# Load schema at module level
_NORMS_SCHEMA = load_norms_schema()

# Extract constants from schema for backward compatibility and convenience
NORMS_SYSTEM = _NORMS_SCHEMA["norms_system"]
SECTOR_TOPIC = _NORMS_SCHEMA["sector_topic"]
STANCE_AGAINST_RECHECK_TEMPLATE = _NORMS_SCHEMA["stance_against_recheck_template"]
STANCE_AGAINST_STRICT_RECHECK_TEMPLATE = _NORMS_SCHEMA["stance_against_strict_recheck_template"]
STANCE_AGAINST_STRICT_RECHECK_OPTIONS = _NORMS_SCHEMA["stance_against_strict_recheck_options"]
NORMS_QUESTIONS: List[Dict[str, Any]] = _NORMS_SCHEMA["norms_questions"]


def load_survey_questions(survey_path: str = "00_vllm_survey_question_final.json", n_per_sector: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load survey questions from JSON, select n_per_sector questions from each sector (or all if None), format as question dicts."""
    survey_file = Path(__file__).parent / survey_path
    with open(survey_file, "r", encoding="utf-8") as f:
        survey_data = json.load(f)
    
    survey_system = survey_data.get("survey_system", "You are an expert annotator. Answer with exactly one of the allowed options; no explanation.")
    
    # Map sector names: FOOD -> food, TRANSPORT -> transport, HOUSING -> housing
    sector_map = {"FOOD": "food", "TRANSPORT": "transport", "HOUSING": "housing"}
    
    survey_questions_by_sector: Dict[str, List[Dict[str, Any]]] = {}
    
    for sector_key, sector_name in sector_map.items():
        if sector_key not in survey_data:
            continue
        
        sector_questions = []
        # Get first question set (there's only one per sector currently)
        for question_set_name, question_set in survey_data[sector_key].items():
            questions = question_set.get("questions", [])
            # Take all questions if n_per_sector is None, otherwise take first n_per_sector
            selected = questions if n_per_sector is None else questions[:n_per_sector]
            
            for q in selected:
                # Format as question dict compatible with call_vllm_single_choice
                formatted_q = {
                    "id": q["id"],
                    "prompt": q["prompt"],
                    "options": ["YES", "NO"],
                    "map_to": {"YES": "1", "NO": "0"},
                }
                sector_questions.append(formatted_q)
            break  # Only process first question set per sector
        
        if sector_questions:
            survey_questions_by_sector[sector_name] = sector_questions
    
    return {"survey_system": survey_system, "questions_by_sector": survey_questions_by_sector}


# Load survey questions at module level (all questions)
_SURVEY_DATA = load_survey_questions(n_per_sector=None)
SURVEY_SYSTEM = _SURVEY_DATA["survey_system"]
SURVEY_QUESTIONS_BY_SECTOR: Dict[str, List[Dict[str, Any]]] = _SURVEY_DATA["questions_by_sector"]


def _parse_single_choice(content: str, options: List[str], map_to: Optional[Dict[str, str]]) -> str:
    """Return first matching option (by substring, longest first); apply map_to if set."""
    c = content.strip().lower()
    # Prefer longest option first so "explicit approval" matches before "explicit"
    for opt in sorted(options, key=len, reverse=True):
        if opt.lower() in c:
            out = opt.lower()
            if map_to:
                out = map_to.get(out, out)
            return out
    return options[0] if options else ""


def _get_prompt_for_question(question: Dict[str, Any], sector: Optional[str] = None) -> str:
    """Use prompt_template + sector_topic when sector is set and template exists; else use prompt."""
    template = question.get("prompt_template")
    if sector is not None and template:
        sector_topic = SECTOR_TOPIC.get(sector, sector)
        return template.format(sector_topic=sector_topic)
    return question.get("prompt", "")


async def call_vllm_single_choice(
    session: aiohttp.ClientSession,
    text: str,
    question: Dict[str, Any],
    base_url: str,
    model_name: str,
    sector: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    """Call vLLM for one question; return (parsed_answer, raw_content). sector used for sector-specific prompts. system_prompt defaults to NORMS_SYSTEM if not provided."""
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    prompt = _get_prompt_for_question(question, sector)
    user_content = prompt + "\n\n---\n\nComment/post:\n\n" + (text[:4000] if text else "")
    system_content = system_prompt if system_prompt is not None else NORMS_SYSTEM
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            parsed = _parse_single_choice(content, question["options"], question.get("map_to"))
            return parsed, content
    except Exception as e:
        qid = question.get("id", "?")
        default = question["options"][0] if question.get("options") else ""
        if question.get("map_to"):
            default = question["map_to"].get(default, default)
        return default, str(e)


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
            if "yes" in content and "no" not in content[: content.find("yes")]:
                return "yes", content
            if "no" in content:
                return "no", content
            return "unknown", content
    except Exception as e:
        return "error", str(e)


async def _recheck_against_is_lack_of_options(
    session: aiohttp.ClientSession,
    text: str,
    sector: str,
    base_url: str,
    model_name: str,
) -> bool:
    """If stance was 'against', ask whether text asks for more options or complains options are insufficient. Returns True if yes (override to pro but lack of options)."""
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    sector_topic = SECTOR_TOPIC.get(sector, sector)
    prompt = STANCE_AGAINST_RECHECK_TEMPLATE.format(sector_topic=sector_topic)
    user_content = prompt + "\n\n---\n\nComment/post:\n\n" + (text[:4000] if text else "")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": NORMS_SYSTEM},
            {"role": "user", "content": user_content},
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
            return "yes" in content and "no" not in content[: content.find("yes")]
    except Exception:
        return False


async def _recheck_against_strict(
    session: aiohttp.ClientSession,
    text: str,
    sector: str,
    base_url: str,
    model_name: str,
) -> str:
    """Second pass for comments still labelled 'against': stringent question. Returns one of: against, frustrated but still pro, unclear stance."""
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    sector_topic = SECTOR_TOPIC.get(sector, sector)
    prompt = STANCE_AGAINST_STRICT_RECHECK_TEMPLATE.format(sector_topic=sector_topic)
    user_content = prompt + "\n\n---\n\nComment/post:\n\n" + (text[:4000] if text else "")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": NORMS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    options = STANCE_AGAINST_STRICT_RECHECK_OPTIONS
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            parsed = _parse_single_choice(content, options, map_to=None)
            return parsed if parsed in options else options[0]
    except Exception:
        return options[0]


async def label_one_item_norms(
    session: aiohttp.ClientSession,
    item: Dict[str, str],
    comment_index: int,
    base_url: str,
    model_name: str,
    sem: asyncio.Semaphore,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all NORMS_QUESTIONS and SURVEY_QUESTIONS for one comment; return { comment_index, comment, answers } (dashboard format). sector used for sector-specific prompts (e.g. stance toward EVs vs solar vs diet). Safety net: if stance is 'against', recheck for 'pro but lack of options'."""
    async with sem:
        answers: Dict[str, str] = {}
        # Run norms questions
        for q in NORMS_QUESTIONS:
            ans, _ = await call_vllm_single_choice(session, item["body"], q, base_url, model_name, sector=sector)
            answers[q["id"]] = ans
        # Run survey questions for this sector (if any)
        if sector and sector in SURVEY_QUESTIONS_BY_SECTOR:
            for q in SURVEY_QUESTIONS_BY_SECTOR[sector]:
                ans, _ = await call_vllm_single_choice(session, item["body"], q, base_url, model_name, sector=sector, system_prompt=SURVEY_SYSTEM)
                answers[q["id"]] = ans
        # Safety net: if model said "against", recheck whether text asks for more options or complains options insufficient
        if sector and answers.get("1.1.1_stance") == "against":
            if await _recheck_against_is_lack_of_options(session, item["body"], sector, base_url, model_name):
                answers["1.1.1_stance"] = "pro but lack of options"
            else:
                # Second pass (stringent): still "against" — recheck with strict question; store result for dashboard
                recheck = await _recheck_against_strict(session, item["body"], sector, base_url, model_name)
                answers["1.1.1_stance_recheck"] = recheck
        result = {
            "comment_index": comment_index,
            "comment": item["body"],
            "answers": answers,
        }
        # Preserve year if present (for temporal analysis)
        if "year" in item:
            result["year"] = item["year"]
        return result


async def label_sector_items_norms(
    items: List[Dict[str, str]],
    sector: str,
    base_url: str,
    model_name: str,
    max_concurrent: int = MAX_CONCURRENT,
) -> List[Dict[str, Any]]:
    """Label each item with all norms questions; return list of { comment_index, comment, answers }. sector used so prompts mention only that sector (e.g. EVs, solar, or veganism/diet)."""
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(label_one_item_norms(session, it, i, base_url, model_name, sem, sector=sector))
            for i, it in enumerate(items)
        ]
        results = []
        with tqdm(total=len(tasks), desc="vLLM norms", unit="item", dynamic_ncols=True) as pbar:
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                pbar.update(1)
    results.sort(key=lambda x: x["comment_index"])
    return results


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
    # limit_total = per-sector cap (500 means 500 of each sector)
    per_sector_cap = limit_total

    for sector, items in tqdm(cache.items(), desc="Sectors", unit="sector", dynamic_ncols=True):
        if limit_per_sector is not None:
            items = items[: limit_per_sector]
        if per_sector_cap is not None:
            items = items[: per_sector_cap]
        if not items:
            all_results[sector] = []
            continue
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


def run_norms_label_cache(
    cache_path: str = CACHE_PATH,
    model_key: str = VLLM_MODEL_KEY,
    max_concurrent: int = MAX_CONCURRENT,
    out_path: str = None,
    limit_per_sector: int = None,
    limit_total: int = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load cache, run all NORMS_QUESTIONS per item via vLLM, save in dashboard format.
    Output: sector -> list of { comment_index, comment, answers: { "1.1_gate": "1", ... } }.
    Use with 00_vLLM_visualize.py to build norms_hierarchical_dashboard.html and examples.
    """
    api = load_api_config(model_key)
    base_url = api["base_url"]
    model_name = api["model_name"]
    survey_q_counts = {s: len(qs) for s, qs in SURVEY_QUESTIONS_BY_SECTOR.items()}
    survey_info = f" + {sum(survey_q_counts.values())} survey questions ({survey_q_counts})" if survey_q_counts else ""
    print(f"vLLM norms: {base_url} model={model_name} ({len(NORMS_QUESTIONS)} norms questions per comment{survey_info})")
    if not check_vllm_health(base_url):
        print("WARNING: vLLM server did not respond. Start the server (e.g. Docker on 8006) or requests may fail.")
    else:
        print("vLLM server OK.")

    cache = load_cache(cache_path)
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    # limit_total = per-sector cap (500 means 500 of each sector)
    per_sector_cap = limit_total

    for sector, items in tqdm(cache.items(), desc="Sectors", unit="sector", dynamic_ncols=True):
        if limit_per_sector is not None:
            items = items[: limit_per_sector]
        if per_sector_cap is not None:
            # Sample equally across years (2010+) for temporal analysis
            items = sample_by_year_equal(items, per_sector_cap, min_year=2010)
        if not items:
            all_results[sector] = []
            continue
        results = asyncio.run(
            label_sector_items_norms(items, sector, base_url, model_name, max_concurrent=max_concurrent)
        )
        all_results[sector] = results

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Saved norms labels: {out_path} (for 00_vLLM_visualize.py)")

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
    p.add_argument("--model-key", default=VLLM_MODEL_KEY, help="Key in local_LLM_api_from_vLLM.json (default 6 = Mistral-7B, 5 = Qwen3-VL-4B)")
    p.add_argument("--7b", "--7B", dest="use_7b", action="store_true", help="Use Mistral 7B (CLASSIFICATIONS #7B, port 8001); now default, kept for compatibility")
    p.add_argument("--qwen", dest="use_qwen", action="store_true", help="Use Qwen3-VL-4B (port 8006) instead of default Mistral-7B")
    p.add_argument("--out", default=None, help="Output JSON path for labels (default: paper4data/disagreement_labels.json)")
    p.add_argument("--limit", type=int, default=None, help="Max items per sector to label (for testing)")
    p.add_argument("--limit-total", type=int, default=None, help="Max items per sector (500 = 500 of each sector, 1500 total)")
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent vLLM requests")
    p.add_argument("--norms", action="store_true", help="Run norms labelling (8 IPCC social-driver questions per comment); output for 00_vLLM_visualize.py")
    p.add_argument("--out-norms", default=None, help="Output JSON for norms labels (default: paper4data/norms_labels.json)")
    p.add_argument("--incomplete", action="store_true", help="Shortcut: label-only, norms, 10k sample cache (equiv. to --label-only --norms --cache paper4data/sector_to_comments_cache_10k_sample.json); use with e.g. --limit 500")
    args = p.parse_args()
    if getattr(args, "use_7b", False):
        args.model_key = "6"
    if getattr(args, "use_qwen", False):
        args.model_key = "5"

    if args.incomplete:
        args.label_only = True
        args.norms = True
        args.cache = SAMPLE_CACHE_PATH
    # When only --limit/--limit-total is given (no other mode), run same as --label-only --norms --cache 10k_sample
    if (args.limit is not None or args.limit_total is not None) and not (
        args.build_only or args.build_sample or args.label_only or args.norms
    ):
        args.label_only = True
        args.norms = True
        args.cache = SAMPLE_CACHE_PATH

    out_path = args.out or os.path.join("paper4data", "disagreement_labels.json")
    out_norms_path = args.out_norms or os.path.join("paper4data", "norms_labels.json")

    if args.build_only:
        print("Building cache from CSVs...")
        build_cache_from_csvs(csv_dir=CSV_DIR, cache_path=args.cache)
        print("Done (build-only).")
        exit(0)

    if args.build_sample:
        print("Building 10k-per-sector sample from cache...")
        build_sample_from_cache(cache_path=args.cache, sample_path=SAMPLE_CACHE_PATH, n_per_sector=SAMPLE_SIZE_PER_SECTOR)
        print("Done (build-sample). Use for API test: --label-only --cache paper4data/sector_to_comments_cache_10k_sample.json --limit-total 100")
        exit(0)

    if args.norms:
        print("Labelling with vLLM (norms: 8 questions per comment)...")
        run_norms_label_cache(
            cache_path=args.cache,
            model_key=args.model_key,
            max_concurrent=args.max_concurrent,
            out_path=out_norms_path,
            limit_per_sector=args.limit,
            limit_total=args.limit_total,
        )
        print("Done. Run: python 00_vLLM_visualize.py --input paper4data/norms_labels.json")
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
