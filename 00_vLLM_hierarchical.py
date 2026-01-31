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
# ---------------------------------------------------------------------------
NORMS_SYSTEM = (
    "You are an expert annotator for social norms and climate-related discourse. "
    "Answer with exactly one of the allowed options; no explanation."
)

# Topic wording per sector so the LLM sees only the relevant sector (not all three)
SECTOR_TOPIC = {
    "transport": "EVs",
    "food": "veganism or vegetarianism / diet",
    "housing": "solar",
}

# Safety net: when stance is "against", recheck if text asks for more options or complains insufficient options
STANCE_AGAINST_RECHECK_TEMPLATE = (
    "Does this comment ask for more {sector_topic} options or complain that current {sector_topic} options are insufficient? "
    "Answer with exactly one word: yes or no."
)

# Second pass (stringent): for comments still labelled "against" after first pass, reclassify with strict question
STANCE_AGAINST_STRICT_RECHECK_OPTIONS = ["against", "frustrated but still pro", "unclear stance"]
STANCE_AGAINST_STRICT_RECHECK_TEMPLATE = (
    "This comment was initially classified as the author being AGAINST {sector_topic}. "
    "Apply a strict second pass. Is the author truly AGAINST {sector_topic} (opposes, rejects, or is hostile)? "
    "Or are they frustrated with aspects (e.g. availability, cost) but still supportive of {sector_topic}? "
    "Or is their stance unclear or ambiguous? "
    "Answer with exactly one of: against, frustrated but still pro, unclear stance."
)

# question_id, user_prompt (or prompt_template with {sector_topic} for sector-specific questions), options, map_to
NORMS_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "1.1_gate",
        "prompt": (
            "Definitions: A social norm is a shared belief or expectation about what is typical or what is approved/disapproved. "
            "Descriptive norm = reference to what people typically do or how common something is (e.g. 'most people here drive EVs'). "
            "Injunctive norm = reference to what people should do, or explicit approval/disapproval (e.g. 'you should go vegan', 'eating meat is wrong'). "
            "Does this comment or post reference what others do or approve, or any social norm (descriptive or injunctive)? Answer with exactly one word: yes or no."
        ),
        "options": ["yes", "no"],
        "map_to": {"yes": "1", "no": "0"},
    },
    {
        "id": "1.1.1_stance",
        "prompt": (
            "What is the author's stance toward the topic (EVs / solar / veganism or diet)? "
            "against = opposes or rejects the topic. pro but lack of options = author is in favor but wants more options or complains current options are insufficient (do not code as against). "
            "Answer with exactly one of: against, against particular but pro, neither/mixed, pro, pro but lack of options."
        ),
        "prompt_template": (
            "What is the author's stance toward {sector_topic}? "
            "Definitions: against = author opposes or rejects {sector_topic}. "
            "pro but lack of options = author is in favor of {sector_topic} but wants more options or complains that current options are insufficient; do NOT code this as against. "
            "Complaints that there are too few {sector_topic} options count as 'pro but lack of options', not 'against'. "
            "Examples: 'I wish there were more {sector_topic} options' → pro but lack of options. '{sector_topic} is stupid' → against. "
            "Answer with exactly one of: against, against particular but pro, neither/mixed, pro, pro but lack of options."
        ),
        "options": ["against", "against particular but pro", "neither/mixed", "pro", "pro but lack of options"],
        "map_to": None,
    },
    {
        "id": "1.2.1_descriptive",
        "prompt": (
            "Descriptive norms refer to what people actually do or how common a behavior is (e.g. 'most people here drive EVs', 'I am a vegetarian'). "
            "They describe behavior or prevalence, not what people should do. Do NOT code as descriptive if the text prescribes or proscribes behavior (that is injunctive). "
            "Answer with exactly one of: explicitly present, absent, unclear."
        ),
        "options": ["explicitly present", "absent", "unclear"],
        "map_to": None,
    },
    {
        "id": "1.2.2_injunctive",
        "prompt": (
            "Injunctive norms are social rules about what behaviors are approved or disapproved—guiding what people should do (or avoid). "
            "They use language like should, must, have to, ought to, or express approval/disapproval (e.g. 'people should go vegan', 'I encourage everyone to go vegan'). "
            "Do NOT code as injunctive mere descriptions of how people act (e.g. 'I am a vegetarian' = describing one's own behavior, not a rule). "
            "Code as injunctive only when the text prescribes or proscribes behavior for others. "
            "Answer with exactly one of: present, absent, unclear."
        ),
        "options": ["present", "absent", "unclear"],
        "map_to": None,
    },
    {
        "id": "1.3.1_reference_group",
        "prompt": "Who is the reference group (who the author refers to as doing or approving something)? Answer with exactly one of: coworkers, family, friends, local community, neighbors, online community, other, other reddit user, partner/spouse, political tribe.",
        "options": ["coworkers", "family", "friends", "local community", "neighbors", "online community", "other", "other reddit user", "partner/spouse", "political tribe"],
        "map_to": None,
    },
    {
        "id": "1.3.1b_perceived_reference_stance",
        "prompt": "What stance does the author attribute to that reference group? Answer with exactly one of: against, neither/mixed, pro.",
        "options": ["against", "neither/mixed", "pro"],
        "map_to": None,
    },
    {
        "id": "1.3.2_mechanism",
        "prompt": "What mechanism is used to convey the norm or social pressure? Answer with exactly one of: blame/shame, community standard, identity/status signaling, other, praise, rule/virtue language, social comparison.",
        "options": ["blame/shame", "community standard", "identity/status signaling", "other", "praise", "rule/virtue language", "social comparison"],
        "map_to": None,
    },
    {
        "id": "1.3.3_second_order",
        "prompt": "Does the text express second-order normative beliefs (beliefs about what others think one should do)? Answer with exactly one of: none, weak, strong.",
        "options": ["none", "weak", "strong"],
        "map_to": {"none": "0", "weak": "1", "strong": "2"},
    },
]


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
) -> Tuple[str, str]:
    """Call vLLM for one norms question; return (parsed_answer, raw_content). sector used for sector-specific prompts."""
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    prompt = _get_prompt_for_question(question, sector)
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
    """Run all NORMS_QUESTIONS for one comment; return { comment_index, comment, answers } (dashboard format). sector used for sector-specific prompts (e.g. stance toward EVs vs solar vs diet). Safety net: if stance is 'against', recheck for 'pro but lack of options'."""
    async with sem:
        answers: Dict[str, str] = {}
        for q in NORMS_QUESTIONS:
            ans, _ = await call_vllm_single_choice(session, item["body"], q, base_url, model_name, sector=sector)
            answers[q["id"]] = ans
        # Safety net: if model said "against", recheck whether text asks for more options or complains options insufficient
        if sector and answers.get("1.1.1_stance") == "against":
            if await _recheck_against_is_lack_of_options(session, item["body"], sector, base_url, model_name):
                answers["1.1.1_stance"] = "pro but lack of options"
            else:
                # Second pass (stringent): still "against" — recheck with strict question; store result for dashboard
                recheck = await _recheck_against_strict(session, item["body"], sector, base_url, model_name)
                answers["1.1.1_stance_recheck"] = recheck
        return {
            "comment_index": comment_index,
            "comment": item["body"],
            "answers": answers,
        }


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
    print(f"vLLM norms: {base_url} model={model_name} ({len(NORMS_QUESTIONS)} questions per comment)")
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
            items = items[: per_sector_cap]
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
    p.add_argument("--model-key", default=VLLM_MODEL_KEY, help="Key in local_LLM_api_from_vLLM.json (e.g. 5 for Qwen3-VL-4B)")
    p.add_argument("--7b", "--7B", dest="use_7b", action="store_true", help="Use Mistral 7B (CLASSIFICATIONS #7B, port 8001); default remains current model")
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
