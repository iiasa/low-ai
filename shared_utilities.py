"""
00_shared_utilities.py

Shared utilities for vLLM labeling and verification scripts.
Provides common functionality for API calls, prompt building, and schema loading.
"""

import json
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CHAT_ENDPOINT = "/v1/chat/completions"
REQUEST_TIMEOUT = 120  # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_norms_schema(schema_path: str = "00_vllm_ipcc_social_norms_schema.json") -> Dict[str, Any]:
    """Load IPCC social norms schema from JSON file."""
    schema_file = Path(__file__).parent / schema_path
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema


def load_survey_questions(survey_path: str = "00_vllm_survey_question_final.json", n_per_sector: Optional[int] = None) -> Dict[str, Any]:
    """
    Load survey questions from JSON, select n_per_sector questions from each sector (or all if None).
    Returns: {"survey_system": str, "questions_by_sector": {sector: [question_dicts]}}
    """
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
                # Format as question dict compatible with call_llm_single_choice
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


def load_api_config(config_path: str = "local_LLM_api_from_vLLM.json") -> Dict[str, Any]:
    """Load API configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load schemas at module level for backward compatibility
_NORMS_SCHEMA = load_norms_schema()
NORMS_SYSTEM = _NORMS_SCHEMA["norms_system"]
SECTOR_TOPIC = _NORMS_SCHEMA["sector_topic"]
STANCE_AGAINST_RECHECK_TEMPLATE = _NORMS_SCHEMA["stance_against_recheck_template"]
STANCE_AGAINST_STRICT_RECHECK_TEMPLATE = _NORMS_SCHEMA["stance_against_strict_recheck_template"]
STANCE_AGAINST_STRICT_RECHECK_OPTIONS = _NORMS_SCHEMA["stance_against_strict_recheck_options"]
NORMS_QUESTIONS: List[Dict[str, Any]] = _NORMS_SCHEMA["norms_questions"]

_SURVEY_DATA = load_survey_questions(n_per_sector=None)
SURVEY_SYSTEM = _SURVEY_DATA["survey_system"]
SURVEY_QUESTIONS_BY_SECTOR: Dict[str, List[Dict[str, Any]]] = _SURVEY_DATA["questions_by_sector"]


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Building & Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _get_prompt_for_question(question: Dict[str, Any], sector: Optional[str] = None) -> str:
    """Use prompt_template + sector_topic when sector is set and template exists; else use prompt."""
    template = question.get("prompt_template")
    if sector is not None and template:
        sector_topic = SECTOR_TOPIC.get(sector, sector)
        return template.format(sector_topic=sector_topic)
    return question.get("prompt", "")


def _parse_single_choice(content: str, options: List[str], map_to: Optional[Dict[str, str]]) -> str:
    """Return first matching option (by substring, longest first); apply map_to if set."""
    import re

    # Clean content: remove special tokens and extra whitespace
    c = content.strip()

    # Remove thinking/reasoning blocks (content between thinking tags)
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reason>.*?</reason>',
        r'<reasoning>.*?</reasoning>',
    ]
    for pattern in thinking_patterns:
        c = re.sub(pattern, ' ', c, flags=re.DOTALL | re.IGNORECASE)

    # Remove common LLM special tokens
    special_tokens = [
        '<|channel|>', '<|endoftext|>', '<|im_end|>', '<|im_start|>',
        '|final', '<|', '|>', '<think>', '</think>', '<thinking>', '</thinking>',
        '<reason>', '</reason>', '<reasoning>', '</reasoning>'
    ]
    for token in special_tokens:
        c = c.replace(token, ' ')

    # Clean up whitespace and lowercase
    c = ' '.join(c.split()).lower()

    # Prefer longest option first so "explicit approval" matches before "explicit"
    for opt in sorted(options, key=len, reverse=True):
        if opt.lower() in c:
            out = opt.lower()
            if map_to:
                out = map_to.get(out, out)
            return out

    # If no match, return first option as default
    return options[0] if options else ""


# ═══════════════════════════════════════════════════════════════════════════════
# API Calling Functions
# ═══════════════════════════════════════════════════════════════════════════════

async def call_llm_single_choice(
    session: aiohttp.ClientSession,
    text: str,
    question: Dict[str, Any],
    base_url: str,
    model_name: str,
    sector: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 64,
) -> Tuple[str, str]:
    """
    Call LLM API for one question; return (parsed_answer, raw_content).

    Args:
        session: aiohttp session
        text: Comment/text to classify
        question: Question dict with 'prompt', 'options', optional 'map_to'
        base_url: API base URL (e.g., "http://localhost:8001")
        model_name: Model identifier
        sector: Optional sector for sector-specific prompts
        system_prompt: Optional system prompt (defaults to NORMS_SYSTEM)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        (parsed_answer, raw_content): Tuple of parsed answer and raw LLM response
    """
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
        "temperature": temperature,
        "max_tokens": max_tokens,
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


async def call_llm_simple(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 16,
) -> Tuple[str, str]:
    """
    Simple LLM API call with system and user prompts.
    Returns: (normalized_label, raw_content)
    """
    url = base_url.rstrip("/") + CHAT_ENDPOINT
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            return content, content
    except Exception as e:
        return "error", str(e)
