"""
vLLM Server Test - Classifier for Survey Questions
Uses the same fake samples as sample_local_llm.py but connects to vLLM server
vLLM server runs on localhost:8000 (as configured in run_vllm_server.bat)

Optimized for maximum throughput using async requests and continuous batching.
vLLM automatically batches concurrent requests together for efficient processing.
"""

import json
import requests
import aiohttp
import asyncio
from typing import List, Dict, Tuple
import time
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import jsonschema
from jsonschema import validate, ValidationError
import concurrent.futures
from functools import partial
from collections import Counter
import sys
import os
import pandas as pd
import random

# Import functions and data from sample_local_llm.py
from sample_local_llm import (
    get_default_questions,
    load_survey_questions,
    load_survey_questions_by_sector,
    get_questions_for_sector,
    SAMPLE_STATEMENTS,
    GROUND_TRUTH,
    RESPONSE_SCHEMA,
    create_relevance_prompt,
    create_relevance_prompt_grouped,
    create_classification_prompt,
    check_relevance,
    classify_statement_against_question,
    check_relevance_task,
    check_relevance_grouped,
    classify_single_task,
    first_pass_relevance,
    second_pass_classification,
    group_related_questions,
    calculate_coherence,
    get_majority_vote,
    save_results_csv,
    calculate_metrics,
    save_metrics_csv,
    print_metrics,
    print_relevance_percentages
)

# Load API configuration (for endpoint structure)
with open('paper4_LOCAL_LLMS_api.json', 'r') as f:
    api_config = json.load(f)

# vLLM Server Configuration - Multiple Models Available
# Available models running on different ports
AVAILABLE_MODELS = {
    "1": {
        "name": "qwen2.5-3b",
        "base_url": "http://127.0.0.1:8000",
        "model_name": "qwen2.5-3b",
        "description": "Qwen2.5-3B-Instruct (port 8000)"
    },
    "2": {
        "name": "qwen3-1p7b",
        "base_url": "http://127.0.0.1:8001",
        "model_name": "qwen3-1p7b",
        "description": "Qwen3 1.7B (port 8001)"
    },
    "3": {
        "name": "gemma2-2b",
        "base_url": "http://127.0.0.1:8002",
        "model_name": "gemma-2-2b-it",
        "description": "Gemma 2 2B Instruct (port 8002)"
    },
    "4": {
        "name": "mistral3-3b",
        "base_url": "http://127.0.0.1:8003",
        "model_name": "mistral3-3b",
        "description": "Mistral MiniStral 3 3B (port 8003)"
    }
}

# Default model (can be overridden via command line or selection)
DEFAULT_MODEL_KEY = "1"
VLLM_BASE_URL = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["base_url"]
VLLM_MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["model_name"]

# Performance tuning for continuous batching
# Optimized for RTX 5090 32GB GPU with Qwen2.5-3B-Instruct model
# Higher concurrency = more requests batched together by vLLM
# vLLM automatically batches concurrent requests in-flight for maximum GPU utilization
# 
# RTX 5090 32GB settings:
#   - 32GB VRAM allows for very high concurrency
#   - 3B model is relatively small, can handle many concurrent requests
#   - Can maximize throughput with high concurrent request limits
MAX_CONCURRENT_REQUESTS = 450  # Optimized for 32GB GPU - can handle high concurrency
REQUEST_TIMEOUT = 60  # Increased timeout for high concurrency scenarios

# Load survey questions (default - will be reloaded in main if flag is set)
SURVEY_QUESTIONS = load_survey_questions(use_real_data=False)

def model_supports_system_messages(model_name: str) -> bool:
    """
    Check if a model supports system messages in its chat template.
    Some models (e.g., Gemma, some Mistral variants) only accept user/assistant roles.
    
    Args:
        model_name: The model name/ID from the server
        
    Returns:
        True if model supports system messages, False otherwise
    """
    model_lower = model_name.lower()
    # Gemma models don't support system messages
    if 'gemma' in model_lower:
        return False
    # Some Mistral variants may not support system messages
    # Add specific checks here if needed
    # For now, assume Mistral supports it unless we know otherwise
    return True

def build_messages_array(prompt: str, model_name: str, system_content: str = None) -> List[Dict]:
    """
    Build messages array for API call, handling models that don't support system messages.
    
    Args:
        prompt: The user prompt
        model_name: The model name/ID
        system_content: Optional system message content
        
    Returns:
        List of message dictionaries
    """
    if system_content is None:
        system_content = "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."
    
    if model_supports_system_messages(model_name):
        # Model supports system messages - use separate system role
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    else:
        # Model doesn't support system messages - merge into user message
        combined_prompt = f"{system_content}\n\n{prompt}"
        return [
            {"role": "user", "content": combined_prompt}
        ]

async def call_vllm_api_async(session: aiohttp.ClientSession, prompt: str, 
                              base_url: str = VLLM_BASE_URL, 
                              model_name: str = VLLM_MODEL_NAME) -> Tuple[str, Dict]:
    """
    Async call to vLLM API using OpenAI-compatible endpoint
    Optimized for continuous batching - vLLM will batch concurrent requests automatically
    
    Args:
        session: aiohttp ClientSession for async requests
        prompt: The prompt to send
        base_url: Base URL for the API (default: localhost:8000)
        model_name: Model name as served by vLLM (default: qwen2.5-3b)
    
    Returns:
        Tuple of (response_text, token_usage_dict) where token_usage_dict contains:
        {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            # Try to extract JSON from response (in case there's extra text)
            content_clean = content.strip()
            # Remove markdown code blocks if present
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            # Validate against schema
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    # Build messages array (handles models that don't support system messages)
    messages = build_messages_array(prompt, model_name)
    
    # Try with JSON schema enforcement first
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    # Fallback payload without schema (if API doesn't support it)
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    # Helper to extract token usage from response
    def extract_token_usage(result: Dict) -> Dict:
        """Extract token usage from vLLM API response"""
        if 'usage' in result:
            usage = result['usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Try with schema first
    try:
        async with session.post(url, json=payload_with_schema, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
            content = result['choices'][0]['message']['content'].strip()
            parsed = parse_and_validate_response(content)
            token_usage = extract_token_usage(result)
            if parsed != "error":
                return parsed, token_usage
    except aiohttp.ClientResponseError as e:
        # If 400 error, might be unsupported response_format, try without
        if e.status == 400:
            try:
                async with session.post(url, json=payload_without_schema, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    parsed = parse_and_validate_response(content)
                    token_usage = extract_token_usage(result)
                    return parsed, token_usage
            except Exception:
                return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        else:
            return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    except Exception:
        return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

async def call_vllm_api_async_with_debug(session: aiohttp.ClientSession, prompt: str, 
                                         base_url: str = VLLM_BASE_URL, 
                                         model_name: str = VLLM_MODEL_NAME,
                                         debug: bool = False) -> Tuple[str, Dict, str]:
    """
    Async call to vLLM API with debug support - returns raw content for inspection
    
    Returns:
        Tuple of (parsed_response, token_usage_dict, raw_content)
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    messages = build_messages_array(prompt, model_name)
    
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    def extract_token_usage(result: Dict) -> Dict:
        """Extract token usage from vLLM API response"""
        if 'usage' in result:
            usage = result['usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Try with schema first
    try:
        async with session.post(url, json=payload_with_schema, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
            content = result['choices'][0]['message']['content'].strip()
            raw_content = content  # Store raw content
            parsed = parse_and_validate_response(content)
            token_usage = extract_token_usage(result)
            if parsed != "error":
                return parsed, token_usage, raw_content
    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            try:
                async with session.post(url, json=payload_without_schema, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    raw_content = content  # Store raw content
                    parsed = parse_and_validate_response(content)
                    token_usage = extract_token_usage(result)
                    return parsed, token_usage, raw_content
            except Exception:
                return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
        else:
            return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
    except Exception:
        return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
    
    return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"

def call_vllm_api(prompt: str, base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME) -> str:
    """
    Call vLLM API using OpenAI-compatible endpoint
    
    Args:
        prompt: The prompt to send
        base_url: Base URL for the API (default: localhost:8000)
        model_name: Model name as served by vLLM (default: qwen2.5-3b)
    
    Returns:
        Response text from the model
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Build messages array (handles models that don't support system messages)
    messages = build_messages_array(prompt, model_name)
    
    # Try with JSON schema enforcement first
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    # Fallback payload without schema (if API doesn't support it)
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            # Try to extract JSON from response (in case there's extra text)
            content_clean = content.strip()
            # Remove markdown code blocks if present
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            # Validate against schema
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    # Try with schema first
    try:
        response = requests.post(url, json=payload_with_schema, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        parsed = parse_and_validate_response(content)
        if parsed != "error":
            return parsed
    except requests.exceptions.HTTPError as e:
        # If 400 error, might be unsupported response_format, try without
        if e.response.status_code == 400:
            try:
                # Try without schema
                response = requests.post(url, json=payload_without_schema, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return parse_and_validate_response(content)
            except Exception as e2:
                print(f"Error calling vLLM API (fallback): {e2}")
                return "error"
        else:
            print(f"HTTP error calling vLLM API: {e}")
            return "error"
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return "error"
    
    return "error"

async def check_relevance_vllm_async(session: aiohttp.ClientSession, statement: str, question_id: str, 
                                    question_info: Dict, base_url: str = VLLM_BASE_URL, 
                                    model_name: str = VLLM_MODEL_NAME) -> Tuple[str, str, Dict]:
    """
    First pass: Check if statement is relevant to question (vLLM async version)
    
    Returns:
        Tuple of (question_id, response, token_usage) where response is 'yes', 'no', or 'error'
    """
    prompt = create_relevance_prompt(statement, question_info)
    response, token_usage = await call_vllm_api_async(session, prompt, base_url, model_name)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes', token_usage)
    elif 'no' in response or response == 'n':
        return (question_id, 'no', token_usage)
    else:
        return (question_id, 'error', token_usage)

async def check_relevance_grouped_vllm_async(session: aiohttp.ClientSession, statement: str, 
                                            grouped_questions: List[Tuple[str, Dict]], 
                                            base_url: str = VLLM_BASE_URL, 
                                            model_name: str = VLLM_MODEL_NAME) -> Dict[str, Tuple[str, Dict]]:
    """
    First pass: Check if statement is relevant to a group of related questions (vLLM async version).
    Returns the same relevance result for all questions in the group.
    
    Returns:
        Dictionary mapping question_id -> (response, token_usage) where response is 'yes'/'no'/'error'
    """
    prompt = create_relevance_prompt_grouped(statement, grouped_questions)
    response, token_usage = await call_vllm_api_async(session, prompt, base_url, model_name)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        result = 'yes'
    elif 'no' in response or response == 'n':
        result = 'no'
    else:
        result = 'error'
    
    # Return same result for all questions in the group
    return {question_id: (result, token_usage) for question_id, _ in grouped_questions}

async def classify_statement_against_question_vllm_async(session: aiohttp.ClientSession, statement: str, 
                                                         question_id: str, question_info: Dict, 
                                                         base_url: str = VLLM_BASE_URL, 
                                                         model_name: str = VLLM_MODEL_NAME) -> Tuple[str, str, Dict]:
    """
    Second pass: Classify a relevant statement against a survey question (vLLM async version)
    
    Returns:
        Tuple of (question_id, response, token_usage) where response is 'yes', 'no', or 'error'
    """
    prompt = create_classification_prompt(statement, question_info)
    
    # DEBUG: Print prompt and question for specific cases
    debug_this = "takes up way too much farmland" in statement and ("Space_Too_Much" in question_id or "Space_Acceptable" in question_id)
    if debug_this:
        print(f"\n{'='*80}")
        print(f"DEBUG - Question ID: {question_id}")
        print(f"DEBUG - Question: {question_info['question']}")
        print(f"DEBUG - Question Description: {question_info.get('description', 'N/A')}")
        print(f"DEBUG - Statement: {statement}")
        print(f"DEBUG - Prompt:\n{prompt}")
        print(f"{'='*80}\n")
    
    # Get response and raw content for debugging
    response, token_usage, raw_content = await call_vllm_api_async_with_debug(session, prompt, base_url, model_name, debug_this)
    
    # DEBUG: Print response for specific cases
    if debug_this:
        print(f"DEBUG - Raw API Response: {raw_content}")
        print(f"DEBUG - Parsed Response: {response}")
        print(f"{'='*80}\n")
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes', token_usage)
    elif 'no' in response or response == 'n':
        return (question_id, 'no', token_usage)
    else:
        return (question_id, 'error', token_usage)

async def check_relevance_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                         stmt_idx: int, statement: str, question_id: str, 
                                         question_info: Dict, base_url: str, model_name: str,
                                         pbar: tqdm, token_counter: Dict) -> Tuple[int, str, str]:
    """First pass: Check relevance of a single statement-question pair (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            _, response, token_usage = await check_relevance_vllm_async(session, statement, question_id, question_info, base_url, model_name)
            # Accumulate token counts
            token_counter['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
            token_counter['completion_tokens'] += token_usage.get('completion_tokens', 0)
            token_counter['total_tokens'] += token_usage.get('total_tokens', 0)
            pbar.update(1)
            return (stmt_idx, question_id, response)
        except Exception:
            pbar.update(1)
            return (stmt_idx, question_id, 'error')

async def check_relevance_grouped_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                                  stmt_idx: int, statement: str, grouped_questions: List[Tuple[str, Dict]], 
                                                  base_url: str, model_name: str,
                                                  pbar: tqdm, token_counter: Dict) -> List[Tuple[int, str, str]]:
    """First pass: Check relevance of a statement to a group of related questions (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            grouped_results = await check_relevance_grouped_vllm_async(session, statement, grouped_questions, base_url, model_name)
            # Accumulate token counts (use first result's token usage, they're all the same)
            if grouped_results:
                first_token_usage = next(iter(grouped_results.values()))[1]
                token_counter['prompt_tokens'] += first_token_usage.get('prompt_tokens', 0)
                token_counter['completion_tokens'] += first_token_usage.get('completion_tokens', 0)
                token_counter['total_tokens'] += first_token_usage.get('total_tokens', 0)
            
            # Return list of (stmt_idx, question_id, response) tuples
            results = [(stmt_idx, question_id, response) for question_id, (response, _) in grouped_results.items()]
            pbar.update(len(results))  # Update progress bar for all questions in group
            return results
        except Exception:
            # Mark all questions in group as error
            results = [(stmt_idx, question_id, 'error') for question_id, _ in grouped_questions]
            pbar.update(len(results))
            return results

async def classify_single_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                         stmt_idx: int, statement: str, question_id: str, 
                                         question_info: Dict, base_url: str, model_name: str,
                                         pbar: tqdm, token_counter: Dict) -> Tuple[int, str, str]:
    """Second pass: Classify a relevant statement-question pair (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            _, response, token_usage = await classify_statement_against_question_vllm_async(
                session, statement, question_id, question_info, base_url, model_name
            )
            # Accumulate token counts
            token_counter['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
            token_counter['completion_tokens'] += token_usage.get('completion_tokens', 0)
            token_counter['total_tokens'] += token_usage.get('total_tokens', 0)
            pbar.update(1)
            return (stmt_idx, question_id, response)
        except Exception:
            pbar.update(1)
            return (stmt_idx, question_id, 'error')

async def first_pass_relevance_vllm_async(statements: List[str], questions: Dict, 
                                         base_url: str = VLLM_BASE_URL, 
                                         model_name: str = VLLM_MODEL_NAME,
                                         max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> Tuple[Dict, Dict]:
    """
    First pass: Determine relevance of all statements to all questions (vLLM async version)
    Uses grouped questions to reduce API calls (e.g., Help/Hurt variants checked together).
    Uses async requests with semaphore for controlled concurrency.
    vLLM's continuous batching will automatically batch concurrent requests together.
    
    Returns:
        Tuple of (results_dict, token_usage_dict) where:
        - results_dict: {statement_index: {question_id: 'yes'/'no'/'error'}}
        - token_usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    results = {}
    
    # Group related questions by base topic
    question_groups = group_related_questions(questions)
    
    # Initialize results structure
    for stmt_idx in range(len(statements)):
        results[stmt_idx] = {}
    
    # Calculate total tasks for progress bar (still one per statement-question pair for display)
    total_tasks = len(statements) * len(questions)
    
    # Token counter (shared across tasks)
    token_counter = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Create semaphore to limit concurrent requests (allows vLLM to batch efficiently)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session with connection pooling for better performance
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    with tqdm(total=total_tasks, desc=f"{model_name[:30]:<30} [Pass 1: Relevance]", unit="task") as pbar:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks (one per statement-group pair)
            tasks = []
            for stmt_idx, statement in enumerate(statements):
                for grouped_questions in question_groups.values():
                    task = check_relevance_grouped_task_vllm_async(
                        session, semaphore, stmt_idx, statement, grouped_questions, 
                        base_url, model_name, pbar, token_counter
                    )
                    tasks.append(task)
            
            # Execute all tasks concurrently (vLLM will batch them)
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in task_results:
                if isinstance(result, Exception):
                    continue
                # Result is a list of (stmt_idx, question_id, response) tuples
                for stmt_idx, question_id, response in result:
                    results[stmt_idx][question_id] = response
    
    return results, token_counter

async def second_pass_classification_vllm_async(statements: List[str], questions: Dict, 
                                                relevance_results: Dict,
                                                base_url: str = VLLM_BASE_URL, 
                                                model_name: str = VLLM_MODEL_NAME,
                                                max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> Tuple[Dict, Dict]:
    """
    Second pass: Detailed classification only for statements marked as relevant (vLLM async version)
    Uses async requests with semaphore for controlled concurrency.
    vLLM's continuous batching will automatically batch concurrent requests together.
    
    Args:
        relevance_results: Results from first pass showing which statements are relevant to which questions
    
    Returns:
        Tuple of (results_dict, token_usage_dict) where:
        - results_dict: {statement_index: {question_id: 'yes'/'no'/'error'}}
        - token_usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    results = {}
    
    # Prepare tasks only for relevant pairs
    tasks = []
    for stmt_idx, statement in enumerate(statements):
        results[stmt_idx] = {}
        for question_id, question_info in questions.items():
            # Only process if marked as relevant in first pass
            if relevance_results.get(stmt_idx, {}).get(question_id) == 'yes':
                tasks.append((stmt_idx, statement, question_id, question_info))
            else:
                # Mark as not relevant (from first pass)
                results[stmt_idx][question_id] = 'no'
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        return results, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Token counter (shared across tasks)
    token_counter = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    with tqdm(total=total_tasks, desc=f"{model_name[:30]:<30} [Pass 2: Classification]", unit="task") as pbar:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks
            async_tasks = []
            for stmt_idx, statement, question_id, question_info in tasks:
                task = classify_single_task_vllm_async(
                    session, semaphore, stmt_idx, statement, question_id, 
                    question_info, base_url, model_name, pbar, token_counter
                )
                async_tasks.append(task)
            
            # Execute all tasks concurrently (vLLM will batch them)
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Process results
            for result in task_results:
                if isinstance(result, Exception):
                    continue
                stmt_idx, question_id, response = result
                results[stmt_idx][question_id] = response
    
    return results, token_counter

# Synchronous wrapper functions for backward compatibility
def first_pass_relevance_vllm(statements: List[str], questions: Dict, 
                              base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME,
                              max_workers: int = None) -> Tuple[Dict, Dict]:
    """
    Synchronous wrapper for async first pass (for backward compatibility)
    Note: For maximum throughput, use first_pass_relevance_vllm_async directly
    
    Returns:
        Tuple of (results_dict, token_usage_dict)
    """
    if max_workers is None:
        max_workers = MAX_CONCURRENT_REQUESTS
    return asyncio.run(first_pass_relevance_vllm_async(statements, questions, base_url, model_name, max_workers))

def second_pass_classification_vllm(statements: List[str], questions: Dict, relevance_results: Dict,
                                   base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME,
                                   max_workers: int = None) -> Tuple[Dict, Dict]:
    """
    Synchronous wrapper for async second pass (for backward compatibility)
    Note: For maximum throughput, use second_pass_classification_vllm_async directly
    
    Returns:
        Tuple of (results_dict, token_usage_dict)
    """
    if max_workers is None:
        max_workers = MAX_CONCURRENT_REQUESTS
    return asyncio.run(second_pass_classification_vllm_async(statements, questions, relevance_results, base_url, model_name, max_workers))

def get_model_id_from_server(base_url: str) -> str:
    """Get the actual model ID from the vLLM server"""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        if 'data' in models and len(models['data']) > 0:
            return models['data'][0]['id']
        return None
    except Exception as e:
        return None

def test_vllm_connection(base_url: str = None) -> bool:
    """Test if vLLM server is accessible and show actual model ID"""
    if base_url is None:
        base_url = VLLM_BASE_URL
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        actual_model_id = get_model_id_from_server(base_url)
        print(f"✓ vLLM server is accessible at {base_url}")
        if actual_model_id:
            print(f"  Actual model ID: {actual_model_id}")
        print(f"  Available models: {models}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to vLLM server at {base_url}")
        print(f"  Error: {e}")
        print(f"  Make sure the server is running on the specified port")
        return False

def list_available_models():
    """List all available vLLM models and verify model IDs from servers"""
    print("\n" + "="*80)
    print("AVAILABLE vLLM MODELS")
    print("="*80)
    for key, model_info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {model_info['description']}")
        print(f"      URL: {model_info['base_url']}")
        print(f"      Configured model name: {model_info['model_name']}")
        # Try to get actual model ID from server
        actual_id = get_model_id_from_server(model_info['base_url'])
        if actual_id:
            print(f"      Actual model ID from server: {actual_id}")
            if actual_id != model_info['model_name']:
                print(f"      ⚠ WARNING: Model name mismatch! Update config to use: {actual_id}")
        else:
            print(f"      ⚠ Could not connect to server to verify model ID")
    print("="*80 + "\n")

def select_model(model_key: str = None) -> Dict:
    """
    Select a model to use. If model_key is provided, use it. Otherwise, prompt user.
    
    Args:
        model_key: Optional model key (1, 2, 3, or 4). If None, prompts user.
    
    Returns:
        Dictionary with model configuration (base_url, model_name, etc.)
    """
    if model_key is None:
        # Interactive selection
        list_available_models()
        while True:
            choice = input("Select model (1-4) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            if choice in AVAILABLE_MODELS:
                selected = AVAILABLE_MODELS[choice]
                print(f"\nSelected: {selected['description']}")
                return selected
            else:
                print(f"Invalid choice. Please enter 1, 2, 3, 4, or 'q'.")
    else:
        # Use provided key
        if model_key in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_key]
        else:
            print(f"Error: Invalid model key '{model_key}'. Available keys: {list(AVAILABLE_MODELS.keys())}")
            list_available_models()
            return select_model()  # Prompt user

async def classify_real_comments_vllm_async(comments_by_sector: Dict[str, List[str]], 
                                           output_prefix: str = "real_comments_vllm", 
                                           base_url: str = VLLM_BASE_URL,
                                           model_name: str = VLLM_MODEL_NAME,
                                           max_concurrent: int = MAX_CONCURRENT_REQUESTS):
    """
    Classify real Reddit comments by sector using survey questions with vLLM (async version).
    Optimized for continuous batching - maximum throughput.
    
    Args:
        comments_by_sector: Dictionary with sector as key and list of comment strings as values
                           e.g., {"Food": ["comment1", "comment2"], "Housing": ["comment3"]}
        output_prefix: Prefix for output CSV files (default: "real_comments_vllm")
        base_url: Base URL for the vLLM API (default: VLLM_BASE_URL)
        model_name: Model name as served by vLLM (default: VLLM_MODEL_NAME)
        max_concurrent: Maximum concurrent requests (default: MAX_CONCURRENT_REQUESTS)
    
    Returns:
        Dictionary with results organized by sector
    """
    print("="*80)
    print("CLASSIFYING REAL REDDIT COMMENTS BY SECTOR (vLLM)")
    print("OPTIMIZED FOR CONTINUOUS BATCHING - MAXIMUM THROUGHPUT")
    print("="*80 + "\n")
    
    # Load survey questions organized by sector
    survey_by_sector = load_survey_questions_by_sector()
    if not survey_by_sector:
        print("Error: Could not load survey questions. Exiting.")
        return {}
    
    # Normalize sector names in input
    sector_mapping = {
        'food': 'Food',
        'transport': 'Transport',
        'housing': 'Housing',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    
    print(f"Processing with vLLM model: {model_name}")
    print(f"Server URL: {base_url}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Total comments by sector:")
    for sector, comments in comments_by_sector.items():
        print(f"  {sector}: {len(comments)} comments")
    print()
    
    # Store all results by sector
    all_sector_results = {}
    
    # Process each sector
    for sector, comments in comments_by_sector.items():
        # Normalize sector name
        normalized_sector = sector_mapping.get(sector.lower(), sector.capitalize())
        
        # Get questions for this sector ONLY (no cross-sector questions)
        sector_questions = get_questions_for_sector(normalized_sector)
        
        if not sector_questions:
            print(f"Warning: No questions found for sector '{sector}'. Skipping.")
            survey_by_sector = load_survey_questions_by_sector()
            print(f"Available sectors: {', '.join(survey_by_sector.keys())}")
            continue
        
        print("\n" + "="*80)
        print(f"SECTOR: {normalized_sector}")
        print(f"Comments: {len(comments)}")
        print(f"Questions: {len(sector_questions)}")
        print("="*80 + "\n")
        
        # ========== FIRST PASS: RELEVANCE CHECK ==========
        print("FIRST PASS: RELEVANCE CHECK\n")
        
        start_time = time.time()
        relevance_results, first_pass_tokens = await first_pass_relevance_vllm_async(
            comments, sector_questions, 
            base_url=base_url, model_name=model_name, 
            max_concurrent=max_concurrent
        )
        first_pass_time = time.time() - start_time
        
        print(f"\nFirst pass completed in {first_pass_time:.2f} seconds")
        print(f"Request throughput: {len(comments) * len(sector_questions) / first_pass_time:.2f} requests/second")
        print(f"Token usage:")
        print(f"  Prompt tokens: {first_pass_tokens['prompt_tokens']:,}")
        print(f"  Completion tokens: {first_pass_tokens['completion_tokens']:,}")
        print(f"  Total tokens: {first_pass_tokens['total_tokens']:,}")
        print(f"Token throughput:")
        print(f"  Output tokens: {first_pass_tokens['completion_tokens'] / first_pass_time:.2f} tokens/second")
        print(f"  Total tokens: {first_pass_tokens['total_tokens'] / first_pass_time:.2f} tokens/second")
        
        # Print relevance percentages
        all_relevance_results = {model_name: relevance_results}
        print_relevance_percentages(all_relevance_results, sector_questions, comments)
        
        # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
        print("="*80)
        print("SECOND PASS: DETAILED CLASSIFICATION (Only Relevant Comments)")
        print("="*80 + "\n")
        
        start_time = time.time()
        classification_results, second_pass_tokens = await second_pass_classification_vllm_async(
            comments, sector_questions, relevance_results,
            base_url=base_url, model_name=model_name, 
            max_concurrent=max_concurrent
        )
        second_pass_time = time.time() - start_time
        
        # Count relevant pairs for throughput calculation
        relevant_pairs = sum(1 for stmt_idx in relevance_results 
                           for q_id in relevance_results[stmt_idx] 
                           if relevance_results[stmt_idx][q_id] == 'yes')
        if relevant_pairs > 0:
            print(f"\nSecond pass completed in {second_pass_time:.2f} seconds")
            print(f"Request throughput: {relevant_pairs / second_pass_time:.2f} requests/second")
            print(f"Token usage:")
            print(f"  Prompt tokens: {second_pass_tokens['prompt_tokens']:,}")
            print(f"  Completion tokens: {second_pass_tokens['completion_tokens']:,}")
            print(f"  Total tokens: {second_pass_tokens['total_tokens']:,}")
            print(f"Token throughput:")
            print(f"  Output tokens: {second_pass_tokens['completion_tokens'] / second_pass_time:.2f} tokens/second")
            print(f"  Total tokens: {second_pass_tokens['total_tokens'] / second_pass_time:.2f} tokens/second")
        
        # Store final results (using same structure as sample_local_llm.py for compatibility)
        all_results = {model_name: classification_results}
        
        # Save results to CSV (no ground truth for real comments)
        output_filename = f"{output_prefix}_{normalized_sector.lower()}_results.csv"
        save_results_csv(all_results, comments, sector_questions, filename=output_filename, has_ground_truth=False)
        
        # Calculate and print metrics (no ground truth)
        metrics = calculate_metrics(all_results, comments, sector_questions, has_ground_truth=False)
        print_metrics(metrics)
        
        # Save metrics to CSV
        metrics_filename = f"{output_prefix}_{normalized_sector.lower()}_metrics.csv"
        save_metrics_csv(metrics, filename=metrics_filename)
        
        # Store results for this sector
        all_sector_results[normalized_sector] = {
            'results': all_results,
            'metrics': metrics,
            'comments': comments,
            'questions': sector_questions
        }
    
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE FOR ALL SECTORS (vLLM)")
    print("="*80 + "\n")
    
    return all_sector_results

def classify_real_comments_vllm(comments_by_sector: Dict[str, List[str]], 
                                output_prefix: str = "real_comments_vllm", 
                                base_url: str = VLLM_BASE_URL,
                                model_name: str = VLLM_MODEL_NAME,
                                max_concurrent: int = MAX_CONCURRENT_REQUESTS):
    """
    Synchronous wrapper for classify_real_comments_vllm_async
    Classify real Reddit comments by sector using survey questions with vLLM.
    
    Works in both regular Python scripts and Jupyter notebooks.
    In notebooks, use await classify_real_comments_vllm_async() directly for better performance.
    
    Args:
        comments_by_sector: Dictionary with sector as key and list of comment strings as values
                           e.g., {"Food": ["comment1", "comment2"], "Housing": ["comment3"]}
        output_prefix: Prefix for output CSV files (default: "real_comments_vllm")
        base_url: Base URL for the vLLM API (default: VLLM_BASE_URL)
        model_name: Model name as served by vLLM (default: VLLM_MODEL_NAME)
        max_concurrent: Maximum concurrent requests (default: MAX_CONCURRENT_REQUESTS)
    
    Returns:
        Dictionary with results organized by sector
    """
    try:
        # Check if we're in a running event loop (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
        # If we're in a notebook, try to use nest_asyncio if available
        try:
            import nest_asyncio
            nest_asyncio.apply()
            # Now we can use asyncio.run() even in a running loop
            return asyncio.run(classify_real_comments_vllm_async(
                comments_by_sector, output_prefix, base_url, model_name, max_concurrent
            ))
        except ImportError:
            # nest_asyncio not available, provide helpful error
            raise RuntimeError(
                "Cannot use classify_real_comments_vllm() in a running event loop (Jupyter notebook).\n"
                "Option 1: Install nest_asyncio and apply it:\n"
                "  pip install nest_asyncio\n"
                "  import nest_asyncio\n"
                "  nest_asyncio.apply()\n"
                "  results = classify_real_comments_vllm(sample_comments)\n\n"
                "Option 2: Use the async version directly:\n"
                "  results = await classify_real_comments_vllm_async(sample_comments)"
            )
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or "get_running_loop" in str(e):
            # This is the specific error we're trying to handle
            raise
        # No running loop, safe to use asyncio.run()
        return asyncio.run(classify_real_comments_vllm_async(
            comments_by_sector, output_prefix, base_url, model_name, max_concurrent
        ))

async def main_async(model_config: Dict = None, use_real_data: bool = False):
    """Async main function to run the two-pass classification with vLLM (optimized for continuous batching)"""
    global SURVEY_QUESTIONS
    
    # Get model configuration
    if model_config is None:
        model_config = {"base_url": VLLM_BASE_URL, "model_name": VLLM_MODEL_NAME}
    
    base_url = model_config["base_url"]
    model_name = model_config["model_name"]
    
    # Reload questions if needed
    if use_real_data:
        SURVEY_QUESTIONS = load_survey_questions(use_real_data=True)
    
    print("="*80)
    print("vLLM SERVER SURVEY QUESTION CLASSIFIER (TWO-PASS APPROACH)")
    print("OPTIMIZED FOR CONTINUOUS BATCHING - MAXIMUM THROUGHPUT")
    print("="*80)
    if use_real_data:
        print(f"Using real survey data from survey_question.json")
    else:
        print(f"Survey: Pew Research - Americans' views on local wind and solar power development")
        print(f"URL: https://www.pewresearch.org/science/2024/06/27/americans-views-on-local-wind-and-solar-power-development/")
    print("="*80 + "\n")
    
    # Test connection to vLLM server
    if not test_vllm_connection(base_url):
        return
    
    print(f"\nProcessing with vLLM model: {model_name}")
    print(f"Server URL: {base_url}")
    print(f"Total questions: {len(SURVEY_QUESTIONS)}")
    print(f"Original statements: {len(SAMPLE_STATEMENTS)}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"vLLM will automatically batch concurrent requests for optimal throughput\n")
    
    # Resample SAMPLE_STATEMENTS from 10 to 1000 for scaling test
    TARGET_SAMPLE_SIZE = 10
    original_count = len(SAMPLE_STATEMENTS)
    
    # Create mapping from resampled index to original index for ground truth lookup
    resampled_to_original = {}
    
    if original_count < TARGET_SAMPLE_SIZE:
        # Resample with replacement to reach target
        resampled_statements = []
        for i in range(TARGET_SAMPLE_SIZE):
            original_idx = random.randint(0, original_count - 1)
            resampled_statements.append(SAMPLE_STATEMENTS[original_idx])
            resampled_to_original[i] = original_idx
        print(f"Resampling statements: {original_count} -> {TARGET_SAMPLE_SIZE} (with replacement)")
        print(f"  Note: Some statements will appear multiple times to test scaling\n")
    else:
        # Sample without replacement if we have enough
        original_indices = list(range(original_count))
        sampled_indices = random.sample(original_indices, TARGET_SAMPLE_SIZE)
        resampled_statements = [SAMPLE_STATEMENTS[idx] for idx in sampled_indices]
        resampled_to_original = {i: idx for i, idx in enumerate(sampled_indices)}
        print(f"Sampling statements: {original_count} -> {TARGET_SAMPLE_SIZE} (without replacement)\n")
    
    # Store relevance results from first pass
    all_relevance_results = {}
    
    # ========== FIRST PASS: RELEVANCE CHECK ==========
    print("="*80)
    print("FIRST PASS: RELEVANCE CHECK")
    print("="*80 + "\n")
    
    start_time = time.time()
    relevance_results, first_pass_tokens = await first_pass_relevance_vllm_async(
        resampled_statements, SURVEY_QUESTIONS, 
        base_url=base_url, model_name=model_name, 
        max_concurrent=MAX_CONCURRENT_REQUESTS
    )
    first_pass_time = time.time() - start_time
    all_relevance_results[model_name] = relevance_results
    
    total_first_pass_tasks = len(resampled_statements) * len(SURVEY_QUESTIONS)
    print(f"\nFirst pass completed in {first_pass_time:.2f} seconds")
    print(f"Request throughput: {total_first_pass_tasks / first_pass_time:.2f} requests/second")
    print(f"Token usage:")
    print(f"  Prompt tokens: {first_pass_tokens['prompt_tokens']:,}")
    print(f"  Completion tokens: {first_pass_tokens['completion_tokens']:,}")
    print(f"  Total tokens: {first_pass_tokens['total_tokens']:,}")
    print(f"Token throughput:")
    print(f"  Output tokens: {first_pass_tokens['completion_tokens'] / first_pass_time:.2f} tokens/second")
    print(f"  Total tokens: {first_pass_tokens['total_tokens'] / first_pass_time:.2f} tokens/second")
    
    # Print relevance percentages
    print_relevance_percentages(all_relevance_results, SURVEY_QUESTIONS, resampled_statements)
    
    # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
    print("="*80)
    print("SECOND PASS: DETAILED CLASSIFICATION (Only Relevant Statements)")
    print("="*80 + "\n")
    
    # Second pass: Detailed classification only for relevant statements
    start_time = time.time()
    classification_results, second_pass_tokens = await second_pass_classification_vllm_async(
        resampled_statements, SURVEY_QUESTIONS, relevance_results,
        base_url=base_url, model_name=model_name, 
        max_concurrent=MAX_CONCURRENT_REQUESTS
    )
    second_pass_time = time.time() - start_time
    
    # Count relevant pairs for throughput calculation
    relevant_pairs = sum(1 for stmt_idx in relevance_results 
                       for q_id in relevance_results[stmt_idx] 
                       if relevance_results[stmt_idx][q_id] == 'yes')
    if relevant_pairs > 0:
        print(f"\nSecond pass completed in {second_pass_time:.2f} seconds")
        print(f"Request throughput: {relevant_pairs / second_pass_time:.2f} requests/second")
        print(f"Token usage:")
        print(f"  Prompt tokens: {second_pass_tokens['prompt_tokens']:,}")
        print(f"  Completion tokens: {second_pass_tokens['completion_tokens']:,}")
        print(f"  Total tokens: {second_pass_tokens['total_tokens']:,}")
        print(f"Token throughput:")
        print(f"  Output tokens: {second_pass_tokens['completion_tokens'] / second_pass_time:.2f} tokens/second")
        print(f"  Total tokens: {second_pass_tokens['total_tokens'] / second_pass_time:.2f} tokens/second")
    
    # Store final results (using same structure as sample_local_llm.py for compatibility)
    all_results = {model_name: classification_results}
    
    # Save results to CSV (using resampled statements)
    # Pass mapping so ground truth can be looked up correctly for resampled statements
    save_results_csv(all_results, resampled_statements, SURVEY_QUESTIONS, 
                    filename="vllm_test_results.csv", has_ground_truth=not use_real_data,
                    statement_to_original_index=resampled_to_original if not use_real_data else None)
    
    # Calculate and print metrics (using resampled statements)
    # Note: Ground truth won't match perfectly due to resampling, but metrics still useful for throughput
    metrics = calculate_metrics(all_results, resampled_statements, SURVEY_QUESTIONS, 
                               has_ground_truth=False)  # Set to False since resampled statements won't match ground truth indices
    print_metrics(metrics)
    
    # Save metrics to CSV
    save_metrics_csv(metrics, filename="vllm_test_metrics.csv")
    
    total_time = first_pass_time + second_pass_time
    total_tokens = first_pass_tokens['total_tokens'] + second_pass_tokens['total_tokens']
    total_output_tokens = first_pass_tokens['completion_tokens'] + second_pass_tokens['completion_tokens']
    
    print(f"\n{'='*80}")
    print(f"TOTAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total requests: {total_first_pass_tasks + relevant_pairs}")
    print(f"Average request throughput: {(total_first_pass_tasks + relevant_pairs) / total_time:.2f} requests/second")
    print(f"\nTotal token usage:")
    print(f"  Prompt tokens: {first_pass_tokens['prompt_tokens'] + second_pass_tokens['prompt_tokens']:,}")
    print(f"  Completion tokens: {total_output_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"\nAverage token throughput:")
    print(f"  Output tokens: {total_output_tokens / total_time:.2f} tokens/second")
    print(f"  Total tokens: {total_tokens / total_time:.2f} tokens/second")
    print(f"{'='*80}\n")
    
    print("Classification complete with vLLM (continuous batching optimized)!")

def main():
    """Main function wrapper (runs async version)"""
    # Check for model selection argument
    model_key = None
    if len(sys.argv) > 1:
        # Check if first arg is a model number (1-4)
        if sys.argv[1] in AVAILABLE_MODELS:
            model_key = sys.argv[1]
        # Check for --model argument
        elif len(sys.argv) > 2 and sys.argv[1] == "--model":
            model_key = sys.argv[2]
    
    # Select model (interactive if no key provided)
    model_config = select_model(model_key)
    
    # Update global defaults for this run
    global VLLM_BASE_URL, VLLM_MODEL_NAME
    VLLM_BASE_URL = model_config["base_url"]
    VLLM_MODEL_NAME = model_config["model_name"]
    
    asyncio.run(main_async(model_config))

def process_reddit_data_and_classify(model_config: Dict = None):
    """
    Process Reddit data from CSV files and classify comments by sector using vLLM.
    This function loads the filtered CSV files, maps keywords to sectors, and runs classification.
    
    Args:
        model_config: Optional model configuration dict. If None, uses global defaults.
    """
    if model_config is None:
        model_config = {"base_url": VLLM_BASE_URL, "model_name": VLLM_MODEL_NAME}
    print("="*80)
    print("PROCESSING REDDIT DATA AND CLASSIFYING WITH vLLM")
    print("="*80 + "\n")
    
    # ========== STEP 1: Define sector keywords ==========
    print("Step 1: Defining sector keywords...")
    sector_keyword_strength = {
        'transport_strong': [
            "electric vehicle", "evs", "bev", "battery electric", "battery-electric vehicle",
            "tesla model", "model 3", "model y", "chevy bolt", "nissan leaf",
            "ioniq 5", "mustang mach-e", "id.4", "rivian", "lucid air",
            "supercharger", "gigafactory", "zero emission vehicle", "zero-emission vehicle",
            "pure electric", "all-electric", "fully electric", "100% electric",
            "electric powertrain", "electric drivetrain", "electric motor vehicle",
            "level 2 charger", "dc fast charger", "public charger", "home charger",
            "charging network", "range anxiety", "mpge",
            "bike lane", "protected cycleway", "car-free", "low emission zone"
        ],
        'transport_weak': [
            "electric car", "electric truck", "electric suv", "plug-in hybrid",
            "phev", "charging station", "charge point", "kw charger", 
            "battery swap", "solid-state battery", "gigacast",
            "tax credit", "zev mandate", "ev rebate", "phase-out ice",
            "e-bike", "micro-mobility", "last-mile delivery", "transit electrification",
            "tesla", "spacex launch price?", "elon says",
            "rail electrification", "hydrogen truck", "low carbon transport"
        ],
        'housing_strong': [
            "rooftop solar", "solar pv", "pv panel", "photovoltaics",
            "solar array", "net metering", "feed-in tariff", "solar inverter",
            "kwh generated", "solar roof", "sunrun", "sunpower",
            r"solar\s+panel(s)?", r"solar\s+pv", r"rooftop\s+solar",
            r"solar\s+power", r"photovoltaic(s)?"
        ],
        'housing_weak': [
            "solar panels", "solar power", "solar installer",
            "battery storage", "powerwall", "home battery", "smart thermostat",
            "energy audit", "energy efficiency upgrade", "led retrofit",
            "green home", "net-zero house", "zero-energy building",
            "solar tax credit", "pvgis", "renewable portfolio standard",
            "community solar", "virtual power plant", "rooftop rebate"
        ],
        'food_strong': [
            "vegan", "plant-based diet", "veganism", "veganuary", "vegetarian", "veg lifestyle",
            "carnivore diet", "meat lover", "steakhouse", "barbecue festival",
            "bacon double", "grass-fed beef", "factory farming",
            "meatless monday", "beyond meat", "impossible burger",
            "plant-based burger", "animal cruelty free"
        ],
        'food_weak': [
            "red meat", "beef consumption", "dairy free", "plant protein",
            "soy burger", "nutritional yeast", "seitan", "tofurky",
            "agricultural emissions", "methane footprint", "carbon hoofprint",
            "cow burps", "livestock emissions", "feedlot",
            "recipe vegan", "tofu scramble", "almond milk", "oat milk",
            "flexitarian", "climatetarian",
            "cultivated meat", "lab-grown meat", "precision fermentation"
        ]
    }
    
    # Combine all keywords for regex search
    sector_keywords = (
        sector_keyword_strength['transport_strong'] +
        sector_keyword_strength['transport_weak'] +
        sector_keyword_strength['housing_strong'] +
        sector_keyword_strength['housing_weak'] +
        sector_keyword_strength['food_strong'] +
        sector_keyword_strength['food_weak']
    )
    
    # ========== STEP 2: Load filtered CSV files ==========
    print("Step 2: Loading filtered CSV files...")
    output_dir = os.path.join('paper4data', 'subreddit_filtered_by_regex')
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found. Please run the filtering step first.")
        return None
    
    filtered_csv_paths = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith('.csv')
    ]
    
    if not filtered_csv_paths:
        print(f"Error: No CSV files found in '{output_dir}'")
        return None
    
    print(f"Found {len(filtered_csv_paths)} CSV files")
    
    # Load all filtered CSVs
    all_filtered = []
    for path in filtered_csv_paths:
        try:
            df = pd.read_csv(path)
            all_filtered.append(df)
        except Exception as e:
            print(f"Error loading filtered csv {path}: {e}")
    
    if not all_filtered:
        print("Error: No data loaded from CSV files")
        return None
    
    df_all_filtered = pd.concat(all_filtered, ignore_index=True)
    print(f"Loaded {len(df_all_filtered)} total rows")
    
    # Fill empty/null body with title
    if 'body' in df_all_filtered.columns and 'title' in df_all_filtered.columns:
        mask = df_all_filtered['body'].isna() | (df_all_filtered['body'] == '') | (df_all_filtered['body'].astype(str).str.strip() == '')
        count_to_fill = mask.sum()
        df_all_filtered.loc[mask, 'body'] = df_all_filtered.loc[mask, 'title']
        if count_to_fill > 0:
            print(f"Filled {count_to_fill} empty/null body values with title")
    
    # Reconstruct keyword_to_comments
    keyword_to_comments = {}
    if not df_all_filtered.empty:
        for kw, group in df_all_filtered.groupby('matched_keyword'):
            keyword_to_comments[kw] = set(group['id'].astype(str))
    
    # Print overall stats
    if not df_all_filtered.empty:
        keyword_counts = df_all_filtered.groupby('matched_keyword')['id'].nunique().sort_values(ascending=False)
        print("\nTotal unique comments/submissions matched per keyword:")
        for kw, count in keyword_counts.head(10).items():  # Show top 10
            print(f"  {kw}: {count}")
        print(f"  ... and {len(keyword_counts) - 10} more keywords")
    
    # ========== STEP 3: Create sector mapping ==========
    print("\nStep 3: Creating sector mapping from matched_keyword...")
    
    # Create reverse mapping: keyword -> sector
    keyword_to_sector = {}
    
    # Map transport keywords
    for kw in sector_keyword_strength['transport_strong'] + sector_keyword_strength['transport_weak']:
        keyword_to_sector[kw] = 'transport'
    
    # Map housing keywords
    for kw in sector_keyword_strength['housing_strong'] + sector_keyword_strength['housing_weak']:
        keyword_to_sector[kw] = 'housing'
    
    # Map food keywords
    for kw in sector_keyword_strength['food_strong'] + sector_keyword_strength['food_weak']:
        keyword_to_sector[kw] = 'food'
    
    # Add sector column to df_all_filtered based on matched_keyword
    if not df_all_filtered.empty and 'matched_keyword' in df_all_filtered.columns:
        df_all_filtered['sector'] = df_all_filtered['matched_keyword'].map(keyword_to_sector)
        print(f"Added 'sector' column to df_all_filtered")
        print(f"Sector distribution:")
        print(df_all_filtered['sector'].value_counts())
    else:
        print("Warning: df_all_filtered is empty or missing 'matched_keyword' column")
        return None
    
    # ========== STEP 4: Prepare comments by sector ==========
    print("\nStep 4: Preparing comments by sector...")
    comments_by_sector = {}
    
    for sector in df_all_filtered['sector'].dropna().unique():
        sector_comments = df_all_filtered[df_all_filtered['sector'] == sector]['body'].dropna().tolist()
        # Filter out empty strings
        sector_comments = [c for c in sector_comments if c and str(c).strip()]
        if sector_comments:
            comments_by_sector[sector] = sector_comments
    
    print(f"Comments by sector:")
    for sector, comments in comments_by_sector.items():
        print(f"  {sector}: {len(comments)} comments")
    
    # ========== STEP 5: Sample comments per sector ==========
    print("\nStep 5: Sampling comments per sector...")
    sample_comments = {}
    for sector, comments in comments_by_sector.items():
        n_samples = min(100, len(comments))
        sampled_comments = random.sample(comments, n_samples) if len(comments) > n_samples else comments
        sample_comments[sector] = sampled_comments
    
    print(f"Sampled comments by sector:")
    for sector, comments in sample_comments.items():
        print(f"  {sector}: {len(comments)} comments")
    
    # ========== STEP 6: Classify with vLLM ==========
    print("\n" + "="*80)
    print("STEP 6: CLASSIFYING WITH vLLM")
    print("="*80)
    print("This will:")
    print("  - Food comments -> only Food survey questions")
    print("  - Transport comments -> only Transport survey questions")
    print("  - Housing comments -> only Housing survey questions")
    print("  NO CROSS-SECTOR CLASSIFICATION")
    print("  Uses vLLM server with continuous batching for maximum throughput")
    print("="*80 + "\n")
    
    # Test connection first
    if not test_vllm_connection(model_config["base_url"]):
        print("Cannot proceed without vLLM server connection")
        return None
    
    # Run classification
    results = asyncio.run(classify_real_comments_vllm_async(
        sample_comments,
        base_url=model_config["base_url"],
        model_name=model_config["model_name"]
    ))
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    args = sys.argv[1:]
    model_key = None
    process_reddit = False
    use_real_data = False
    
    # Check for help
    if "--help" in args or "-h" in args:
        print("\n" + "="*80)
        print("vLLM TEST - USAGE")
        print("="*80)
        print("\nUsage:")
        print("  python vllm_test.py [model_number] [options]")
        print("\nModel Selection:")
        print("  [1] Qwen2.5-3B-Instruct (port 8000)")
        print("  [2] Qwen3 1.7B (port 8001)")
        print("  [3] Gemma 2 2B Instruct (port 8002)")
        print("  [4] Mistral MiniStral 3 3B (port 8003)")
        print("\nOptions:")
        print("  --process-reddit    Process Reddit data from CSV files")
        print("  --run-real-data     Use real survey questions from survey_question.json")
        print("  --model N           Select model by number (1-4)")
        print("  --help, -h          Show this help message")
        print("\nExamples:")
        print("  python vllm_test.py                    # Interactive model selection")
        print("  python vllm_test.py 1                  # Use model 1 (Qwen2.5-3B)")
        print("  python vllm_test.py 2 --process-reddit  # Use model 2, process Reddit data")
        print("  python vllm_test.py --model 3          # Use model 3 (Gemma 2)")
        print("="*80 + "\n")
        sys.exit(0)
    
    # Check for model selection
    if args and args[0] in AVAILABLE_MODELS:
        model_key = args[0]
        args = args[1:]
    elif len(args) > 1 and args[0] == "--model":
        model_key = args[1]
        args = args[2:]
    
    # Check for other flags
    if "--process-reddit" in args:
        process_reddit = True
    if "--run-real-data" in args or "-run_real_data" in args:
        use_real_data = True
    
    # Select model (interactive if no key provided)
    model_config = select_model(model_key)
    
    if process_reddit:
        # Process Reddit data and classify
        results = process_reddit_data_and_classify(model_config)
        if results:
            print("\n" + "="*80)
            print("PROCESSING COMPLETE")
            print("="*80)
            print("Results saved to CSV files with prefix 'real_comments_vllm_'")
    else:
        # Default: run with sample statements
        asyncio.run(main_async(model_config, use_real_data))
