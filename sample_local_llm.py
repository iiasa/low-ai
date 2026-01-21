"""
Sample Local LLM Classifier for Pew Research Survey Questions
Classifies statements about wind and solar power into relevant survey questions
Based on: https://www.pewresearch.org/science/2024/06/27/americans-views-on-local-wind-and-solar-power-development/
"""

import json
import requests
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
import jsonschema
from jsonschema import validate, ValidationError
import concurrent.futures
from functools import partial
from collections import Counter
import sys
import os

# Load API configuration
with open('paper4_LOCAL_LLMS_api.json', 'r') as f:
    api_config = json.load(f)

def get_default_questions() -> Dict:
    """Return default hardcoded survey questions"""
    return {
    "Local_Economy_Help": {
        "question": "Would installing a wind or solar power development in your community help your local economy?",
        "description": "Belief that nearby wind/solar project would help local economy. Cues: 'help local economy', 'jobs', 'boost', 'economic growth'."
    },
    "Local_Economy_Hurt": {
        "question": "Would installing a wind or solar power development in your community hurt your local economy?",
        "description": "Belief that nearby wind/solar project would hurt local economy. Cues: 'hurt local economy', 'harm', 'negative impact', 'economic decline'."
    },
    "Local_Economy_No_Difference": {
        "question": "Would installing a wind or solar power development in your community make no difference to your local economy?",
        "description": "Belief that nearby wind/solar project would make no difference to local economy. Cues: 'make no difference', 'no impact', 'neutral'."
    },
    "Landscape_Unattractive": {
        "question": "Would installing a wind or solar power development in your community make the landscape unattractive?",
        "description": "Belief that project would make landscape unattractive. Cues: 'ugly', 'ruin the view', 'eyesore', 'unattractive', 'aesthetic harm'."
    },
    "Landscape_Not_Unattractive": {
        "question": "Would installing a wind or solar power development in your community NOT make the landscape unattractive?",
        "description": "Belief that project would NOT harm landscape aesthetics. Cues: 'would not make unattractive', 'beautiful', 'aesthetic benefit'."
    },
    "Space_Too_Much": {
        "question": "Would installing a wind or solar power development in your community take up too much space?",
        "description": "Concern that project takes too much space. Cues: 'take up too much space', 'footprint', 'land use', 'too large'."
    },
    "Space_Acceptable": {
        "question": "Would installing a wind or solar power development in your community NOT take up too much space?",
        "description": "Belief that space taken is acceptable. Cues: 'would not take too much space', 'reasonable footprint', 'acceptable size'."
    },
    "Utility_Bill_Lower": {
        "question": "Would installing a wind or solar power development in your community lower the price you pay for electricity?",
        "description": "Expectation that local renewables will lower electricity bills. Cues: 'lower my bill', 'cheaper power', 'reduce costs', 'savings'."
    },
    "Utility_Bill_Higher": {
        "question": "Would installing a wind or solar power development in your community raise the price you pay for electricity?",
        "description": "Expectation that local renewables will raise electricity bills. Cues: 'higher bills', 'more expensive power', 'cost increase'."
    },
    "Tax_Revenue_Help": {
        "question": "Would installing a wind or solar power development in your community help local tax revenue?",
        "description": "Belief that project will boost local tax revenue. Cues: 'tax revenue', 'municipal income', 'local taxes', 'revenue boost'."
    }
}

def load_survey_questions(use_real_data: bool = False) -> Dict:
    """Load survey questions from JSON file or use default hardcoded questions"""
    if use_real_data:
        # Load from survey_question.json
        if not os.path.exists('survey_question.json'):
            print("Error: survey_question.json not found. Using default questions.")
            return get_default_questions()
        
        with open('survey_question.json', 'r', encoding='utf-8') as f:
            survey_data = json.load(f)
        
        # Flatten the nested structure (Food, Housing, Energy) into a single dict
        flattened = {}
        for sector, questions in survey_data.items():
            for question_id, question_info in questions.items():
                # Keep the question_id as-is, or prefix with sector if needed
                flattened[question_id] = question_info
        
        print(f"Loaded {len(flattened)} questions from survey_question.json")
        print(f"Sectors: {', '.join(survey_data.keys())}")
        return flattened
    else:
        return get_default_questions()

# Load survey questions (default - will be reloaded in main if flag is set)
SURVEY_QUESTIONS = load_survey_questions(use_real_data=False)

# Generate 10 sample statements about wind and solar power
SAMPLE_STATEMENTS = [
    "Installing a solar farm in our town would create hundreds of local jobs and boost our economy.",
    "Those wind turbines are an eyesore and completely ruin the beautiful countryside view.",
    "I'm not sure if a solar panel farm would really help or hurt our local economy - it's hard to say.",
    "The solar development takes up way too much farmland that we need for agriculture.",
    "Having renewable energy nearby would definitely lower my electricity bills, which is great.",
    "I'm worried that building a wind farm will increase our electricity costs because of infrastructure expenses.",
    "The new solar panels on the community center look great and don't detract from the area at all.",
    "A wind farm would bring in significant tax revenue for our county, helping fund schools and roads.",
    "I don't think a solar development would make any real difference to our local economy one way or another.",
    "The space used for the solar farm is reasonable compared to the benefits we'll get from clean energy."
]

# Ground truth labels: {statement_index: {question_id: 'yes'/'no'}}
GROUND_TRUTH = {
    0: {  # "Installing a solar farm in our town would create hundreds of local jobs and boost our economy."
        "Local_Economy_Help": "yes",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    1: {  # "Those wind turbines are an eyesore and completely ruin the beautiful countryside view."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "yes",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    2: {  # "I'm not sure if a solar panel farm would really help or hurt our local economy - it's hard to say."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "yes",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    3: {  # "The solar development takes up way too much farmland that we need for agriculture."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "yes",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    4: {  # "Having renewable energy nearby would definitely lower my electricity bills, which is great."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "yes",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    5: {  # "I'm worried that building a wind farm will increase our electricity costs because of infrastructure expenses."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "yes",
        "Tax_Revenue_Help": "no"
    },
    6: {  # "The new solar panels on the community center look great and don't detract from the area at all."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "yes",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    7: {  # "A wind farm would bring in significant tax revenue for our county, helping fund schools and roads."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "yes"
    },
    8: {  # "I don't think a solar development would make any real difference to our local economy one way or another."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "yes",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    9: {  # "The space used for the solar farm is reasonable compared to the benefits we'll get from clean energy."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "yes",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    }
}

# JSON Schema for API response validation
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "string",
            "enum": ["yes", "no"]
        }
    },
    "required": ["relevant"],
    "additionalProperties": False
}

def call_local_llm_api(model_config: Dict, prompt: str, base_url: str = "http://127.0.0.1:1234") -> str:
    """
    Call local LLM API using OpenAI-compatible endpoint
    
    Args:
        model_config: Model configuration from JSON
        prompt: The prompt to send
        base_url: Base URL for the API (default: localhost:8000)
    
    Returns:
        Response text from the model
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if model_config.get('api_key'):
        headers["Authorization"] = f"Bearer {model_config['api_key']}"
    
    # Each request is independent - no conversation history maintained
    # Fresh messages array for each API call to ensure no context from previous prompts
    
    # Try with JSON schema enforcement first
    payload_with_schema = {
        "model": model_config['name'],
        "messages": [
            {"role": "system", "content": "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
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
        "model": model_config['name'],
        "messages": [
            {"role": "system", "content": "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
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
            # Don't print errors during normal operation
            return "error"
        except ValidationError:
            # Don't print errors during normal operation
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
                # Try fallback without schema
                
                # Try without schema
                response = requests.post(url, json=payload_without_schema, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return parse_and_validate_response(content)
            except requests.exceptions.HTTPError as e2:
                # If fallback also fails with 400, try different model name formats
                if e2.response.status_code == 400:
                    # Try different model name variations
                    model_name_variations = [
                        model_config.get('id'),  # Try ID first
                        model_config['name'].split('/')[-1],  # Try without prefix
                        model_config['name']  # Try original
                    ]
                    # Remove duplicates and None
                    model_name_variations = [m for m in model_name_variations if m and m not in [None, '']]
                    model_name_variations = list(dict.fromkeys(model_name_variations))  # Remove duplicates
                    
                    for model_variant in model_name_variations:
                        try:
                            payload_variant = {
                                "model": model_variant,
                                "messages": [
                                    {"role": "system", "content": "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."},
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": api_config['config']['default_temperature'],
                                "max_tokens": api_config['config']['default_max_tokens']
                            }
                            response = requests.post(url, json=payload_variant, headers=headers, timeout=30)
                            response.raise_for_status()
                            result = response.json()
                            content = result['choices'][0]['message']['content'].strip()
                            parsed = parse_and_validate_response(content)
                            if parsed != "error":
                                return parsed
                        except:
                            continue  # Try next variation
                    
                    # All variations failed
                    return "error"
                else:
                    # Non-400 error in fallback
                    return "error"
            except Exception as e2:
                print(f"Error calling API (fallback) for {model_config['name']}: {e2}")
                return "error"
        else:
            print(f"HTTP error calling API for {model_config['name']}: {e}")
            return "error"
    except Exception as e:
        print(f"Error calling API for {model_config['name']}: {e}")
        return "error"
    
    return "error"

def group_related_questions(questions: Dict) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Group related questions by their base topic.
    Questions like "Local_Economy_Help" and "Local_Economy_Hurt" are grouped together
    since they share the same base topic "Local_Economy".
    
    Handles special cases like:
    - "Landscape_Unattractive" and "Landscape_Not_Unattractive" -> both grouped as "Landscape"
    - "Space_Too_Much" and "Space_Acceptable" -> both grouped as "Space"
    
    Returns:
        Dictionary mapping base_topic -> list of (question_id, question_info) tuples
    """
    # Define variant suffixes that should be stripped to find the base topic
    variant_suffixes = [
        '_Help', '_Hurt', '_No_Difference',
        '_Lower', '_Higher',
        '_Unattractive', '_Not_Unattractive',
        '_Too_Much', '_Acceptable'
    ]
    
    groups = {}
    for question_id, question_info in questions.items():
        # Try to find base topic by removing known variant suffixes
        base_topic = question_id
        for suffix in variant_suffixes:
            if question_id.endswith(suffix):
                base_topic = question_id[:-len(suffix)]
                break
        
        # If no variant suffix matched, try splitting on last underscore as fallback
        if base_topic == question_id:
            parts = question_id.split('_')
            if len(parts) > 1:
                base_topic = '_'.join(parts[:-1])
        
        if base_topic not in groups:
            groups[base_topic] = []
        groups[base_topic].append((question_id, question_info))
    
    return groups

def create_relevance_prompt(statement: str, question_info: Dict) -> str:
    """Create prompt for first pass: determine if statement is relevant to question"""
    prompt = f"""Please determine if this statement is relevant to the survey question below.

Survey Question: {question_info['question']}

Statement: "{statement}"

Is this statement relevant to the survey question? Does it express an opinion, belief, or concern related to this question?

Please respond with valid JSON:
{{
  "relevant": "yes"  // or "no"
}}

- "yes" if the statement is relevant to this survey question
- "no" if the statement is not relevant to this survey question

Response (JSON only):"""
    return prompt

def create_relevance_prompt_grouped(statement: str, grouped_questions: List[Tuple[str, Dict]]) -> str:
    """
    Create prompt for first pass: determine if statement is relevant to a group of related questions.
    This is more efficient than checking each question separately.
    
    Args:
        statement: The statement to check
        grouped_questions: List of (question_id, question_info) tuples for related questions
    
    Returns:
        Prompt string
    """
    questions_text = "\n".join([
        f"- {q_info['question']}" for _, q_info in grouped_questions
    ])
    
    prompt = f"""Please determine if this statement is relevant to ANY of the following related survey questions.

Related Survey Questions:
{questions_text}

Statement: "{statement}"

Is this statement relevant to any of these survey questions? Does it express an opinion, belief, or concern related to any of these questions?

Please respond with valid JSON:
{{
  "relevant": "yes"  // or "no"
}}

- "yes" if the statement is relevant to any of these survey questions
- "no" if the statement is not relevant to any of these survey questions

Response (JSON only):"""
    return prompt

def create_classification_prompt(statement: str, question_info: Dict) -> str:
    """Create prompt for second pass: detailed classification (only for relevant statements)"""
    prompt = f"""Please classify this statement about wind and solar power development.

Survey Question: {question_info['question']}

Question Description: {question_info['description']}

Statement to classify: "{statement}"

This statement has been identified as relevant to the survey question above. Please confirm the classification.

Please respond with valid JSON in the following format:
{{
  "relevant": "yes"  // or "no"
}}

- Please use "yes" if the statement is relevant to this survey question
- Please use "no" if the statement is not relevant to this survey question

Response (JSON only):"""
    return prompt

def check_relevance(statement: str, question_id: str, question_info: Dict, 
                    model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Tuple[str, str]:
    """
    First pass: Check if statement is relevant to question
    
    Returns:
        Tuple of (question_id, response) where response is 'yes', 'no', or 'error'
    """
    prompt = create_relevance_prompt(statement, question_info)
    response = call_local_llm_api(model_config, prompt, base_url)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes')
    elif 'no' in response or response == 'n':
        return (question_id, 'no')
    else:
        return (question_id, 'error')

def classify_statement_against_question(statement: str, question_id: str, question_info: Dict, 
                                        model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Tuple[str, str]:
    """
    Second pass: Classify a relevant statement against a survey question
    
    Returns:
        Tuple of (question_id, response) where response is 'yes', 'no', or 'error'
    """
    prompt = create_classification_prompt(statement, question_info)
    response = call_local_llm_api(model_config, prompt, base_url)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes')
    elif 'no' in response or response == 'n':
        return (question_id, 'no')
    else:
        return (question_id, 'error')

def check_relevance_grouped(statement: str, grouped_questions: List[Tuple[str, Dict]], 
                            model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Dict[str, str]:
    """
    First pass: Check if statement is relevant to a group of related questions.
    Returns the same relevance result for all questions in the group.
    
    Returns:
        Dictionary mapping question_id -> 'yes'/'no'/'error'
    """
    prompt = create_relevance_prompt_grouped(statement, grouped_questions)
    response = call_local_llm_api(model_config, prompt, base_url)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        result = 'yes'
    elif 'no' in response or response == 'n':
        result = 'no'
    else:
        result = 'error'
    
    # Return same result for all questions in the group
    return {question_id: result for question_id, _ in grouped_questions}

def check_relevance_task(args: Tuple) -> Tuple[int, str, str]:
    """First pass: Check relevance of a single statement-question pair"""
    stmt_idx, statement, question_id, question_info, model_config, base_url = args
    _, response = check_relevance(statement, question_id, question_info, model_config, base_url)
    return (stmt_idx, question_id, response)

def check_relevance_grouped_task(args: Tuple) -> List[Tuple[int, str, str]]:
    """First pass: Check relevance of a statement to a group of related questions"""
    stmt_idx, statement, grouped_questions, model_config, base_url = args
    results = check_relevance_grouped(statement, grouped_questions, model_config, base_url)
    # Return list of (stmt_idx, question_id, response) tuples
    return [(stmt_idx, question_id, response) for question_id, response in results.items()]

def classify_single_task(args: Tuple) -> Tuple[int, str, str]:
    """Second pass: Classify a relevant statement-question pair"""
    stmt_idx, statement, question_id, question_info, model_config, base_url = args
    _, response = classify_statement_against_question(
        statement, question_id, question_info, model_config, base_url
    )
    return (stmt_idx, question_id, response)

def first_pass_relevance(statements: List[str], questions: Dict, 
                        model_config: Dict, base_url: str = "http://127.0.0.1:1234",
                        max_workers: int = 50) -> Dict:
    """
    First pass: Determine relevance of all statements to all questions
    Uses grouped questions to reduce API calls (e.g., Help/Hurt variants checked together)
    
    Returns:
        Dictionary with structure: {statement_index: {question_id: 'yes'/'no'/'error'}}
    """
    results = {}
    
    # Group related questions by base topic
    question_groups = group_related_questions(questions)
    
    # Prepare all tasks (one per statement-group pair)
    tasks = []
    for stmt_idx, statement in enumerate(statements):
        for grouped_questions in question_groups.values():
            tasks.append((stmt_idx, statement, grouped_questions, model_config, base_url))
    
    # Initialize results structure
    for stmt_idx in range(len(statements)):
        results[stmt_idx] = {}
    
    # Calculate total tasks for progress bar (still one per statement-question pair for display)
    total_tasks = len(statements) * len(questions)
    
    # Process tasks in parallel with progress bar
    model_short_name = model_config['name'].split('/')[-1] if '/' in model_config['name'] else model_config['name']
    with tqdm(total=total_tasks, desc=f"{model_short_name[:30]:<30} [Pass 1: Relevance]", unit="task") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(check_relevance_grouped_task, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    grouped_results = future.result()
                    for stmt_idx, question_id, response in grouped_results:
                        results[stmt_idx][question_id] = response
                        pbar.update(1)
                except Exception:
                    task = future_to_task[future]
                    stmt_idx = task[0]
                    grouped_questions = task[2]
                    # Mark all questions in group as error
                    for question_id, _ in grouped_questions:
                        results[stmt_idx][question_id] = "error"
                        pbar.update(1)
    
    return results

def second_pass_classification(statements: List[str], questions: Dict, relevance_results: Dict,
                              model_config: Dict, base_url: str = "http://127.0.0.1:1234",
                              max_workers: int = 50) -> Dict:
    """
    Second pass: Detailed classification only for statements marked as relevant
    
    Args:
        relevance_results: Results from first pass showing which statements are relevant to which questions
    
    Returns:
        Dictionary with structure: {statement_index: {question_id: 'yes'/'no'/'error'}}
    """
    results = {}
    
    # Prepare tasks only for relevant pairs
    tasks = []
    for stmt_idx, statement in enumerate(statements):
        results[stmt_idx] = {}
        for question_id, question_info in questions.items():
            # Only process if marked as relevant in first pass
            if relevance_results.get(stmt_idx, {}).get(question_id) == 'yes':
                tasks.append((stmt_idx, statement, question_id, question_info, model_config, base_url))
            else:
                # Mark as not relevant (from first pass)
                results[stmt_idx][question_id] = 'no'
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        return results
    
    # Process tasks in parallel with progress bar
    model_short_name = model_config['name'].split('/')[-1] if '/' in model_config['name'] else model_config['name']
    with tqdm(total=total_tasks, desc=f"{model_short_name[:30]:<30} [Pass 2: Classification]", unit="task") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(classify_single_task, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    stmt_idx, question_id, response = future.result()
                    results[stmt_idx][question_id] = response
                    pbar.update(1)
                except Exception:
                    task = future_to_task[future]
                    stmt_idx = task[0]
                    question_id = task[2]
                    results[stmt_idx][question_id] = "error"
                    pbar.update(1)
    
    return results

def calculate_coherence(responses: List[str]) -> float:
    """Calculate coherence as percentage of majority label / total models"""
    if not responses:
        return 0.0
    
    # Count occurrences of each response (excluding errors)
    valid_responses = [r for r in responses if r != 'error']
    if not valid_responses:
        return 0.0
    
    counts = Counter(valid_responses)
    majority_count = max(counts.values())
    total_models = len(responses)
    
    # Coherence = majority count / total models
    coherence = (majority_count / total_models) * 100.0
    return round(coherence, 2)

def get_majority_vote(responses: List[str]) -> str:
    """Get majority vote from responses (excluding errors)"""
    valid_responses = [r for r in responses if r != 'error']
    if not valid_responses:
        return 'error'
    counts = Counter(valid_responses)
    return counts.most_common(1)[0][0]

def save_results_csv(all_results: Dict[str, Dict], statements: List[str], questions: Dict, filename: str = "sample_local_llm_results.csv", has_ground_truth: bool = True, statement_to_original_index: Dict[int, int] = None):
    """
    Save results to CSV file with statement text, model classifications, ground truth (if available), and coherence
    
    Args:
        statement_to_original_index: Optional mapping from resampled statement index to original index.
                                    Used when statements have been resampled and ground truth indices need to be mapped.
    """
    import csv
    
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    # Create CSV with header: Statement, Question, Ground_Truth (if available), Model1, Model2, Model3, Coherence
    fieldnames = ['Statement', 'Question']
    if has_ground_truth:
        fieldnames.append('Ground_Truth')
    fieldnames.extend([model_name for model_name in model_names])
    fieldnames.append('Coherence')
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write results for each statement-question pair
        for stmt_idx, statement in enumerate(statements):
            for question_id in question_ids:
                row = {
                    'Statement': statement,
                    'Question': question_id
                }
                
                # Get ground truth (if available)
                if has_ground_truth:
                    # Use mapping if provided (for resampled statements), otherwise use stmt_idx directly
                    original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
                    ground_truth = GROUND_TRUTH.get(original_idx, {}).get(question_id, 'unknown')
                    row['Ground_Truth'] = ground_truth
                
                # Get classification from each model
                responses = []
                for model_name in model_names:
                    model_result = all_results[model_name].get(stmt_idx, {})
                    response = model_result.get(question_id, 'error')
                    row[model_name] = response
                    responses.append(response)
                
                # Calculate coherence
                coherence = calculate_coherence(responses)
                row['Coherence'] = coherence
                
                writer.writerow(row)
    
    print(f"Results saved to {filename}")

def calculate_metrics(all_results: Dict[str, Dict], statements: List[str], questions: Dict, has_ground_truth: bool = True) -> Dict:
    """Calculate accuracy (if ground truth available) and coherence metrics"""
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    total_pairs = len(statements) * len(question_ids)
    
    # Initialize counters
    model_correct = {model_name: 0 for model_name in model_names}
    majority_correct = 0
    coherence_correct = []
    coherence_wrong = []
    coherence_all = []
    
    # Process each statement-question pair
    for stmt_idx in range(len(statements)):
        for question_id in question_ids:
            # Get responses from all models
            responses = []
            for model_name in model_names:
                model_result = all_results[model_name].get(stmt_idx, {})
                response = model_result.get(question_id, 'error')
                responses.append(response)
            
            # Calculate coherence (always calculated)
            coherence = calculate_coherence(responses)
            coherence_all.append(coherence)
            
            # Accuracy calculations only if ground truth is available
            if has_ground_truth:
                ground_truth = GROUND_TRUTH.get(stmt_idx, {}).get(question_id, 'unknown')
                if ground_truth == 'unknown':
                    continue
                
                # Check if each model is correct
                for model_name, response in zip(model_names, responses):
                    if response == ground_truth:
                        model_correct[model_name] += 1
                
                # Check majority vote
                majority = get_majority_vote(responses)
                if majority == ground_truth:
                    majority_correct += 1
                    coherence_correct.append(coherence)
                else:
                    coherence_wrong.append(coherence)
    
    # Calculate percentages
    metrics = {
        'total_pairs': total_pairs,
        'has_ground_truth': has_ground_truth,
        'total_coherence': sum(coherence_all) / len(coherence_all) if coherence_all else 0,
    }
    
    # Add accuracy metrics only if ground truth is available
    if has_ground_truth:
        metrics['model_accuracy'] = {model_name: (model_correct[model_name] / total_pairs * 100) 
                                       for model_name in model_names}
        metrics['majority_accuracy'] = (majority_correct / total_pairs * 100) if total_pairs > 0 else 0
        metrics['coherence_when_correct'] = sum(coherence_correct) / len(coherence_correct) if coherence_correct else 0
        metrics['coherence_when_wrong'] = sum(coherence_wrong) / len(coherence_wrong) if coherence_wrong else 0
    
    return metrics

def save_metrics_csv(metrics: Dict, filename: str = "sample_local_llm_metrics.csv"):
    """Save metrics to CSV file"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Pairs', metrics['total_pairs']])
        writer.writerow(['Total Coherence (%)', round(metrics['total_coherence'], 2)])
        
        # Only include accuracy metrics if ground truth is available
        if metrics.get('has_ground_truth', True):
            writer.writerow(['Majority Accuracy (%)', round(metrics['majority_accuracy'], 2)])
            writer.writerow(['Coherence When Correct (%)', round(metrics['coherence_when_correct'], 2)])
            writer.writerow(['Coherence When Wrong (%)', round(metrics['coherence_when_wrong'], 2)])
            
            for model_name, accuracy in metrics['model_accuracy'].items():
                writer.writerow([f'{model_name} Accuracy (%)', round(accuracy, 2)])
    
    print(f"Metrics saved to {filename}")

def print_metrics(metrics: Dict):
    """Print metrics at the end"""
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"\nTotal Statement-Question Pairs: {metrics['total_pairs']}")
    print(f"\nTotal Coherence: {metrics['total_coherence']:.2f}%")
    
    # Only print accuracy metrics if ground truth is available
    if metrics.get('has_ground_truth', True):
        print(f"\nModel Accuracy:")
        for model_name, accuracy in metrics['model_accuracy'].items():
            print(f"  {model_name}: {accuracy:.2f}%")
        print(f"\nMajority Vote Accuracy: {metrics['majority_accuracy']:.2f}%")
        print(f"\nCoherence When Correct: {metrics['coherence_when_correct']:.2f}%")
        print(f"Coherence When Wrong: {metrics['coherence_when_wrong']:.2f}%")
    else:
        print("\n(No ground truth available - accuracy metrics not calculated)")
    
    print("="*80 + "\n")


def load_survey_questions_by_sector() -> Dict[str, Dict]:
    """Load survey questions from JSON file organized by sector"""
    if not os.path.exists('survey_question.json'):
        print("Error: survey_question.json not found.")
        return {}
    
    with open('survey_question.json', 'r', encoding='utf-8') as f:
        survey_data = json.load(f)
    
    # Normalize sector names (handle case differences)
    sector_mapping = {
        'food': 'Food',
        'housing': 'Housing',
        'transport': 'Transport',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    
    normalized_data = {}
    for sector, questions in survey_data.items():
        # Normalize to title case
        normalized_sector = sector.capitalize()
        # Ensure each question has the sector field set
        for question_id, question_info in questions.items():
            if 'sector' not in question_info:
                question_info['sector'] = normalized_sector
        normalized_data[normalized_sector] = questions
    
    return normalized_data

def get_questions_for_sector(sector: str) -> Dict:
    """
    Get survey questions for a specific sector only (no cross-sector questions)
    
    Args:
        sector: Sector name (Food, Transport, Housing, or lowercase variants)
    
    Returns:
        Dictionary of questions for the specified sector only
    """
    survey_by_sector = load_survey_questions_by_sector()
    
    # Normalize sector name
    sector_mapping = {
        'food': 'Food',
        'transport': 'Transport',
        'housing': 'Housing',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    normalized_sector = sector_mapping.get(sector.lower(), sector.capitalize())
    
    if normalized_sector not in survey_by_sector:
        print(f"Warning: Sector '{sector}' not found. Available sectors: {', '.join(survey_by_sector.keys())}")
        return {}
    
    # Get questions for this sector and filter to ensure they all have the correct sector
    sector_questions = survey_by_sector[normalized_sector].copy()
    
    # Double-check: filter out any questions that don't match the sector
    filtered_questions = {}
    for question_id, question_info in sector_questions.items():
        # Ensure sector field matches
        if question_info.get('sector', '').capitalize() == normalized_sector:
            filtered_questions[question_id] = question_info
        else:
            print(f"Warning: Question {question_id} has sector '{question_info.get('sector')}' but expected '{normalized_sector}'. Skipping.")
    
    return filtered_questions

def print_relevance_percentages(all_relevance_results: Dict[str, Dict], questions: Dict, statements: List[str] = None):
    """Print relevance percentages for each question (by question ID only)"""
    if statements is None:
        statements = SAMPLE_STATEMENTS
    
    question_ids = list(questions.keys())
    model_names = list(all_relevance_results.keys())
    
    print("\n" + "="*80)
    print("RELEVANCE PERCENTAGES BY QUESTION (First Pass)")
    print("="*80)
    print(f"{'Question ID':<40} " + " ".join([f"{m.split('/')[-1][:15]:<15}" for m in model_names]) + " Average")
    print("-" * 80)
    
    for question_id in question_ids:
        row = f"{question_id:<40} "
        percentages = []
        
        for model_name in model_names:
            # Count relevant (yes) for this question across all statements
            relevant_count = sum(1 for stmt_idx in all_relevance_results[model_name] 
                               if all_relevance_results[model_name][stmt_idx].get(question_id) == 'yes')
            total_statements = len(statements)
            percentage = (relevant_count / total_statements * 100) if total_statements > 0 else 0
            percentages.append(percentage)
            row += f"{percentage:>6.1f}%        "
        
        avg_percentage = sum(percentages) / len(percentages) if percentages else 0
        row += f"{avg_percentage:>6.1f}%"
        print(row)
    
    print("="*80 + "\n")

def main():
    """Main function to run the two-pass classification for all models"""
    global SURVEY_QUESTIONS
    
    # Check for -run_real_data flag and reload questions if needed
    use_real_data = len(sys.argv) > 1 and sys.argv[1] == "-run_real_data"
    if use_real_data:
        SURVEY_QUESTIONS = load_survey_questions(use_real_data=True)
    
    print("="*80)
    print("LOCAL LLM SURVEY QUESTION CLASSIFIER (TWO-PASS APPROACH)")
    print("="*80)
    if use_real_data:
        print(f"Using real survey data from survey_question.json")
    else:
        print(f"Survey: Pew Research - Americans' views on local wind and solar power development")
        print(f"URL: https://www.pewresearch.org/science/2024/06/27/americans-views-on-local-wind-and-solar-power-development/")
    print("="*80 + "\n")
    
    # Get all enabled models from config
    enabled_models = [m for m in api_config['models'] if m.get('enabled', True)]
    if not enabled_models:
        print("Error: No enabled models found in configuration")
        return
    
    print(f"Processing {len(enabled_models)} models sequentially")
    print(f"Total questions: {len(SURVEY_QUESTIONS)}")
    print(f"Total statements: {len(SAMPLE_STATEMENTS)}\n")
    
    # Base URL - using local server
    base_url = "http://127.0.0.1:1234"
    
    # Store relevance results from first pass (all models)
    all_relevance_results = {}
    
    # ========== FIRST PASS: RELEVANCE CHECK ==========
    print("="*80)
    print("FIRST PASS: RELEVANCE CHECK")
    print("="*80 + "\n")
    
    for model_idx, model_config in enumerate(enabled_models, 1):
        print(f"\n{'='*80}")
        print(f"Model {model_idx}/{len(enabled_models)}: {model_config['name']}")
        print(f"{'='*80}\n")
        
        # First pass: Check relevance
        relevance_results = first_pass_relevance(SAMPLE_STATEMENTS, SURVEY_QUESTIONS, model_config, base_url, max_workers=50)
        all_relevance_results[model_config['name']] = relevance_results
    
    # Print relevance percentages
    print_relevance_percentages(all_relevance_results, SURVEY_QUESTIONS, SAMPLE_STATEMENTS)
    
    # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
    print("="*80)
    print("SECOND PASS: DETAILED CLASSIFICATION (Only Relevant Statements)")
    print("="*80 + "\n")
    
    # Store final results from all models
    all_results = {}
    
    for model_idx, model_config in enumerate(enabled_models, 1):
        print(f"\n{'='*80}")
        print(f"Model {model_idx}/{len(enabled_models)}: {model_config['name']}")
        print(f"{'='*80}\n")
        
        # Second pass: Detailed classification only for relevant statements
        relevance_results = all_relevance_results[model_config['name']]
        classification_results = second_pass_classification(
            SAMPLE_STATEMENTS, SURVEY_QUESTIONS, relevance_results, model_config, base_url, max_workers=50
        )
        
        # Store final results
        all_results[model_config['name']] = classification_results
    
    # Save results to CSV
    save_results_csv(all_results, SAMPLE_STATEMENTS, SURVEY_QUESTIONS, has_ground_truth=not use_real_data)
    
    # Calculate and print metrics
    metrics = calculate_metrics(all_results, SAMPLE_STATEMENTS, SURVEY_QUESTIONS, has_ground_truth=not use_real_data)
    print_metrics(metrics)
    
    # Save metrics to CSV
    save_metrics_csv(metrics)
    
    print("Classification complete for all models!")

def call_local_llm_api_detailed(model_config: Dict, prompt: str, base_url: str = "http://127.0.0.1:1234", verbose: bool = False) -> Tuple[str, Dict]:
    """
    Call local LLM API with detailed response information
    
    Returns:
        Tuple of (parsed_response, full_api_response_dict)
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    
    if model_config.get('api_key'):
        headers["Authorization"] = f"Bearer {model_config['api_key']}"
    
    payload_with_schema = {
        "model": model_config['name'],
        "messages": [
            {"role": "system", "content": "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
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
        "model": model_config['name'],
        "messages": [
            {"role": "system", "content": "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
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
        except (json.JSONDecodeError, ValidationError) as e:
            if verbose:
                print(f"Parse/validation error: {e}")
                print(f"Response content: {content[:200]}")
            return "error"
    
    # Try with schema first
    try:
        if verbose:
            print(f"Request URL: {url}")
            print(f"Request payload (with schema): {json.dumps(payload_with_schema, indent=2)}")
        
        response = requests.post(url, json=payload_with_schema, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if verbose:
            print(f"\nFull API Response:")
            print(json.dumps(result, indent=2))
        
        content = result['choices'][0]['message']['content'].strip()
        parsed = parse_and_validate_response(content)
        if parsed != "error":
            return parsed, result
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            if verbose:
                print(f"400 error with schema, trying without schema...")
            try:
                response = requests.post(url, json=payload_without_schema, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                if verbose:
                    print(f"\nFull API Response (fallback):")
                    print(json.dumps(result, indent=2))
                content = result['choices'][0]['message']['content'].strip()
                parsed = parse_and_validate_response(content)
                return parsed, result
            except Exception as e2:
                if verbose:
                    print(f"Error in fallback: {e2}")
                return "error", {}
        else:
            if verbose:
                print(f"HTTP error: {e}")
            return "error", {}
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return "error", {}
    
    return "error", {}

def test_single_example():
    """Test function to run a single example with all models (two-pass approach)"""
    print("="*80)
    print("TESTING SINGLE EXAMPLE - ALL MODELS (TWO-PASS)")
    print("="*80 + "\n")
    
    # Get all enabled models from config
    enabled_models = [m for m in api_config['models'] if m.get('enabled', True)]
    if not enabled_models:
        print("Error: No enabled models found in configuration")
        return
    
    base_url = "http://127.0.0.1:1234"
    
    # Use first statement and first question
    statement = SAMPLE_STATEMENTS[0]
    question_id = list(SURVEY_QUESTIONS.keys())[0]
    question_info = SURVEY_QUESTIONS[question_id]
    
    print(f"Statement: {statement}\n")
    print(f"Question ID: {question_id}")
    print(f"Question: {question_info['question']}\n")
    print("-"*80 + "\n")
    
    # Test all models - First pass: Relevance
    print("FIRST PASS: RELEVANCE CHECK\n")
    all_relevance = {}
    for model_config in enabled_models:
        print(f"Testing Model: {model_config['name']}")
        prompt = create_relevance_prompt(statement, question_info)
        response, full_result = call_local_llm_api_detailed(model_config, prompt, base_url, verbose=False)
        print(f"Relevance: {response} ({'YES' if response == 'yes' else 'NO' if response == 'no' else 'ERROR'})")
        if full_result and 'choices' in full_result and len(full_result['choices']) > 0:
            content = full_result['choices'][0]['message']['content']
            print(f"Raw JSON: {content}")
        print("-"*80 + "\n")
        all_relevance[model_config['name']] = {0: {question_id: response}}
    
    # Second pass: Detailed classification (only if relevant)
    print("\nSECOND PASS: DETAILED CLASSIFICATION\n")
    all_results = {}
    for model_config in enabled_models:
        relevance = all_relevance[model_config['name']][0][question_id]
        if relevance == 'yes':
            print(f"Testing Model: {model_config['name']} (Relevant - proceeding with classification)")
            prompt = create_classification_prompt(statement, question_info)
            response, full_result = call_local_llm_api_detailed(model_config, prompt, base_url, verbose=False)
            print(f"Classification: {response}")
            if full_result and 'choices' in full_result and len(full_result['choices']) > 0:
                content = full_result['choices'][0]['message']['content']
                print(f"Raw JSON: {content}")
            all_results[model_config['name']] = {0: {question_id: response}}
        else:
            print(f"Model: {model_config['name']} (Not relevant - skipping classification)")
            all_results[model_config['name']] = {0: {question_id: 'no'}}
        print("-"*80 + "\n")
    
    # Save to CSV (test always uses ground truth if available)
    save_results_csv(all_results, [statement], {question_id: question_info}, "test_single_example.csv", has_ground_truth=True)
    print("\nTest complete!")

def prepare_comments_from_dataframe(df, text_column: str = 'body', sector_column: str = 'sector') -> Dict[str, List[str]]:
    """
    Helper function to prepare comments from a pandas DataFrame.
    
    Args:
        df: pandas DataFrame with comments and sector information
        text_column: Name of the column containing comment text (default: 'body')
        sector_column: Name of the column containing sector labels (default: 'sector')
    
    Returns:
        Dictionary with sector as key and list of comment strings as values
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    if sector_column not in df.columns:
        raise ValueError(f"Column '{sector_column}' not found in DataFrame")
    
    comments_by_sector = {}
    for sector in df[sector_column].dropna().unique():
        sector_comments = df[df[sector_column] == sector][text_column].dropna().tolist()
        # Filter out empty strings
        sector_comments = [c for c in sector_comments if c and str(c).strip()]
        if sector_comments:
            comments_by_sector[sector] = sector_comments
    
    return comments_by_sector

def classify_real_comments(comments_by_sector: Dict[str, List[str]], output_prefix: str = "real_comments", base_url: str = "http://127.0.0.1:1234"):
    """
    Classify real Reddit comments by sector using survey questions.
    
    Args:
        comments_by_sector: Dictionary with sector as key and list of comment strings as values
                           e.g., {"Food": ["comment1", "comment2"], "Housing": ["comment3"]}
        output_prefix: Prefix for output CSV files (default: "real_comments")
        base_url: Base URL for the LLM API (default: "http://127.0.0.1:1234")
    
    Returns:
        Dictionary with results organized by sector
    """
    print("="*80)
    print("CLASSIFYING REAL REDDIT COMMENTS BY SECTOR")
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
    
    # Get all enabled models from config
    enabled_models = [m for m in api_config['models'] if m.get('enabled', True)]
    if not enabled_models:
        print("Error: No enabled models found in configuration")
        return {}
    
    print(f"Processing {len(enabled_models)} models sequentially")
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
        
        # Store relevance results from first pass (all models)
        all_relevance_results = {}
        
        # ========== FIRST PASS: RELEVANCE CHECK ==========
        print("FIRST PASS: RELEVANCE CHECK\n")
        
        for model_idx, model_config in enumerate(enabled_models, 1):
            print(f"\n{'='*80}")
            print(f"Model {model_idx}/{len(enabled_models)}: {model_config['name']}")
            print(f"{'='*80}\n")
            
            # First pass: Check relevance
            relevance_results = first_pass_relevance(comments, sector_questions, model_config, base_url, max_workers=50)
            all_relevance_results[model_config['name']] = relevance_results
        
        # Print relevance percentages
        print_relevance_percentages(all_relevance_results, sector_questions, comments)
        
        # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
        print("="*80)
        print("SECOND PASS: DETAILED CLASSIFICATION (Only Relevant Comments)")
        print("="*80 + "\n")
        
        # Store final results from all models
        all_results = {}
        
        for model_idx, model_config in enumerate(enabled_models, 1):
            print(f"\n{'='*80}")
            print(f"Model {model_idx}/{len(enabled_models)}: {model_config['name']}")
            print(f"{'='*80}\n")
            
            # Second pass: Detailed classification only for relevant comments
            relevance_results = all_relevance_results[model_config['name']]
            classification_results = second_pass_classification(
                comments, sector_questions, relevance_results, model_config, base_url, max_workers=50
            )
            
            # Store final results
            all_results[model_config['name']] = classification_results
        
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
    print("CLASSIFICATION COMPLETE FOR ALL SECTORS")
    print("="*80 + "\n")
    
    return all_sector_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_example()
    else:
        main()
