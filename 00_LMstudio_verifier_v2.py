"""
00_LMstudio_verifier_v2.py

═══════════════════════════════════════════════════════════════════════════════
WHY: Comprehensive Verification & Quality Assurance for Hierarchical Labeling
═══════════════════════════════════════════════════════════════════════════════

In automated annotation pipelines, smaller/faster models (Mistral-7B, Qwen3-VL-4B via vLLM)
are used for bulk labeling due to cost and throughput efficiency. However, we need to validate
that these labels are reliable and accurate across BOTH labeling tasks:
  1. Survey questions: attitude labels (diet/EV/solar attitudes)
  2. IPCC social norms schema: hierarchical labels (gate, stance, norms, etc.)

This script implements a comprehensive verification strategy:
  1. Sample a subset of labeled data (50 comments per question/task, or max available)
  2. Re-label using a larger reasoning model (GPT-OSS-20B via LM Studio)
  3. Use EXACT same prompting as 00_vLLM_hierarchical.py via shared_utilities.py
  4. Treat reasoning model output as "ground truth"
  5. Calculate standard ML evaluation metrics (accuracy, precision, recall, F1, kappa)

This gives us:
  - Confidence intervals for automated label quality across all tasks
  - Identification of problematic questions/categories
  - Category-level estimation errors (over/underestimation)
  - Evidence for methodology section in papers
  - Continuous quality monitoring capability
  - Sample-level data for error analysis and example extraction

═══════════════════════════════════════════════════════════════════════════════
HOW: Comprehensive Verification Pipeline
═══════════════════════════════════════════════════════════════════════════════

1. SAMPLING STRATEGY:
   - Random stratified sampling: 50 comments per question across all sectors
     * Norms questions: sampled from all comments (norms labels apply to all)
     * Survey questions: sampled per sector (diet/housing/transport)
   - Ensures coverage of different comment types and sectors
   - Balance between statistical power and verification cost
   - Higher sample size (50 vs 20) for better statistical confidence

2. RE-LABELING WITH REASONING MODEL:
   - Use larger model (GPT-OSS-20B) hosted in LM Studio as local API
   - CRITICAL: Use shared_utilities.py to ensure exact same prompting as vLLM
     * Same system prompts (NORMS_SYSTEM, SURVEY_SYSTEM)
     * Same question structures (NORMS_QUESTIONS, SURVEY_QUESTIONS_BY_SECTOR)
     * Same answer extraction logic (call_llm_single_choice)
   - Reasoning model assumed to be more accurate due to size and chain-of-thought
   - Hierarchical labeling support (gate -> stance -> norms cascade)

3. METRIC CALCULATION (Per Question, Both Task Types):
   - Overall Accuracy: % of labels that match between fast model and reasoning model
   - Per-Class Metrics (for each answer option):
     * Precision: Of labels predicted as X, how many were truly X?
     * Recall: Of true X labels, how many were predicted as X?
     * F1 Score: Harmonic mean of precision and recall
   - Cohen's Kappa: Inter-rater agreement adjusting for chance
   - Category Estimation Error: Over/underestimation by fast model for each label
     * Positive error: fast model over-predicts this category
     * Negative error: fast model under-predicts this category
   - Empty Response Tracking: Percentage of cases where reasoning model failed to respond

4. OUTPUT:
   - 00_verification_results.json: Aggregated metrics by question
     * Summary statistics (mean/std/min/max accuracy, kappa)
     * Per-question metrics (accuracy, precision, recall, F1, kappa)
     * Category estimation errors
     * Broken down by task type (norms vs survey)
   - 00_verification_samples.json: Full sample-level data
     * All verified comments with both vLLM and reasoning labels
     * Raw reasoning model responses
     * Enables detailed error analysis and example extraction

═══════════════════════════════════════════════════════════════════════════════
WHAT: Metrics Interpretation
═══════════════════════════════════════════════════════════════════════════════

ACCURACY:
- Accuracy ≥ 0.85: Excellent agreement, vLLM labels are reliable
- Accuracy 0.70-0.85: Good agreement, minor issues
- Accuracy < 0.70: Poor agreement, investigate prompts or model choice

PER-CLASS METRICS (identify specific problems):
- Low precision for class X: vLLM over-predicts X (false positives)
- Low recall for class X: vLLM under-predicts X (misses true X)
- Low F1: Class X is generally problematic in vLLM

COHEN'S KAPPA (inter-rater agreement):
- κ > 0.80: Almost perfect agreement
- κ 0.60-0.80: Substantial agreement
- κ 0.40-0.60: Moderate agreement
- κ < 0.40: Poor agreement

CATEGORY ESTIMATION ERROR:
- |error| ≤ 2%: Accurate category size estimation
- error > 2%: Fast model overestimates this category
- error < -2%: Fast model underestimates this category
- Important for population-level inferences (e.g., "X% of comments support policy Y")

EMPTY RESPONSES:
- Monitor % of cases where reasoning model fails to respond
- High empty rate (>10%) may indicate prompt issues or model limitations
- Valid responses only are used for accuracy calculations

═══════════════════════════════════════════════════════════════════════════════
KEY DIFFERENCES FROM v1:
═══════════════════════════════════════════════════════════════════════════════

v1 (00_LMstudio_verifier.py):
- Single task type: survey questions only
- 20 samples per question
- Direct prompting from survey JSON
- One output file

v2 (this script):
- BOTH task types: norms + survey questions
- 50 samples per question (2.5x more for better statistics)
- Shared utilities: exact same prompting as 00_vLLM_hierarchical.py
- Two output files: aggregated metrics + full sample data
- Hierarchical labeling support
- Category estimation error tracking
- Empty response monitoring

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import random
import asyncio
import aiohttp
import time
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
import numpy as np
from tqdm import tqdm

# Import shared utilities
from shared_utilities import (
    load_api_config,
    load_norms_schema,
    load_survey_questions,
    call_llm_single_choice,
    NORMS_SYSTEM,
    SURVEY_SYSTEM,
    NORMS_QUESTIONS,
    SURVEY_QUESTIONS_BY_SECTOR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_CONFIG_PATH = "local_LLM_api_from_vLLM.json"
NORMS_LABELS_PATH = "paper4data/norms_labels.json"
OUTPUT_PATH = "paper4data/00_verification_results.json"
SAMPLES_OUTPUT_PATH = "paper4data/00_verification_samples.json"  # Full sample-level results
SAMPLES_PER_QUESTION = 50  # Number of comments to verify per question (or max available)
MAX_CONCURRENT = 16  # Concurrent requests (matching LM Studio configuration)
RANDOM_SEED = 42


# Load LM Studio API configuration
def get_lm_studio_config() -> Tuple[str, str]:
    """Get LM Studio API URL and model name from config."""
    config = load_api_config(API_CONFIG_PATH)
    verification_key = config.get("verification_model_key", "7")
    model_config = config["available_models"][verification_key]
    return model_config["base_url"], model_config["model_name"]


LM_STUDIO_BASE_URL, LM_STUDIO_MODEL = get_lm_studio_config()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def load_norms_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load vLLM-labeled data (sector -> list of labeled comments with hierarchical answers)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_comments_for_verification(
    norms_data: Dict[str, List[Dict[str, Any]]],
    samples_per_question: int = SAMPLES_PER_QUESTION,
    seed: int = RANDOM_SEED
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Sample comments for verification from norms_labels.json.

    Returns: {
        "norms": {question_id: [comments with answers]},
        "survey": {question_id: [comments with answers]}
    }

    Each comment has: comment, sector, year, answers dict with all labels
    """
    random.seed(seed)

    sampled = {"norms": defaultdict(list), "survey": defaultdict(list)}

    # Sample for norms questions (questions apply to all comments)
    all_comments = []
    for sector, comments in norms_data.items():
        for c in comments:
            if "answers" in c:
                all_comments.append({**c, "sector": sector})

    # Sample per norms question
    for norms_q in NORMS_QUESTIONS:
        qid = norms_q["id"]
        # Filter comments that have this question answered
        eligible = [c for c in all_comments if qid in c.get("answers", {})]
        if len(eligible) > samples_per_question:
            sampled_comments = random.sample(eligible, samples_per_question)
        else:
            sampled_comments = eligible
        sampled["norms"][qid] = sampled_comments

    # Sample for survey questions (sector-specific)
    for sector, survey_questions in SURVEY_QUESTIONS_BY_SECTOR.items():
        sector_comments = []
        for c in norms_data.get(sector, []):
            if "answers" in c:
                sector_comments.append({**c, "sector": sector})

        for survey_q in survey_questions:
            qid = survey_q["id"]
            # Filter comments that have this question answered
            eligible = [c for c in sector_comments if qid in c.get("answers", {})]
            if len(eligible) > samples_per_question:
                sampled_comments = random.sample(eligible, samples_per_question)
            else:
                sampled_comments = eligible
            sampled["survey"][qid] = sampled_comments

    return sampled


# ═══════════════════════════════════════════════════════════════════════════════
# Re-labeling with Reasoning Model
# ═══════════════════════════════════════════════════════════════════════════════

async def relabel_with_reasoning_model(
    sampled_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    max_concurrent: int = MAX_CONCURRENT
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Re-label sampled comments using LM Studio reasoning model.
    Uses exact same prompting structure as 00_vLLM_hierarchical.py via shared utilities.

    Returns: {
        "norms": {question_id: [comments with vllm_label and reasoning_label]},
        "survey": {question_id: [comments with vllm_label and reasoning_label]}
    }
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def label_comment_for_question(session, comment, question, task_type, system_prompt):
        """Label one comment for one question."""
        async with semaphore:
            text = comment["comment"]
            sector = comment.get("sector")
            qid = question["id"]

            # Get vLLM label from existing answers
            vllm_label = comment.get("answers", {}).get(qid, "")

            # Get reasoning model label using shared utility
            reasoning_label, raw_content, _ = await call_llm_single_choice(
                session=session,
                text=text,
                question=question,
                base_url=LM_STUDIO_BASE_URL,
                model_name=LM_STUDIO_MODEL,
                sector=sector,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1024,  # High limit for thinking model to complete reasoning + answer
            )

            return {
                **comment,
                "question_id": qid,
                "vllm_label": vllm_label,
                "reasoning_label": reasoning_label,
                "raw_reasoning_response": raw_content,
            }

    # Build all tasks
    tasks = []
    task_metadata = []  # Track which task belongs to which question/type

    # Norms questions
    for qid, comments in sampled_data["norms"].items():
        question = next((q for q in NORMS_QUESTIONS if q["id"] == qid), None)
        if not question:
            continue
        for comment in comments:
            tasks.append(label_comment_for_question(
                None, comment, question, "norms", NORMS_SYSTEM
            ))
            task_metadata.append(("norms", qid))

    # Survey questions
    for qid, comments in sampled_data["survey"].items():
        # Find question in survey questions
        question = None
        for sector_questions in SURVEY_QUESTIONS_BY_SECTOR.values():
            question = next((q for q in sector_questions if q["id"] == qid), None)
            if question:
                break
        if not question:
            continue
        for comment in comments:
            tasks.append(label_comment_for_question(
                None, comment, question, "survey", SURVEY_SYSTEM
            ))
            task_metadata.append(("survey", qid))

    total_tasks = len(tasks)
    print(f"\n{'='*80}")
    print(f"Re-labeling {total_tasks} comment-question pairs with reasoning model")
    print(f"Model: {LM_STUDIO_MODEL}")
    print(f"Concurrency: {max_concurrent} | Batch size: 50")
    print(f"{'='*80}\n")

    start_time = time.time()
    results = []

    async with aiohttp.ClientSession() as session:
        # Update tasks to use the session
        tasks_with_session = []
        for i, task in enumerate(tasks):
            # Extract the arguments from the original task coroutine
            task_type, qid = task_metadata[i]
            if task_type == "norms":
                comments = sampled_data["norms"][qid]
                question = next((q for q in NORMS_QUESTIONS if q["id"] == qid), None)
                system_prompt = NORMS_SYSTEM
            else:
                comments = sampled_data["survey"][qid]
                question = None
                for sector_questions in SURVEY_QUESTIONS_BY_SECTOR.values():
                    question = next((q for q in sector_questions if q["id"] == qid), None)
                    if question:
                        break
                system_prompt = SURVEY_SYSTEM

            # Find which comment this task is for
            comment_idx = len([m for m in task_metadata[:i] if m == (task_type, qid)])
            if comment_idx < len(comments):
                comment = comments[comment_idx]
                tasks_with_session.append(label_comment_for_question(
                    session, comment, question, task_type, system_prompt
                ))

        with tqdm(total=total_tasks, desc="Re-labeling", unit="labels",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i in range(0, total_tasks, 50):
                batch = tasks_with_session[i:i+50]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
                pbar.update(len(batch))

                # Calculate performance metrics
                elapsed = time.time() - start_time
                processed = len(results)
                rate = processed / elapsed if elapsed > 0 else 0
                remaining_tasks = total_tasks - processed
                eta_seconds = remaining_tasks / rate if rate > 0 else 0

                pbar.set_postfix_str(f"{rate:.1f} labels/s | ETA: {eta_seconds/60:.1f} min")

    # Reorganize results by task type and question
    relabeled = {"norms": defaultdict(list), "survey": defaultdict(list)}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue
        task_type, qid = task_metadata[i]
        relabeled[task_type][qid].append(result)

    return {
        "norms": dict(relabeled["norms"]),
        "survey": dict(relabeled["survey"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_label(label: str) -> str:
    """Normalize labels for comparison."""
    return str(label).lower().strip()


def calculate_metrics_for_question(
    comments: List[Dict[str, Any]],
    question_id: str,
    question_short_form: str = None,
) -> Dict[str, Any]:
    """
    Calculate verification metrics for a single question.
    Treats reasoning_label as ground truth, vllm_label as predictions.
    """
    # Extract labels
    total_samples = len(comments)
    y_true = [normalize_label(c["reasoning_label"]) for c in comments]
    y_pred = [normalize_label(c["vllm_label"]) for c in comments]

    # Filter out any empty labels (reasoning model didn't respond) and collect examples
    valid_comments = []
    for c in comments:
        t = normalize_label(c["reasoning_label"])
        p = normalize_label(c["vllm_label"])
        if t and p:
            valid_comments.append({
                "comment": c.get("comment", "")[:300],  # Truncate long comments
                "sector": c.get("sector", ""),
                "vllm_label": p,
                "reasoning_label": t,
                "match": t == p
            })

    n_empty = total_samples - len(valid_comments)
    empty_pct = (n_empty / total_samples * 100) if total_samples > 0 else 0

    if not valid_comments:
        return {
            "question_id": question_id,
            "question_short_form": question_short_form or question_id,
            "n_samples_total": total_samples,
            "n_samples_valid": 0,
            "n_empty_responses": n_empty,
            "empty_response_pct": round(empty_pct, 1),
            "accuracy": 0.0,
            "error": "No valid label pairs",
            "examples": {"correct": [], "incorrect": []}
        }

    y_true = [c["reasoning_label"] for c in valid_comments]
    y_pred = [c["vllm_label"] for c in valid_comments]

    # Get unique labels
    labels = sorted(set(y_true) | set(y_pred))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except:
        kappa = 0.0

    # Category size estimation (over/underestimation)
    reasoning_dist = Counter(y_true)
    fast_model_dist = Counter(y_pred)

    category_estimation = {}
    for label in labels:
        reasoning_pct = (reasoning_dist[label] / len(y_true)) * 100
        fast_model_pct = (fast_model_dist[label] / len(y_pred)) * 100
        diff = fast_model_pct - reasoning_pct

        if abs(diff) <= 2:
            est_type = "accurate"
        elif diff > 2:
            est_type = "overestimation"
        else:
            est_type = "underestimation"

        category_estimation[label] = {
            "reasoning_model_pct": round(reasoning_pct, 1),
            "fast_model_pct": round(fast_model_pct, 1),
            "estimation_error": round(diff, 1),
            "estimation_type": est_type,
        }

    return {
        "question_id": question_id,
        "question_short_form": question_short_form or question_id,
        "n_samples_total": total_samples,
        "n_samples_valid": len(valid_comments),
        "n_empty_responses": n_empty,
        "empty_response_pct": round(empty_pct, 1),
        "accuracy": round(accuracy, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        "cohen_kappa": round(kappa, 3),
        "category_estimation": category_estimation,
    }


def calculate_all_metrics(
    relabeled: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Calculate metrics for all questions (norms + survey)."""
    results = {"norms": {}, "survey": {}}

    # Norms questions
    for qid, comments in relabeled["norms"].items():
        question = next((q for q in NORMS_QUESTIONS if q["id"] == qid), None)
        short_form = question.get("id", qid) if question else qid
        results["norms"][qid] = calculate_metrics_for_question(comments, qid, short_form)

    # Survey questions
    for qid, comments in relabeled["survey"].items():
        # Find question to get short_form
        short_form = qid
        for sector_questions in SURVEY_QUESTIONS_BY_SECTOR.values():
            question = next((q for q in sector_questions if q["id"] == qid), None)
            if question:
                # Try to find short_form in original survey data
                short_form = qid.replace("_", " ")
                break
        results["survey"][qid] = calculate_metrics_for_question(comments, qid, short_form)

    # Calculate summary statistics
    all_accuracies = []
    all_kappas = []
    total_samples = 0
    total_empty = 0
    for task_results in [results["norms"], results["survey"]]:
        for metrics in task_results.values():
            if metrics.get("accuracy"):
                all_accuracies.append(metrics["accuracy"])
            if metrics.get("cohen_kappa") and not np.isnan(metrics["cohen_kappa"]):
                all_kappas.append(metrics["cohen_kappa"])
            total_samples += metrics.get("n_samples_total", 0)
            total_empty += metrics.get("n_empty_responses", 0)

    overall_empty_pct = (total_empty / total_samples * 100) if total_samples > 0 else 0

    summary = {
        "total_questions": len(all_accuracies),
        "total_norms_questions": len(results["norms"]),
        "total_survey_questions": len(results["survey"]),
        "total_samples": total_samples,
        "total_empty_responses": total_empty,
        "empty_response_pct": round(overall_empty_pct, 1),
        "mean_accuracy": round(np.mean(all_accuracies), 3) if all_accuracies else 0.0,
        "std_accuracy": round(np.std(all_accuracies), 3) if all_accuracies else 0.0,
        "min_accuracy": round(np.min(all_accuracies), 3) if all_accuracies else 0.0,
        "max_accuracy": round(np.max(all_accuracies), 3) if all_accuracies else 0.0,
        "mean_kappa": round(np.mean(all_kappas), 3) if all_kappas else 0.0,
    }

    return {
        "summary": summary,
        "by_task": results,
        "config": {
            "samples_per_question": SAMPLES_PER_QUESTION,
            "vllm_model": "Mistral-7B / Qwen3-VL-4B",
            "reasoning_model": LM_STUDIO_MODEL,
            "random_seed": RANDOM_SEED,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "="*80)
    print("vLLM Label Verification Pipeline (Comprehensive)")
    print("="*80)

    # 1. Load data
    print("\n[1/5] Loading data...")
    norms_data = load_norms_labels(NORMS_LABELS_PATH)
    total_comments = sum(len(comments) for comments in norms_data.values())
    print(f"  Loaded {total_comments} labeled comments")
    print(f"  Loaded {len(NORMS_QUESTIONS)} norms questions")
    total_survey_q = sum(len(qs) for qs in SURVEY_QUESTIONS_BY_SECTOR.values())
    print(f"  Loaded {total_survey_q} survey questions")

    # 2. Sample comments
    print(f"\n[2/5] Sampling {SAMPLES_PER_QUESTION} comments per question...")
    sampled = sample_comments_for_verification(norms_data, SAMPLES_PER_QUESTION, RANDOM_SEED)
    total_norms_samples = sum(len(comments) for comments in sampled["norms"].values())
    total_survey_samples = sum(len(comments) for comments in sampled["survey"].values())
    print(f"  Sampled {total_norms_samples} norms labels")
    print(f"  Sampled {total_survey_samples} survey labels")
    print(f"  Total: {total_norms_samples + total_survey_samples} labels to verify")

    # 3. Re-label with reasoning model
    print(f"\n[3/5] Re-labeling with reasoning model ({LM_STUDIO_MODEL})...")
    relabeled = await relabel_with_reasoning_model(sampled, MAX_CONCURRENT)

    # 4. Calculate metrics
    print("\n[4/5] Calculating verification metrics...")
    results = calculate_all_metrics(relabeled)
    print(f"  Total samples: {results['summary']['total_samples']}")
    print(f"  Empty responses: {results['summary']['total_empty_responses']} ({results['summary']['empty_response_pct']:.1f}%)")
    print(f"  Mean accuracy (valid only): {results['summary']['mean_accuracy']:.3f}")
    print(f"  Mean Cohen's kappa: {results['summary']['mean_kappa']:.3f}")

    # 5. Save results
    print(f"\n[5/5] Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # Save aggregated metrics
    print(f"  - Aggregated metrics: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save full sample-level data for later analysis and example extraction
    print(f"  - Full sample data: {SAMPLES_OUTPUT_PATH}")
    with open(SAMPLES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(relabeled, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("Verification complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("="*80)

    # Print summary table
    print("\n" + "Summary by Question".center(80))
    print("-"*80)
    print(f"{'Question':<40} {'Acc':<8} {'Kappa':<8} {'N':<5} {'Type':<10}")
    print("-"*80)

    # Print norms questions
    for qid, metrics in sorted(results["by_task"]["norms"].items()):
        short = metrics.get("question_short_form", qid)[:38]
        acc = metrics.get("accuracy", 0)
        kappa = metrics.get("cohen_kappa", 0)
        n = metrics.get("n_samples_valid", 0)
        kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "n/a"
        print(f"{short:<40} {acc:.3f}    {kappa_str:<8} {n:<5} {'norms':<10}")

    # Print survey questions
    for qid, metrics in sorted(results["by_task"]["survey"].items()):
        short = metrics.get("question_short_form", qid)[:38]
        acc = metrics.get("accuracy", 0)
        kappa = metrics.get("cohen_kappa", 0)
        n = metrics.get("n_samples_valid", 0)
        kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "n/a"
        print(f"{short:<40} {acc:.3f}    {kappa_str:<8} {n:<5} {'survey':<10}")

    print("-"*80)


if __name__ == "__main__":
    asyncio.run(main())
