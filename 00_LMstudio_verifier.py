"""
00_LMstudio_verifier.py

═══════════════════════════════════════════════════════════════════════════════
WHY: Verification & Quality Assurance for Automated Labeling
═══════════════════════════════════════════════════════════════════════════════

In automated annotation pipelines, smaller/faster models (Mistral-7B via vLLM) are used
for bulk labeling due to cost and throughput efficiency. However, we need to validate
that these labels are reliable and accurate.

This script implements a verification strategy:
  1. Sample a subset of labeled data (20 comments per question/task)
  2. Re-label using a larger reasoning model (GPT-OSS-20B via LM Studio)
  3. Treat reasoning model output as "ground truth"
  4. Calculate standard ML evaluation metrics (accuracy, precision, recall, F1)

This gives us:
  - Confidence intervals for automated label quality
  - Identification of problematic questions/categories
  - Evidence for methodology section in papers
  - Continuous quality monitoring capability

═══════════════════════════════════════════════════════════════════════════════
HOW: Verification Pipeline
═══════════════════════════════════════════════════════════════════════════════

1. SAMPLING STRATEGY:
   - Random stratified sampling: 20 comments per question across all sectors
   - Ensures coverage of different comment types and sectors
   - Balance between statistical power and verification cost

2. RE-LABELING WITH REASONING MODEL:
   - Use larger model (GPT-OSS-20B) hosted in LM Studio as local API
   - Same prompts as original vLLM labeling (from 00_vllm_survey_question_final.json)
   - Reasoning model assumed to be more accurate due to size and training

3. METRIC CALCULATION (Per Question):
   - Overall Accuracy: % of labels that match between models
   - Per-Class Metrics (for each answer option):
     * Precision: Of labels predicted as X, how many were truly X?
     * Recall: Of true X labels, how many were predicted as X?
     * F1 Score: Harmonic mean of precision and recall
   - Confusion Matrix: Cross-tabulation of predicted vs true labels
   - Cohen's Kappa: Inter-rater agreement adjusting for chance

4. OUTPUT:
   - 00_verification_results.json: Structured results for downstream use
   - Compact dashboard visualization showing quality across all questions

═══════════════════════════════════════════════════════════════════════════════
WHAT: Metrics Interpretation
═══════════════════════════════════════════════════════════════════════════════

- Accuracy ≥ 0.85: Excellent agreement, vLLM labels are reliable
- Accuracy 0.70-0.85: Good agreement, minor issues
- Accuracy < 0.70: Poor agreement, investigate prompts or model choice

Per-class metrics identify specific problems:
- Low precision for class X: vLLM over-predicts X (false positives)
- Low recall for class X: vLLM under-predicts X (misses true X)
- Low F1: Class X is generally problematic in vLLM

Cohen's Kappa:
- κ > 0.80: Almost perfect agreement
- κ 0.60-0.80: Substantial agreement
- κ 0.40-0.60: Moderate agreement
- κ < 0.40: Poor agreement

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_CONFIG_PATH = "local_LLM_api_from_vLLM.json"
NORMS_LABELS_PATH = "paper4data/norms_labels.json"
SURVEY_QUESTIONS_PATH = "00_vllm_survey_question_final.json"
OUTPUT_PATH = "paper4data/00_verification_results.json"
SAMPLES_PER_QUESTION = 20  # Number of comments to verify per question
MAX_CONCURRENT = 12  # Concurrent requests (matching LM Studio configuration)
REQUEST_TIMEOUT = 120  # Longer timeout for reasoning model
RANDOM_SEED = 42


def load_api_config(path: str) -> Tuple[str, str]:
    """Load LM Studio API configuration from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    verification_key = config.get("verification_model_key", "7")
    model_config = config["available_models"][verification_key]

    api_url = f"{model_config['base_url']}/v1/chat/completions"
    model_name = model_config["model_name"]

    return api_url, model_name


# Load API configuration
LM_STUDIO_API_URL, LM_STUDIO_MODEL = load_api_config(API_CONFIG_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_norms_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load vLLM-labeled data (sector -> list of labeled comments)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_survey_questions(path: str) -> Dict[str, Dict[str, Any]]:
    """Load survey questions with prompts (question_id -> question_data)."""
    with open(path, "r", encoding="utf-8") as f:
        survey = json.load(f)

    questions = {}
    for sector_data in survey.values():
        if not isinstance(sector_data, dict):
            continue
        for question_set in sector_data.values():
            if not isinstance(question_set, dict) or "questions" not in question_set:
                continue
            for q in question_set["questions"]:
                questions[q["id"]] = q

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling Strategy
# ═══════════════════════════════════════════════════════════════════════════════

def sample_comments_for_verification(
    norms_data: Dict[str, List[Dict[str, Any]]],
    questions: Dict[str, Dict[str, Any]],
    samples_per_question: int = 20,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample comments for verification, stratified by question.
    Returns: {question_id: [list of comment dicts with vLLM labels]}
    """
    random.seed(seed)

    # Collect all comments that have labels for each question
    question_to_comments = defaultdict(list)

    for sector, comments in norms_data.items():
        for comment in comments:
            if "answers" not in comment:
                continue
            for qid, label in comment["answers"].items():
                if qid in questions:  # Only include questions we have prompts for
                    question_to_comments[qid].append({
                        "sector": sector,
                        "comment": comment.get("comment", ""),
                        "vllm_label": str(label).strip(),
                        "comment_index": comment.get("comment_index"),
                    })

    # Sample N comments per question
    sampled = {}
    for qid, comments in question_to_comments.items():
        if len(comments) >= samples_per_question:
            sampled[qid] = random.sample(comments, samples_per_question)
        else:
            sampled[qid] = comments  # Use all if less than N

    return sampled


# ═══════════════════════════════════════════════════════════════════════════════
# LM Studio API for Re-labeling
# ═══════════════════════════════════════════════════════════════════════════════

async def call_lm_studio_async(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str = LM_STUDIO_MODEL,
    timeout: int = REQUEST_TIMEOUT
) -> str:
    """Call LM Studio API with reasoning model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Deterministic for verification
        "max_tokens": 50,
    }

    try:
        async with session.post(
            LM_STUDIO_API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LM Studio API error {resp.status}: {text}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling LM Studio: {e}")
        return ""


async def relabel_with_reasoning_model(
    sampled_data: Dict[str, List[Dict[str, Any]]],
    questions: Dict[str, Dict[str, Any]],
    max_concurrent: int = MAX_CONCURRENT
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Re-label sampled comments using reasoning model.
    Returns: {question_id: [comments with both vllm_label and reasoning_label]}
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def label_comment(session, qid, comment):
        async with semaphore:
            question = questions[qid]
            prompt = question["prompt"]

            # Build full prompt with comment
            full_prompt = f"{prompt}\n\nText: {comment['comment']}"

            reasoning_label = await call_lm_studio_async(session, full_prompt)
            comment["reasoning_label"] = reasoning_label
            return comment

    # Relabel all sampled comments
    async with aiohttp.ClientSession() as session:
        tasks = []
        for qid, comments in sampled_data.items():
            for comment in comments:
                tasks.append(label_comment(session, qid, comment))

        total_tasks = len(tasks)
        print(f"Re-labeling {total_tasks} comments with reasoning model...")
        print(f"Concurrency: {max_concurrent} | Batch size: 50")

        start_time = time.time()
        results = []

        with tqdm(total=total_tasks, desc="Re-labeling", unit="comments",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i in range(0, total_tasks, 50):
                batch = tasks[i:i+50]
                batch_start = time.time()
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                batch_time = time.time() - batch_start

                results.extend(batch_results)
                pbar.update(len(batch))

                # Calculate and display performance metrics
                elapsed = time.time() - start_time
                processed = len(results)
                rate = processed / elapsed if elapsed > 0 else 0
                remaining_tasks = total_tasks - processed
                eta_seconds = remaining_tasks / rate if rate > 0 else 0

                pbar.set_postfix_str(f"{rate:.1f} comments/s | ETA: {eta_seconds/60:.1f} min")

    # Reorganize by question_id
    relabeled = defaultdict(list)
    task_idx = 0
    for qid, comments in sampled_data.items():
        for _ in comments:
            if task_idx < len(results) and not isinstance(results[task_idx], Exception):
                relabeled[qid].append(results[task_idx])
            task_idx += 1

    return dict(relabeled)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Calculation
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_label(label: str) -> str:
    """Normalize labels for comparison (lowercase, strip)."""
    return str(label).lower().strip()


def calculate_metrics_for_question(
    comments: List[Dict[str, Any]],
    question_id: str
) -> Dict[str, Any]:
    """
    Calculate verification metrics for a single question.
    Treats reasoning_label as ground truth, fast_model_label as predictions.
    Also computes category size over/underestimation.
    """
    # Extract labels
    y_true = [normalize_label(c["reasoning_label"]) for c in comments]
    y_pred = [normalize_label(c["vllm_label"]) for c in comments]

    # Filter out any empty labels
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
    if not valid_pairs:
        return {
            "question_id": question_id,
            "n_samples": 0,
            "accuracy": 0.0,
            "error": "No valid label pairs"
        }

    y_true, y_pred = zip(*valid_pairs)

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

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    # Category size estimation: over/underestimation by fast model
    reasoning_dist = Counter(y_true)
    fast_model_dist = Counter(y_pred)
    total = len(y_true)

    category_estimation = {}
    for label in labels:
        reasoning_pct = (reasoning_dist.get(label, 0) / total) * 100
        fast_model_pct = (fast_model_dist.get(label, 0) / total) * 100
        diff = fast_model_pct - reasoning_pct

        category_estimation[label] = {
            "reasoning_model_pct": float(reasoning_pct),
            "fast_model_pct": float(fast_model_pct),
            "estimation_error": float(diff),  # Positive = overestimation, Negative = underestimation
            "estimation_type": "overestimation" if diff > 2 else "underestimation" if diff < -2 else "accurate"
        }

    # Disagreement examples (first 5)
    disagreements = [
        {
            "comment": c["comment"][:200] + "..." if len(c["comment"]) > 200 else c["comment"],
            "fast_model": c["vllm_label"],
            "reasoning_model": c["reasoning_label"],
            "sector": c["sector"]
        }
        for c in comments
        if normalize_label(c["vllm_label"]) != normalize_label(c["reasoning_label"])
    ][:5]

    return {
        "question_id": question_id,
        "n_samples": len(valid_pairs),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "per_class_metrics": per_class_metrics,
        "category_estimation": category_estimation,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist()
        },
        "disagreement_examples": disagreements
    }


def calculate_all_metrics(
    relabeled_data: Dict[str, List[Dict[str, Any]]],
    questions: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate metrics for all questions."""
    results = {}

    for qid, comments in relabeled_data.items():
        if qid in questions:
            metrics = calculate_metrics_for_question(comments, qid)
            metrics["question_wording"] = questions[qid].get("wording", qid)
            metrics["question_short_form"] = questions[qid].get("short_form", qid)
            results[qid] = metrics

    # Overall summary
    accuracies = [m["accuracy"] for m in results.values() if "accuracy" in m]
    kappas = [m["cohen_kappa"] for m in results.values() if "cohen_kappa" in m]

    summary = {
        "total_questions": len(results),
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
        "min_accuracy": float(np.min(accuracies)) if accuracies else 0.0,
        "max_accuracy": float(np.max(accuracies)) if accuracies else 0.0,
        "mean_kappa": float(np.mean(kappas)) if kappas else 0.0,
    }

    return {
        "summary": summary,
        "by_question": results,
        "config": {
            "samples_per_question": SAMPLES_PER_QUESTION,
            "vllm_model": "Mistral-7B",
            "reasoning_model": LM_STUDIO_MODEL,
            "random_seed": RANDOM_SEED
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 80)
    print("vLLM Label Verification Pipeline")
    print("=" * 80)

    # 1. Load data
    print("\n[1/5] Loading data...")
    norms_data = load_norms_labels(NORMS_LABELS_PATH)
    questions = load_survey_questions(SURVEY_QUESTIONS_PATH)
    print(f"  Loaded {sum(len(v) for v in norms_data.values())} labeled comments")
    print(f"  Loaded {len(questions)} survey questions")

    # 2. Sample comments for verification
    print(f"\n[2/5] Sampling {SAMPLES_PER_QUESTION} comments per question...")
    sampled = sample_comments_for_verification(norms_data, questions, SAMPLES_PER_QUESTION, RANDOM_SEED)
    total_to_verify = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total_to_verify} comments across {len(sampled)} questions")

    # 3. Re-label with reasoning model
    print(f"\n[3/5] Re-labeling with reasoning model ({LM_STUDIO_MODEL})...")
    relabeled = await relabel_with_reasoning_model(sampled, questions, MAX_CONCURRENT)

    # 4. Calculate metrics
    print("\n[4/5] Calculating verification metrics...")
    results = calculate_all_metrics(relabeled, questions)
    print(f"  Mean accuracy: {results['summary']['mean_accuracy']:.3f}")
    print(f"  Mean Cohen's kappa: {results['summary']['mean_kappa']:.3f}")

    # 5. Save results
    print(f"\n[5/5] Saving results to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Verification complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 80)

    # Print summary table (use ASCII-safe output for Windows console)
    print("\n" + "Summary by Question".center(80))
    print("-" * 80)
    print(f"{'Question':<40} {'Acc':<8} {'Kappa':<8} {'N':<5}")
    print("-" * 80)
    for qid, metrics in sorted(results["by_question"].items()):
        short = metrics.get("question_short_form", qid)[:38]
        acc = metrics.get("accuracy", 0)
        kappa = metrics.get("cohen_kappa", 0)
        n = metrics.get("n_samples", 0)
        print(f"{short:<40} {acc:.3f}    {kappa:.3f}    {n}")
    print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
