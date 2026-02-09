"""
temp.py
Load verification samples for each question (lowest accuracy first) and analyze mismatches.
"""

import json

# Load verification results
with open('paper4data/00_verification_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Load verification samples
with open('paper4data/00_verification_samples.json', 'r', encoding='utf-8') as f:
    samples_data = json.load(f)

# Flatten samples from nested structure
all_samples = []
for task_type in ['norms', 'survey']:
    if task_type in samples_data:
        for question_id, question_samples in samples_data[task_type].items():
            for sample in question_samples:
                all_samples.append(sample)

# Get questions sorted by accuracy (lowest first)
questions = []
for task_type in ['norms', 'survey']:
    if task_type in results['by_task']:
        for qid, qdata in results['by_task'][task_type].items():
            questions.append({
                'id': qid,
                'type': task_type,
                'accuracy': qdata['accuracy'],
                'kappa': qdata['cohen_kappa'],
                'n_samples': qdata['n_samples_valid']
            })

questions.sort(key=lambda x: x['accuracy'])

# Analyze each question
import sys
if len(sys.argv) > 1:
    # Analyze specific question
    target_qid = sys.argv[1]
    questions = [q for q in questions if q['id'] == target_qid]

for q in questions[:10]:  # Top 10 lowest accuracy
    qid = q['id']
    print(f"\n{'='*100}")
    print(f"QUESTION: {qid}")
    print(f"Accuracy: {q['accuracy']:.2%} | Kappa: {q['kappa']:.3f} | Type: {q['type']}")
    print(f"{'='*100}\n")

    # Get samples for this question
    question_samples = [s for s in all_samples if s.get('question_id') == qid]

    # Separate matches and mismatches
    matches = [s for s in question_samples if s['vllm_label'] == s['reasoning_label']]
    mismatches = [s for s in question_samples if s['vllm_label'] != s['reasoning_label']]

    print(f"Total samples: {len(question_samples)}")
    print(f"Matches: {len(matches)} ({len(matches)/len(question_samples)*100:.1f}%)")
    print(f"Mismatches: {len(mismatches)} ({len(mismatches)/len(question_samples)*100:.1f}%)")

    # Show mismatches
    print(f"\n{'-'*100}")
    print("MISMATCHES (Fast model WRONG, Reasoning model CORRECT):")
    print(f"{'-'*100}\n")

    for i, sample in enumerate(mismatches, 1):
        comment = sample.get('comment', sample.get('comment_text', 'N/A'))
        if len(comment) > 300:
            comment = comment[:300] + "..."

        print(f"MISMATCH #{i}:")
        print(f"Comment: {comment}")
        print(f"Sector: {sample.get('sector', 'N/A')}")
        print(f"Fast model (WRONG): {sample['vllm_label']}")
        print(f"Reasoning model (CORRECT): {sample['reasoning_label']}")
        print(f"Raw reasoning response: {sample.get('raw_reasoning_response', 'N/A')[:100]}")
        print()

    print(f"\n{'-'*100}")
    print("CATEGORY DISTRIBUTION:")
    print(f"{'-'*100}\n")

    # Count category distributions
    from collections import Counter
    vllm_counts = Counter([s['vllm_label'] for s in question_samples])
    reasoning_counts = Counter([s['reasoning_label'] for s in question_samples])

    all_categories = set(vllm_counts.keys()) | set(reasoning_counts.keys())

    print(f"{'Category':<30} {'Fast Model':<15} {'Reasoning Model':<15} {'Difference':<15}")
    print(f"{'-'*75}")
    for cat in sorted(all_categories):
        vllm_pct = vllm_counts.get(cat, 0) / len(question_samples) * 100
        reasoning_pct = reasoning_counts.get(cat, 0) / len(question_samples) * 100
        diff = vllm_pct - reasoning_pct
        print(f"{cat:<30} {vllm_pct:>6.1f}% ({vllm_counts.get(cat, 0):>2}) {reasoning_pct:>6.1f}% ({reasoning_counts.get(cat, 0):>2}) {diff:>+6.1f}%")

    print(f"\n{'-'*100}")
    print("CORRECT PREDICTIONS (for reference):")
    print(f"{'-'*100}\n")

    # Show a few correct predictions for comparison
    for i, sample in enumerate(matches[:3], 1):
        comment = sample.get('comment', sample.get('comment_text', 'N/A'))
        if len(comment) > 200:
            comment = comment[:200] + "..."

        print(f"CORRECT #{i}:")
        print(f"Comment: {comment}")
        print(f"Both models agreed: {sample['vllm_label']}")
        print()

    print("\n" + "="*100 + "\n")
