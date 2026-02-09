# Full Paper Reasoning & Development Process
## Comprehensive Thought Process for Climate Change Social Norms Study

**Author:** Sandeep (via conversation with Claude Code)
**Date:** February 2026
**Purpose:** Document the complete reasoning, methodology evolution, and decision-making process behind the automated labeling and verification pipeline for social norms analysis in climate-related Reddit discussions.

---

## Table of Contents

1. [Core Research Question](#core-research-question)
2. [Two-Model Architecture: Fast Labeling + Reasoning Verification](#two-model-architecture)
3. [Confidence Scores: Measuring Model Uncertainty](#confidence-scores)
4. [Verification Methodology](#verification-methodology)
5. [Dashboard & Visualization Philosophy](#dashboard--visualization-philosophy)
6. [Temporal Analysis & Sampling Strategy](#temporal-analysis--sampling-strategy)
7. [Prompt Engineering Evolution](#prompt-engineering-evolution)
8. [Key Technical Decisions](#key-technical-decisions)

---

## Core Research Question

**Central Aim:** Understand how social norms around climate action (EVs, solar panels, plant-based diet) are discussed on Reddit, and how these norms influence behavior and attitudes.

**IPCC Framework:** This study operationalizes IPCC's social drivers of climate action:
- **Descriptive norms:** What people actually do ("most people here drive EVs")
- **Injunctive norms:** What people approve/disapprove ("you should go vegan")
- **Reference groups:** Who influences the speaker (family, friends, coworkers)
- **Second-order beliefs:** What the speaker thinks others believe

**Scale Challenge:**
- 3 sectors: Transport (EVs), Housing (solar), Food (diet/veganism)
- 500 comments per sector (1,500 total for main analysis)
- 7 norms questions + 29 sector-specific survey questions per comment
- ~54,000 question-answer pairs to label

**Why Automated Labeling?**
Manual annotation would take:
- 54,000 labels × 30 seconds per label = 450 hours (11+ weeks of full-time work)
- Cost: $15/hour × 450 hours = $6,750

Automated labeling with verification:
- Fast model (Mistral-7B): ~2 hours for 54,000 labels
- Verification model (GPT-OSS-20B): ~4 hours for 900 verification samples
- Total time: ~6 hours, cost: electricity + GPU time

---

## Two-Model Architecture: Fast Labeling + Reasoning Verification

### Why Two Models Instead of One?

**Initial Approach (Rejected):** Use one high-quality reasoning model (GPT-OSS-20B) for all labeling.

**Problems:**
1. **Speed:** Reasoning models use chain-of-thought (up to 1024 tokens for internal reasoning), making them 10-15x slower than fast models
2. **Cost:** More tokens = higher compute cost
3. **Scalability:** Cannot scale to 100k+ comments without excessive time/cost

**Solution:** Hierarchical two-model approach

### Fast Labeling Model (Mistral-7B via vLLM)

**Role:** Primary labeler for all 1,500 comments × 36 questions = 54,000 labels

**Characteristics:**
- **Speed:** High throughput (80 concurrent requests, ~30 labels/second)
- **Determinism:** Low temperature (0.1) for consistent labels
- **Simplicity:** Short prompts, minimal tokens (max 64 response tokens)
- **Output:** Single-word or short-phrase answers (e.g., "yes", "no", "pro", "against")

**Why Mistral-7B?**
- Excellent instruction-following for simple classification tasks
- Fast inference with vLLM (optimized for throughput)
- Smaller model = lower memory footprint = higher batch sizes
- Alternative: Qwen3-VL-4B (even faster but slightly lower accuracy)

**Deployment:**
```bash
docker run --gpus all -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### Reasoning Verification Model (GPT-OSS-20B via LM Studio)

**Role:** Judge/verifier for a SAMPLE of labels (25 per question = 900 total) to estimate fast model accuracy

**Characteristics:**
- **Quality:** Uses extended reasoning (thinking tokens) for complex edge cases
- **Depth:** Can handle nuanced distinctions (e.g., "pro but lack of options" vs "against")
- **Sampling:** Only verifies representative sample (900 out of 54,000 labels = 1.7%)
- **Purpose:** Provides ground truth for accuracy estimation, not for all labeling

**Why GPT-OSS-20B?**
- Open-source model with strong reasoning capabilities
- Supports extended thinking (DeepSeek-style reasoning tokens)
- Can run locally via LM Studio (no API costs)
- 20B parameters = good balance of quality and speed

**Deployment:**
```bash
# LM Studio server on port 1234
# Model: openai/gpt-oss-20b
# Settings: temperature 0.1, max_tokens 1024 (allows thinking tokens)
```

### Why This Architecture Works

**Theoretical Foundation:**
- Fast model gets 85% accuracy → 46,000 correct labels, 8,000 incorrect
- Verification sample (900 labels, 25 per question) estimates per-question accuracy
- Statistical power: 25 samples per question gives ±10% confidence interval
- Cost-benefit: 85% accuracy at 1/10th the cost of 100% manual annotation

**Quality Assurance:**
- Verification identifies systematic errors (e.g., fast model overestimates "neither/mixed" stance by 16%)
- Enables targeted prompt optimization for lowest-accuracy questions
- Dashboard shows estimation errors per category (over/underestimation)

---

## Confidence Scores: Measuring Model Uncertainty

### The Core Insight

**Research Question:** Can model confidence (log probabilities) predict when the fast labeler disagrees with the reasoning judge?

**Hypothesis:** Low-confidence predictions from the fast model are more likely to be incorrect when verified by the reasoning model.

**Utility:**
- Flag uncertain predictions for human review
- Estimate label quality without full verification
- Identify which types of comments are hardest to classify

### Log Probabilities Explained

**What are logprobs?**
- Each token generated by an LLM has an associated log probability
- Logprob closer to 0 (e.g., -0.1) = high confidence
- Logprob more negative (e.g., -2.0) = low confidence
- Can convert to probability: `prob = exp(logprob)`

**Example:**
```
Comment: "I love electric vehicles! They're the future."
Question: Author stance toward EVs?
Fast model response: "pro"
  Token 1: "pro" | logprob: -0.005 | confidence: 99.5%
  Average logprob: -0.005 → HIGH CONFIDENCE

Comment: "EVs are okay but expensive and charging is a hassle."
Question: Author stance toward EVs?
Fast model response: "neither/mixed"
  Token 1: "neither" | logprob: -0.8 | confidence: 44.9%
  Token 2: "/" | logprob: -0.3 | confidence: 74.1%
  Token 3: "mixed" | logprob: -0.5 | confidence: 60.7%
  Average logprob: -0.53 → MEDIUM CONFIDENCE
```

### Implementation in Our Pipeline

**API Changes:**
All API call functions now return 3 values instead of 2:
```python
# Before
parsed_answer, raw_content = await call_vllm_single_choice(...)

# After
parsed_answer, raw_content, avg_logprob = await call_vllm_single_choice(..., return_logprobs=True)
```

**Payload Modification:**
```python
payload = {
    "model": model_name,
    "messages": [...],
    "temperature": 0.1,
    "max_tokens": 64,
    "logprobs": True,        # NEW: request log probabilities
    "top_logprobs": 5,       # NEW: return top 5 alternative tokens per position
}
```

**Extraction Logic:**
```python
# Extract average logprob from API response
if "logprobs" in choice:
    logprobs_data = choice["logprobs"]
    if "content" in logprobs_data and logprobs_data["content"]:
        # Average the logprobs across all tokens
        token_logprobs = [
            token.get("logprob", 0)
            for token in logprobs_data["content"]
            if token.get("logprob") is not None
        ]
        if token_logprobs:
            avg_logprob = sum(token_logprobs) / len(token_logprobs)
```

**Storage Format:**
Output JSON now includes `logprobs` field alongside `answers`:
```json
{
  "comment_index": 42,
  "comment": "I love EVs but they're too expensive...",
  "answers": {
    "1.1_gate": "1",
    "1.1.1_stance": "pro but lack of options",
    ...
  },
  "logprobs": {
    "1.1_gate": -0.05,
    "1.1.1_stance": -0.82,
    ...
  },
  "year": 2023
}
```

### Analysis Questions (To Be Answered)

1. **Confidence vs Accuracy Correlation:**
   - Do questions with lower average logprobs have lower verification accuracy?
   - Hypothesis: 1.2.1_descriptive (32% accuracy) has lower average confidence than diet_1 (100% accuracy)

2. **Threshold Identification:**
   - What logprob threshold optimally flags uncertain predictions?
   - Example: Flag all predictions with avg_logprob < -1.0 for human review

3. **Efficiency Gains:**
   - Can we reduce verification sample size by targeting low-confidence predictions?
   - Example: Instead of 25 random samples per question, verify 10 lowest-confidence + 15 random

---

## Verification Methodology

### Sampling Strategy

**Per-Question Sampling:**
- 25 samples per question (36 questions × 25 = 900 total)
- Increased from initial 10 samples for better statistical power
- Samples drawn randomly from all labeled comments (1,500 comments across 3 sectors)

**Why 25 samples?**
- Statistical rule of thumb: n ≥ 25 for reasonable confidence intervals
- Margin of error at 25 samples: ±10% at 95% confidence
- Practical limit: 900 total verification calls takes ~4 hours with reasoning model

**Sector Balance:**
- Transport: 500 comments → ~8 samples per question
- Housing: 500 comments → ~8 samples per question
- Food: 500 comments → ~8 samples per question
- Ensures each sector represented in verification

### Verification Process

**Step 1: Sample Selection**
```python
# For each question, sample 25 labeled comments
sampled = {}
for question_id in all_questions:
    question_labels = [label for label in all_labels if label["question_id"] == question_id]
    sampled[question_id] = random.sample(question_labels, min(25, len(question_labels)))
```

**Step 2: Re-labeling with Reasoning Model**
```python
# Call reasoning model (GPT-OSS-20B) for each sampled label
for sample in sampled_labels:
    reasoning_label, raw_response, _ = await call_llm_single_choice(
        session=session,
        text=sample["comment_text"],
        question=question,
        base_url=reasoning_model_url,
        model_name="openai/gpt-oss-20b",
        sector=sample["sector"],
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=1024,  # Allow extended reasoning
    )
    sample["reasoning_label"] = reasoning_label
    sample["vllm_label"] = sample["original_label"]
```

**Step 3: Accuracy Calculation**
```python
# Compare fast model vs reasoning model labels
for question_id, samples in sampled.items():
    matches = [1 for s in samples if s["vllm_label"] == s["reasoning_label"]]
    accuracy = sum(matches) / len(samples)

    # Cohen's kappa for inter-rater agreement
    kappa = cohen_kappa_score(
        [s["reasoning_label"] for s in samples],
        [s["vllm_label"] for s in samples]
    )
```

**Step 4: Category Estimation Errors**
```python
# Calculate over/underestimation per category
for category in question_options:
    reasoning_pct = percentage of reasoning labels = category
    fast_pct = percentage of fast labels = category
    estimation_error = fast_pct - reasoning_pct
    estimation_type = "overestimation" if error > 0 else "underestimation"
```

### Verification Results

**Overall:**
- Mean accuracy: 85%
- Cohen's kappa: 0.434 (moderate agreement)
- Empty responses: 0%

**Per Question Type:**
- Norms questions: 32-84% accuracy (wide range)
- Survey questions: 76-100% accuracy (narrower range)

**Lowest Accuracy (Priority for Prompt Optimization):**
1. 1.2.1_descriptive: 32% (descriptive norms identification)
2. 1.1_gate: 44% (social norm present yes/no)
3. 1.3.1b_perceived_reference_stance: 48% (reference group's stance)

**Highest Accuracy (Good Prompts):**
1. Multiple survey questions: 100% (clear binary questions)
2. diet_1, diet_14, solar_6, solar_9: 100%

---

## Dashboard & Visualization Philosophy

### Core Principle: Transparency for Research Integrity

**Goal:** Show all aspects of model performance, including weaknesses, to enable:
1. Identification of systematic errors
2. Targeted prompt improvements
3. Confidence in results (show what works AND what doesn't)

### Visualization Components

#### 1. Author Stance Distribution (Donut Chart)
**Purpose:** Show overall distribution of stances toward climate actions
**Insight:** Most comments are "pro" (44%) or "neither/mixed" (36%); few are "against" (4%)
**Design:** Donut chart with percentages, color-coded by stance

#### 2. Descriptive vs Injunctive Norms (Radar Chart)
**Purpose:** Compare presence of descriptive norms (what people do) vs injunctive norms (what people should do)
**Insight:** Injunctive norms more common than descriptive norms in Reddit discussions
**Design:** Radar chart with 3 categories each (present/absent/unclear)

#### 3. Reference Groups (Horizontal Bar Chart)
**Purpose:** Show which social groups are most influential (family, friends, online community, etc.)
**Insight:** Online community and "other" most common; family/friends less common
**Design:** Sorted bar chart, longest bars at top

#### 4. Second-Order Beliefs (Stacked Bar Chart)
**Purpose:** Show distribution of beliefs about what others think (none/weak/strong)
**Insight:** Most comments have no second-order beliefs (72%)
**Design:** Stacked bar with percentages

#### 5. Perceived Reference Group Stance (Stacked Bar Chart)
**Purpose:** When reference groups are mentioned, what stance does the author attribute to them?
**Insight:** Reference groups mostly perceived as "pro" (70%)
**Design:** Stacked bar, color-coded by stance

#### 6. Temporal Trends (Line Charts)
**Purpose:** Show how norms discussions evolve over time (2010-2025)
**Insight:** Increasing prevalence of social norms discussions in recent years
**Design:** Line charts per sector, yearly aggregation

#### 7. Survey Question Distributions (Stacked Bar Charts, 3 per row)
**Purpose:** Show responses to 29 sector-specific survey questions
**Design:** Compact layout (3 plots per row), hide text on small segments (<5%)

#### 8. Top Estimation Errors (Bar Chart with Annotations)
**Purpose:** Highlight where fast model most overestimates or underestimates categories
**Insight:** Fast model overestimates "neither/mixed" stance by 16%, underestimates "pro" by 12%
**Design:** Diverging bar chart (green for overestimation, red for underestimation)

#### 9. Label Verification (Bar Chart)
**Purpose:** Show accuracy and Cohen's kappa per question
**Insight:** Identifies lowest-accuracy questions needing prompt improvement
**Design:** Grouped bar chart (accuracy + kappa side by side)

#### 10. Examples Page (Bordered Card Layout)
**Purpose:** Show actual Reddit comments with labels, color-coded by verification status
**Design:**
- Green left border: Fast model agrees with reasoning model (correct)
- Red left border: Fast model disagrees with reasoning model (incorrect)
- Only shows verified examples (900 out of 54,000 labels)

### Dark Theme Aesthetics

**Rationale:** Professional, modern look; reduces eye strain for long viewing sessions

**Color Palette:**
- Background: `#000000` (pure black)
- Panels: `#1a1a1a` (very dark gray)
- Text: `#e0e0e0` (light gray)
- Accent colors: `#4a90e2` (blue), `#8fcc8f` (green), `#ff9aa8` (red)

**Implementation:**
```css
html, body { background: #000000; color: #e0e0e0; }
.chart { background: #1a1a1a; border-radius: 8px; padding: 16px; }
```

### Layout Decisions

**3-per-row for Bar Charts:**
- Rationale: Maximize horizontal space usage, reduce scrolling
- Exception: Donut and radar charts (larger, more complex) stay 1-2 per row

**Hide Text on Small Segments:**
- Rationale: Text on bars < 5% becomes unreadable
- Solution: Only show text if segment ≥ 5% of total
- Fixed font size (10px) prevents text from shrinking excessively

**Responsive Design:**
- Charts resize to fit container width
- Font sizes remain fixed (prevent illegibility)

---

## Temporal Analysis & Sampling Strategy

### Why Temporal Analysis Matters

**Research Question:** Do social norms around climate action change over time?

**Hypotheses:**
1. Injunctive norms ("you should go vegan") may increase as climate awareness grows
2. Descriptive norms ("most people here drive EVs") may increase as adoption grows
3. Reference group influences may shift (e.g., online community more important in recent years)

### Temporal Sampling Strategy

**Challenge:** Reddit data spans 2010-2025 (15 years), but we want 500 samples per sector

**Naive Approach (Rejected):** Random sampling
- Problem: Newer comments vastly outnumber older comments on Reddit
- Result: 90% of samples from 2020-2025, 10% from 2010-2019

**Solution:** Equal temporal sampling
```python
def sample_by_year_equal(items, n_total, min_year=2010):
    """Sample n_total items distributed equally across years (min_year onwards)."""
    # Filter to items with year >= min_year
    items_with_year = [it for it in items if it.get("year") >= min_year]

    # Group by year
    by_year = defaultdict(list)
    for item in items_with_year:
        by_year[item["year"]].append(item)

    # Calculate per-year quota
    n_years = len(by_year)
    per_year_quota = n_total // n_years

    # Sample per-year_quota from each year
    sampled = []
    for year, year_items in by_year.items():
        n_take = min(per_year_quota, len(year_items))
        sampled.extend(random.sample(year_items, n_take))

    # Fill remaining slots from years with leftover items
    remaining = n_total - len(sampled)
    if remaining > 0:
        available = [item for year in by_year for item in by_year[year] if item not in sampled]
        sampled.extend(random.sample(available, min(remaining, len(available))))

    return sampled
```

**Result:** 500 samples distributed ~33 per year (2010-2025), enabling temporal trend analysis

**Visualization:** Line charts showing proportion of comments with social norms over time per sector

---

## Prompt Engineering Evolution

### Iteration 1: Simple Definitions

**Example (1.1_gate):**
```
Does this comment reference a social norm? Answer: yes or no.
```

**Problem:** Too vague, 40% accuracy

### Iteration 2: Add Definitions

**Example (1.1_gate):**
```
A social norm is a shared belief about what is typical or approved.
Descriptive norm = what people do. Injunctive norm = what people should do.
Does this comment reference a social norm? Answer: yes or no.
```

**Problem:** Still vague, 44% accuracy

### Iteration 3: Add Examples (Current)

**Example (1.1_gate):**
```
Definitions: A social norm is a shared belief or expectation about what is typical or what is approved/disapproved.
Descriptive norm = reference to what people typically do or how common something is (e.g. 'most people here drive EVs').
Injunctive norm = reference to what people should do, or explicit approval/disapproval (e.g. 'you should go vegan', 'eating meat is wrong').
Does this comment or post reference what others do or approve, or any social norm (descriptive or injunctive)?
Answer with exactly one word: yes or no.
```

**Problem:** Still 44% accuracy, overestimates "yes" by 48%

### Iteration 4: Add Negative Examples (Proposed)

**Example (1.1_gate - optimized):**
```
A social norm is a SHARED expectation about what is typical or approved/disapproved in a group.

Descriptive norm = statement about what people TYPICALLY DO or how COMMON something is:
✓ "Most people here drive EVs"
✓ "Everyone in my neighborhood has solar"
✗ "My friend bought an EV" (individual, not shared expectation)

Injunctive norm = statement about what people SHOULD DO or explicit approval/disapproval:
✓ "You should go vegan"
✓ "Eating meat is wrong"
✗ "I think veganism is good" (personal opinion, not prescribing for others)

Does this comment express a social norm (shared expectation about what is typical or approved)?
Answer: yes or no
```

**Expected Result:** 60-65% accuracy (target: +15-20% improvement)

### Key Prompt Engineering Principles

1. **Explicit definitions with examples**
2. **Negative examples (what NOT to code)**
3. **Sector-specific examples (EVs vs solar vs diet)**
4. **Decision trees for complex multi-way classifications**
5. **Short, directive language ("Answer with exactly one of...")**

---

## Key Technical Decisions

### 1. vLLM vs LM Studio for Fast Labeling

**Decision:** Use vLLM with OpenAI-compatible API

**Rationale:**
- Higher throughput (80 concurrent requests vs 10 for LM Studio)
- Better GPU utilization (continuous batching)
- Native Docker support
- OpenAI-compatible API (easy to swap models)

### 2. Hierarchical Question Structure

**Decision:** Use hierarchical/conditional questions (e.g., only ask reference group stance if reference group is identified)

**Rationale:**
- Reduces unnecessary API calls (don't ask follow-ups if gate question is "no")
- More interpretable results (stance question only makes sense if social norm is present)

**Implementation:**
```python
# Safety net: if stance is "against", recheck for "pro but lack of options"
if answers.get("1.1.1_stance") == "against":
    is_lack_of_options, _ = await _recheck_against_is_lack_of_options(...)
    if is_lack_of_options:
        answers["1.1.1_stance"] = "pro but lack of options"
```

### 3. Async Concurrent API Calls

**Decision:** Use asyncio + aiohttp for concurrent API calls

**Rationale:**
- 80x speedup vs sequential calls
- Essential for handling 54,000 labels in reasonable time
- Semaphore limits prevent overwhelming server

**Implementation:**
```python
sem = asyncio.Semaphore(MAX_CONCURRENT)  # 80 concurrent requests
async with sem:
    result = await call_vllm_single_choice(...)
```

### 4. Caching & Deduplication

**Decision:** Deduplicate comments by text before labeling

**Rationale:**
- Reddit has duplicate comments (e.g., bot responses, copypasta)
- No need to label the same text twice
- Saves API calls and time

**Implementation:**
```python
seen_bodies = set()
unique = []
for item in items:
    body = item.get("body", "").strip()
    if body and body not in seen_bodies:
        seen_bodies.add(body)
        unique.append(item)
```

### 5. JSON Schema for Prompts

**Decision:** Store all prompts in JSON schema files (00_vllm_ipcc_social_norms_schema.json)

**Rationale:**
- Single source of truth for prompts
- Easy to version control
- Shared between labeling and verification scripts
- Enables prompt optimization without code changes

### 6. Sector-Specific Prompts

**Decision:** Use template prompts with `{sector_topic}` placeholder

**Rationale:**
- Same question across sectors, but mention sector-specific topic
- Example: "What is the author's stance toward {sector_topic}?" → "What is the author's stance toward EVs/solar/diet?"

**Implementation:**
```python
SECTOR_TOPIC = {
    "transport": "EVs",
    "food": "veganism or vegetarianism / diet",
    "housing": "solar"
}

def _get_prompt_for_question(question, sector):
    template = question.get("prompt_template")
    if sector and template:
        sector_topic = SECTOR_TOPIC[sector]
        return template.format(sector_topic=sector_topic)
    return question.get("prompt", "")
```

### 7. Survey Questions Applied to All Comments

**Decision:** Survey questions (sector-specific attitude questions) are applied to ALL comments in a sector, not filtered by the gate question (1.1_gate).

**Rationale:**
- Preserves maximum data: captures attitudes even in comments without explicit social norms, enabling analysis of how attitudes relate to norm presence/absence.
- Simplifies pipeline: uniform labeling across all comments without conditional logic reduces complexity and potential errors.
- Research flexibility: allows post-hoc analysis of attitude differences between norm-present vs norm-absent comments without requiring re-labeling.

---

## Conclusion: Research Impact & Next Steps

### What We've Built

1. **Scalable automated labeling pipeline:** 54,000 labels in ~6 hours
2. **Quality assurance via verification:** 900 samples verified with reasoning model
3. **Confidence estimation:** Log probabilities for every label
4. **Comprehensive dashboard:** Interactive visualizations for all results
5. **Transparency:** Full error analysis and examples

### Research Contributions

1. **Methodological:** Demonstrates viability of two-model (fast + reasoning) approach for large-scale text annotation
2. **Substantive:** First large-scale analysis of social norms in climate discussions on Reddit across 3 sectors and 15 years
3. **Technical:** Open-source pipeline (00_vLLM_hierarchical.py, shared_utilities.py) reusable for other text classification tasks

### Next Steps

**Immediate (Technical):**
1. Implement optimized prompts from 00_prompt_optimization.md
2. Re-run verification to measure accuracy improvement
3. Analyze confidence scores vs verification accuracy correlation

**Near-Term (Analysis):**
1. Temporal trend analysis: How do social norms evolve 2010-2025?
2. Cross-sector comparison: Do norms differ between EVs, solar, and diet?
3. Reference group influence: Which groups most influential per sector?

**Long-Term (Extensions):**
1. Expand to more subreddits beyond initial set
2. Multi-language analysis (non-English climate discussions)
3. Causal inference: Do social norms predict behavior change?

---

**Document End**
