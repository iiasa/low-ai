# Survey-frame Reddit labeling pipeline — running notes (vllm_test)

## 0) Objective
0.1 Use *survey questions* as an exogenous, empirical frame set.
0.2 Measure, in Reddit text, two distinct random variables:
- **R(q, x)** = relevance of comment x to survey item q.
- **S(q, x)** = stance / outcome category *conditional on relevance*.
0.3 Distill expensive teacher LLM decisions into fast student encoders (LoRA) for scale.

---

## 1) Current state in `vllm_test.py` (what is true today)
1.1 Pass 1 (relevance)
- Prompt asks: “Is statement relevant to survey question?”
- Output schema forces JSON: `{ "relevant": "yes|no" }`.

1.2 Pass 2 (classification)
- Prompt asks: “Does this statement support/agree/align with this survey question?”
- Output schema still forces `{ "relevant": "yes|no" }`.
- For pairs marked irrelevant in Pass 1, Pass 2 fills `'no'` by default.

1.3 Scientific consequence
- The key `relevant` is used for two different semantics across passes.
- Pass-2 `'no'` conflates:
  - irrelevant pairs (filtered out), and
  - relevant-but-not-supporting pairs.

---

## 2) Primary repair: make an explicit ontology of base topics (not per-variant yes/no)
2.1 Motivation
- Survey items often come in mutually exclusive variants (help vs hurt vs no difference).
- Classifying variants independently is noisy and can be logically inconsistent.

2.2 Proposed ontology (base topic → label set)
2.2.1 **Local_Economy** → {`help`, `hurt`, `no_difference`, `mixed`, `mention_only`, `uncertain`}
2.2.2 **Landscape** → {`unattractive`, `not_unattractive`, `mixed`, `mention_only`, `uncertain`}
2.2.3 **Space** → {`too_much`, `acceptable`, `mixed`, `mention_only`, `uncertain`}
2.2.4 **Utility_Bill** → {`lower`, `higher`, `mixed`, `mention_only`, `uncertain`}
2.2.5 **Tax_Revenue** → {`help`, `hurt`, `mention_only`, `uncertain`} (adjust if you have a “no_difference” variant)

2.3 Deterministic mapping back to original survey-item IDs
- Example for Local_Economy:
  - `Local_Economy_Help = yes` iff label==`help`
  - `Local_Economy_Hurt = yes` iff label==`hurt`
  - `Local_Economy_No_Difference = yes` iff label==`no_difference`
  - otherwise all three = `no` and keep base label for analysis.

2.4 Payoff
- Coherent outcomes by construction.
- Lower label noise.
- Clear semantics for distillation.

---

## 3) Prompt corrections (method-first)
3.1 Pass-1 relevance prompt: tighten the definition of “relevant”
- Relevant = the statement expresses *a belief, evaluation, prediction, experience, or concern* about the proposition(s) in q.
- Not relevant = merely mentions renewables/climate without touching the q-specific attribute.

3.2 Pass-2 prompt: stop asking “support yes/no” for a variant question
- Ask for one label from the base-topic label set (Section 2).
- Add `mention_only` explicitly (relevant but no directional commitment).
- Add `uncertain` explicitly (insufficient evidence).

3.3 Evidence anchoring (auditability)
- Require “evidence spans” (short verbatim fragments) as a field.
- This enables: (i) human audit, (ii) debugging systematic failure modes.

3.4 Schema design recommendation
- Use different keys for different variables:
  - Pass 1: `{ "relevant": "yes|no|uncertain" }`
  - Pass 2 (base topic): `{ "label": "...", "evidence": ["..."], "confidence": 0-1 }`

---

## 4) Step-1 relevance: can we do better with embeddings?
Yes. Treat relevance as approximate nearest-neighbor retrieval in a semantic space, then (optionally) verify.

4.1 Prototype embedding method (fast, interpretable)
4.1.1 For each base topic t (or each survey item q), define a small set of **prototypes**:
- `p_yes`: 5–20 short, unambiguous statements that are maximally relevant.
- `p_no`: 5–20 confusable-but-irrelevant statements.
4.1.2 Embed all prototypes and each comment.
4.1.3 Score relevance by similarity margin:
- `score = max_sim(x, p_yes) - max_sim(x, p_no)`
4.1.4 Threshold with a validation set.

4.2 Hybrid retrieval (what an ML person would do)
4.2.1 Stage A (retrieval): embeddings + BM25 keyword triggers.
4.2.2 Stage B (verification): a small cross-encoder (MiniLM / DistilRoBERTa) trained on labeled pairs.

4.3 Why this is “scientific”
- Embeddings define an explicit measurement geometry.
- Thresholds can be calibrated.
- Prototypes give mechanistic interpretability (“nearest evidence”).

---

## 5) Step-2 stance: can we do something analogous with prototypes / poles?
Yes.

5.1 Pole prototypes per base topic
- For each label in the ontology (e.g., Local_Economy: help/hurt/no_difference), define 5–20 pole exemplars.
- Then classify by nearest-pole similarity *conditional on relevance*.

5.2 Pairwise model is cleaner (recommended for the student)
- Input: `[CLS] (survey item or base-topic description) [SEP] comment [SEP]`.
- Output: softmax over ontology labels.
- This generalizes to new questions and reduces per-topic overfitting.

---

## 6) Can we use logprobs / completion-style scoring?
Yes, if your inference stack exposes token logprobs reliably.

6.1 Logprob-as-classifier (teacher or student)
6.1.1 Prompt ends with: `label =` and restrict labels to a small set.
6.1.2 Compute logprob for each candidate label token/string.
6.1.3 Choose argmax; use normalized margin as confidence.

6.2 Benefits
- Produces a *real-valued* confidence aligned with the model’s internal distribution.
- Removes JSON parsing fragility.

6.3 Caveat
- Requires stable tokenization + consistent label strings.

---

## 7) Teacher–student distillation plan (how an ML expert would do it)
7.1 Data construction
7.1.1 Candidate generation (retrieval): for each comment, propose top-k base topics.
7.1.2 Teacher labeling: for each (topic, comment) pair, get:
- Pass 1: relevance (yes/no/uncertain)
- Pass 2: ontology label (help/hurt/…/mention_only/uncertain)
- optional: evidence spans
- optional: confidence/logprob margin

7.2 Student models
7.2.1 **Relevance student**: binary/3-way classifier on (topic, comment).
7.2.2 **Stance student**: multiclass classifier on (topic, comment), trained only on teacher-relevant pairs.

7.3 Distillation losses
7.3.1 Hard labels (cross-entropy) + soft targets (teacher confidence).
7.3.2 Calibration objective (temperature scaling) on a small human-labeled set.

7.4 Active learning loop
- Sample more teacher labels where student uncertainty is high or where prototypes disagree.
- 2–3 rounds usually saturate.

---

## 8) Measurement reporting (what you publish)
8.1 Separate channel variables
- Report **visibility**: P(relevant) per topic.
- Report **stance distribution**: P(label | relevant) per topic.

8.2 Survey comparability
- Map ontology labels to survey response bins deterministically.
- Always condition on relevance when comparing “agree rates” to survey.

8.3 Robustness
- Run multiple teachers (or multiple seeds) and report agreement.
- Audit a stratified sample (high-confidence, borderline, rejected-but-close).

---

## 9) Concrete change list (next edits)
9.1 Rename semantics (even if you keep the JSON key)
- Decide: does `relevant` in pass-2 mean “aligns”? If yes, rename key or document clearly.

9.2 Replace pass-2 per-variant with pass-2 per-base-topic
- Implement ontology labels + deterministic mapping.

9.3 Introduce `mention_only` and `uncertain` states
- Critical for identifiability.

9.4 Optional: add embedding prototype retrieval before pass-1
- Use as a filter or as a candidate generator.

9.5 Preserve both pass outputs
- Never collapse irrelevant into pass-2 `no` without keeping pass-1.

---

## 10) Open questions to resolve (quick)
10.1 Do you want “policy support” (build solar farm) as its own base topic distinct from attribute topics (economy, space, landscape)?
10.2 Do you want to treat rhetorical questions / sarcasm as `mixed` or `uncertain` by default?
10.3 For the paper: do you report mention_only as part of relevance (yes) or as stance (a separate bin)?

