"""
temp.py - Comprehensive visualization experiments with all dashboard data
"""
import json
from collections import defaultdict

# Load data
with open("paper4data/00_verification_results.json", "r", encoding="utf-8") as f:
    vr = json.load(f)

with open("paper4data/00_verification_samples.json", "r", encoding="utf-8") as f:
    vs = json.load(f)

# Load full labeled data for temporal analysis
with open("paper4data/norms_labels.json", "r", encoding="utf-8") as f:
    labeled_data = json.load(f)

# Load survey question metadata for labels
with open("00_vllm_survey_question_final.json", "r", encoding="utf-8") as f:
    survey_meta = json.load(f)

# ═══════════════════════════════════════════════════════════════════
# Compute temporal data for all variables
# ═══════════════════════════════════════════════════════════════════
YEARS = list(range(2010, 2025))

# Survey question short labels
SURVEY_LABELS = {}
SURVEY_IDS = {"food": [], "transport": [], "housing": []}
for sector_key, sec_lower in [("FOOD", "food"), ("TRANSPORT", "transport"), ("HOUSING", "housing")]:
    for frame_data in survey_meta[sector_key].values():
        for q in frame_data["questions"]:
            qid = q["id"]
            # Short label from wording
            wording = q["wording"]
            # Truncate to ~30 chars
            if len(wording) > 35:
                wording = wording[:32] + "..."
            SURVEY_LABELS[qid] = wording
            SURVEY_IDS[sec_lower].append(qid)

# Compute survey question proportions by year per sector
# temporal_survey[sector][qid] = {year: proportion_yes}
temporal_survey = {}
for sec in ["food", "transport", "housing"]:
    temporal_survey[sec] = {}
    for qid in SURVEY_IDS[sec]:
        year_yes = defaultdict(int)
        year_total = defaultdict(int)
        for rec in labeled_data[sec]:
            y = rec.get("year")
            if y is None or y < 2010:
                continue
            ans = rec.get("answers", {}).get(qid)
            if ans is not None:
                year_total[y] += 1
                if str(ans).strip().lower() == "yes":
                    year_yes[y] += 1
        props = {}
        for y in YEARS:
            if year_total[y] > 0:
                props[y] = round(year_yes[y] / year_total[y] * 100, 1)
            else:
                props[y] = 0
        temporal_survey[sec][qid] = props

# Compute norm stance proportions by year per sector (already in dashboard but as counts)
# temporal_stance[sector] = {stance_label: {year: proportion}}
STANCE_LABELS = ["pro", "against", "against particular but pro", "neither/mixed", "pro but lack of options"]
STANCE_COLORS = {"pro": "#8fcc8f", "against": "#ff9aa8", "against particular but pro": "#ffb87a",
                 "neither/mixed": "#ffe87a", "pro but lack of options": "#8fbfd9"}

temporal_stance = {}
for sec in ["food", "transport", "housing"]:
    temporal_stance[sec] = {sl: {} for sl in STANCE_LABELS}
    year_counts = defaultdict(lambda: defaultdict(int))
    year_totals = defaultdict(int)
    for rec in labeled_data[sec]:
        y = rec.get("year")
        if y is None or y < 2010:
            continue
        stance = rec.get("answers", {}).get("1.1.1_stance", "")
        if stance:
            year_counts[y][stance.strip().lower()] += 1
            year_totals[y] += 1
    for sl in STANCE_LABELS:
        for y in YEARS:
            if year_totals[y] > 0:
                temporal_stance[sec][sl][y] = round(year_counts[y].get(sl, 0) / year_totals[y] * 100, 1)
            else:
                temporal_stance[sec][sl][y] = 0

# Compute norm dimension proportions by year per sector
# For descriptive/injunctive: "explicitly present" vs "absent" vs "unclear"
# For second_order: "2" (strong) vs "1" (weak) vs "0" (none)
NORM_DIMS = {
    "1.2.1_descriptive": {"title": "Descriptive Norm", "cats": ["explicitly present", "absent", "unclear"],
                          "colors": ["#7caed6", "#566a7a", "#8a9aa8"]},
    "1.2.2_injunctive": {"title": "Injunctive Norm", "cats": ["present", "absent", "unclear"],
                         "colors": ["#8fbfd9", "#566a7a", "#8a9aa8"]},
    "1.3.3_second_order": {"title": "Second-Order Belief", "cats": ["2", "1", "0"],
                           "display": ["strong", "weak", "none"],
                           "colors": ["#b8a8c8", "#8fbfd9", "#8a9aa8"]},
}

temporal_norms = {}
for sec in ["food", "transport", "housing"]:
    temporal_norms[sec] = {}
    for qid, meta in NORM_DIMS.items():
        temporal_norms[sec][qid] = {cat: {} for cat in meta["cats"]}
        year_counts = defaultdict(lambda: defaultdict(int))
        year_totals = defaultdict(int)
        for rec in labeled_data[sec]:
            y = rec.get("year")
            if y is None or y < 2010:
                continue
            val = rec.get("answers", {}).get(qid, "")
            if val is not None:
                val_str = str(val).strip().lower()
                year_counts[y][val_str] += 1
                year_totals[y] += 1
        for cat in meta["cats"]:
            for y in YEARS:
                if year_totals[y] > 0:
                    temporal_norms[sec][qid][cat][y] = round(year_counts[y].get(cat, 0) / year_totals[y] * 100, 1)
                else:
                    temporal_norms[sec][qid][cat][y] = 0

# Serialize temporal data to JSON for JS injection
temporal_survey_js = json.dumps(temporal_survey)
temporal_stance_js = json.dumps(temporal_stance)
temporal_norms_js = json.dumps(temporal_norms)
norm_dims_js = json.dumps(NORM_DIMS)
survey_labels_js = json.dumps(SURVEY_LABELS)
survey_ids_js = json.dumps(SURVEY_IDS)
years_js = json.dumps(YEARS)

# ═══════════════════════════════════════════════════════════════════
# Compute dynamic chart data from labeled_data (gauge/treemap/sankey/bubble/radial)
# ═══════════════════════════════════════════════════════════════════
# 1. Gate % (gauge)
gate_pct = {}
for _sec in ["food", "transport", "housing"]:
    _total = len(labeled_data[_sec])
    _yes = sum(1 for _r in labeled_data[_sec]
               if str(_r.get("answers", {}).get("1.1_gate", "")).strip().lower()
               in ("yes", "present", "1"))
    gate_pct[_sec] = round(_yes / _total * 100, 1) if _total else 0

# 2. Author stance counts (treemap)
stance_counts = {}
for _sec in ["food", "transport", "housing"]:
    _c = defaultdict(int)
    for _r in labeled_data[_sec]:
        _v = str(_r.get("answers", {}).get("1.1.1_stance", "")).strip().lower()
        if _v: _c[_v] += 1
    stance_counts[_sec.capitalize()] = {
        "pro":           _c.get("pro", 0),
        "against":       _c.get("against", 0),
        "abp":           _c.get("against particular but pro", 0),
        "neither/mixed": _c.get("neither/mixed", 0),
        "plo":           _c.get("pro but lack of options", 0),
    }

# 3. Sankey per-sector dimension counts
sankey_data = {}
for _sec in ["food", "transport", "housing"]:
    _dc, _ic, _sc = defaultdict(int), defaultdict(int), defaultdict(int)
    for _r in labeled_data[_sec]:
        _ans = _r.get("answers", {})
        _v = str(_ans.get("1.2.1_descriptive", "")).strip().lower()
        if _v: _dc[_v] += 1
        _v = str(_ans.get("1.2.2_injunctive", "")).strip().lower()
        if _v: _ic[_v] += 1
        _v = str(_ans.get("1.3.3_second_order", "")).strip()
        if _v: _sc[_v] += 1
    sankey_data[_sec.capitalize()] = {
        "desc": [_dc.get("explicitly present", 0), _dc.get("absent", 0), _dc.get("unclear", 0)],
        "inj":  [_ic.get("present", 0), _ic.get("absent", 0), _ic.get("unclear", 0)],
        "sec":  [_sc.get("2", 0), _sc.get("1", 0), _sc.get("0", 0)],
    }

# 4. Reference group bubble counts
_REF_CATS = ['family','partner/spouse','friends','coworkers','neighbors',
             'local community','political tribe','online community','other reddit user']
bubble_counts = {}
for _sec in ["food", "transport", "housing"]:
    _c = defaultdict(int)
    for _r in labeled_data[_sec]:
        _v = str(_r.get("answers", {}).get("1.3.1_reference_group", "")).strip().lower()
        if _v in _REF_CATS: _c[_v] += 1
    bubble_counts[_sec.capitalize()] = [_c.get(_cat, 0) for _cat in _REF_CATS]

# 5. Sector totals
sector_totals = {
    "Food":      len(labeled_data["food"]),
    "Transport": len(labeled_data["transport"]),
    "Housing":   len(labeled_data["housing"]),
}

# 6. Radial chart values (% yes per survey question, same order as SURVEY_IDS)
radial_vals = {}
for _sec in ["food", "transport", "housing"]:
    _total = len(labeled_data[_sec])
    _pcts = []
    for _qid in SURVEY_IDS[_sec]:
        _yes = sum(1 for _r in labeled_data[_sec]
                   if str(_r.get("answers", {}).get(_qid, "")).strip().lower() == "yes")
        _pcts.append(round(_yes / _total * 100, 1) if _total else 0)
    radial_vals[_sec] = _pcts

_food_max_r      = max(5, round(max(radial_vals["food"],      default=25) * 1.3 / 5) * 5)
_transport_max_r = max(5, round(max(radial_vals["transport"], default=10) * 1.3 / 5) * 5)
_housing_max_r   = max(5, round(max(radial_vals["housing"],   default=15) * 1.3 / 5) * 5)

# 7. Descriptive labels and full-question tooltips for radial charts
def _radial_label(q):
    """Use short_form directly (already captures question essence); fall back to wording."""
    sf = (q.get("short_form") or "").strip()
    if sf:
        return sf
    w = q.get("wording", "").strip().rstrip("?.,:;")
    words = w.split()
    return " ".join(words[:5])

radial_display_labels = {}
radial_hover_tooltips = {}
# Food: merge willingness-to-reduce-meat/dairy spokes into one
_FOOD_MERGE_IDS = {"diet_4", "diet_5", "diet_6", "diet_8"}
_FOOD_MERGE_LABEL = "willingness - low-carbon diet"
for _sec_lower, _sec_key in [("food","FOOD"),("transport","TRANSPORT"),("housing","HOUSING")]:
    _qid_to_q = {}
    for _frame in survey_meta[_sec_key].values():
        for _q in _frame["questions"]:
            _qid_to_q[_q["id"]] = _q
    _triples = []  # (pct, label, tooltip)
    _merge_pcts, _merge_qs = [], []
    for _qid, _pct in zip(SURVEY_IDS[_sec_lower], radial_vals[_sec_lower]):
        if _sec_lower == "food" and _qid in _FOOD_MERGE_IDS:
            _merge_pcts.append(_pct)
            _merge_qs.append(_qid_to_q.get(_qid, {}))
            continue
        _q = _qid_to_q.get(_qid, {})
        _triples.append((_pct, _radial_label(_q), _q.get("wording", _qid).strip()))
    if _merge_pcts:  # insert merged spoke
        _merged_pct = round(sum(_merge_pcts) / len(_merge_pcts), 1)
        _merged_tip = " · ".join(q.get("wording", "").strip().rstrip("?") for q in _merge_qs if q.get("wording"))
        _triples.append((_merged_pct, _FOOD_MERGE_LABEL, _merged_tip))
    _triples.sort(key=lambda x: x[0])  # ascending % — lowest to highest
    radial_vals[_sec_lower]              = [t[0] for t in _triples]
    radial_display_labels[_sec_lower]   = [t[1] for t in _triples]
    radial_hover_tooltips[_sec_lower]   = [t[2] for t in _triples]

radial_food_labels_js       = json.dumps(radial_display_labels["food"])
radial_transport_labels_js  = json.dumps(radial_display_labels["transport"])
radial_housing_labels_js    = json.dumps(radial_display_labels["housing"])
radial_food_tips_js         = json.dumps(radial_hover_tooltips["food"])
radial_transport_tips_js    = json.dumps(radial_hover_tooltips["transport"])
radial_housing_tips_js      = json.dumps(radial_hover_tooltips["housing"])

# Serialize
gate_pct_js          = json.dumps(gate_pct)
stance_counts_js     = json.dumps(stance_counts)
sankey_data_js       = json.dumps(sankey_data)
bubble_food_js       = json.dumps(bubble_counts.get("Food", []))
bubble_transport_js  = json.dumps(bubble_counts.get("Transport", []))
bubble_housing_js    = json.dumps(bubble_counts.get("Housing", []))
sector_totals_js     = json.dumps(sector_totals)
radial_food_vals_js      = json.dumps(radial_vals["food"])
radial_transport_vals_js = json.dumps(radial_vals["transport"])
radial_housing_vals_js   = json.dumps(radial_vals["housing"])

# ═══════════════════════════════════════════════════════════════════
# Build examples HTML from verification samples - 9-column layout
# ═══════════════════════════════════════════════════════════════════
import math as _math

# Load norms schema for prompt text
with open("00_vllm_ipcc_social_norms_schema.json", "r", encoding="utf-8") as f:
    norms_schema = json.load(f)
norms_prompts = {q["id"]: q["prompt"] for q in norms_schema["norms_questions"]}

NORMS_Q_LABELS = [
    ("1.1_gate", "Norm Signal (Gate)"),
    ("1.2.1_descriptive", "Descriptive Norm"),
    ("1.2.2_injunctive", "Injunctive Norm"),
    ("1.3.1_reference_group", "Reference Group"),
    ("1.3.3_second_order", "Second-Order Belief"),
]

# Map raw vllm_label values to human-readable display names
LABEL_DISPLAY = {
    "1.1_gate": {"0": "absent", "1": "present"},
    "1.3.3_second_order": {"0": "none", "1": "weak", "2": "strong"},
}

ALL_QS = []
# Norms questions (all sectors)
for qid, label in NORMS_Q_LABELS:
    ALL_QS.append(("norms", qid, label, None))

# Survey questions (sector-specific)
survey_prompts = {}
for sector_key, sec_lower in [("FOOD", "food"), ("TRANSPORT", "transport"), ("HOUSING", "housing")]:
    for frame_data in survey_meta[sector_key].values():
        for q in frame_data["questions"]:
            qid = q["id"]
            display = q.get("short_form") or q.get("wording", qid)
            survey_prompts[qid] = q.get("prompt", "")
            ALL_QS.append(("survey", qid, display, sec_lower))

EX_SECTORS = ["food", "transport", "housing"]
EX_SEC_LABELS = {"food": "FOOD", "transport": "TRANSPORT", "housing": "HOUSING"}
EX_SEC_COLORS = {"food": "#5ab4ac", "transport": "#af8dc3", "housing": "#f4a460"}

# Count full-dataset label occurrences per question from labeled_data (9000 records)
full_label_counts = defaultdict(lambda: defaultdict(int))
for _sec in ["food", "transport", "housing"]:
    for _rec in labeled_data[_sec]:
        for _qid, _val in _rec.get("answers", {}).items():
            if _val is not None and str(_val).strip():
                full_label_counts[_qid][str(_val).strip().lower()] += 1

def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _get_conf(sample, qid):
    lp = sample.get("logprobs", {})
    for k in [qid, qid.replace(" ", "_"), qid.replace("_", " ")]:
        if k in lp:
            return _math.exp(lp[k])
    return None

examples_html = ""
for task_type, qid, display_label, sector_filter in ALL_QS:
    samples = vs.get(task_type, {}).get(qid, [])
    if not samples:
        continue

    # Accuracy
    task_results = vr.get("by_task", {}).get(task_type, {}).get(qid, {})
    acc = task_results.get("accuracy", 0)
    acc_color = "#8fcc8f" if acc >= 0.85 else "#ffb87a" if acc >= 0.70 else "#ff9aa8"

    # All unique answer labels (sorted)
    seen_labels = {}
    for s in samples:
        lbl = str(s.get("vllm_label", "")).strip()
        seen_labels[lbl] = seen_labels.get(lbl, 0) + 1
    all_labels = sorted(seen_labels.keys())

    # Group samples by (label, sector): 2 agree + 1 disagree examples
    # Collect all, split by agreement, then pick 2 agree + 1 disagree (fallback to available)
    _all_grouped = {}
    for s in samples:
        lbl = str(s.get("vllm_label", "")).strip()
        sec = str(s.get("sector", "")).strip().lower()
        key = (lbl, sec)
        if key not in _all_grouped:
            _all_grouped[key] = {"agree": [], "disagree": []}
        is_agree = str(s.get("vllm_label","")).strip().lower() == str(s.get("reasoning_label","")).strip().lower()
        bucket = "agree" if is_agree else "disagree"
        _all_grouped[key][bucket].append(s)
    grouped = {}
    for key, buckets in _all_grouped.items():
        agree2 = buckets["agree"][:2]
        disagree1 = buckets["disagree"][:1]
        # fallback: if not enough of one type, fill with the other
        combined = agree2 + disagree1
        if len(combined) < 3:
            extra = [s for s in (buckets["agree"][2:] + buckets["disagree"][1:]) if s not in combined]
            combined += extra[:3 - len(combined)]
        grouped[key] = combined

    # Prompt text
    if task_type == "norms":
        prompt_text = norms_prompts.get(qid, "")
    else:
        prompt_text = survey_prompts.get(qid, "")

    prompt_esc = _esc(prompt_text[:800] + ("\u2026" if len(prompt_text) > 800 else ""))
    prompt_drop = (
        f'<details class="ex-prompt-drop" open>'
        f'<summary>Read exact prompt &amp; choices sent to the LLM</summary>'
        f'<div class="ex-prompt-box">{prompt_esc}</div>'
        f'</details>'
    )

    # Column headers: sector name once (span 3) + Ex 1/2/3 subheaders
    sec_headers = ""
    col_headers = ""
    for sec in EX_SECTORS:
        color = EX_SEC_COLORS[sec]
        lbl_str = EX_SEC_LABELS[sec]
        if sector_filter and sector_filter != sec:
            sec_headers += '<div class="ex-sec-hdr ex-col-na" style="grid-column:span 3">—</div>'
            col_headers += '<div class="ex-col-hdr ex-col-na">—</div>' * 3
        else:
            sec_headers += f'<div class="ex-sec-hdr" style="color:{color};grid-column:span 3">{lbl_str}</div>'
            for ci in range(1, 4):
                col_headers += f'<div class="ex-col-hdr" style="color:{color}">Ex {ci}</div>'

    # Answer rows
    # Label display map for this question (e.g. "0"→"none" for second_order)
    q_label_map = LABEL_DISPLAY.get(qid, {})

    answer_rows_html = ""
    for lbl in all_labels:
        lbl_display = q_label_map.get(lbl, lbl)
        # Also map reasoning_label for display
        row_cells = ""
        for sec in EX_SECTORS:
            if sector_filter and sector_filter != sec:
                row_cells += '<div class="ex-cell ex-cell-na"></div>' * 3
            else:
                cell_samples = grouped.get((lbl, sec), [])
                for ci in range(3):
                    if ci < len(cell_samples):
                        s = cell_samples[ci]
                        comment_full = str(s.get("comment", ""))
                        comment = comment_full[:220] + ("\u2026" if len(comment_full) > 220 else "")
                        is_match = str(s.get("vllm_label","")).strip().lower() == str(s.get("reasoning_label","")).strip().lower()
                        border_color = "#4ade80" if is_match else "#f87171"
                        conf = _get_conf(s, qid)
                        conf_badge = f'<span class="ex-conf">{conf*100:.0f}%</span>' if conf is not None else ""
                        rsn_raw = str(s.get("reasoning_label",""))
                        rsn = q_label_map.get(rsn_raw, rsn_raw)[:25]
                        row_cells += (
                            f'<div class="ex-cell" style="border-left:3px solid {border_color}">'
                            f'{conf_badge}'
                            f'<div class="ex-comment">{_esc(comment)}</div>'
                            f'<div class="ex-reason-lbl">Verify: <b>{_esc(rsn)}</b></div>'
                            f'</div>'
                        )
                    else:
                        row_cells += '<div class="ex-cell ex-cell-empty"></div>'

        lbl_n = full_label_counts[qid].get(lbl.lower(), 0)
        answer_rows_html += (
            f'<div class="ex-answer-row">'
            f'<div class="ex-answer-label">{_esc(lbl_display)}'
            f' <span class="ex-label-n">n={lbl_n:,}</span></div>'
            f'<div class="ex-9grid">{row_cells}</div>'
            f'</div>'
        )

    n_total = sum(full_label_counts[qid].values()) if qid in full_label_counts else len(samples)
    examples_html += (
        f'<details class="ex-section">'
        f'<summary>{_esc(display_label)}'
        f' <span class="ex-sum-meta">'
        f'<span style="color:#4a6a8a">{_esc(qid)}</span>'
        f'&nbsp;&nbsp;n={n_total:,}'
        f'&nbsp;&nbsp;acc=<span style="color:{acc_color};font-weight:700">{acc*100:.0f}%</span>'
        f'</span></summary>'
        f'<div class="ex-content">'
        f'{prompt_drop}'
        f'<div class="ex-9grid ex-sec-hdr-row">{sec_headers}</div>'
        f'<div class="ex-9grid ex-hdr-row">{col_headers}</div>'
        f'{answer_rows_html}'
        f'</div></details>\n'
    )

# ═══════════════════════════════════════════════════════════════════
# Build verification data for JS
# ═══════════════════════════════════════════════════════════════════
# Questions excluded from verification displays
SKIP_QIDS = {"1.3.1b_perceived_reference_stance"}

# Accuracy by question sorted
acc_data = []
for task_type in ["norms", "survey"]:
    for qid, m in vr["by_task"][task_type].items():
        if qid in SKIP_QIDS:
            continue
        acc_data.append({"q": qid[:35], "acc": m["accuracy"], "type": task_type})
acc_data.sort(key=lambda x: x["acc"])
acc_questions = json.dumps([d["q"] for d in acc_data])
acc_values = json.dumps([d["acc"] for d in acc_data])
acc_colors = json.dumps(["#ff9aa8" if d["acc"]<0.70 else "#ffb87a" if d["acc"]<0.85 else "#8fcc8f" for d in acc_data])

# Estimation errors - top 30
est_data = []
for task_type in ["norms", "survey"]:
    for qid, m in vr["by_task"][task_type].items():
        if qid in SKIP_QIDS:
            continue
        for cat, ce in m.get("category_estimation", {}).items():
            err = ce.get("estimation_error", 0)
            if abs(err) >= 2:
                short = qid[:25] + " - " + cat[:10]
                est_data.append({"label": short, "error": err})
est_data.sort(key=lambda x: -abs(x["error"]))
est_data = est_data[:40]
est_labels = json.dumps([d["label"] for d in est_data])
est_values = json.dumps([d["error"] for d in est_data])
est_colors = json.dumps(["#ff9aa8" if d["error"]>0 else "#5ab4ac" for d in est_data])

summary = vr["summary"]

# ═══════════════════════════════════════════════════════════════════
# Compute confidence metrics from samples (logprobs -> probability)
# ═══════════════════════════════════════════════════════════════════
import math

conf_match_vals = []
conf_mismatch_vals = []
# Per-question avg confidence and mismatch rate
q_conf_mismatch = []
# Per-question high-conf accuracy
q_highconf_acc = []

for task_type in ["norms", "survey"]:
    for qid, samples in vs.get(task_type, {}).items():
        if qid in SKIP_QIDS:
            continue
        q_confs = []
        q_matches = 0
        q_total = 0
        q_hc_match = 0
        q_hc_total = 0
        for s in samples:
            # Get logprob for this question
            lp = s.get("logprobs", {})
            # Try different key formats
            logprob = None
            for k in [qid, qid.replace(" ", "_")]:
                if k in lp:
                    logprob = lp[k]
                    break
            # Also try survey key format (e.g., "diet 4" -> "diet_4")
            if logprob is None:
                for k, v in lp.items():
                    if k.replace("_", " ") == qid or k == qid:
                        logprob = v
                        break
            if logprob is None:
                continue
            conf = math.exp(logprob)  # convert logprob to probability
            is_match = str(s.get("vllm_label","")).strip().lower() == str(s.get("reasoning_label","")).strip().lower()
            if is_match:
                conf_match_vals.append(conf)
            else:
                conf_mismatch_vals.append(conf)
            q_confs.append(conf)
            q_total += 1
            if is_match:
                q_matches += 1
            if conf > 0.9:
                q_hc_total += 1
                if is_match:
                    q_hc_match += 1
        if q_total > 0:
            avg_conf = sum(q_confs) / len(q_confs)
            mismatch_pct = (q_total - q_matches) / q_total * 100
            q_conf_mismatch.append({"q": qid[:30], "conf": avg_conf, "mismatch": mismatch_pct})
        if q_hc_total > 0:
            q_highconf_acc.append({"q": qid[:35], "acc": q_hc_match / q_hc_total, "n": q_hc_total})

avg_conf_match = sum(conf_match_vals) / len(conf_match_vals) if conf_match_vals else 0
avg_conf_mismatch = sum(conf_mismatch_vals) / len(conf_mismatch_vals) if conf_mismatch_vals else 0
high_conf_samples = sum(1 for c in conf_match_vals + conf_mismatch_vals if c > 0.9)
high_conf_matches = sum(1 for c in conf_match_vals if c > 0.9)
high_conf_mismatches = sum(1 for c in conf_mismatch_vals if c > 0.9)
high_conf_acc = high_conf_matches / high_conf_samples if high_conf_samples > 0 else 0

# Confidence bins for box plot
conf_bins = {"50-60%": [], "60-70%": [], "70-80%": [], "80-90%": [], "90-100%": []}
all_confs = [(c, True) for c in conf_match_vals] + [(c, False) for c in conf_mismatch_vals]
for c, is_m in all_confs:
    pct = c * 100
    if pct < 60: conf_bins["50-60%"].append(0 if is_m else 1)
    elif pct < 70: conf_bins["60-70%"].append(0 if is_m else 1)
    elif pct < 80: conf_bins["70-80%"].append(0 if is_m else 1)
    elif pct < 90: conf_bins["80-90%"].append(0 if is_m else 1)
    else: conf_bins["90-100%"].append(0 if is_m else 1)

bin_labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
bin_mismatch_pcts = []
for bl in bin_labels:
    vals = conf_bins[bl]
    bin_mismatch_pcts.append(sum(vals) / len(vals) * 100 if vals else 0)

# Sort high-conf accuracy
q_highconf_acc.sort(key=lambda x: x["acc"])
hc_questions = json.dumps([d["q"] for d in q_highconf_acc])
hc_values = json.dumps([d["acc"] for d in q_highconf_acc])
hc_colors = json.dumps(["#ff9aa8" if d["acc"]<0.70 else "#ffb87a" if d["acc"]<0.85 else "#8fcc8f" for d in q_highconf_acc])

# Scatter data
q_conf_mismatch.sort(key=lambda x: x["conf"])
scatter_x = json.dumps([d["conf"] for d in q_conf_mismatch])
scatter_y = json.dumps([d["mismatch"] for d in q_conf_mismatch])
scatter_labels = json.dumps([d["q"] for d in q_conf_mismatch])
bin_labels_js = json.dumps(bin_labels)
bin_mismatch_js = json.dumps(bin_mismatch_pcts)

# ═══════════════════════════════════════════════════════════════════
# Write HTML
# ═══════════════════════════════════════════════════════════════════
html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Social Norms in Climate Discussions</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{background:#0a1628}
body{font-family:"Inter","Segoe UI",system-ui,sans-serif;background:#0a1628;color:#e0e0e0;padding:15px 20px}
h1{text-align:center;font-size:1.3em;color:#fff;margin-bottom:2px;font-weight:300;letter-spacing:2px}
.subtitle{text-align:center;color:#6a8caf;font-size:0.78em;margin-bottom:18px}
/* Tabs with grouped structure */
.tabs{display:flex;gap:2px;margin-bottom:0;flex-wrap:wrap;justify-content:center;align-items:end}
.tab-btn{padding:7px 14px;cursor:pointer;border:none;background:#12203a;color:#6a8caf;font-size:0.78em;border-radius:8px 8px 0 0;transition:all 0.3s;font-family:inherit;font-weight:500}
.tab-btn:hover{background:#1a3050;color:#a0c4e8}
.tab-btn.active{background:#1a3050;color:#fff;border-bottom:2px solid #5ab4ac}
.tab-sep{width:1px;height:20px;background:#1a3050;margin:0 4px;align-self:center}
.tab-content{display:none;background:#0f1d33;border-radius:0 8px 8px 8px;padding:20px;min-height:400px}
.tab-content.active{display:block}
.sec{background:#12203a;border-radius:10px;padding:16px;border:1px solid #1a3050;position:relative}
.sec:hover{border-color:#2a4060}
.sec-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
.sec-title{font-size:0.95em;color:#fff;font-weight:400}
.info-btn{width:20px;height:20px;border-radius:50%;border:1px solid #3a5a7a;background:none;color:#5a8aaf;font-size:11px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-family:serif;font-style:italic;flex-shrink:0;transition:all 0.2s}
.info-btn:hover{background:#1a3050;color:#fff;border-color:#5ab4ac}
.info-popup{display:none;position:absolute;top:40px;right:10px;background:#1a3050;border:1px solid #2a4060;border-radius:8px;padding:12px 14px;font-size:0.78em;color:#b0c4d8;line-height:1.5;max-width:350px;z-index:10;box-shadow:0 4px 20px rgba(0,0,0,0.5)}
.info-popup.show{display:block}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.grid2>*,.grid3>*{min-width:0;overflow:hidden}
.full{grid-column:1/-1}
.gauge-clip{height:240px;overflow:hidden;position:relative}
.bubble-chart{width:100%;height:260px}
.bubble-chart svg{width:100%;height:100%}
.bubble-label{fill:#fff;font-family:Inter,sans-serif;pointer-events:none;text-anchor:middle}
.bubble-sector-label{fill:#6a8caf;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px}
.sankey-toggles{display:flex;gap:6px;margin-bottom:8px;justify-content:flex-start}
.sankey-toggle{padding:5px 14px;border:1px solid #2a4060;border-radius:14px;background:none;color:#6a8caf;font-size:0.75em;cursor:pointer;font-family:inherit;transition:all 0.2s}
.sankey-toggle:hover{border-color:#5ab4ac;color:#b0c4d8}
.sankey-toggle.active{background:#1a3050;border-color:#5ab4ac;color:#fff}
.legend{display:flex;gap:12px;justify-content:center;margin-top:6px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px;font-size:0.75em;color:#b0c4d8}
.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
/* Animations */
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes scaleIn{from{opacity:0;transform:scale(0.85)}to{opacity:1;transform:scale(1)}}
.sec{animation:fadeSlideUp 0.5s ease-out both}
.tab-content.active .sec{animation:fadeSlideUp 0.5s ease-out both}
.stat-box{animation:scaleIn 0.4s ease-out both}
.ex-section{animation:fadeSlideUp 0.4s ease-out both}
/* Stat boxes */
.stat-grid{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:15px 0}
.stat-box{background:#12203a;padding:10px 16px;border-radius:8px;text-align:center;min-width:80px}
.stat-val{font-size:1.3em;font-weight:600}
.stat-label{font-size:0.65em;color:#6a8caf;margin-top:2px}
/* Examples */
.ex-section{margin-bottom:8px}
.ex-section summary{cursor:pointer;padding:8px 12px;background:#12203a;border-radius:6px;font-size:0.9em;color:#b0c4d8;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:10px}
.ex-section summary::-webkit-details-marker{display:none}
.ex-section summary::before{content:"\25B6 ";font-size:0.7em;color:#5ab4ac}
.ex-section[open] summary::before{content:"\25BC ";font-size:0.7em;color:#5ab4ac}
.ex-content{padding:8px 0 8px 12px}
.ex-group-title{font-size:0.8em;font-weight:600;margin:8px 0 4px;padding-left:4px}
.ex-card{background:#0a1628;border-radius:6px;padding:10px 12px;margin-bottom:6px;font-size:0.8em;line-height:1.4}
.ex-card.mismatch{border-left:3px solid #ff9aa8}
.ex-card.match{border-left:3px solid #8fcc8f}
.ex-comment{color:#c0d0e0;margin-bottom:6px;white-space:pre-wrap}
.ex-labels{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.ex-tag{background:#1a2a40;padding:2px 8px;border-radius:10px;font-size:0.85em;color:#b0c4d8}
.ex-tag.reason{background:#2a1a30;color:#c49fc4}
.ex-sector{font-size:0.75em;color:#4a6a8a;margin-left:auto}
/* 9-col Examples grid */
.ex-9grid{display:grid;grid-template-columns:repeat(9,1fr);gap:4px}
.ex-sec-hdr-row{margin-bottom:0}
.ex-sec-hdr{font-size:0.72em;font-weight:700;letter-spacing:1px;padding:4px 2px 2px;text-align:center;text-transform:uppercase;border-radius:3px 3px 0 0}
.ex-hdr-row{margin-bottom:2px}
.ex-col-hdr{font-size:0.6em;font-weight:700;letter-spacing:1px;padding:3px 2px;text-align:center;text-transform:uppercase;border-radius:3px 3px 0 0}
.ex-col-na{color:#1a3050!important}
.ex-cell{background:#0a1628;border-radius:3px;padding:7px 7px 5px;font-size:0.7em;line-height:1.45;min-height:70px;position:relative;border-left:3px solid transparent;word-break:break-word}
.ex-cell-na{background:#070d1a;border-radius:3px;min-height:70px}
.ex-cell-empty{background:#080f1c;border-radius:3px;min-height:70px;border-left:3px solid #0a1628}
.ex-conf{position:absolute;top:3px;right:3px;font-size:0.65em;background:#1a3050;padding:1px 4px;border-radius:8px;color:#8fcc8f;font-weight:600}
.ex-reason-lbl{margin-top:5px;font-size:0.8em;color:#5a7a9a}
.ex-answer-row{margin-bottom:8px}
.ex-answer-label{font-size:0.78em;font-weight:700;color:#c0d0e0;padding:4px 0 3px;text-transform:capitalize;letter-spacing:0.5px;border-bottom:1px solid #1a3050;margin-bottom:3px;display:flex;align-items:center;gap:8px}
.ex-label-n{font-size:0.75em;font-weight:400;color:#4a6a8a;text-transform:none;letter-spacing:0}
.ex-sum-meta{float:right;font-size:0.75em;font-weight:400;color:#6a8caf}
.ex-prompt-drop{margin-bottom:8px}
.ex-prompt-drop>summary{font-size:0.72em;color:#4a7aaf;cursor:pointer;padding:3px 0;user-select:none}
.ex-prompt-box{background:#060c18;border:1px solid #1a3050;border-radius:6px;padding:10px;font-size:0.68em;color:#7a8a9a;font-family:monospace;white-space:pre-wrap;margin-top:5px;max-height:180px;overflow-y:auto}
</style>
</head>
<body>
<h1>SOCIAL NORMS IN CLIMATE DISCUSSIONS</h1>
<p class="subtitle">Reddit analysis across Food, Transport &amp; Housing &mdash; 9,000 comments &mdash; LLM-labeled with verification</p>

<div class="tabs">
<button class="tab-btn active" onclick="showTab('norms')">Social Norms</button>
<span class="tab-sep"></span>
<button class="tab-btn" onclick="showTab('survey')">Lifestyle Adoption Factors</button>
<span class="tab-sep"></span>
<button class="tab-btn" onclick="showTab('temporal')">Temporal</button>
<span class="tab-sep"></span>
<button class="tab-btn" onclick="showTab('examples')">Examples</button>
</div>

<!-- TAB 1: NORMS (presence + dimensions + reference groups) -->
<div class="tab-content active" id="tab-norms">
<div style="display:flex;gap:16px;align-items:stretch">
<div class="sec" style="flex:1;min-width:0">
<div class="sec-header">
<div class="sec-title">Does the comment reference a social norm?</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Gate question (1.1): Each comment is first classified as containing a social norm or not. A social norm is any reference to what people do (descriptive) or should do (injunctive) regarding climate actions. Food discussions contain norms most often (40%), while transport/housing are lower.</div>
<div class="gauge-clip"><div id="gauge-chart"></div></div>
<div style="text-align:center;margin-top:6px">
<span style="color:#5ab4ac;font-size:0.85em">&#9679; Food FOOD_GATE_PCT_VAL%</span> &nbsp;
<span style="color:#af8dc3;font-size:0.85em">&#9679; Transport TRANSPORT_GATE_PCT_VAL%</span> &nbsp;
<span style="color:#f4a460;font-size:0.85em">&#9679; Housing HOUSING_GATE_PCT_VAL%</span>
</div>
</div>
<div class="sec" style="flex:2;min-width:0">
<div class="sec-header">
<div class="sec-title">Social groups mentioned by Reddit users</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Which social groups does the speaker reference when discussing climate actions? Shows the count of mentions per reference group category across sectors. Size = frequency of mention.</div>
<div id="bubble-pack" class="bubble-chart"></div>
<div style="text-align:center;color:#6a8caf;font-size:0.72em;margin-top:4px">Count = comments mentioning that social group &nbsp;&middot;&nbsp; % = share of all cross-sector group mentions</div>
<div class="legend" id="bubble-legend"></div>
</div>
</div>
<div class="sec" style="margin-top:16px">
<div class="sec-header">
<div class="sec-title">Descriptive or Injunctive Norm</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup"><b>Descriptive norm</b>: references to what people do (observed behaviour). <b>Injunctive norm</b>: references to what people should do (social approval/disapproval). Area = comment count; colour intensity = Present (bright) / Absent (mid) / Unclear (faint).</div>
<div class="grid2" style="margin-top:8px">
<div><div style="text-align:center;color:#b0c4d8;font-size:0.82em;font-weight:600;margin-bottom:4px;letter-spacing:1px">DESCRIPTIVE NORM</div><div id="treemap-desc"></div></div>
<div><div style="text-align:center;color:#b0c4d8;font-size:0.82em;font-weight:600;margin-bottom:4px;letter-spacing:1px">INJUNCTIVE NORM</div><div id="treemap-inj"></div></div>
</div>
</div>
</div>

<!-- TAB 4: FACTORS & BARRIERS -->
<div class="tab-content" id="tab-survey">
<div class="sec">
<div class="sec-header">
<div class="sec-title">Attitude &amp; Barrier Factors by Sector</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Each comment is also labeled against sector-specific survey questions about attitudes, barriers, and behavioral factors. Shows % of comments where the factor is present. Food has 13 factors, Transport 6, Housing 10.</div>
<div style="display:flex;justify-content:center;gap:0;flex-wrap:wrap">
<div style="flex:1;min-width:280px"><div style="text-align:center;color:#5ab4ac;font-size:0.85em;font-weight:600;letter-spacing:1px;margin-bottom:4px">FOOD</div><svg id="radial-food" viewBox="0 0 460 430" style="width:100%;max-width:460px;display:block;margin:auto"></svg></div>
<div style="flex:1;min-width:280px"><div style="text-align:center;color:#af8dc3;font-size:0.85em;font-weight:600;letter-spacing:1px;margin-bottom:4px">TRANSPORT</div><svg id="radial-transport" viewBox="0 0 460 430" style="width:100%;max-width:460px;display:block;margin:auto"></svg></div>
<div style="flex:1;min-width:280px"><div style="text-align:center;color:#f4a460;font-size:0.85em;font-weight:600;letter-spacing:1px;margin-bottom:4px">HOUSING</div><svg id="radial-housing" viewBox="0 0 460 430" style="width:100%;max-width:460px;display:block;margin:auto"></svg></div>
</div>
</div>
</div>

<!-- TAB: TEMPORAL TRENDS -->
<div class="tab-content" id="tab-temporal">
<div class="sec">
<div class="sec-header">
<div class="sec-title">Norm Dimensions Over Time (% by year)</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">% of comments per year where the norm dimension is <b>present</b>. For Second-Order, strong+weak are merged. Each line is a sector.</div>
<div class="temporal-dim-toggles" style="display:flex;gap:6px;margin-bottom:10px;justify-content:flex-start">
<button class="sankey-toggle active" data-dim="1.2.1_descriptive" onclick="window._setTemporalDim(this.dataset.dim)">Descriptive</button>
<button class="sankey-toggle" data-dim="1.2.2_injunctive" onclick="window._setTemporalDim(this.dataset.dim)">Injunctive</button>
<button class="sankey-toggle" data-dim="1.3.3_second_order" onclick="window._setTemporalDim(this.dataset.dim)">Second-Order</button>
</div>
<div id="temporal-dim-combined"></div>
</div>
<div class="sec" style="margin-top:16px">
<div class="sec-header">
<div class="sec-title">Survey Factors Over Time (% yes by year)</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Proportion of comments matching each survey question factor per year. Food has 13 factors, Transport 6, Housing 10. Stacked areas show how these behavioral/attitudinal factors evolve over time.</div>
<div class="grid3">
<div><div style="text-align:center;color:#5ab4ac;font-size:0.8em;font-weight:600;margin-bottom:2px">FOOD</div><div id="temporal-survey-food"></div></div>
<div><div style="text-align:center;color:#af8dc3;font-size:0.8em;font-weight:600;margin-bottom:2px">TRANSPORT</div><div id="temporal-survey-transport"></div></div>
<div><div style="text-align:center;color:#f4a460;font-size:0.8em;font-weight:600;margin-bottom:2px">HOUSING</div><div id="temporal-survey-housing"></div></div>
</div>
</div>
</div>

<!-- TAB 6: EXAMPLES -->
<div class="tab-content" id="tab-examples">
<div style="margin-bottom:12px;color:#6a8caf;font-size:0.82em">
All 36 questions (7 norm dimensions + 29 survey frames) &mdash; 3 examples per answer label per sector.
Each cell shows <b style="color:#e0e0e0">2 cases where labeler &amp; verifier agree</b> and <b style="color:#e0e0e0">1 where they disagree</b>, so you can judge model reliability directly.
&nbsp;&nbsp;<span style="color:#4ade80">&#9646;</span> Labeler &amp; verifier agree &nbsp;&nbsp;
<span style="color:#f87171">&#9646;</span> Labeler &amp; verifier disagree &nbsp;&nbsp;
Confidence % = logprob of fast model prediction.
</div>
EXAMPLES_PLACEHOLDER
</div>

<script>
function showTab(id) {
    document.querySelectorAll('.tab-content').forEach(function(t){t.classList.remove('active')});
    document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active')});
    document.getElementById('tab-'+id).classList.add('active');
    event.target.classList.add('active');
    // Re-trigger section animations
    document.querySelectorAll('#tab-'+id+' .sec').forEach(function(c,i){
        c.style.animation='none';c.offsetHeight;
        c.style.animation='fadeSlideUp 0.5s ease-out '+(i*0.08)+'s both';
    });
    document.querySelectorAll('#tab-'+id+' .stat-box').forEach(function(c,i){
        c.style.animation='none';c.offsetHeight;
        c.style.animation='scaleIn 0.4s ease-out '+(i*0.05)+'s both';
    });
    window.dispatchEvent(new Event('resize'));
    if(id==='norms'){setTimeout(function(){if(window._animateGauge)window._animateGauge();},100);setTimeout(function(){if(window._animateNorms)window._animateNorms();},100);setTimeout(initBubbles,200);}
    if(id==='temporal') setTimeout(function(){if(window._animateTemporal)window._animateTemporal();},200);
    if(id==='survey') setTimeout(function(){if(window._animateSurvey)window._animateSurvey();},200);
}
function toggleInfo(btn){var p=btn.parentElement.nextElementSibling;if(p&&p.classList.contains('info-popup'))p.classList.toggle('show');}
var PC={displayModeBar:false,responsive:true};
var DB='rgba(0,0,0,0)';
var DL={paper_bgcolor:DB,plot_bgcolor:DB,font:{color:'#ffffff',family:'Inter,sans-serif',size:10},legend:{font:{size:14,color:'#ffffff'},orientation:'h',x:0.5,xanchor:'center',y:-0.22,yanchor:'top'}};

// ═══════════════ GAUGE (animated from 0, replayable) ═══════════════
(function(){
    var _gp=GATE_PCT_DATA;var f=_gp.food,t=_gp.transport,h=_gp.housing,avg=(f+t+h)/3,vis=f+t+h;
    Plotly.newPlot('gauge-chart',[{
        values:[0.01,0.01,0.01,0.03],labels:['Food','Transport','Housing',''],
        marker:{colors:['#5ab4ac','#af8dc3','#f4a460','rgba(0,0,0,0)'],line:{color:'#12203a',width:4}},
        hole:0.65,direction:'clockwise',sort:false,rotation:270,
        textposition:'none',type:'pie',showlegend:false,
        hovertemplate:'<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    }],Object.assign({},DL,{height:440,margin:{t:5,b:5,l:5,r:5},
        annotations:[
            {text:'<b>0.0%</b>',x:0.5,y:0.55,font:{size:46,color:'#fff'},showarrow:false},
            {text:'of comments contain social norm',x:0.5,y:0.48,font:{size:12,color:'#b0c4d8'},showarrow:false},
        ]
    }),PC);
    var _gTimer=null;
    window._animateGauge=function(){
        if(_gTimer)clearInterval(_gTimer);
        Plotly.restyle('gauge-chart',{values:[[0.01,0.01,0.01,0.03]]});
        Plotly.relayout('gauge-chart',{'annotations[0].text':'<b>0.0%</b>'});
        var steps=30,step=0;
        _gTimer=setInterval(function(){
            step++;var p=step/steps;var ease=1-Math.pow(1-p,3);
            var cf=f*ease,ct=t*ease,ch=h*ease,cv=(cf+ct+ch);
            Plotly.restyle('gauge-chart',{values:[[cf,ct,ch,cv]]});
            Plotly.relayout('gauge-chart',{'annotations[0].text':'<b>'+((cf+ct+ch)/3).toFixed(1)+'%</b>'});
            if(step>=steps)clearInterval(_gTimer);
        },30);
    };
    window._animateGauge();
})();

// ═══════════════ BUBBLES ═══════════════
var bubbleData={
    categories:['family','partner/spouse','friends','coworkers','neighbors','local community','political tribe','online community','other reddit user'],
    colors:['#e07878','#c8b448','#4db86e','#48aed0','#7878cc','#d05890','#96c040','#e09850','#9898a8'],
    Food:BUBBLE_FOOD_DATA,Transport:BUBBLE_TRANSPORT_DATA,Housing:BUBBLE_HOUSING_DATA
};
function initBubbles(){
    var el=document.getElementById('bubble-pack');if(!el)return;el.innerHTML='';
    var w=el.clientWidth||900,h=el.clientHeight||260;
    var svg=d3.select('#bubble-pack').append('svg').attr('viewBox','0 0 '+w+' '+h);
    var sectors=['Food','Transport','Housing'],cw=w/3,nodes=[];
    var grandTotal=0;
    sectors.forEach(function(sec){bubbleData[sec].forEach(function(v){grandTotal+=v;});});
    // Adaptive scale: largest bubble radius = 42% of column width, so chart stays bounded as data grows
    var maxVal=0;
    sectors.forEach(function(sec){bubbleData[sec].forEach(function(v){if(v>maxVal)maxVal=v;});});
    var maxAllowedR=cw*0.18;
    var rScale=maxVal>0?maxAllowedR/Math.sqrt(maxVal):1.5;
    sectors.forEach(function(sec,si){
        var cx=cw*si+cw/2;
        svg.append('text').attr('x',cx).attr('y',18).attr('class','bubble-sector-label').attr('text-anchor','middle').text(sec.toUpperCase());
        bubbleData.categories.forEach(function(cat,ci){
            var val=bubbleData[sec][ci];if(val===0)return;
            var pct=grandTotal>0?(val/grandTotal*100).toFixed(1):0;
            nodes.push({sector:sec,category:cat,value:val,pct:pct,r:Math.sqrt(val)*rScale,color:bubbleData.colors[ci],cx:cx,cy:h/2+5,x:cx+(Math.random()-0.5)*40,y:h/2+5+(Math.random()-0.5)*40});
        });
    });
    // Mark top-3 per sector by value
    sectors.forEach(function(sec){
        var sn=nodes.filter(function(d){return d.sector===sec;}).sort(function(a,b){return b.value-a.value;});
        sn.slice(0,3).forEach(function(d){d.top3=true;});
    });
    svg.selectAll('.sep').data([1,2]).enter().append('line')
        .attr('x1',function(d){return cw*d}).attr('x2',function(d){return cw*d})
        .attr('y1',28).attr('y2',h-5).attr('stroke','#1a3050').attr('stroke-width',1).attr('stroke-dasharray','4,4');
    // Shared hover tooltip
    var _bTip=document.getElementById('_bubble_tip');
    if(!_bTip){_bTip=document.createElement('div');_bTip.id='_bubble_tip';
        _bTip.style.cssText='position:fixed;background:#1a3050;border:1px solid #3a6080;border-radius:6px;padding:6px 10px;font-size:0.78em;color:#e0e0e0;pointer-events:none;z-index:9999;display:none;line-height:1.5';
        document.body.appendChild(_bTip);}
    var g=svg.selectAll('.b').data(nodes).enter().append('g').style('cursor','default');
    g.append('circle').attr('cx',function(d){return d.x}).attr('cy',function(d){return d.y}).attr('r',0)
        .attr('fill',function(d){return d.color}).attr('fill-opacity',0.85).attr('stroke',function(d){return d.color}).attr('stroke-opacity',0.3).attr('stroke-width',2)
        .transition().duration(800).delay(function(d,i){return i*30}).attr('r',function(d){return d.r});
    // Hover on circle shows tooltip for ALL bubbles
    g.select('circle')
        .on('mouseover',function(event,d){
            _bTip.innerHTML='<b>'+d.category+'</b><br>'+d.value+' comments ('+d.pct+'%)';
            _bTip.style.display='block';})
        .on('mousemove',function(event){_bTip.style.left=(event.clientX+14)+'px';_bTip.style.top=(event.clientY-8)+'px';})
        .on('mouseout',function(){_bTip.style.display='none';});
    // Category label — top-3 only
    g.append('text').attr('x',function(d){return d.x}).attr('y',function(d){return d.y+(d.r>25?-12:d.r>15?-6:-2)}).attr('class','bubble-label')
        .attr('font-size',function(d){return Math.max(8,Math.min(d.r*0.6,13))+'px'}).attr('opacity',0)
        .text(function(d){return d.top3?d.category:''}).transition().duration(600).delay(function(d,i){return 800+i*30}).attr('opacity',1);
    // Count — top-3 only
    g.append('text').attr('x',function(d){return d.x}).attr('y',function(d){return d.y+(d.r>25?4:d.r>15?7:10)}).attr('class','bubble-label')
        .attr('font-size',function(d){return Math.max(9,Math.min(d.r*0.55,14))+'px'}).attr('font-weight','700').attr('opacity',0)
        .text(function(d){return d.top3?d.value:''}).transition().duration(600).delay(function(d,i){return 800+i*30}).attr('opacity',1);
    // Percentage — top-3 only
    g.append('text').attr('x',function(d){return d.x}).attr('y',function(d){return d.y+(d.r>25?17:d.r>15?19:22)}).attr('class','bubble-label')
        .attr('font-size',function(d){return Math.max(7,Math.min(d.r*0.42,11))+'px'}).attr('opacity',0)
        .text(function(d){return d.top3?'('+d.pct+'%)':''}).transition().duration(600).delay(function(d,i){return 800+i*30}).attr('opacity',1);
    d3.forceSimulation(nodes).force('x',d3.forceX(function(d){return d.cx}).strength(0.1))
        .force('y',d3.forceY(h/2+10).strength(0.08)).force('collide',d3.forceCollide(function(d){return d.r+2}).strength(0.95))
        .force('charge',d3.forceManyBody().strength(-1)).alpha(0.8)
        .on('tick',function(){
            g.select('circle').attr('cx',function(d){return d.x}).attr('cy',function(d){return d.y});
            g.selectAll('text').attr('x',function(d){return d.x});
            g.select('text:nth-child(2)').attr('y',function(d){return d.y+(d.r>25?-12:d.r>15?-6:-2)});
            g.select('text:nth-child(3)').attr('y',function(d){return d.y+(d.r>25?4:d.r>15?7:10)});
            g.select('text:nth-child(4)').attr('y',function(d){return d.y+(d.r>25?17:d.r>15?19:22)});
        });
    var le=document.getElementById('bubble-legend');le.innerHTML='';
    bubbleData.categories.forEach(function(c,i){le.innerHTML+='<span class="legend-item"><span class="legend-dot" style="background:'+bubbleData.colors[i]+'"></span>'+c+'</span>';});
}

// ═══════════════ NORM TREEMAPS (descriptive + injunctive) ═══════════════
(function(){
    var SD=SANKEY_SD_DATA;
    var secs=['Food','Transport','Housing'];
    var SC={Food:'90,180,172',Transport:'175,141,195',Housing:'244,164,96'};
    var catNames=['Present','Absent','Unclear'];
    var catAlphas=[0.88,0.45,0.18];

    function makeNormTreemap(divId,dimKey){
        var labels=[],parents=[],values=[],colors=[],texts=[];
        secs.forEach(function(sec){
            var d=SD[sec][dimKey];
            var total=d[0]+d[1]+d[2];
            labels.push(sec);parents.push('');values.push(0);
            colors.push('rgba('+SC[sec]+',0.08)');
            texts.push('<b>'+sec+'</b>');
            catNames.forEach(function(cat,ci){
                var v=d[ci];
                var pct=total>0?Math.round(v/total*100):0;
                labels.push(sec+'_'+cat);parents.push(sec);values.push(v);
                colors.push('rgba('+SC[sec]+','+catAlphas[ci]+')');
                texts.push(cat+'<br>'+v.toLocaleString()+'<br>('+pct+'%)');
            });
        });
        Plotly.newPlot(divId,[{
            type:'treemap',labels:labels,parents:parents,values:values,
            text:texts,textinfo:'text',branchvalues:'remainder',
            marker:{colors:colors,line:{width:1.5,color:'#0f1d33'}},
            textfont:{size:12,color:'#ffffff'},
            hovertemplate:'%{text}<extra></extra>',
            tiling:{packing:'squarify',pad:3},
        }],Object.assign({},DL,{height:340,margin:{t:5,b:5,l:5,r:5},uniformtext:{minsize:9,mode:'hide'}}),PC);
    }

    makeNormTreemap('treemap-desc','desc');
    makeNormTreemap('treemap-inj','inj');
    window._animateNorms=function(){
        makeNormTreemap('treemap-desc','desc');
        makeNormTreemap('treemap-inj','inj');
    };
})();

// ═══════════════ SURVEY RADIALS (D3 sequential spoke animation) ═══════════════
// D3.arc: 0 = 12 o'clock, angles go clockwise
// To place labels with Math.cos/sin, convert: trigAngle = d3Angle - PI/2
function drawRadial(svgId, labels, tooltips, values, baseColor, maxR, delay){
    var svg=d3.select('#'+svgId);
    var cx=230,cy=210,outerR=130;
    var n=labels.length, sliceAngle=2*Math.PI/n;
    var gapFrac=0.15; // fraction of slice that is gap
    var spokeAngle=sliceAngle*(1-gapFrac);
    var bc=baseColor;
    var arc=d3.arc();
    // Shared floating tooltip (created once for all radial charts)
    var _rtip=document.getElementById('_radial_tip');
    if(!_rtip){
        _rtip=document.createElement('div');
        _rtip.id='_radial_tip';
        _rtip.style.cssText='position:fixed;background:#1a3050;border:1px solid #3a6080;border-radius:7px;padding:10px 14px;font-size:0.8em;color:#b0c4d8;line-height:1.55;max-width:300px;pointer-events:none;z-index:9999;display:none;box-shadow:0 4px 20px rgba(0,0,0,0.7)';
        document.body.appendChild(_rtip);
    }
    function _showRTip(tip,pct){
        _rtip.innerHTML='<div style="color:#fff;font-weight:600;margin-bottom:5px">'+pct.toFixed(1)+'% of comments</div><div style="color:#8ab8d8;font-size:0.95em">'+tip+'</div>';
        _rtip.style.display='block';
    }
    function _moveRTip(e){_rtip.style.left=(e.clientX+16)+'px';_rtip.style.top=(e.clientY-8)+'px';}
    function _hideRTip(){_rtip.style.display='none';}
    // Grid circles
    [0.25,0.5,0.75,1.0].forEach(function(s){
        svg.append('circle').attr('cx',cx).attr('cy',cy).attr('r',outerR*s)
            .attr('fill','none').attr('stroke','#1a2a40').attr('stroke-width',0.5);
    });
    // Faint radial guide lines at each spoke center angle
    values.forEach(function(v,i){
        var centerD3=i*sliceAngle;
        var trigMid=centerD3-Math.PI/2;
        svg.append('line')
            .attr('x1',cx).attr('y1',cy)
            .attr('x2',cx+Math.cos(trigMid)*outerR)
            .attr('y2',cy+Math.sin(trigMid)*outerR)
            .attr('stroke','#1a2a40').attr('stroke-width',0.5);
    });
    // Grid tick labels (at top)
    [0.25,0.5,0.75,1.0].forEach(function(s){
        svg.append('text').attr('x',cx+3).attr('y',cy-outerR*s-1)
            .attr('fill','#ffffff').attr('font-size','9px').attr('font-family','Inter,sans-serif')
            .text((maxR*s).toFixed(0)+'%');
    });
    // Build spoke data - all angles in D3 convention (0=top, clockwise)
    var spokes=[];
    values.forEach(function(v,i){
        // D3 arc angles: each spoke centered at i*sliceAngle
        var centerD3=i*sliceAngle;
        var startD3=centerD3-spokeAngle/2;
        var endD3=centerD3+spokeAngle/2;
        var r=Math.max(3, outerR*(v/maxR));
        var alpha=0.3+0.65*(v/maxR); if(alpha>0.95)alpha=0.95;
        // Convert center to trig angle for label placement
        var trigMid=centerD3-Math.PI/2;
        var g=svg.append('g').style('cursor','pointer');
        var path=g.append('path')
            .attr('d',arc({innerRadius:0,outerRadius:r,startAngle:startD3,endAngle:endD3}))
            .attr('transform','translate('+cx+','+cy+')')
            .attr('fill',bc).attr('fill-opacity',alpha)
            .attr('stroke',bc).attr('stroke-opacity',0.6).attr('stroke-width',1)
            .attr('opacity',0);
        // Hover highlight + custom tooltip
        g.on('mouseover',function(event){
            d3.select(this).select('path').attr('stroke','#fff').attr('stroke-width',2).attr('stroke-opacity',1);
            _showRTip(tooltips[i]||labels[i],v);_moveRTip(event);
        })
        .on('mousemove',function(event){_moveRTip(event);})
        .on('mouseout',function(){
            d3.select(this).select('path').attr('stroke',bc).attr('stroke-width',1).attr('stroke-opacity',0.6);
            _hideRTip();
        });
        spokes.push({g:g,path:path,label:labels[i],value:v,startD3:startD3,endD3:endD3,r:r,trigMid:trigMid,alpha:alpha});
    });
    // Labels at spoke centers — word-wrapped to avoid horizontal clipping
    var MAX_LINE_CH=16;
    spokes.forEach(function(s,i){
        var lx=cx+Math.cos(s.trigMid)*(outerR+20);
        var ly=cy+Math.sin(s.trigMid)*(outerR+20);
        var anchor=Math.abs(Math.cos(s.trigMid))<0.15?'middle':(Math.cos(s.trigMid)>0?'start':'end');
        var words=s.label.split(' ');
        var lines=[],cur='';
        words.forEach(function(w){
            if(cur&&(cur+' '+w).length>MAX_LINE_CH){lines.push(cur);cur=w;}
            else{cur=cur?cur+' '+w:w;}
        });
        if(cur)lines.push(cur);
        var lh=11; // px between lines
        var startY=ly-(lines.length-1)*lh/2;
        var textEl=svg.append('text').attr('x',lx).attr('y',startY)
            .attr('text-anchor',anchor).attr('fill','#ffffff')
            .attr('font-size','9px').attr('font-family','Inter,sans-serif')
            .attr('opacity',0).attr('class','rl-'+svgId+'-'+i);
        lines.forEach(function(line,li){
            textEl.append('tspan').attr('x',lx).attr('dy',li===0?0:lh).text(line);
        });
        textEl.style('cursor','help')
            .on('mouseover',function(event){_showRTip(tooltips[i]||labels[i],s.value);_moveRTip(event);})
            .on('mousemove',function(event){_moveRTip(event);})
            .on('mouseout',_hideRTip);
    });
    // Sequential animation
    window['_animRadial_'+svgId]=function(){
        spokes.forEach(function(s,i){
            s.path.transition()
                .delay(delay+i*120).duration(500).ease(d3.easeCubicOut)
                .attr('opacity',1)
                .attrTween('d',function(){
                    return function(t){
                        return arc({innerRadius:0,outerRadius:s.r*t,startAngle:s.startD3,endAngle:s.endD3});
                    };
                });
            // Value label — placed near spoke tip, min 42% of outerR to avoid centre crowding
            var labelR=Math.max(s.r*0.92, outerR*0.42);
            var vx=cx+Math.cos(s.trigMid)*labelR;
            var vy=cy+Math.sin(s.trigMid)*labelR+3;
            var showLabel=s.value>=2.5 && s.r>15;
            svg.append('text').attr('x',vx).attr('y',vy)
                .attr('text-anchor','middle').attr('fill','#fff').attr('font-weight','600')
                .attr('font-size',s.r>50?'11px':s.r>30?'9px':'7px').attr('font-family','Inter,sans-serif')
                .attr('pointer-events','none')
                .attr('opacity',0).text(showLabel?s.value.toFixed(1)+'%':'')
                .transition().delay(delay+i*120+300).duration(300).attr('opacity',showLabel?0.9:0);
            // Outer label fade in
            d3.select('.rl-'+svgId+'-'+i)
                .transition().delay(delay+i*120+200).duration(300).attr('opacity',1);
        });
    };
}
drawRadial('radial-food',RADIAL_FOOD_LABELS,RADIAL_FOOD_TOOLTIPS,RADIAL_FOOD_VALS,'#5ab4ac',RADIAL_FOOD_MAX,0);
drawRadial('radial-transport',RADIAL_TRANSPORT_LABELS,RADIAL_TRANSPORT_TOOLTIPS,RADIAL_TRANSPORT_VALS,'#af8dc3',RADIAL_TRANSPORT_MAX,0);
drawRadial('radial-housing',RADIAL_HOUSING_LABELS,RADIAL_HOUSING_TOOLTIPS,RADIAL_HOUSING_VALS,'#f4a460',RADIAL_HOUSING_MAX,0);
window._animateSurvey=function(){
    // Reset and replay - clear dynamic elements, redraw
    ['radial-food','radial-transport','radial-housing'].forEach(function(id){
        var svg=document.getElementById(id);
        svg.innerHTML='';
    });
    drawRadial('radial-food',RADIAL_FOOD_LABELS,RADIAL_FOOD_TOOLTIPS,RADIAL_FOOD_VALS,'#5ab4ac',RADIAL_FOOD_MAX,0);
    drawRadial('radial-transport',RADIAL_TRANSPORT_LABELS,RADIAL_TRANSPORT_TOOLTIPS,RADIAL_TRANSPORT_VALS,'#af8dc3',RADIAL_TRANSPORT_MAX,0);
    drawRadial('radial-housing',RADIAL_HOUSING_LABELS,RADIAL_HOUSING_TOOLTIPS,RADIAL_HOUSING_VALS,'#f4a460',RADIAL_HOUSING_MAX,0);
    window['_animRadial_radial-food']();
    window['_animRadial_radial-transport']();
    window['_animRadial_radial-housing']();
};

// ═══════════════ TEMPORAL CHARTS ═══════════════
(function(){
    var YEARS=TEMPORAL_YEARS;
    var tNorms=TEMPORAL_NORMS;
    var normDims=TEMPORAL_NORM_DIMS;
    var tSurvey=TEMPORAL_SURVEY;
    var surveyLabels=TEMPORAL_SURVEY_LABELS;
    var surveyIds=TEMPORAL_SURVEY_IDS;

    // 1. Norm Dimensions temporal — single plot, 3 sector lines, present only
    var currentDim='1.2.1_descriptive';
    var _secCol={food:'#5ab4ac',transport:'#af8dc3',housing:'#f4a460'};
    var _secLbl={food:'Food',transport:'Transport',housing:'Housing'};
    function drawNormDim(dim){
        var meta=normDims[dim];
        var traces=[];
        ['food','transport','housing'].forEach(function(sec){
            var catData=tNorms[sec][dim];
            var yvals;
            if(dim==='1.3.3_second_order'){
                // merge strong(cats[0]="2") + weak(cats[1]="1")
                yvals=YEARS.map(function(y){
                    return (catData[meta.cats[0]]?catData[meta.cats[0]][y]||0:0)
                          +(catData[meta.cats[1]]?catData[meta.cats[1]][y]||0:0);
                });
            }else{
                var pc=meta.cats[0]; // "explicitly present" or "present"
                yvals=YEARS.map(function(y){return catData[pc]?catData[pc][y]||0:0});
            }
            var col=_secCol[sec];
            traces.push({x:YEARS,y:yvals,name:_secLbl[sec],type:'scatter',mode:'lines+markers',
                line:{width:2.5,color:col,shape:'spline',smoothing:0.8},
                marker:{size:4,color:col},showlegend:true,
                hovertemplate:'<b>%{fullData.name}</b><br>%{x}: %{y:.1f}%<extra></extra>'
            });
        });
        var yLbl=dim==='1.3.3_second_order'?'% present (strong+weak)':'% present';
        Plotly.newPlot('temporal-dim-combined',traces,Object.assign({},DL,{
            height:260,margin:{t:10,b:50,l:40,r:10},
            xaxis:{tickfont:{size:9,color:'#ffffff'},gridcolor:'#1a2a40',dtick:3},
            yaxis:{title:{text:yLbl,font:{size:10,color:'#ffffff'}},tickfont:{size:9,color:'#ffffff'},gridcolor:'#1a2a40',rangemode:'tozero'},
        }),PC);
    }
    drawNormDim(currentDim);
    window._setTemporalDim=function(dim){
        currentDim=dim;
        drawNormDim(dim);
        document.querySelectorAll('.temporal-dim-toggles .sankey-toggle').forEach(function(b){
            b.classList.toggle('active',b.dataset.dim===dim);
        });
    };

    // 3. Survey factors temporal — individual spline curves (not stacked)
    var surveyColorPalette=['#5ab4ac','#af8dc3','#f4a460','#8fcc8f','#ff9aa8','#8fbfd9','#ffb87a','#c49fc4','#7cadc6','#ffe87a','#a8b8c2','#ffa8c2','#b8a8c8'];
    function drawSurveyLines(sec,i){
        var qids=surveyIds[sec];
        var traces=[];
        qids.forEach(function(qid,qi){
            var yvals=YEARS.map(function(y){return tSurvey[sec][qid]?tSurvey[sec][qid][y]||0:0});
            var shortLabel=surveyLabels[qid]||qid;
            if(shortLabel.length>22) shortLabel=shortLabel.substring(0,20)+'...';
            var col=surveyColorPalette[qi%surveyColorPalette.length];
            traces.push({x:YEARS,y:yvals,name:shortLabel,type:'scatter',mode:'lines+markers',
                line:{width:2,color:col,shape:'spline',smoothing:0.8},
                marker:{size:4,color:col},
                showlegend:true,
                hovertemplate:'<b>'+(surveyLabels[qid]||qid)+'</b><br>%{x}: %{y:.1f}%<extra></extra>'
            });
        });
        Plotly.newPlot('temporal-survey-'+sec,traces,Object.assign({},DL,{
            height:340,margin:{t:10,b:40,l:38,r:5},
            xaxis:{tickfont:{size:9,color:'#ffffff'},gridcolor:'#1a2a40',dtick:3},
            yaxis:{title:{text:'% yes',font:{size:10,color:'#ffffff'}},tickfont:{size:9,color:'#ffffff'},gridcolor:'#1a2a40',rangemode:'tozero'},
            legend:{font:{size:14,color:'#ffffff'},orientation:'h',y:-0.22,x:0.5,xanchor:'center',yanchor:'top',traceorder:'normal'}
        }),PC);
    }
    ['food','transport','housing'].forEach(function(sec,i){ drawSurveyLines(sec,i); });

    window._animateTemporal=function(){
        drawNormDim(currentDim);
        ['food','transport','housing'].forEach(function(sec,i){ drawSurveyLines(sec,i); });
    };
})();

window.addEventListener('load',function(){
    setTimeout(initBubbles,300);
    setTimeout(function(){if(window._animateNorms)window._animateNorms();},400);
});
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════
# Build stat boxes
# ═══════════════════════════════════════════════════════════════════
stats = [
    (f"{summary['mean_accuracy']*100:.1f}%", "Accuracy", "#8fcc8f" if summary['mean_accuracy']>=0.85 else "#ffb87a"),
    (f"{summary['mean_kappa']:.2f}", "Cohen's k", "#ff9aa8" if summary['mean_kappa']<0.4 else "#8fcc8f"),
    (f"{summary['empty_response_pct']:.1f}%", "No Response", "#8fcc8f"),
    (str(summary['total_samples']), "Total Samples", "#b0c4d8"),
    (f"{avg_conf_match:.3f}", "Conf (Match)", "#8fcc8f"),
    (f"{avg_conf_mismatch:.3f}", "Conf (Mismatch)", "#ffb87a"),
    (f"{high_conf_acc*100:.1f}%", "Acc (Conf>0.9)", "#8fcc8f" if high_conf_acc>=0.9 else "#ffb87a"),
    (str(high_conf_samples), "High-Conf Samples", "#b0c4d8"),
]
stat_html = ""
for val, label, color in stats:
    stat_html += f'<div class="stat-box"><div class="stat-val" style="color:{color}">{val}</div><div class="stat-label">{label}</div></div>'

# Inject data
html = html.replace("STAT_BOXES", stat_html)
html = html.replace("EXAMPLES_PLACEHOLDER", examples_html)
html = html.replace("ACC_QUESTIONS", acc_questions)
html = html.replace("ACC_VALUES", acc_values)
html = html.replace("ACC_COLORS", acc_colors)
html = html.replace("EST_LABELS", est_labels)
html = html.replace("EST_VALUES", est_values)
html = html.replace("EST_COLORS", est_colors)
html = html.replace("CONF_BIN_LABELS", bin_labels_js)
html = html.replace("CONF_BIN_MISMATCH", bin_mismatch_js)
html = html.replace("SCATTER_X", scatter_x)
html = html.replace("SCATTER_Y", scatter_y)
html = html.replace("SCATTER_LABELS", scatter_labels)
html = html.replace("HC_QUESTIONS", hc_questions)
html = html.replace("HC_VALUES", hc_values)
html = html.replace("HC_COLORS", hc_colors)
# Temporal data
html = html.replace("TEMPORAL_YEARS", years_js)
html = html.replace("TEMPORAL_STANCE", temporal_stance_js)
html = html.replace("TEMPORAL_NORMS", temporal_norms_js)
html = html.replace("TEMPORAL_NORM_DIMS", norm_dims_js)
html = html.replace("TEMPORAL_SURVEY_LABELS", survey_labels_js)
html = html.replace("TEMPORAL_SURVEY_IDS", survey_ids_js)
html = html.replace("TEMPORAL_SURVEY", temporal_survey_js)
# Dynamic data from labeled_data
html = html.replace("GATE_PCT_DATA",         gate_pct_js)
html = html.replace("STANCE_COUNTS_DATA",    stance_counts_js)
html = html.replace("SANKEY_SD_DATA",        sankey_data_js)
html = html.replace("BUBBLE_FOOD_DATA",      bubble_food_js)
html = html.replace("BUBBLE_TRANSPORT_DATA", bubble_transport_js)
html = html.replace("BUBBLE_HOUSING_DATA",   bubble_housing_js)
html = html.replace("SECTOR_TOTALS_DATA",    sector_totals_js)
html = html.replace("FOOD_GATE_PCT_VAL",     str(gate_pct["food"]))
html = html.replace("TRANSPORT_GATE_PCT_VAL",str(gate_pct["transport"]))
html = html.replace("HOUSING_GATE_PCT_VAL",  str(gate_pct["housing"]))
html = html.replace("RADIAL_FOOD_LABELS",      radial_food_labels_js)
html = html.replace("RADIAL_TRANSPORT_LABELS", radial_transport_labels_js)
html = html.replace("RADIAL_HOUSING_LABELS",   radial_housing_labels_js)
html = html.replace("RADIAL_FOOD_TOOLTIPS",    radial_food_tips_js)
html = html.replace("RADIAL_TRANSPORT_TOOLTIPS",radial_transport_tips_js)
html = html.replace("RADIAL_HOUSING_TOOLTIPS", radial_housing_tips_js)
html = html.replace("RADIAL_FOOD_VALS",        radial_food_vals_js)
html = html.replace("RADIAL_TRANSPORT_VALS", radial_transport_vals_js)
html = html.replace("RADIAL_HOUSING_VALS",   radial_housing_vals_js)
html = html.replace("RADIAL_FOOD_MAX",       str(_food_max_r))
html = html.replace("RADIAL_TRANSPORT_MAX",  str(_transport_max_r))
html = html.replace("RADIAL_HOUSING_MAX",    str(_housing_max_r))

with open("temp.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated temp.html - 6 tabs:")
print(f"  1. Norm Presence (gauge)")
print(f"  2. Norm Dimensions (4 stacked bars + reference group bubbles)")
print(f"  3. Author Stance (treemap)")
print(f"  4. Factors & Barriers (3 radial charts)")
print(f"  5. Verification (stat boxes + 4 charts)")
print(f"  6. Examples ({len(ALL_QS)} questions, 9-column layout)")
