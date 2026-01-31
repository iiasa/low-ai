"""
00_vLLM_visualize.py

Load norms_labels.json (from 00_vLLM_hierarchical.py --norms) and generate:
- 00_dashboardv2.html — pie and bar charts per question/sector
- 00_dashboard_examples.html — one example comment per (question, category, sector)

Input JSON format: { "food": [ { "comment_index", "comment", "answers": { "1.1_gate": "1", ... } }, ... ], "transport": ..., "housing": ... }
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Map answer codes to display labels (dashboard and examples)
CODE_TO_LABEL = {
    "1.1_gate": {"0": "No", "1": "Yes"},
    "1.1.1_stance": {"pushing for": "pro"},
    "1.3.1b_perceived_reference_stance": {"pushing for": "pro"},
    "1.2.1_descriptive": {"0": "none", "1": "implied", "2": "explicit"},
    "1.2.2_injunctive": {
        "0": "none",
        "1": "implied approval",
        "2": "implied disapproval",
        "3": "explicit approval",
        "4": "explicit disapproval",
    },
    "1.3.3_second_order": {"0": "none", "1": "weak", "2": "strong"},
}

SECTOR_DISPLAY = {"food": "FOOD", "transport": "TRANSPORT", "housing": "HOUSING"}
QUESTION_TITLES = {
    "1.1_gate": "Norm signal present",
    "1.1.1_stance": "Author stance",
    "1.2.1_descriptive": "Descriptive norm",
    "1.2.2_injunctive": "Injunctive norm",
    "1.3.1_reference_group": "Reference group",
    "1.3.1b_perceived_reference_stance": "Perceived reference stance",
    "1.3.2_mechanism": "Mechanism",
    "1.3.3_second_order": "Second-order normative belief",
}

# Exact prompt and choices sent to the LLM (mirrors 00_vLLM_hierarchical.NORMS_QUESTIONS for display)
NORMS_PROMPTS = {
    "1.1_gate": {
        "prompt": "Definitions: A social norm is a shared belief or expectation about what is typical or what is approved/disapproved. Descriptive norm = reference to what people typically do or how common something is (e.g. 'most people here drive EVs'). Injunctive norm = reference to what people should do, or explicit approval/disapproval (e.g. 'you should go vegan', 'eating meat is wrong'). Does this comment or post reference what others do or approve, or any social norm (descriptive or injunctive)? Answer with exactly one word: yes or no.",
        "options": ["yes", "no"],
    },
    "1.1.1_stance": {
        "prompt": "What is the author's stance toward {sector_topic}? Definitions: against = author opposes or rejects {sector_topic}. pro but lack of options = author is in favor of {sector_topic} but wants more options or complains that current options are insufficient; do NOT code this as against. Complaints that there are too few {sector_topic} options count as 'pro but lack of options', not 'against'. Examples: 'I wish there were more {sector_topic} options' → pro but lack of options. '{sector_topic} is stupid' → against. Answer with exactly one of: against, against particular but pro, neither/mixed, pro, pro but lack of options.",
        "options": ["against", "against particular but pro", "neither/mixed", "pro", "pro but lack of options"],
        "sector_specific": True,
        "sector_topic": {"transport": "EVs", "food": "veganism or vegetarianism / diet", "housing": "solar"},
    },
    "1.2.1_descriptive": {
        "prompt": "Does the text express a descriptive norm (what people do / how common something is)? Answer with exactly one of: none, implied, explicit.",
        "options": ["none", "implied", "explicit"],
    },
    "1.2.2_injunctive": {
        "prompt": "Does the text express an injunctive norm (what people should do / approval or disapproval)? Answer with exactly one of: none, implied approval, implied disapproval, explicit approval, explicit disapproval.",
        "options": ["none", "implied approval", "implied disapproval", "explicit approval", "explicit disapproval"],
    },
    "1.3.1_reference_group": {
        "prompt": "Who is the reference group (who the author refers to as doing or approving something)? Answer with exactly one of: coworkers, family, friends, local community, neighbors, online community, other, other reddit user, partner/spouse, political tribe.",
        "options": ["coworkers", "family", "friends", "local community", "neighbors", "online community", "other", "other reddit user", "partner/spouse", "political tribe"],
    },
    "1.3.1b_perceived_reference_stance": {
        "prompt": "What stance does the author attribute to that reference group? Answer with exactly one of: against, neither/mixed, pro.",
        "options": ["against", "neither/mixed", "pro"],
    },
    "1.3.2_mechanism": {
        "prompt": "What mechanism is used to convey the norm or social pressure? Answer with exactly one of: blame/shame, community standard, identity/status signaling, other, praise, rule/virtue language, social comparison.",
        "options": ["blame/shame", "community standard", "identity/status signaling", "other", "praise", "rule/virtue language", "social comparison"],
    },
    "1.3.3_second_order": {
        "prompt": "Does the text express second-order normative beliefs (beliefs about what others think one should do)? Answer with exactly one of: none, weak, strong.",
        "options": ["none", "weak", "strong"],
    },
}


def load_norms_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def answer_to_label(qid: str, value: str) -> str:
    if qid in CODE_TO_LABEL and value in CODE_TO_LABEL[qid]:
        return CODE_TO_LABEL[qid][value]
    return value


def compute_counts(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """counts[question_id][sector][answer_label] = count"""
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for sector, items in data.items():
        for rec in items:
            ans = rec.get("answers") or {}
            for qid, val in ans.items():
                label = answer_to_label(qid, str(val).strip())
                counts[qid][sector][label] += 1
    return counts


def compute_recheck_counts(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """Among comments labelled 'against' (1st pass) in author stance, count second-pass recheck labels per sector.
    Returns recheck_counts[sector][recheck_label] = count. recheck_label in: against, frustrated but still pro, unclear stance.
    """
    recheck_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sector, items in data.items():
        for rec in items:
            ans = rec.get("answers") or {}
            if answer_to_label("1.1.1_stance", str(ans.get("1.1.1_stance", "")).strip()) != "against":
                continue
            recheck = (ans.get("1.1.1_stance_recheck") or "").strip().lower()
            if not recheck:
                continue
            # Normalize to display label
            if "frustrated" in recheck and "pro" in recheck:
                recheck = "frustrated but still pro"
            elif "unclear" in recheck:
                recheck = "unclear stance"
            elif "against" in recheck:
                recheck = "against"
            else:
                recheck = "unclear stance"  # fallback
            recheck_counts[sector][recheck] += 1
    return dict(recheck_counts)


def _chart_id(qid: str, sector: str = "") -> str:
    """Safe HTML id and JS variable name: no dots (e.g. 1.1_gate -> 1_1_gate)."""
    safe_q = qid.replace(".", "_")
    return f"{safe_q}_{sector}" if sector else safe_q


# Recheck (second pass) labels and colors for "against" author-stance recheck chart
RECHECK_LABELS_ORDER = ["against", "frustrated but still pro", "unclear stance"]
RECHECK_COLORS = {
    "against": "#e74c3c",
    "frustrated but still pro": "#3498db",
    "unclear stance": "#95a5a6",
}


def build_dashboard_html(
    counts: Dict[str, Dict[str, Dict[str, int]]],
    out_path: str,
    recheck_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Write dashboard HTML with Plotly charts (inline JSON + plotly.js). If recheck_counts is set, add bottom chart for against recheck."""
    sectors = ["food", "transport", "housing"]
    question_order = [
        "1.1_gate",
        "1.1.1_stance",
        "1.3.1_reference_group",
        "1.3.1b_perceived_reference_stance",
        "1.2.1_descriptive",
        "1.2.2_injunctive",
        "1.3.2_mechanism",
        "1.3.3_second_order",
    ]
    colors = {
        "Yes": "#3498db",
        "No": "#9b59b6",
        "pro": "#2ecc71",
        "against": "#e74c3c",
        "against particular but pro": "#e67e22",
        "neither/mixed": "#f1c40f",
        "pro but lack of options": "#3498db",
        "none": "#95a5a6",
        "implied": "#673ab7",
        "explicit": "#3f51b5",
        "implied approval": "#3498db",
        "implied disapproval": "#9b59b6",
        "explicit approval": "#34495e",
        "explicit disapproval": "#8e44ad",
        # Reference group (distinct per category)
        "family": "#2c3e50",
        "partner/spouse": "#9b59b6",
        "friends": "#3498db",
        "coworkers": "#8e44ad",
        "neighbors": "#2980b9",
        "local community": "#1abc9c",
        "political tribe": "#5d6d7e",
        "online community": "#95a5a6",
        "other reddit user": "#e91e63",
        "other": "#546e7a",
        # Mechanism
        "social comparison": "#5c6bc0",
        "praise": "#66bb6a",
        "blame/shame": "#ef5350",
        "community standard": "#26a69a",
        "identity/status signaling": "#ffa726",
        "rule/virtue language": "#ab47bc",
        # Second-order
        "weak": "#42a5f5",
        "strong": "#ff7043",
    }
    # Fallback palette for any label not in colors (distinct hues)
    _palette_extra = [
        "#7e57c2", "#ec407a", "#26c6da", "#9ccc65", "#ffca28",
        "#8d6e63", "#78909c", "#5c6bc0", "#66bb6a", "#ef5350",
    ]

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Norms Hierarchical Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
html { background: #0d1117; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 10px; background: #0d1117; color: #e6edf3; min-height: 100vh; font-size: 14px; }
h1 { text-align: center; color: #e6edf3; font-size: 1.6em; margin: 0.4em 0; }
h2 { margin: 6px 0 4px; color: #c9d1d9; font-size: 1.2em; }
.chart { margin: 4px 0; background: #161b22; border-radius: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.3); padding: 8px; }
.charts-row { display: flex; flex-wrap: nowrap; gap: 10px; margin: 8px 0; }
.charts-row .chart { flex: 1; min-width: 0; margin: 0; }
a { color: #58a6ff; font-size: 0.9em; }
</style>
</head>
<body>
<h1>Norms Hierarchical Dashboard</h1>
<p style="text-align:center"><a href="00_dashboard_examples.html">Example comments by category</a></p>
"""
    ]

    for qid in question_order:
        if qid not in counts:
            continue
        title = QUESTION_TITLES.get(qid, qid)
        html_parts.append(f"<h2>{title}</h2>\n")

        c = counts[qid]
        if qid == "1.1_gate":
            # One figure with 3 donuts (domains) and a single shared legend; bigger height
            gate_traces = []
            for i, sector in enumerate(sectors):
                yes_count = c.get(sector, {}).get("Yes", 0) + c.get(sector, {}).get("1", 0)
                no_count = c.get(sector, {}).get("No", 0) + c.get(sector, {}).get("0", 0)
                x0, x1 = i / 3, (i + 1) / 3
                # Shrink donut vertically (y domain) so sector titles sit above with clear gap
                gate_traces.append({
                    "labels": ["Yes", "No"],
                    "values": [yes_count, no_count],
                    "type": "pie",
                    "hole": 0.5,
                    "marker": {"colors": [colors.get("Yes", "#3498db"), colors.get("No", "#9b59b6")]},
                    "domain": {"x": [x0, x1], "y": [0.18, 0.92]},
                    "showlegend": i == 0,
                })
            cid = _chart_id(qid)
            annotations = [
                {"text": SECTOR_DISPLAY.get(s, s), "x": (i + 0.5) / 3, "y": 1.04, "showarrow": False, "xanchor": "center", "font": {"color": "#e6edf3", "size": 11}}
                for i, s in enumerate(sectors)
            ]
            gate_layout = {
                "height": 380,
                "paper_bgcolor": "#161b22",
                "plot_bgcolor": "#161b22",
                "font": {"color": "#e6edf3", "size": 11},
                "showlegend": True,
                "legend": {"orientation": "h", "yanchor": "top", "y": -0.08, "xanchor": "center", "x": 0.5, "font": {"size": 10}},
                "margin": {"t": 44, "b": 32, "l": 8, "r": 8},
                "annotations": annotations,
            }
            gate_fig = {"data": gate_traces, "layout": gate_layout}
            html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
            html_parts.append(
                f'<script>var fig_{cid} = {json.dumps(gate_fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
            )
        else:
            labels_seen = set()
            for sector in sectors:
                for label in c.get(sector, {}).keys():
                    labels_seen.add(label)
            labels_order = [l for l in colors if l in labels_seen] + [l for l in sorted(labels_seen) if l not in colors]
            # Assign distinct color per label (from colors dict or fallback palette)
            label_to_color = {}
            unseen_idx = 0
            for label in labels_order:
                if label in colors:
                    label_to_color[label] = colors[label]
                else:
                    label_to_color[label] = _palette_extra[unseen_idx % len(_palette_extra)]
                    unseen_idx += 1
            x_by_label = {label: [c.get(s, {}).get(label, 0) for s in sectors] for label in labels_order}
            y = [SECTOR_DISPLAY.get(s, s) for s in sectors]
            traces = []
            for label in labels_order:
                x = x_by_label[label]
                if any(x):
                    traces.append({
                        "x": x,
                        "y": y,
                        "name": label,
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": label_to_color[label]},
                        "text": [str(v) if v else "" for v in x],
                        "textposition": "inside",
                    })
            if traces:
                fig = {
                    "data": traces,
                    "layout": {
                        "barmode": "stack",
                        "height": 240,
                        "margin": {"l": 85, "t": 20, "b": 28},
                        "xaxis": {"title": "Count", "color": "#e6edf3", "gridcolor": "#30363d", "titlefont": {"size": 11}, "tickfont": {"size": 10}},
                        "yaxis": {"color": "#e6edf3", "gridcolor": "#30363d", "tickfont": {"size": 10}},
                        "showlegend": True,
                        "paper_bgcolor": "#161b22",
                        "plot_bgcolor": "#161b22",
                        "font": {"color": "#e6edf3", "size": 11},
                        "legend": {"font": {"color": "#e6edf3", "size": 10}},
                    },
                }
                cid = _chart_id(qid)
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )

    # Bottom section: "against" (1st pass) second-pass recheck percentages and chart
    if recheck_counts:
        recheck_traces = []
        labels_order = [l for l in RECHECK_LABELS_ORDER if any(recheck_counts.get(s, {}).get(l) for s in sectors)]
        for label in labels_order:
            x_vals = [recheck_counts.get(s, {}).get(label, 0) for s in sectors]
            if any(x_vals):
                recheck_traces.append({
                    "x": [SECTOR_DISPLAY.get(s, s) for s in sectors],
                    "y": x_vals,
                    "name": label,
                    "type": "bar",
                    "marker": {"color": RECHECK_COLORS.get(label, "#95a5a6")},
                    "text": [str(v) for v in x_vals],
                    "textposition": "inside",
                })
        if recheck_traces:
            html_parts.append('<h2>Against (1st pass) — second-pass recheck</h2>\n')
            # Summary line: Of N against (per sector): X% confirmed, Y% frustrated but still pro, Z% unclear
            summary_parts = []
            for sector in sectors:
                total = sum(recheck_counts.get(sector, {}).values())
                if total == 0:
                    continue
                pct_against = 100 * recheck_counts.get(sector, {}).get("against", 0) / total
                pct_frustrated = 100 * recheck_counts.get(sector, {}).get("frustrated but still pro", 0) / total
                pct_unclear = 100 * recheck_counts.get(sector, {}).get("unclear stance", 0) / total
                summary_parts.append(
                    f"{SECTOR_DISPLAY.get(sector, sector)}: of {total} &quot;against&quot; → {pct_against:.0f}% confirmed against, {pct_frustrated:.0f}% frustrated but still pro, {pct_unclear:.0f}% unclear"
                )
            if summary_parts:
                html_parts.append(f'<p style="font-size: 0.95em; color: #c9d1d9;">{" | ".join(summary_parts)}</p>\n')
            recheck_layout = {
                "barmode": "stack",
                "height": 240,
                "margin": {"l": 85, "t": 20, "b": 28},
                "xaxis": {"title": "Sector", "color": "#e6edf3", "gridcolor": "#30363d", "titlefont": {"size": 11}, "tickfont": {"size": 10}},
                "yaxis": {"title": "Count", "color": "#e6edf3", "gridcolor": "#30363d", "tickfont": {"size": 10}},
                "showlegend": True,
                "paper_bgcolor": "#161b22",
                "plot_bgcolor": "#161b22",
                "font": {"color": "#e6edf3", "size": 11},
                "legend": {"font": {"color": "#e6edf3", "size": 10}},
            }
            recheck_fig = {"data": recheck_traces, "layout": recheck_layout}
            html_parts.append('<div class="chart" id="chart_stance_recheck"></div>\n')
            html_parts.append(
                f'<script>var fig_stance_recheck = {json.dumps(recheck_fig)}; Plotly.newPlot("chart_stance_recheck", fig_stance_recheck.data, fig_stance_recheck.layout);</script>\n'
            )

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote dashboard: {out_path}")


def build_examples_html(data: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    """One example comment per (question, category, sector)."""
    sectors = ["food", "transport", "housing"]
    question_order = [
        "1.1_gate",
        "1.1.1_stance",
        "1.3.1_reference_group",
        "1.3.1b_perceived_reference_stance",
        "1.2.1_descriptive",
        "1.2.2_injunctive",
        "1.3.2_mechanism",
    ]
    # Collect by (qid, label, sector) -> list of comment texts, then pick one at random per slot
    by_cat: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sector, items in data.items():
        for rec in items:
            comment = (rec.get("comment") or "").strip()
            if not comment:
                continue
            text = comment[:500] + ("..." if len(comment) > 500 else "")
            ans = rec.get("answers") or {}
            for qid, val in ans.items():
                label = answer_to_label(qid, str(val).strip())
                by_cat[qid][label][sector].append(text)
    # Up to 3 random samples per (qid, label, sector) for display
    N_EXAMPLES = 3
    samples: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for qid in by_cat:
        for label in by_cat[qid]:
            for sector in by_cat[qid][label]:
                cands = by_cat[qid][label][sector]
                if cands:
                    n = min(N_EXAMPLES, len(cands))
                    samples[qid][label][sector] = random.sample(cands, n)

    # Against recheck (2nd pass): (recheck_label, sector) -> list of comment texts (author stance was "against")
    recheck_by_cat: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for sector, items in data.items():
        for rec in items:
            ans = rec.get("answers") or {}
            if answer_to_label("1.1.1_stance", str(ans.get("1.1.1_stance", "")).strip()) != "against":
                continue
            recheck_raw = (ans.get("1.1.1_stance_recheck") or "").strip().lower()
            if not recheck_raw:
                continue
            if "frustrated" in recheck_raw and "pro" in recheck_raw:
                recheck_label = "frustrated but still pro"
            elif "unclear" in recheck_raw:
                recheck_label = "unclear stance"
            elif "against" in recheck_raw:
                recheck_label = "against"
            else:
                recheck_label = "unclear stance"
            comment = (rec.get("comment") or "").strip()
            if comment:
                text = comment[:500] + ("..." if len(comment) > 500 else "")
                recheck_by_cat[recheck_label][sector].append(text)
    recheck_samples: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for recheck_label in recheck_by_cat:
        for sector in recheck_by_cat[recheck_label]:
            cands = recheck_by_cat[recheck_label][sector]
            if cands:
                n = min(N_EXAMPLES, len(cands))
                recheck_samples[recheck_label][sector] = random.sample(cands, n)

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Norms Hierarchical - Example Comments</title>
<style>
* { box-sizing: border-box; }
html { background: #0d1117; }
body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 12px; background: #0d1117; color: #e6edf3; min-height: 100vh; font-size: 13px; }
h1 { text-align: center; color: #e6edf3; font-size: 1.28em; margin: 0.4em 0; }
.back-link { text-align: center; margin-bottom: 14px; font-size: 0.8em; }
.back-link a { color: #58a6ff; }
.question-section { max-width: 1400px; margin: 0 auto 10px; background: #161b22; border-radius: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.question-section summary { font-size: 14px; font-weight: 600; color: #c9d1d9; padding: 12px 18px; cursor: pointer; list-style: none; border-bottom: 2px solid transparent; }
.question-section summary::-webkit-details-marker { display: none; }
.question-section summary::before { content: "▶ "; color: #58a6ff; font-size: 10px; }
.question-section[open] summary::before { content: "▼ "; }
.question-section[open] summary { border-bottom-color: #30363d; }
.question-content { padding: 18px; padding-top: 10px; }
.category-group { margin-bottom: 14px; }
.category-title { font-weight: 600; font-size: 11px; margin-bottom: 8px; padding: 6px 10px; border-radius: 0; display: inline-block; color: #e6edf3; }
.sectors-row { display: grid; grid-template-columns: repeat(9, 1fr); gap: 8px; }
.sector-column { display: flex; flex-direction: column; min-width: 0; }
.sector-header { font-weight: 600; font-size: 9px; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; }
.example-comment { padding: 6px 8px; border-radius: 0; font-size: 9.5px; line-height: 1.4; white-space: pre-wrap; word-break: break-word; border-left: 3px solid #58a6ff; background: #0d1117; color: #e6edf3; }
.prompt-dropdown { margin-bottom: 14px; }
.prompt-dropdown summary { cursor: pointer; color: #58a6ff; font-size: 11px; user-select: none; }
.prompt-dropdown summary:hover { text-decoration: underline; }
.prompt-box { margin-top: 10px; padding: 10px; border-radius: 0; background: #0d1117; border: 1px solid #30363d; font-size: 10.4px; }
.prompt-box .prompt-text { color: #e6edf3; line-height: 1.5; margin-bottom: 8px; }
.prompt-box .choices-label { color: #8b949e; font-weight: 600; margin-bottom: 4px; font-size: 10.4px; }
.prompt-box .choices-list { color: #c9d1d9; list-style: none; padding-left: 0; font-size: 10.4px; }
.prompt-box .choices-list li { margin: 2px 0; }
</style>
</head>
<body>
<h1>Example Comments by Category</h1>
<div class="back-link"><a href="00_dashboardv2.html">&lt;- Back to Dashboard</a></div>
"""
    ]

    for qid in question_order:
        if qid not in samples:
            continue
        title = QUESTION_TITLES.get(qid, qid)
        html_parts.append(f'<details class="question-section"><summary>{html_escape(title)}</summary>\n')
        html_parts.append('<div class="question-content">\n')
        # Dropdown: exact prompt and choices sent to the LLM
        prompt_info = NORMS_PROMPTS.get(qid)
        if prompt_info:
            html_parts.append('<details class="prompt-dropdown"><summary>Read exact prompt &amp; choices sent to the LLM</summary>\n')
            html_parts.append('<div class="prompt-box">\n')
            prompt_text = prompt_info["prompt"]
            if prompt_info.get("sector_specific") and prompt_info.get("sector_topic"):
                html_parts.append('<div class="choices-label">Prompt uses only the comment\'s sector topic (not all three):</div>\n')
                st = prompt_info["sector_topic"]
                for sec, topic in st.items():
                    html_parts.append(f'<div class="prompt-text"><strong>{sec}:</strong> {html_escape(topic)}</div>\n')
                html_parts.append('<div class="choices-label">Prompt template:</div>\n')
            html_parts.append(f'<div class="prompt-text">{html_escape(prompt_text)}</div>\n')
            html_parts.append('<div class="choices-label">Valid choices:</div>\n<ul class="choices-list">\n')
            for opt in prompt_info["options"]:
                html_parts.append(f'<li>{html_escape(opt)}</li>\n')
            html_parts.append('</ul>\n</div>\n</details>\n')
        for label in sorted(samples[qid].keys()):
            html_parts.append(f'<div class="category-group"><div class="category-title">{label}</div><div class="sectors-row">\n')
            for sector in sectors:
                texts = (samples[qid][label].get(sector) or [])[:N_EXAMPLES]
                while len(texts) < N_EXAMPLES:
                    texts.append("")
                sector_name = SECTOR_DISPLAY.get(sector, sector)
                for i, text in enumerate(texts):
                    header = f"{sector_name} · {i + 1}"
                    html_parts.append(f'<div class="sector-column"><div class="sector-header">{header}</div>')
                    html_parts.append(f'<div class="example-comment">{html_escape(text) if text else "—"}</div></div>\n')
            html_parts.append("</div></div>\n")
        html_parts.append("</div></details>\n")

    # Against recheck (2nd pass) examples: comments labelled "against" (1st pass) and their recheck label
    if recheck_samples:
        html_parts.append('<details class="question-section"><summary>Against recheck (2nd pass)</summary>\n')
        html_parts.append('<div class="question-content">\n')
        html_parts.append('<p style="font-size: 11px; color: #8b949e; margin-bottom: 12px;">Comments that were labelled &quot;against&quot; in Author stance (1st pass) and re-labelled in a stringent second LLM pass.</p>\n')
        for recheck_label in RECHECK_LABELS_ORDER:
            if recheck_label not in recheck_samples:
                continue
            html_parts.append(f'<div class="category-group"><div class="category-title">{html_escape(recheck_label)}</div><div class="sectors-row">\n')
            for sector in sectors:
                texts = (recheck_samples[recheck_label].get(sector) or [])[:N_EXAMPLES]
                while len(texts) < N_EXAMPLES:
                    texts.append("")
                sector_name = SECTOR_DISPLAY.get(sector, sector)
                for i, text in enumerate(texts):
                    header = f"{sector_name} · {i + 1}"
                    html_parts.append(f'<div class="sector-column"><div class="sector-header">{header}</div>')
                    html_parts.append(f'<div class="example-comment">{html_escape(text) if text else "—"}</div></div>\n')
            html_parts.append("</div></div>\n")
        html_parts.append("</div></details>\n")

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote examples: {out_path}")


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Build norms dashboard and examples from norms_labels.json")
    p.add_argument("--input", default="paper4data/norms_labels.json", help="Input JSON from 00_vLLM_hierarchical.py --norms (default, used when omitted)")
    p.add_argument("--dashboard", default="00_dashboardv2.html", help="Output dashboard HTML path")
    p.add_argument("--examples", default="00_dashboard_examples.html", help="Output examples HTML path")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        print("Run first: python 00_vLLM_hierarchical.py --label-only --norms --limit-total 1000")
        return

    data = load_norms_labels(args.input)
    counts = compute_counts(data)
    recheck_counts = compute_recheck_counts(data)
    build_dashboard_html(counts, args.dashboard, recheck_counts=recheck_counts)
    build_examples_html(data, args.examples)
    print("Done.")


if __name__ == "__main__":
    main()
