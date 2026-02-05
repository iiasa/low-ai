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
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional


def _collapse_newlines(s: str) -> str:
    """Collapse 2+ newlines to a single newline (for example comment display)."""
    return re.sub(r"\n{2,}", "\n", s)

# Map answer codes to display labels (dashboard and examples)
CODE_TO_LABEL = {
    "1.1_gate": {"0": "No", "1": "Yes"},
    "1.1.1_stance": {"pushing for": "pro"},
    "1.3.1b_perceived_reference_stance": {"pushing for": "pro"},
    "1.2.1_descriptive": {
        "0": "absent", "1": "unclear", "2": "explicitly present",
        "none": "absent", "implied": "unclear", "explicit": "explicitly present",
    },
    "1.2.2_injunctive": {
        "0": "none", "1": "implied approval", "2": "implied disapproval",
        "3": "explicit approval", "4": "explicit disapproval",
        "none": "absent", "implied approval": "present", "implied disapproval": "present",
        "explicit approval": "present", "explicit disapproval": "present",
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
        "prompt": "Descriptive norms refer to what people actually do or how common a behavior is (e.g. 'most people here drive EVs', 'I am a vegetarian'). They describe behavior or prevalence, not what people should do. Do NOT code as descriptive if the text prescribes or proscribes behavior (that is injunctive). Answer with exactly one of: explicitly present, absent, unclear.",
        "options": ["explicitly present", "absent", "unclear"],
    },
    "1.2.2_injunctive": {
        "prompt": "Injunctive norms are social rules about what behaviors are approved or disapproved—guiding what people should do (or avoid). They use language like should, must, have to, ought to, or express approval/disapproval (e.g. 'people should go vegan', 'I encourage everyone to go vegan'). Do NOT code as injunctive mere descriptions of how people act (e.g. 'I am a vegetarian' = describing one's own behavior, not a rule). Code as injunctive only when the text prescribes or proscribes behavior for others. Answer with exactly one of: present, absent, unclear.",
        "options": ["present", "absent", "unclear"],
    },
    "1.3.1_reference_group": {
        "prompt": "Who is the reference group? This refers to a group that the author has a personal relationship with (e.g. their own family, their own coworkers, their own friends). Do NOT code as a reference group if the text describes someone else's relationship (e.g. 'my neighbor's family' → not 'family' as reference group) or describes someone else who has that relationship (e.g. 'a coworker' without indicating the author's relationship). The reference group is who the author refers to as doing or approving something, where the author has a personal connection to that group. Answer with exactly one of: coworkers, family, friends, local community, neighbors, online community, other, other reddit user, partner/spouse, political tribe.",
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


def load_survey_metadata(survey_path: str = "00_vllm_survey_question_final.json") -> Dict[str, Dict[str, str]]:
    """Load survey question metadata (id -> wording, short_form, sector) for display titles."""
    survey_file = Path(survey_path)
    if not survey_file.exists():
        return {}
    with open(survey_file, "r", encoding="utf-8") as f:
        survey_data = json.load(f)
    
    metadata: Dict[str, Dict[str, str]] = {}
    for sector_key in ["FOOD", "TRANSPORT", "HOUSING"]:
        if sector_key not in survey_data:
            continue
        for question_set_name, question_set in survey_data[sector_key].items():
            for q in question_set.get("questions", []):
                metadata[q["id"]] = {
                    "wording": q.get("wording", q["id"]),
                    "short_form": q.get("short_form", q.get("wording", q["id"])),
                    "sector": sector_key.lower(),
                }
    return metadata


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


def compute_temporal_counts(
    data: Dict[str, List[Dict[str, Any]]],
    qid: str,
    min_year: int = 2010,
) -> Dict[str, Dict[int, Dict[str, int]]]:
    """
    Compute counts by year for a question. Returns temporal_counts[sector][year][label] = count.
    Only includes items with year >= min_year.
    """
    temporal_counts: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for sector, items in data.items():
        for rec in items:
            year = rec.get("year")
            if not year or not isinstance(year, (int, float)) or int(year) < min_year:
                continue
            year_int = int(year)
            ans = rec.get("answers") or {}
            val = ans.get(qid)
            if val is None:
                continue
            label = answer_to_label(qid, str(val).strip())
            temporal_counts[sector][year_int][label] += 1
    return dict(temporal_counts)


def build_dashboard_html(
    counts: Dict[str, Dict[str, Dict[str, int]]],
    out_path: str,
    recheck_counts: Optional[Dict[str, Dict[str, int]]] = None,
    data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    survey_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """Write dashboard HTML with Plotly charts (inline JSON + plotly.js). If recheck_counts is set, add bottom chart for against recheck. If data is set, add temporal plots."""
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
        # Descriptive / injunctive (new schema)
        "explicitly present": "#3f51b5",
        "absent": "#95a5a6",
        "unclear": "#78909c",
        "present": "#3498db",
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
                        "showlegend": True,
                    })
            if traces:
                fig = {
                    "data": traces,
                    "layout": {
                        "barmode": "stack",
                        "bargap": 0.2,
                        "height": 150,
                        "margin": {"l": 85, "t": 15, "b": 20},
                        "xaxis": {"title": "Count", "color": "#e6edf3", "gridcolor": "#30363d", "titlefont": {"size": 11}, "tickfont": {"size": 10}},
                        "yaxis": {"color": "#e6edf3", "gridcolor": "#30363d", "tickfont": {"size": 10}},
                        "showlegend": True,
                        "paper_bgcolor": "#161b22",
                        "plot_bgcolor": "#161b22",
                        "font": {"color": "#e6edf3", "size": 11},
                        "legend": {
                            "font": {"color": "#e6edf3", "size": 10},
                            "traceorder": "normal",
                            "itemclick": "toggleothers",
                            "itemdoubleclick": "toggle",
                        },
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
                "bargap": 0.2,
                "height": 150,
                "margin": {"l": 85, "t": 15, "b": 20},
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

    # Temporal plots for Author stance and Reference group
    if data:
        temporal_qids = ["1.1.1_stance", "1.3.1_reference_group"]
        for qid in temporal_qids:
            temporal_counts = compute_temporal_counts(data, qid, min_year=2010)
            if not temporal_counts:
                continue
            
            title = QUESTION_TITLES.get(qid, qid)
            html_parts.append(f'<h2>{title} (temporal)</h2>\n')
            
            # Collect all years and labels across sectors
            all_years = set()
            all_labels = set()
            for sector_counts in temporal_counts.values():
                all_years.update(sector_counts.keys())
                for year_counts in sector_counts.values():
                    all_labels.update(year_counts.keys())
            
            if not all_years:
                continue
            
            years_sorted = sorted(all_years)
            labels_order = [l for l in colors if l in all_labels] + [l for l in sorted(all_labels) if l not in colors]
            
            # One chart per sector (3 charts in a row)
            html_parts.append('<div class="charts-row">\n')
            for sector in sectors:
                sector_counts = temporal_counts.get(sector, {})
                if not sector_counts:
                    continue
                
                traces = []
                label_to_color_map = {}
                unseen_idx = 0
                for label in labels_order:
                    if label in colors:
                        label_to_color_map[label] = colors[label]
                    else:
                        label_to_color_map[label] = _palette_extra[unseen_idx % len(_palette_extra)]
                        unseen_idx += 1
                    
                    y_vals = [sector_counts.get(year, {}).get(label, 0) for year in years_sorted]
                    if any(y_vals):
                        trace = {
                            "x": years_sorted,
                            "y": y_vals,
                            "name": label,
                            "type": "scatter",
                            "mode": "lines+markers",
                            "stackgroup": "one",
                            "marker": {"color": label_to_color_map[label], "size": 6},
                            "line": {"color": label_to_color_map[label], "width": 2},
                        }
                        if traces:
                            trace["fill"] = "tonexty"
                        else:
                            trace["fill"] = "tozeroy"
                        traces.append(trace)
                
                if traces:
                    cid = _chart_id(qid, sector)
                    temporal_layout = {
                        "height": 300,
                        "margin": {"l": 60, "t": 40, "b": 40, "r": 20},
                        "xaxis": {
                            "title": "Year",
                            "color": "#e6edf3",
                            "gridcolor": "#30363d",
                            "titlefont": {"size": 11},
                            "tickfont": {"size": 10},
                        },
                        "yaxis": {
                            "title": "Count",
                            "color": "#e6edf3",
                            "gridcolor": "#30363d",
                            "tickfont": {"size": 10},
                        },
                        "showlegend": True,
                        "paper_bgcolor": "#161b22",
                        "plot_bgcolor": "#161b22",
                        "font": {"color": "#e6edf3", "size": 11},
                        "legend": {"font": {"color": "#e6edf3", "size": 9}, "orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
                        "title": {"text": SECTOR_DISPLAY.get(sector, sector), "font": {"color": "#e6edf3", "size": 12}},
                    }
                    temporal_fig = {"data": traces, "layout": temporal_layout}
                    html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                    html_parts.append(
                        f'<script>var fig_{cid} = {json.dumps(temporal_fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                    )
            html_parts.append("</div>\n")

    # Survey questions section (at bottom)
    survey_metadata = survey_metadata or {}
    survey_qids = [qid for qid in counts.keys() if qid.startswith(("diet_", "ev_", "solar_"))]
    if survey_qids:
        html_parts.append('<h2 style="margin-top: 40px;">Survey Questions</h2>\n')
        # Group by sector
        survey_by_sector: Dict[str, List[str]] = defaultdict(list)
        for qid in sorted(survey_qids):
            sector = survey_metadata.get(qid, {}).get("sector", "unknown")
            survey_by_sector[sector].append(qid)
        
        for sector in ["food", "transport", "housing"]:
            if sector not in survey_by_sector:
                continue
            
            sector_title = SECTOR_DISPLAY.get(sector, sector.upper())
            html_parts.append(f'<h3 style="margin-top: 20px; color: #c9d1d9;">{sector_title}</h3>\n')
            
            for qid in survey_by_sector[sector]:
                if qid not in counts:
                    continue
                
                short_form = survey_metadata.get(qid, {}).get("short_form", qid)
                html_parts.append(f'<h4 style="margin-top: 12px; margin-bottom: 4px; color: #b1bac4; font-size: 1.0em;">{short_form}</h4>\n')
                
                c = counts[qid]
                # Create horizontal bar chart: YES vs NO for this question's sector only
                if sector not in c:
                    continue
                
                yes_count = c[sector].get("YES", 0) + c[sector].get("1", 0) + c[sector].get("yes", 0)
                no_count = c[sector].get("NO", 0) + c[sector].get("0", 0) + c[sector].get("no", 0)
                total = yes_count + no_count
                
                if total == 0:
                    continue
                
                yes_pct = (yes_count / total) * 100
                no_pct = (no_count / total) * 100
                
                traces = [
                    {
                        "x": [yes_pct],
                        "y": [SECTOR_DISPLAY.get(sector, sector)],
                        "name": "YES",
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": "#2ecc71"},
                        "text": [f"{yes_pct:.1f}% ({yes_count})"],
                        "textposition": "inside",
                        "showlegend": True,
                    },
                    {
                        "x": [no_pct],
                        "y": [SECTOR_DISPLAY.get(sector, sector)],
                        "name": "NO",
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": "#e74c3c"},
                        "text": [f"{no_pct:.1f}% ({no_count})"],
                        "textposition": "inside",
                        "showlegend": True,
                    }
                ]
                
                cid = _chart_id(qid, "")
                survey_layout = {
                    "barmode": "stack",
                    "bargap": 0.2,
                    "height": 100,
                    "margin": {"l": 85, "t": 10, "b": 15, "r": 20},
                    "xaxis": {"title": "Percentage", "range": [0, 100], "color": "#e6edf3", "gridcolor": "#30363d", "titlefont": {"size": 10}, "tickfont": {"size": 9}},
                    "yaxis": {"color": "#e6edf3", "gridcolor": "#30363d", "tickfont": {"size": 9}},
                    "showlegend": True,
                    "paper_bgcolor": "#161b22",
                    "plot_bgcolor": "#161b22",
                    "font": {"color": "#e6edf3", "size": 10},
                    "legend": {"font": {"color": "#e6edf3", "size": 9}, "orientation": "h", "yanchor": "bottom", "y": -0.35, "xanchor": "center", "x": 0.5},
                }
                survey_fig = {"data": traces, "layout": survey_layout}
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(survey_fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote dashboard: {out_path}")


def build_examples_html(data: Dict[str, List[Dict[str, Any]]], out_path: str, survey_metadata: Optional[Dict[str, Dict[str, str]]] = None) -> None:
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
    
    # Add survey questions to question_order if they exist in data
    survey_metadata = survey_metadata or {}
    survey_qids = []
    for sector, items in data.items():
        for rec in items:
            ans = rec.get("answers") or {}
            for qid in ans.keys():
                if qid.startswith(("diet_", "ev_", "solar_")) and qid not in survey_qids:
                    survey_qids.append(qid)
    if survey_qids:
        question_order.extend(sorted(survey_qids))
    # Collect by (qid, label, sector) -> list of comment texts, then pick one at random per slot
    by_cat: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sector, items in data.items():
        for rec in items:
            comment = _collapse_newlines((rec.get("comment") or "").strip())
            if not comment:
                continue
            text = comment[:500] + ("..." if len(comment) > 500 else "")
            ans = rec.get("answers") or {}
            for qid, val in ans.items():
                label = answer_to_label(qid, str(val).strip())
                # Normalize survey question labels: "1" -> "YES", "0" -> "NO"
                if qid.startswith(("diet_", "ev_", "solar_")):
                    if label in ("1", "yes"):
                        label = "YES"
                    elif label in ("0", "no"):
                        label = "NO"
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
            comment = _collapse_newlines((rec.get("comment") or "").strip())
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
        # Use short_form for survey questions, QUESTION_TITLES for norms questions
        if qid.startswith(("diet_", "ev_", "solar_")):
            title = survey_metadata.get(qid, {}).get("short_form", qid)
        else:
            title = QUESTION_TITLES.get(qid, qid)
        html_parts.append(f'<details class="question-section"><summary>{html_escape(title)}</summary>\n')
        html_parts.append('<div class="question-content">\n')
        # Dropdown: exact prompt and choices sent to the LLM
        prompt_info = NORMS_PROMPTS.get(qid)
        # For survey questions, show prompt from metadata
        if qid.startswith(("diet_", "ev_", "solar_")):
            survey_q_meta = survey_metadata.get(qid, {})
            if survey_q_meta:
                html_parts.append('<details class="prompt-dropdown"><summary>Read exact prompt &amp; choices sent to the LLM</summary>\n')
                html_parts.append('<div class="prompt-box">\n')
                # Load full survey data to get prompt
                survey_file = Path("00_vllm_survey_question_final.json")
                if survey_file.exists():
                    with open(survey_file, "r", encoding="utf-8") as f:
                        survey_data = json.load(f)
                    for sector_key in ["FOOD", "TRANSPORT", "HOUSING"]:
                        if sector_key not in survey_data:
                            continue
                        for question_set_name, question_set in survey_data[sector_key].items():
                            for q in question_set.get("questions", []):
                                if q["id"] == qid:
                                    html_parts.append(f'<div class="prompt-text">{html_escape(q.get("prompt", ""))}</div>\n')
                                    html_parts.append('<div class="choices-label">Valid choices:</div>\n<ul class="choices-list">\n')
                                    html_parts.append('<li>YES</li>\n<li>NO</li>\n')
                                    html_parts.append('</ul>\n</div>\n</details>\n')
                                    break
                            else:
                                continue
                            break
        elif prompt_info:
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
        # For survey questions, only show examples for the matching sector
        survey_q_sector = None
        if qid.startswith(("diet_", "ev_", "solar_")):
            survey_q_sector = survey_metadata.get(qid, {}).get("sector", None)
        
        for label in sorted(samples[qid].keys()):
            html_parts.append(f'<div class="category-group"><div class="category-title">{label}</div><div class="sectors-row">\n')
            for sector in sectors:
                # Skip sectors that don't match for survey questions
                if survey_q_sector and sector != survey_q_sector:
                    # Fill with empty slots for non-matching sectors
                    for i in range(N_EXAMPLES):
                        sector_name = SECTOR_DISPLAY.get(sector, sector)
                        header = f"{sector_name} · {i + 1}"
                        html_parts.append(f'<div class="sector-column"><div class="sector-header">{header}</div>')
                        html_parts.append(f'<div class="example-comment">—</div></div>\n')
                    continue
                
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
    survey_metadata = load_survey_metadata()
    build_dashboard_html(counts, args.dashboard, recheck_counts=recheck_counts, data=data, survey_metadata=survey_metadata)
    build_examples_html(data, args.examples, survey_metadata=survey_metadata)
    print("Done.")


if __name__ == "__main__":
    main()
