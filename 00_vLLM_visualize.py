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
import numpy as np
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
    "against": "#ff9aa8",
    "frustrated but still pro": "#8fbfd9",
    "unclear stance": "#d0d0d0",
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
        "1.1.1_stance",
        "1.1_gate",
        "1.3.1_reference_group",
        "1.3.1b_perceived_reference_stance",
        "1.2.1_descriptive",
        "1.2.2_injunctive",
        "1.3.2_mechanism",
        "1.3.3_second_order",
    ]
    colors = {
        "Yes": "#7cbcbe",
        "No": "#c49fc4",
        "pro": "#8fcc8f",
        "against": "#ff9aa8",
        "against particular but pro": "#ffb87a",
        "neither/mixed": "#ffe87a",
        "pro but lack of options": "#8fbfd9",
        "none": "#c2c2c2",
        "implied": "#b88fb8",
        "explicit": "#7caed6",
        "implied approval": "#8fbfd9",
        "implied disapproval": "#d99fd9",
        "explicit approval": "#7cadc6",
        "explicit disapproval": "#c692c6",
        # Descriptive / injunctive (new schema)
        "explicitly present": "#7caed6",
        "absent": "#d0d0d0",
        "unclear": "#b8c1c6",
        "present": "#8fbfd9",
        # Reference group (distinct per category)
        "family": "#a8b8c2",
        "partner/spouse": "#c49fc4",
        "friends": "#8fbfd9",
        "coworkers": "#c692c6",
        "neighbors": "#7cadc6",
        "local community": "#7cc2b8",
        "political tribe": "#a8b1b8",
        "online community": "#b8b8b8",
        "other reddit user": "#ffa8c2",
        "other": "#a0acb4",
        # Mechanism
        "social comparison": "#92a8c6",
        "praise": "#a8cca8",
        "blame/shame": "#ffa8a8",
        "community standard": "#7cc2b8",
        "identity/status signaling": "#ffc8a0",
        "rule/virtue language": "#c6a8c6",
        # Second-order
        "weak": "#8fbfd9",
        "strong": "#ffb0a0",
    }
    # Fallback palette for any label not in colors (distinct hues)
    _palette_extra = [
        "#b8a8c6", "#ffa8c2", "#a8d4dc", "#c2d4b0", "#ffd4a8",
        "#c2a89c", "#a8b4b8", "#a0acc6", "#b0ccb0", "#ffb0b0",
    ]

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Norms Hierarchical Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
html { background: #000000; }
body { font-family: "Inter", "Segoe UI", system-ui, sans-serif; margin: 0; padding: 8px; background: #000000; color: #e0e0e0; min-height: 100vh; font-size: 13px; }
h1 { text-align: center; color: #e0e0e0; font-size: 1.3em; margin: 0.3em 0; font-weight: 600; }
h2 { margin: 4px 0 3px; color: #b0b0b0; font-size: 1.05em; font-weight: 600; }
h3 { color: #b0b0b0; }
.chart { margin: 3px 0; background: #1a1a1a; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); padding: 6px; }
.charts-row { display: flex; flex-wrap: nowrap; gap: 24px; margin: 6px 0; }
.charts-row .chart { flex: 1; min-width: 0; margin: 0; }
a { color: #6c9bcf; font-size: 0.85em; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>Norms Hierarchical Dashboard</h1>
<p style="text-align:center; margin: 6px 0;"><a href="00_dashboard_examples.html">Example comments by category</a></p>
"""
    ]

    # Group bar charts into pairs for 2-per-row layout
    bar_chart_buffer = []

    for qid in question_order:
        if qid not in counts:
            continue
        title = QUESTION_TITLES.get(qid, qid)
        c = counts[qid]

        if qid == "1.1_gate":
            # Flush any pending bar charts before gate chart
            if bar_chart_buffer:
                html_parts.append('<div class="charts-row">\n')
                for buffered_html in bar_chart_buffer:
                    html_parts.append(buffered_html)
                html_parts.append('</div>\n')
                bar_chart_buffer = []

            html_parts.append(f"<h2>{title}</h2>\n")
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
                {"text": SECTOR_DISPLAY.get(s, s), "x": (i + 0.5) / 3, "y": 1.04, "showarrow": False, "xanchor": "center", "font": {"color": "#b0b0b0", "size": 10}}
                for i, s in enumerate(sectors)
            ]
            gate_layout = {
                "height": 280,
                "paper_bgcolor": "#1a1a1a",
                "plot_bgcolor": "#1a1a1a",
                "font": {"color": "#e0e0e0", "size": 10},
                "showlegend": True,
                "legend": {"orientation": "h", "yanchor": "top", "y": -0.06, "xanchor": "center", "x": 0.5, "font": {"size": 9, "color": "#e0e0e0"}},
                "margin": {"t": 32, "b": 24, "l": 6, "r": 6},
                "annotations": annotations,
            }
            gate_fig = {"data": gate_traces, "layout": gate_layout}
            html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
            html_parts.append(
                f'<script>var fig_{cid} = {json.dumps(gate_fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
            )
        else:
            # Bar chart - buffer for 2-per-row layout
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

            # Calculate totals per sector for percentages
            sector_totals = {s: sum(c.get(s, {}).values()) for s in sectors}

            traces = []
            for label in labels_order:
                x = x_by_label[label]
                if any(x):
                    # Format text as "n (x%)" - hide for segments < 5%
                    text_labels = []
                    for i, s in enumerate(sectors):
                        count = x[i]
                        if count > 0:
                            total = sector_totals.get(s, 0)
                            pct = (count / total * 100) if total > 0 else 0
                            # Only show text if segment is >= 5% (readable size)
                            if pct >= 5:
                                text_labels.append(f"{count} ({pct:.0f}%)")
                            else:
                                text_labels.append("")
                        else:
                            text_labels.append("")

                    traces.append({
                        "x": x,
                        "y": y,
                        "name": label,
                        "type": "bar",
                        "orientation": "h",
                        "marker": {"color": label_to_color[label]},
                        "text": text_labels,
                        "textposition": "inside",
                        "textfont": {"size": 10},  # Fixed minimum font size
                        "showlegend": True,
                    })
            if traces:
                fig = {
                    "data": traces,
                    "layout": {
                        "barmode": "stack",
                        "bargap": 0.15,
                        "height": 160,
                        "width": 450,
                        "margin": {"l": 75, "t": 30, "b": 75},
                        "xaxis": {"title": "", "color": "#b0b0b0", "gridcolor": "#3a3a3a", "titlefont": {"size": 10, "color": "#b0b0b0"}, "tickfont": {"size": 9, "color": "#b0b0b0"}},
                        "yaxis": {"color": "#b0b0b0", "gridcolor": "#3a3a3a", "tickfont": {"size": 9, "color": "#b0b0b0"}},
                        "showlegend": True,
                        "paper_bgcolor": "#1a1a1a",
                        "plot_bgcolor": "#1a1a1a",
                        "font": {"color": "#e0e0e0", "size": 10},
                        "legend": {
                            "font": {"color": "#e0e0e0", "size": 9},
                            "traceorder": "normal",
                            "itemclick": "toggleothers",
                            "itemdoubleclick": "toggle",
                            "orientation": "h",
                            "yanchor": "top",
                            "y": -0.35,
                            "xanchor": "center",
                            "x": 0.5,
                        },
                        "title": {"text": title, "font": {"color": "#b0b0b0", "size": 13, "family": "Inter, Segoe UI, sans-serif"}, "x": 0.02, "xanchor": "left"},
                    },
                }
                cid = _chart_id(qid)
                chart_html = f'<div class="chart" id="chart_{cid}"></div>\n'
                chart_html += f'<script>var fig_{cid} = {json.dumps(fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'

                # Buffer this chart
                bar_chart_buffer.append(chart_html)

                # If we have 3 charts buffered, emit them in a row
                if len(bar_chart_buffer) == 3:
                    html_parts.append('<div class="charts-row">\n')
                    for buffered_html in bar_chart_buffer:
                        html_parts.append(buffered_html)
                    html_parts.append('</div>\n')
                    bar_chart_buffer = []

    # Bottom section: "against" (1st pass) second-pass recheck percentages and chart
    if recheck_counts:
        recheck_traces = []
        labels_order = [l for l in RECHECK_LABELS_ORDER if any(recheck_counts.get(s, {}).get(l) for s in sectors)]

        # Calculate totals per sector for percentages
        recheck_sector_totals = {s: sum(recheck_counts.get(s, {}).values()) for s in sectors}

        for label in labels_order:
            counts_vals = [recheck_counts.get(s, {}).get(label, 0) for s in sectors]
            if any(counts_vals):
                # Format text as "n (x%)" - hide for segments < 5%
                text_labels = []
                for i, s in enumerate(sectors):
                    count = counts_vals[i]
                    if count > 0:
                        total = recheck_sector_totals.get(s, 0)
                        pct = (count / total * 100) if total > 0 else 0
                        # Only show text if segment is >= 5% (readable size)
                        if pct >= 5:
                            text_labels.append(f"{count} ({pct:.0f}%)")
                        else:
                            text_labels.append("")
                    else:
                        text_labels.append("")

                recheck_traces.append({
                    "x": counts_vals,
                    "y": [SECTOR_DISPLAY.get(s, s) for s in sectors],
                    "name": label,
                    "type": "bar",
                    "orientation": "h",
                    "marker": {"color": RECHECK_COLORS.get(label, "#e8e8e8")},
                    "text": text_labels,
                    "textposition": "inside",
                    "textfont": {"size": 10},  # Fixed minimum font size
                    "showlegend": True,
                })
        if recheck_traces:
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
            summary_text = " | ".join(summary_parts) if summary_parts else ""

            recheck_layout = {
                "barmode": "stack",
                "bargap": 0.15,
                "height": 160,
                "width": 450,
                "margin": {"l": 75, "t": 30, "b": 90},
                "xaxis": {"title": "", "color": "#b0b0b0", "gridcolor": "#3a3a3a", "titlefont": {"size": 10, "color": "#b0b0b0"}, "tickfont": {"size": 9, "color": "#b0b0b0"}},
                "yaxis": {"color": "#b0b0b0", "gridcolor": "#3a3a3a", "tickfont": {"size": 9, "color": "#b0b0b0"}},
                "showlegend": True,
                "paper_bgcolor": "#1a1a1a",
                "plot_bgcolor": "#1a1a1a",
                "font": {"color": "#e0e0e0", "size": 10},
                "legend": {
                    "font": {"color": "#e0e0e0", "size": 9},
                    "traceorder": "normal",
                    "itemclick": "toggleothers",
                    "itemdoubleclick": "toggle",
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.35,
                    "xanchor": "center",
                    "x": 0.5,
                },
                "title": {"text": "Against (1st pass) — second-pass recheck", "font": {"color": "#b0b0b0", "size": 13, "family": "Inter, Segoe UI, sans-serif"}, "x": 0.02, "xanchor": "left"},
                "annotations": [{"text": summary_text, "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.48, "showarrow": False, "xanchor": "center", "font": {"size": 9, "color": "#a0a0a0"}}] if summary_text else [],
            }
            recheck_fig = {"data": recheck_traces, "layout": recheck_layout}

            # Add recheck chart to buffer
            chart_html = '<div class="chart" id="chart_stance_recheck"></div>\n'
            chart_html += f'<script>var fig_stance_recheck = {json.dumps(recheck_fig)}; Plotly.newPlot("chart_stance_recheck", fig_stance_recheck.data, fig_stance_recheck.layout);</script>\n'
            bar_chart_buffer.append(chart_html)

            # If we have 3 charts buffered, emit them
            if len(bar_chart_buffer) == 3:
                html_parts.append('<div class="charts-row">\n')
                for buffered_html in bar_chart_buffer:
                    html_parts.append(buffered_html)
                html_parts.append('</div>\n')
                bar_chart_buffer = []

    # Flush any remaining buffered charts
    if bar_chart_buffer:
        html_parts.append('<div class="charts-row">\n')
        for buffered_html in bar_chart_buffer:
            html_parts.append(buffered_html)
        html_parts.append('</div>\n')
        bar_chart_buffer = []

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
            
            # Create single figure with 3 subplots using domain positioning
            all_traces = []
            annotations = []
            sector_idx = 0
            total_sectors = len([s for s in sectors if s in temporal_counts and temporal_counts[s]])

            # Build color map once
            label_to_color_map = {}
            unseen_idx = 0
            for label in labels_order:
                if label in colors:
                    label_to_color_map[label] = colors[label]
                else:
                    label_to_color_map[label] = _palette_extra[unseen_idx % len(_palette_extra)]
                    unseen_idx += 1

            for i, sector in enumerate(sectors):
                sector_counts = temporal_counts.get(sector, {})
                if not sector_counts:
                    continue

                sector_idx += 1
                is_last_sector = (sector_idx == total_sectors)

                # Calculate domain for this subplot with gaps
                gap = 0.02  # 2% gap between plots
                subplot_width = (1.0 - gap * 2) / 3  # Divide remaining space by 3
                x0 = i * (subplot_width + gap)
                x1 = x0 + subplot_width

                # Define axis names for this subplot
                xaxis_name = "xaxis" if i == 0 else f"xaxis{i+1}"
                yaxis_name = "yaxis" if i == 0 else f"yaxis{i+1}"

                sector_traces = []
                for label in labels_order:
                    y_vals = [sector_counts.get(year, {}).get(label, 0) for year in years_sorted]
                    if any(y_vals):
                        trace = {
                            "x": years_sorted,
                            "y": y_vals,
                            "name": label,
                            "type": "scatter",
                            "mode": "lines+markers",
                            "stackgroup": f"one_{i}",  # Unique stackgroup per subplot
                            "marker": {"color": label_to_color_map[label], "size": 6},
                            "line": {"color": label_to_color_map[label], "width": 2},
                            "showlegend": is_last_sector,
                            "xaxis": f"x{i+1 if i > 0 else ''}",
                            "yaxis": f"y{i+1 if i > 0 else ''}",
                        }
                        if sector_traces:
                            trace["fill"] = "tonexty"
                        else:
                            trace["fill"] = "tozeroy"
                        sector_traces.append(trace)
                        all_traces.append(trace)

                # Add sector title annotation
                annotations.append({
                    "text": SECTOR_DISPLAY.get(sector, sector),
                    "x": (i + 0.5) / 3,
                    "y": 1.02,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "xanchor": "center",
                    "yanchor": "bottom",
                    "font": {"color": "#b0b0b0", "size": 11},
                })

            if all_traces:
                # Create layout with three subplots
                temporal_layout = {
                    "height": 280,
                    "paper_bgcolor": "#1a1a1a",
                    "plot_bgcolor": "#1a1a1a",
                    "font": {"color": "#e0e0e0", "size": 10},
                    "showlegend": True,
                    "margin": {"t": 50, "b": 70, "l": 50, "r": 20},
                    "annotations": annotations,
                    "legend": {"font": {"color": "#e0e0e0", "size": 8}, "orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
                }

                # Add axis configuration for each subplot
                gap = 0.02  # 2% gap between plots
                subplot_width = (1.0 - gap * 2) / 3
                for i, sector in enumerate(sectors):
                    if sector not in temporal_counts or not temporal_counts[sector]:
                        continue

                    x0 = i * (subplot_width + gap)
                    x1 = x0 + subplot_width
                    xaxis_key = "xaxis" if i == 0 else f"xaxis{i+1}"
                    yaxis_key = "yaxis" if i == 0 else f"yaxis{i+1}"

                    temporal_layout[xaxis_key] = {
                        "domain": [x0, x1],
                        "anchor": f"y{i+1 if i > 0 else ''}",
                        "title": "Year",
                        "color": "#b0b0b0",
                        "gridcolor": "#3a3a3a",
                        "titlefont": {"size": 10, "color": "#b0b0b0"},
                        "tickfont": {"size": 9, "color": "#b0b0b0"},
                    }
                    temporal_layout[yaxis_key] = {
                        "domain": [0, 0.85],
                        "anchor": f"x{i+1 if i > 0 else ''}",
                        "title": "Count" if i == 0 else "",
                        "color": "#b0b0b0",
                        "gridcolor": "#3a3a3a",
                        "tickfont": {"size": 9, "color": "#b0b0b0"},
                    }

                temporal_fig = {"data": all_traces, "layout": temporal_layout}
                cid = _chart_id(qid, "temporal")
                html_parts.append(f'<div class="chart" id="chart_{cid}"></div>\n')
                html_parts.append(
                    f'<script>var fig_{cid} = {json.dumps(temporal_fig)}; Plotly.newPlot("chart_{cid}", fig_{cid}.data, fig_{cid}.layout);</script>\n'
                )

    # Survey questions section (at bottom) - radial bar charts using domain positioning like gate chart
    survey_metadata = survey_metadata or {}
    survey_qids = [qid for qid in counts.keys() if qid.startswith(("diet_", "ev_", "solar_"))]
    if survey_qids:
        html_parts.append('<h2 style="margin-top: 28px;">Survey Questions</h2>\n')
        # Group by sector
        survey_by_sector: Dict[str, List[str]] = defaultdict(list)
        for qid in sorted(survey_qids):
            sector = survey_metadata.get(qid, {}).get("sector", "unknown")
            survey_by_sector[sector].append(qid)

        # Collect data for all three sectors
        radial_traces = []
        annotations = []

        for i, sector in enumerate(["food", "transport", "housing"]):
            if sector not in survey_by_sector:
                continue

            sector_title = SECTOR_DISPLAY.get(sector, sector.upper())

            # Collect YES percentages for all questions in this sector
            question_labels = []
            yes_percentages = []

            for qid in survey_by_sector[sector]:
                if qid not in counts:
                    continue

                c = counts[qid]
                if sector not in c:
                    continue

                yes_count = c[sector].get("YES", 0) + c[sector].get("1", 0) + c[sector].get("yes", 0)
                no_count = c[sector].get("NO", 0) + c[sector].get("0", 0) + c[sector].get("no", 0)
                total = yes_count + no_count

                if total == 0:
                    continue

                yes_pct = (yes_count / total) * 100
                short_form = survey_metadata.get(qid, {}).get("short_form", qid)

                question_labels.append(short_form)
                yes_percentages.append(yes_pct)

            if not yes_percentages:
                continue

            # Find max percentage for this sector
            max_pct = max(yes_percentages)
            max_range = max(10, ((int(max_pct) // 5) + 1) * 5)

            # Calculate angles
            n_questions = len(question_labels)
            angles = [360 * i / n_questions for i in range(n_questions)]

            # Generate tick marks
            num_ticks = 5
            tick_step = max_range / num_ticks
            tickvals = [i * tick_step for i in range(num_ticks + 1)]
            ticktext = [f"{t:.0f}%" for t in tickvals]

            # Calculate domain positioning (x: horizontal, y: vertical)
            x0, x1 = i / 3, (i + 1) / 3

            trace = {
                "type": "barpolar",
                "r": yes_percentages,
                "theta": angles,
                "name": sector_title,
                "subplot": f"polar{i+1 if i > 0 else ''}",
                "marker": {
                    "color": yes_percentages,
                    "colorscale": [[0, "#ff9aa8"], [0.5, "#ffc8a0"], [1, "#8fcc8f"]],
                    "cmin": 0,
                    "cmax": max_range,
                    "line": {"color": "#000000", "width": 1},
                },
                "hovertemplate": "<b>%{theta}</b><br>%{r:.1f}%<extra></extra>",
            }
            radial_traces.append(trace)

            # Add sector title annotation
            annotations.append({
                "text": sector_title,
                "x": (i + 0.5) / 3,
                "y": 1.02,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "bottom",
                "font": {"color": "#b0b0b0", "size": 12, "family": "Inter, Segoe UI, sans-serif"},
            })

        # Create layout with three polar subplots
        if radial_traces:
            radial_layout = {
                "height": 500,
                "paper_bgcolor": "#1a1a1a",
                "plot_bgcolor": "#1a1a1a",
                "font": {"color": "#e0e0e0", "size": 9},
                "showlegend": False,
                "margin": {"t": 50, "b": 80, "l": 80, "r": 80},
                "annotations": annotations,
            }

            # Add polar axis configuration for each subplot
            for i, sector in enumerate(["food", "transport", "housing"]):
                if sector not in survey_by_sector:
                    continue

                # Recalculate data for axis config
                question_labels = []
                yes_percentages = []
                for qid in survey_by_sector[sector]:
                    if qid not in counts:
                        continue
                    c = counts[qid]
                    if sector not in c:
                        continue
                    yes_count = c[sector].get("YES", 0) + c[sector].get("1", 0) + c[sector].get("yes", 0)
                    no_count = c[sector].get("NO", 0) + c[sector].get("0", 0) + c[sector].get("no", 0)
                    total = yes_count + no_count
                    if total == 0:
                        continue
                    yes_pct = (yes_count / total) * 100
                    short_form = survey_metadata.get(qid, {}).get("short_form", qid)
                    question_labels.append(short_form)
                    yes_percentages.append(yes_pct)

                if not yes_percentages:
                    continue

                max_pct = max(yes_percentages)
                max_range = max(10, ((int(max_pct) // 5) + 1) * 5)
                n_questions = len(question_labels)
                angles = [360 * j / n_questions for j in range(n_questions)]

                num_ticks = 5
                tick_step = max_range / num_ticks
                tickvals = [j * tick_step for j in range(num_ticks + 1)]
                ticktext = [f"{t:.0f}%" for t in tickvals]

                x0, x1 = i / 3, (i + 1) / 3

                polar_key = f"polar{i+1 if i > 0 else ''}"
                radial_layout[polar_key] = {
                    "domain": {"x": [x0, x1], "y": [0, 0.92]},
                    "bgcolor": "#1a1a1a",
                    "radialaxis": {
                        "range": [0, max_range],
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                        "tickfont": {"size": 9, "color": "#b0b0b0"},
                        "gridcolor": "#3a3a3a",
                        "linecolor": "#505050",
                    },
                    "angularaxis": {
                        "tickmode": "array",
                        "tickvals": angles,
                        "ticktext": question_labels,
                        "tickfont": {"size": 9, "color": "#b0b0b0"},
                        "gridcolor": "#3a3a3a",
                        "linecolor": "#505050",
                        "rotation": 90,
                        "direction": "counterclockwise",
                    },
                }

            radial_fig = {"data": radial_traces, "layout": radial_layout}
            html_parts.append('<div class="chart" id="chart_survey_radials"></div>\n')
            html_parts.append(
                f'<script>var fig_survey_radials = {json.dumps(radial_fig)}; Plotly.newPlot("chart_survey_radials", fig_survey_radials.data, fig_survey_radials.layout);</script>\n'
            )

    # Add verification results section with confidence analysis
    verification_path = "paper4data/00_verification_results.json"
    verification_samples_path = "paper4data/00_verification_samples.json"
    if os.path.exists(verification_path):
        html_parts.extend(build_verification_section(
            verification_path,
            verification_samples_path if os.path.exists(verification_samples_path) else None
        ))

    html_parts.append("</body>\n</html>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote dashboard: {out_path}")


def build_confidence_plots_only(samples_path: str) -> List[str]:
    """Build just the confidence analysis plots for 4-column layout - two vertically stacked plots."""
    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            samples_data = json.load(f)
    except Exception as e:
        return [f'<div style="width: 24%;"><p style="color: #ff6b6b; text-align: center;">Error loading samples</p></div>\n']

    # Load question labels
    try:
        with open("00_vllm_survey_question_final.json", "r", encoding="utf-8") as f:
            survey_data = json.load(f)
        question_labels = {}
        for sector_data in survey_data.values():
            if not isinstance(sector_data, dict):
                continue
            for question_set in sector_data.values():
                if not isinstance(question_set, dict) or "questions" not in question_set:
                    continue
                for q in question_set["questions"]:
                    question_labels[q["id"]] = q.get("short_form", q["id"])
    except:
        question_labels = {}

    try:
        with open("00_vllm_ipcc_social_norms_schema.json", "r", encoding="utf-8") as f:
            norms_schema = json.load(f)
        for q in norms_schema.get("norms_questions", []):
            question_labels[q["id"]] = q["id"]
    except:
        pass

    # Analyze confidence vs mismatch
    confidence_data = []

    for task_type in ["norms", "survey"]:
        if task_type not in samples_data:
            continue
        for qid, samples in samples_data[task_type].items():
            for sample in samples:
                vllm_label = str(sample.get("vllm_label", "")).strip().lower()
                reasoning_label = str(sample.get("reasoning_label", "")).strip().lower()
                is_match = (vllm_label == reasoning_label)

                logprobs = sample.get("logprobs", {})
                if qid in logprobs:
                    logprob = logprobs[qid]
                    confidence = min(1.0, max(0.0, np.exp(logprob)))
                    confidence_data.append({
                        "qid": qid,
                        "confidence": confidence,
                        "is_match": is_match,
                        "sector": sample.get("sector", "unknown")
                    })

    if not confidence_data:
        return [f'<div style="width: 24%;"><p style="color: #ff6b6b; text-align: center;">No data</p></div>\n']

    html = []
    html.append('<div style="width: 24%; display: flex; flex-direction: column;">\n')
    html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Confidence vs Mismatch</h3>\n')

    # === TOP PLOT: Bar chart (aggregate by confidence bins) ===

    # Bin by confidence
    confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    bin_stats = {label: {"total": 0, "mismatches": 0} for label in bin_labels}

    for item in confidence_data:
        conf = item["confidence"]
        for i in range(len(confidence_bins) - 1):
            if confidence_bins[i] <= conf < confidence_bins[i+1]:
                bin_label = bin_labels[i]
                break
        else:
            bin_label = bin_labels[-1]

        bin_stats[bin_label]["total"] += 1
        if not item["is_match"]:
            bin_stats[bin_label]["mismatches"] += 1

    mismatch_rates = []
    bin_counts = []
    for label in bin_labels:
        total = bin_stats[label]["total"]
        if total > 0:
            rate = bin_stats[label]["mismatches"] / total
            mismatch_rates.append(rate * 100)
            bin_counts.append(total)
        else:
            mismatch_rates.append(0)
            bin_counts.append(0)

    # Bar chart
    confidence_trace = {
        "x": bin_labels,
        "y": mismatch_rates,
        "type": "bar",
        "marker": {
            "color": mismatch_rates,
            "colorscale": [[0, "#8fcc8f"], [0.5, "#ffb87a"], [1, "#ff9aa8"]],
            "cmin": 0,
            "cmax": max(mismatch_rates) if mismatch_rates else 50,
            "line": {"width": 0}
        },
        "text": [f"{r:.1f}%" if c > 0 else "" for r, c in zip(mismatch_rates, bin_counts)],
        "textposition": "outside",
        "textfont": {"size": 9, "color": "#e0e0e0"},
        "hovertemplate": "Confidence: %{x}<br>Mismatch: %{y:.1f}%<extra></extra>",
    }

    confidence_layout = {
        "height": 240,  # Reduced from 500 to ~0.5x
        "margin": {"l": 50, "t": 5, "b": 50, "r": 20},
        "xaxis": {
            "title": "Confidence",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "titlefont": {"size": 9, "color": "#b0b0b0"},
            "tickfont": {"size": 8, "color": "#b0b0b0"},
        },
        "yaxis": {
            "title": "Mismatch (%)",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 9, "color": "#b0b0b0"},
            "tickfont": {"size": 8, "color": "#b0b0b0"},
        },
        "paper_bgcolor": "#1a1a1a",
        "plot_bgcolor": "#1a1a1a",
        "font": {"color": "#e0e0e0"},
        "showlegend": False,
    }

    confidence_fig = {"data": [confidence_trace], "layout": confidence_layout}
    html.append('<div class="chart" id="chart_confidence_bar"></div>\n')
    html.append(f'<script>var fig_confidence_bar = {json.dumps(confidence_fig)}; Plotly.newPlot("chart_confidence_bar", fig_confidence_bar.data, fig_confidence_bar.layout);</script>\n')

    # === BOTTOM PLOT: Bubble chart (per-question breakdown) ===

    # Calculate per-question stats
    question_stats = defaultdict(lambda: {"total": 0, "mismatches": 0, "confidences": []})

    for item in confidence_data:
        qid = item["qid"]
        question_stats[qid]["total"] += 1
        question_stats[qid]["confidences"].append(item["confidence"])
        if not item["is_match"]:
            question_stats[qid]["mismatches"] += 1

    # Prepare data for bubble chart
    question_plot_data = []
    for qid, stats in question_stats.items():
        if stats["total"] > 0:
            mismatch_rate = stats["mismatches"] / stats["total"] * 100
            avg_conf = np.mean(stats["confidences"])
            label = question_labels.get(qid, qid)
            question_plot_data.append({
                "qid": qid,
                "label": label[:30],
                "mismatch_rate": mismatch_rate,
                "avg_confidence": avg_conf,
                "total": stats["total"]
            })

    q_labels = [q["label"] for q in question_plot_data]
    q_mismatch_rates = [q["mismatch_rate"] for q in question_plot_data]
    q_avg_confidences = [q["avg_confidence"] for q in question_plot_data]
    q_totals = [q["total"] for q in question_plot_data]

    question_trace = {
        "x": q_avg_confidences,
        "y": q_mismatch_rates,
        "mode": "markers",
        "type": "scatter",
        "marker": {
            "size": [min(12.5, max(3, t / 6)) for t in q_totals],  # 50% smaller
            "color": q_mismatch_rates,
            "colorscale": [[0, "#8fcc8f"], [0.5, "#ffb87a"], [1, "#ff9aa8"]],
            "cmin": 0,
            "cmax": max(q_mismatch_rates) if q_mismatch_rates else 50,
            "line": {"width": 0.5, "color": "#e0e0e0"},
            "showscale": False,
        },
        "text": q_labels,
        "hovertemplate": "<b>%{text}</b><br>Conf: %{x:.3f}<br>Mismatch: %{y:.1f}%<extra></extra>",
    }

    question_layout = {
        "height": 240,  # Match top plot height
        "margin": {"l": 50, "t": 10, "b": 50, "r": 20},
        "xaxis": {
            "title": "Avg Confidence",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 9, "color": "#b0b0b0"},
            "tickfont": {"size": 8, "color": "#b0b0b0"},
            "range": [0.5, 1.05],  # Start from 0.5 instead of 0
        },
        "yaxis": {
            "title": "Mismatch (%)",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 9, "color": "#b0b0b0"},
            "tickfont": {"size": 8, "color": "#b0b0b0"},
        },
        "paper_bgcolor": "#1a1a1a",
        "plot_bgcolor": "#1a1a1a",
        "font": {"color": "#e0e0e0"},
        "showlegend": False,
    }

    question_fig = {"data": [question_trace], "layout": question_layout}
    html.append('<div class="chart" id="chart_confidence_bubble" style="margin-top: 10px;"></div>\n')
    html.append(f'<script>var fig_confidence_bubble = {json.dumps(question_fig)}; Plotly.newPlot("chart_confidence_bubble", fig_confidence_bubble.data, fig_confidence_bubble.layout);</script>\n')

    html.append('</div>\n')  # Close width: 24% div (column 3)

    return html


def build_low_confidence_accuracy(samples_path: str, question_labels: Dict[str, str]) -> List[str]:
    """Build accuracy chart for high-confidence samples (confidence > 0.9)."""
    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            samples_data = json.load(f)
    except Exception as e:
        return [f'<div style="width: 24%;"><p style="color: #ff6b6b; text-align: center;">Error loading samples</p></div>\n']

    # Calculate accuracy per question for samples with confidence > 0.9
    high_conf_threshold = 0.9
    question_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for task_type in ["norms", "survey"]:
        if task_type not in samples_data:
            continue
        for qid, samples in samples_data[task_type].items():
            for sample in samples:
                vllm_label = str(sample.get("vllm_label", "")).strip().lower()
                reasoning_label = str(sample.get("reasoning_label", "")).strip().lower()

                logprobs = sample.get("logprobs", {})
                if qid in logprobs:
                    confidence = min(1.0, max(0.0, np.exp(logprobs[qid])))

                    # Only include samples with high confidence
                    if confidence > high_conf_threshold:
                        question_accuracy[qid]["total"] += 1
                        if vllm_label == reasoning_label:
                            question_accuracy[qid]["correct"] += 1

    # Calculate accuracy percentages
    accuracy_data = []
    for qid, stats in question_accuracy.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            label = question_labels.get(qid, qid)
            accuracy_data.append({
                "qid": qid,
                "label": label[:32],
                "accuracy": acc,
                "total": stats["total"]
            })

    # Sort by accuracy (lowest first)
    accuracy_data.sort(key=lambda x: x["accuracy"])

    if not accuracy_data:
        html = []
        html.append('<div style="width: 24%;">\n')
        html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Accuracy (Conf &gt; 0.9)</h3>\n')
        html.append('<p style="color: #a0a0a0; text-align: center; font-size: 0.85em;">No samples with confidence &gt; 0.9</p>\n')
        html.append('</div>\n')
        return html

    html = []
    html.append('<div style="width: 24%;">\n')
    html.append(f'<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Accuracy (Conf &gt; {high_conf_threshold})</h3>\n')

    # Bar chart
    q_labels = [d["label"] for d in accuracy_data]
    accuracies = [d["accuracy"] for d in accuracy_data]
    colors = []
    for acc in accuracies:
        if acc >= 0.7:
            colors.append("#8fcc8f")
        elif acc >= 0.5:
            colors.append("#ffb87a")
        else:
            colors.append("#ff9aa8")

    trace = {
        "y": q_labels,
        "x": accuracies,
        "type": "bar",
        "orientation": "h",
        "marker": {"color": colors, "line": {"width": 0}},
        "text": [f"{v:.0%}" for v in accuracies],
        "textposition": "outside",
        "textfont": {"size": 8, "color": "#e0e0e0"},
        "hovertemplate": "%{y}<br>Accuracy: %{x:.1%}<extra></extra>",
    }

    layout = {
        "height": 500,
        "margin": {"l": 180, "t": 5, "b": 35, "r": 40},
        "xaxis": {
            "title": "",
            "range": [0, 1.05],
            "tickformat": ".0%",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 9, "color": "#b0b0b0"},
            "tickfont": {"size": 8, "color": "#b0b0b0"},
        },
        "yaxis": {
            "color": "#b0b0b0",
            "tickfont": {"size": 8, "color": "#b0b0b0"},
        },
        "paper_bgcolor": "#1a1a1a",
        "plot_bgcolor": "#1a1a1a",
        "font": {"color": "#e0e0e0"},
        "showlegend": False,
    }

    fig = {"data": [trace], "layout": layout}
    html.append('<div class="chart" id="chart_low_conf_accuracy"></div>\n')
    html.append(f'<script>var fig_low_conf = {json.dumps(fig)}; Plotly.newPlot("chart_low_conf_accuracy", fig_low_conf.data, fig_low_conf.layout);</script>\n')
    html.append('</div>\n')  # Close width: 24% div (column 4)

    return html


def build_confidence_analysis_section(samples_path: str) -> List[str]:
    """Build confidence vs verifier mismatch analysis section."""
    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            samples_data = json.load(f)
    except Exception as e:
        return [f'<p style="color: #ff6b6b; text-align: center;">Error loading verification samples: {e}</p>\n']

    # Load survey questions for labels
    try:
        with open("00_vllm_survey_question_final.json", "r", encoding="utf-8") as f:
            survey_data = json.load(f)
        question_labels = {}
        for sector_data in survey_data.values():
            if not isinstance(sector_data, dict):
                continue
            for question_set in sector_data.values():
                if not isinstance(question_set, dict) or "questions" not in question_set:
                    continue
                for q in question_set["questions"]:
                    question_labels[q["id"]] = q.get("short_form", q["id"])
    except:
        question_labels = {}

    # Load norms schema for norms question labels
    try:
        with open("00_vllm_ipcc_social_norms_schema.json", "r", encoding="utf-8") as f:
            norms_schema = json.load(f)
        for q in norms_schema.get("norms_questions", []):
            question_labels[q["id"]] = q["id"]
    except:
        pass

    # Analyze confidence vs mismatch
    # Collect all (question_id, confidence, match) tuples
    confidence_data = []  # [(qid, confidence, is_match, sector)]

    for task_type in ["norms", "survey"]:
        if task_type not in samples_data:
            continue
        for qid, samples in samples_data[task_type].items():
            for sample in samples:
                vllm_label = str(sample.get("vllm_label", "")).strip().lower()
                reasoning_label = str(sample.get("reasoning_label", "")).strip().lower()
                is_match = (vllm_label == reasoning_label)

                # Get confidence for this specific question
                logprobs = sample.get("logprobs", {})
                if qid in logprobs:
                    # Convert logprob to confidence: exp(logprob), closer to 1 = higher confidence
                    # Logprobs are negative, so exp(-0.01) ≈ 0.99, exp(-2) ≈ 0.135
                    logprob = logprobs[qid]
                    confidence = min(1.0, max(0.0, np.exp(logprob)))

                    confidence_data.append({
                        "qid": qid,
                        "confidence": confidence,
                        "is_match": is_match,
                        "sector": sample.get("sector", "unknown")
                    })

    if not confidence_data:
        return ['<p style="color: #ff6b6b; text-align: center;">No confidence data available for analysis.</p>\n']

    html = []
    html.append('<div style="margin-top: 50px; padding-top: 30px; border-top: 3px solid #d0d0d0;">\n')
    html.append('<h2 style="text-align: center;">Confidence vs Verifier Mismatch Analysis</h2>\n')
    html.append('<p style="text-align: center; font-size: 0.85em; color: #a0a0a0; margin: 8px 0;">Does Mistral\'s self-reported confidence predict when the verifier disagrees?</p>\n')

    # === PANEL 1: Overall aggregate statistics ===

    # Bin by confidence (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
    confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    # Overall statistics by confidence bin
    bin_stats = {label: {"total": 0, "mismatches": 0} for label in bin_labels}

    for item in confidence_data:
        conf = item["confidence"]
        # Find bin
        for i in range(len(confidence_bins) - 1):
            if confidence_bins[i] <= conf < confidence_bins[i+1]:
                bin_label = bin_labels[i]
                break
        else:
            bin_label = bin_labels[-1]  # Last bin includes 1.0

        bin_stats[bin_label]["total"] += 1
        if not item["is_match"]:
            bin_stats[bin_label]["mismatches"] += 1

    # Calculate mismatch rates
    mismatch_rates = []
    bin_counts = []
    for label in bin_labels:
        total = bin_stats[label]["total"]
        if total > 0:
            rate = bin_stats[label]["mismatches"] / total
            mismatch_rates.append(rate * 100)
            bin_counts.append(total)
        else:
            mismatch_rates.append(0)
            bin_counts.append(0)

    # Overall stats card
    total_samples = len(confidence_data)
    total_mismatches = sum(not item["is_match"] for item in confidence_data)
    overall_mismatch_rate = (total_mismatches / total_samples * 100) if total_samples > 0 else 0

    avg_confidence = np.mean([item["confidence"] for item in confidence_data])
    avg_confidence_match = np.mean([item["confidence"] for item in confidence_data if item["is_match"]])
    avg_confidence_mismatch = np.mean([item["confidence"] for item in confidence_data if not item["is_match"]])

    html.append('<div style="display: flex; justify-content: center; gap: 15px; margin: 20px 0; flex-wrap: wrap;">\n')

    # Overall mismatch rate
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 140px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: {"#8fcc8f" if overall_mismatch_rate < 15 else "#ffb87a" if overall_mismatch_rate < 30 else "#ff9aa8"};">{overall_mismatch_rate:.1f}%</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Mismatch Rate</div></div>\n')

    # Avg confidence (match)
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 140px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: #8fcc8f;">{avg_confidence_match:.3f}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Avg Conf (Match)</div></div>\n')

    # Avg confidence (mismatch)
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 140px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: #ff9aa8;">{avg_confidence_mismatch:.3f}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Avg Conf (Mismatch)</div></div>\n')

    # Total samples
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 140px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: #e0e0e0;">{total_samples}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Total Comparisons</div></div>\n')

    html.append('</div>\n')

    # Panel 1: Overall mismatch rate by confidence bin
    html.append('<div style="margin-top: 25px;">\n')
    html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Mismatch Rate by Confidence Level (All Questions)</h3>\n')

    confidence_trace = {
        "x": bin_labels,
        "y": mismatch_rates,
        "type": "bar",
        "marker": {
            "color": mismatch_rates,
            "colorscale": [[0, "#8fcc8f"], [0.5, "#ffb87a"], [1, "#ff9aa8"]],
            "cmin": 0,
            "cmax": max(mismatch_rates) if mismatch_rates else 50,
            "line": {"width": 0}
        },
        "text": [f"{r:.1f}%<br>n={c}" for r, c in zip(mismatch_rates, bin_counts)],
        "textposition": "outside",
        "textfont": {"size": 10, "color": "#e0e0e0"},
        "hovertemplate": "Confidence: %{x}<br>Mismatch Rate: %{y:.1f}%<br>Samples: %{text}<extra></extra>",
    }

    confidence_layout = {
        "height": 300,
        "margin": {"l": 60, "t": 10, "b": 70, "r": 40},
        "xaxis": {
            "title": "Mistral Confidence Level",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "titlefont": {"size": 11, "color": "#b0b0b0"},
            "tickfont": {"size": 10, "color": "#b0b0b0"},
        },
        "yaxis": {
            "title": "Verifier Mismatch Rate (%)",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 11, "color": "#b0b0b0"},
            "tickfont": {"size": 10, "color": "#b0b0b0"},
        },
        "paper_bgcolor": "#1a1a1a",
        "plot_bgcolor": "#1a1a1a",
        "font": {"color": "#e0e0e0"},
        "showlegend": False,
    }

    confidence_fig = {"data": [confidence_trace], "layout": confidence_layout}
    html.append('<div class="chart" id="chart_confidence_overall"></div>\n')
    html.append(f'<script>var fig_confidence_overall = {json.dumps(confidence_fig)}; Plotly.newPlot("chart_confidence_overall", fig_confidence_overall.data, fig_confidence_overall.layout);</script>\n')
    html.append('</div>\n')

    # === PANEL 2: Per-question breakdown ===

    # Calculate mismatch rate and avg confidence per question
    question_stats = defaultdict(lambda: {"total": 0, "mismatches": 0, "confidences": []})

    for item in confidence_data:
        qid = item["qid"]
        question_stats[qid]["total"] += 1
        question_stats[qid]["confidences"].append(item["confidence"])
        if not item["is_match"]:
            question_stats[qid]["mismatches"] += 1

    # Prepare data for plotting
    question_plot_data = []
    for qid, stats in question_stats.items():
        if stats["total"] > 0:
            mismatch_rate = stats["mismatches"] / stats["total"] * 100
            avg_conf = np.mean(stats["confidences"])
            label = question_labels.get(qid, qid)
            question_plot_data.append({
                "qid": qid,
                "label": label[:35],  # Truncate for display
                "mismatch_rate": mismatch_rate,
                "avg_confidence": avg_conf,
                "total": stats["total"]
            })

    # Sort by mismatch rate (highest first)
    question_plot_data.sort(key=lambda x: x["mismatch_rate"], reverse=True)

    html.append('<div style="margin-top: 35px;">\n')
    html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Mismatch Rate & Confidence by Question</h3>\n')

    # Create scatter plot: x=avg_confidence, y=mismatch_rate, size=total
    q_labels = [q["label"] for q in question_plot_data]
    q_mismatch_rates = [q["mismatch_rate"] for q in question_plot_data]
    q_avg_confidences = [q["avg_confidence"] for q in question_plot_data]
    q_totals = [q["total"] for q in question_plot_data]

    question_trace = {
        "x": q_avg_confidences,
        "y": q_mismatch_rates,
        "mode": "markers",
        "type": "scatter",
        "marker": {
            "size": [min(30, max(8, t / 3)) for t in q_totals],  # Scale size by sample count
            "color": q_mismatch_rates,
            "colorscale": [[0, "#8fcc8f"], [0.5, "#ffb87a"], [1, "#ff9aa8"]],
            "cmin": 0,
            "cmax": max(q_mismatch_rates) if q_mismatch_rates else 50,
            "line": {"width": 1, "color": "#e0e0e0"},
            "showscale": True,
            "colorbar": {
                "title": "Mismatch<br>Rate (%)",
                "titlefont": {"size": 9, "color": "#b0b0b0"},
                "tickfont": {"size": 9, "color": "#b0b0b0"},
                "x": 1.02,
            }
        },
        "text": q_labels,
        "hovertemplate": "<b>%{text}</b><br>Avg Confidence: %{x:.3f}<br>Mismatch Rate: %{y:.1f}%<extra></extra>",
    }

    question_layout = {
        "height": 400,
        "margin": {"l": 70, "t": 10, "b": 80, "r": 100},
        "xaxis": {
            "title": "Average Mistral Confidence",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 11, "color": "#b0b0b0"},
            "tickfont": {"size": 10, "color": "#b0b0b0"},
            "range": [0, 1.05],
        },
        "yaxis": {
            "title": "Verifier Mismatch Rate (%)",
            "color": "#b0b0b0",
            "gridcolor": "#3a3a3a",
            "showgrid": True,
            "titlefont": {"size": 11, "color": "#b0b0b0"},
            "tickfont": {"size": 10, "color": "#b0b0b0"},
        },
        "paper_bgcolor": "#1a1a1a",
        "plot_bgcolor": "#1a1a1a",
        "font": {"color": "#e0e0e0"},
        "showlegend": False,
        "annotations": [
            {
                "text": "Bubble size = sample count",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.18,
                "showarrow": False,
                "font": {"size": 9, "color": "#7f8c8d"},
            }
        ]
    }

    question_fig = {"data": [question_trace], "layout": question_layout}
    html.append('<div class="chart" id="chart_confidence_by_question"></div>\n')
    html.append(f'<script>var fig_confidence_by_question = {json.dumps(question_fig)}; Plotly.newPlot("chart_confidence_by_question", fig_confidence_by_question.data, fig_confidence_by_question.layout);</script>\n')
    html.append('</div>\n')

    html.append('</div>\n')
    return html


def build_verification_section(verification_path: str, samples_path: Optional[str] = None) -> List[str]:
    """Build beautiful verification results visualization for dashboard bottom."""
    try:
        with open(verification_path, "r", encoding="utf-8") as f:
            verification = json.load(f)
    except Exception as e:
        return [f'<p style="color: #ff6b6b; text-align: center;">Error loading verification results: {e}</p>\n']

    # Load survey questions to get short_form labels (will be reused for column 4)
    question_labels = {}
    try:
        with open("00_vllm_survey_question_final.json", "r", encoding="utf-8") as f:
            survey_data = json.load(f)
        for sector_data in survey_data.values():
            if not isinstance(sector_data, dict):
                continue
            for question_set in sector_data.values():
                if not isinstance(question_set, dict) or "questions" not in question_set:
                    continue
                for q in question_set["questions"]:
                    question_labels[q["id"]] = q.get("short_form", q["id"])
    except:
        pass

    # Load norms schema for norms question labels
    try:
        with open("00_vllm_ipcc_social_norms_schema.json", "r", encoding="utf-8") as f:
            norms_schema = json.load(f)
        for q in norms_schema.get("norms_questions", []):
            question_labels[q["id"]] = q["id"]  # Use ID for norms questions
    except:
        pass

    html = []
    summary = verification.get("summary", {})

    # Handle both old (by_question) and new (by_task) formats
    by_task = verification.get("by_task", {})
    if by_task:
        # New format: combine norms and survey
        by_question = {**by_task.get("norms", {}), **by_task.get("survey", {})}
    else:
        # Old format
        by_question = verification.get("by_question", {})

    html.append('<div style="margin-top: 50px; padding-top: 30px; border-top: 3px solid #d0d0d0;">\n')
    html.append('<h2 style="text-align: center;">Label Verification: Non-Reasoning (Fast Labelling) Model vs Reasoning (Slow, More Accurate Judge)</h2>\n')
    html.append(f'<p style="text-align: center; font-size: 0.85em; color: #a0a0a0; margin: 8px 0;">Mistral-7B labels verified against {verification.get("config", {}).get("reasoning_model", "GPT-OSS-20B")} • {verification.get("config", {}).get("samples_per_question", 20)} samples per question</p>\n')

    # Summary metrics (compact cards)
    mean_acc = summary.get("mean_accuracy", 0)
    mean_kappa = summary.get("mean_kappa", 0)
    min_acc = summary.get("min_accuracy", 0)
    max_acc = summary.get("max_accuracy", 0)
    empty_pct = summary.get("empty_response_pct", 0)
    total_samples = summary.get("total_samples", 0)

    # Calculate confidence stats for the summary row
    avg_confidence_match = None
    avg_confidence_mismatch = None
    high_conf_accuracy = None
    high_conf_samples = None
    if samples_path and os.path.exists(samples_path):
        try:
            with open(samples_path, "r", encoding="utf-8") as f:
                samples_data = json.load(f)
            confidence_data = []
            high_conf_correct = 0
            high_conf_total = 0
            high_conf_threshold = 0.9

            for task_type in ["norms", "survey"]:
                if task_type not in samples_data:
                    continue
                for qid, samples in samples_data[task_type].items():
                    for sample in samples:
                        vllm_label = str(sample.get("vllm_label", "")).strip().lower()
                        reasoning_label = str(sample.get("reasoning_label", "")).strip().lower()
                        is_match = (vllm_label == reasoning_label)
                        logprobs = sample.get("logprobs", {})
                        if qid in logprobs:
                            confidence = min(1.0, max(0.0, np.exp(logprobs[qid])))
                            confidence_data.append({"confidence": confidence, "is_match": is_match})

                            # Track high-confidence accuracy
                            if confidence > high_conf_threshold:
                                high_conf_total += 1
                                if is_match:
                                    high_conf_correct += 1

            if confidence_data:
                avg_confidence_match = np.mean([item["confidence"] for item in confidence_data if item["is_match"]])
                avg_confidence_mismatch = np.mean([item["confidence"] for item in confidence_data if not item["is_match"]])

            if high_conf_total > 0:
                high_conf_accuracy = high_conf_correct / high_conf_total
                high_conf_samples = high_conf_total
        except:
            pass

    html.append('<div style="display: flex; justify-content: center; gap: 15px; margin: 20px 0; flex-wrap: wrap;">\n')

    # Accuracy card
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: {"#8fcc8f" if mean_acc >= 0.85 else "#ffb87a" if mean_acc >= 0.70 else "#ff9aa8"};">{mean_acc:.1%}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Accuracy</div></div>\n')

    # Cohen's kappa card
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: {"#8fcc8f" if mean_kappa >= 0.80 else "#ffb87a" if mean_kappa >= 0.60 else "#ff9aa8"};">{mean_kappa:.2f}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Cohen\'s κ</div></div>\n')

    # Empty responses card
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: {"#8fcc8f" if empty_pct < 5 else "#ffb87a" if empty_pct < 10 else "#ff9aa8"};">{empty_pct:.1f}%</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">No Response</div></div>\n')

    # Sample size card
    html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
    html.append(f'<div style="font-size: 2em; font-weight: 600; color: #e0e0e0;">{total_samples}</div>')
    html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Total Samples</div></div>\n')

    # Confidence (Match) card
    if avg_confidence_match is not None:
        html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
        html.append(f'<div style="font-size: 2em; font-weight: 600; color: #8fcc8f;">{avg_confidence_match:.3f}</div>')
        html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Conf (Match)</div></div>\n')

    # Confidence (Mismatch) card
    if avg_confidence_mismatch is not None:
        html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
        html.append(f'<div style="font-size: 2em; font-weight: 600; color: #ff9aa8;">{avg_confidence_mismatch:.3f}</div>')
        html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Conf (Mismatch)</div></div>\n')

    # High-confidence accuracy card
    if high_conf_accuracy is not None:
        html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
        html.append(f'<div style="font-size: 2em; font-weight: 600; color: {"#8fcc8f" if high_conf_accuracy >= 0.85 else "#ffb87a" if high_conf_accuracy >= 0.70 else "#ff9aa8"};">{high_conf_accuracy:.1%}</div>')
        html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">Acc (Conf &gt; 0.9)</div></div>\n')

    # High-confidence samples card
    if high_conf_samples is not None:
        html.append(f'<div style="background: #1a1a1a; padding: 15px 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); text-align: center; min-width: 120px;">')
        html.append(f'<div style="font-size: 2em; font-weight: 600; color: #e0e0e0;">{high_conf_samples}</div>')
        html.append(f'<div style="font-size: 0.8em; color: #a0a0a0;">High-Conf Samples</div></div>\n')

    html.append('</div>\n')

    # Charts side by side - four columns with confidence analysis
    if by_question:
        html.append('<div style="display: flex; gap: 1%; margin-top: 25px;">\n')

        # COLUMN 1: Accuracy by Question (24% width)
        html.append('<div style="width: 24%;">\n')
        html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Accuracy by Question</h3>\n')

        # Sort by accuracy (lowest first)
        sorted_questions = sorted(
            by_question.items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=False
        )

        # Build data with short labels
        q_labels = []
        accuracies = []
        colors = []

        for qid, metrics in sorted_questions:
            # Use short_form from mapping, fall back to question_short_form from metrics
            label = question_labels.get(qid, metrics.get("question_short_form", qid))
            acc = metrics.get("accuracy", 0)
            q_labels.append(label[:32])  # Shorter truncation for side-by-side
            accuracies.append(acc)
            # Color based on accuracy
            if acc >= 0.7:
                colors.append("#8fcc8f")
            elif acc >= 0.5:
                colors.append("#ffb87a")
            else:
                colors.append("#ff9aa8")

        # Create bar chart
        verification_trace = {
            "y": q_labels,
            "x": accuracies,
            "type": "bar",
            "orientation": "h",
            "marker": {"color": colors, "line": {"width": 0}},
            "text": [f"{v:.0%}" for v in accuracies],
            "textposition": "outside",
            "textfont": {"size": 9, "color": "#e0e0e0"},
            "hovertemplate": "%{y}<br>Accuracy: %{x:.1%}<extra></extra>",
        }

        verification_layout = {
            "height": 500,
            "margin": {"l": 220, "t": 5, "b": 35, "r": 50},
            "xaxis": {
                "title": "",
                "range": [0, 1.05],
                "tickformat": ".0%",
                "color": "#b0b0b0",
                "gridcolor": "#3a3a3a",
                "showgrid": True,
                "titlefont": {"size": 10, "color": "#b0b0b0"},
                "tickfont": {"size": 9, "color": "#b0b0b0"},
            },
            "yaxis": {
                "color": "#b0b0b0",
                "tickfont": {"size": 9, "color": "#b0b0b0"},
            },
            "paper_bgcolor": "#1a1a1a",
            "plot_bgcolor": "#1a1a1a",
            "font": {"color": "#e0e0e0"},
            "showlegend": False,
        }

        verification_fig = {"data": [verification_trace], "layout": verification_layout}
        html.append('<div class="chart" id="chart_verification"></div>\n')
        html.append(f'<script>var fig_verification = {json.dumps(verification_fig)}; Plotly.newPlot("chart_verification", fig_verification.data, fig_verification.layout);</script>\n')
        html.append('</div>\n')  # Close column 1

        # COLUMN 2: Top Estimation Errors (24% width)
        html.append('<div style="width: 24%;">\n')
        html.append('<h3 style="font-size: 1.05em; color: #b0b0b0; text-align: center; margin-bottom: 10px;">Top Estimation Errors</h3>\n')

        # Collect estimation errors with short labels
        estimation_data = []
        for qid, metrics in by_question.items():
            if "category_estimation" not in metrics:
                continue
            # Use short label from mapping
            label = question_labels.get(qid, metrics.get("question_short_form", qid))
            for category, est in metrics["category_estimation"].items():
                estimation_data.append({
                    "question": label,
                    "category": category,
                    "error": est["estimation_error"],
                    "type": est["estimation_type"]
                })

        if estimation_data:
            # Show ALL questions sorted by absolute error (worst first)
            estimation_data_sorted = sorted(estimation_data, key=lambda x: abs(x["error"]), reverse=True)

            est_labels = [f"{d['question'][:25]} - {d['category']}" for d in estimation_data_sorted]
            est_errors = [d["error"] for d in estimation_data_sorted]
            est_colors = ["#ff9aa8" if e > 0 else "#7cbcbe" for e in est_errors]

            estimation_trace = {
                "y": est_labels,
                "x": est_errors,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": est_colors, "line": {"width": 0}},
                "text": [f"{e:+.0f}pp" for e in est_errors],
                "textposition": "outside",
                "textfont": {"size": 9, "color": "#e0e0e0"},
                "showlegend": False,
                "hovertemplate": "%{y}<br>Error: %{x:+.1f}pp<extra></extra>",
            }

            estimation_layout = {
                "height": 500,
                "margin": {"l": 220, "t": 25, "b": 35, "r": 50},
                "xaxis": {
                    "title": "",
                    "color": "#b0b0b0",
                    "gridcolor": "#3a3a3a",
                    "showgrid": True,
                    "titlefont": {"size": 10, "color": "#b0b0b0"},
                    "tickfont": {"size": 9, "color": "#b0b0b0"},
                    "zeroline": True,
                    "zerolinecolor": "#b0b0b0",
                    "zerolinewidth": 1.5,
                },
                "yaxis": {
                    "color": "#b0b0b0",
                    "tickfont": {"size": 9, "color": "#b0b0b0"},
                },
                "paper_bgcolor": "#1a1a1a",
                "plot_bgcolor": "#1a1a1a",
                "font": {"color": "#e0e0e0"},
                "showlegend": False,
                "annotations": [
                    {
                        "x": -30,
                        "y": 1.02,
                        "xref": "x",
                        "yref": "paper",
                        "text": "← Underestimation",
                        "showarrow": False,
                        "font": {"size": 9, "color": "#5a9c9c"},
                        "bgcolor": "#1a3333",
                        "borderpad": 4,
                        "xanchor": "center",
                    },
                    {
                        "x": 30,
                        "y": 1.02,
                        "xref": "x",
                        "yref": "paper",
                        "text": "Overestimation →",
                        "showarrow": False,
                        "font": {"size": 9, "color": "#cc6666"},
                        "bgcolor": "#331a1a",
                        "borderpad": 4,
                        "xanchor": "center",
                    },
                ],
            }

            estimation_fig = {"data": [estimation_trace], "layout": estimation_layout}
            html.append('<div class="chart" id="chart_estimation_errors"></div>\n')
            html.append(f'<script>var fig_estimation = {json.dumps(estimation_fig)}; Plotly.newPlot("chart_estimation_errors", fig_estimation.data, fig_estimation.layout);</script>\n')

        html.append('</div>\n')  # Close column 2

        # COLUMN 3: Confidence Analysis (24% width)
        if samples_path:
            confidence_html = build_confidence_plots_only(samples_path)
            html.extend(confidence_html)

        # COLUMN 4: Low-Confidence Accuracy (24% width)
        if samples_path:
            low_conf_html = build_low_confidence_accuracy(samples_path, question_labels)
            html.extend(low_conf_html)

        html.append('</div>\n')  # Close flex container

    html.append('</div>\n')
    return html


def build_examples_html(data: Dict[str, List[Dict[str, Any]]], out_path: str, survey_metadata: Optional[Dict[str, Dict[str, str]]] = None) -> None:
    """One example comment per (question, category, sector).
    Adds green border if verifier agrees with fast model, red if disagrees.
    Shows confidence score in top corner."""

    # Load verification samples for border coloring and confidence scores
    verification_map = {}  # comment_text -> (vllm_label, reasoning_label, match)
    confidence_map = {}  # (comment_text, qid) -> confidence
    samples_path = "paper4data/00_verification_samples.json"
    if os.path.exists(samples_path):
        with open(samples_path, "r", encoding="utf-8") as f:
            samples_data = json.load(f)
        for task_type in ["norms", "survey"]:
            for qid, qsamples in samples_data.get(task_type, {}).items():
                for sample in qsamples:
                    comment_key = sample.get("comment", "")[:200].strip().lower()
                    vllm = str(sample.get("vllm_label", "")).lower()
                    reasoning = str(sample.get("reasoning_label", "")).lower()
                    verification_map[comment_key] = (vllm, reasoning, vllm == reasoning)

                    # Store confidence for this question
                    logprobs = sample.get("logprobs", {})
                    if qid in logprobs:
                        confidence = min(1.0, max(0.0, np.exp(logprobs[qid])))
                        confidence_map[(comment_key, qid)] = confidence

    sectors = ["food", "transport", "housing"]
    question_order = [
        "1.1.1_stance",
        "1.1_gate",
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
    # ONLY use verified examples (those in verification_map) so all have colored borders
    by_cat: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sector, items in data.items():
        for rec in items:
            comment = _collapse_newlines((rec.get("comment") or "").strip())
            if not comment:
                continue

            # Check if this comment was verified
            comment_key = comment[:200].strip().lower()
            if comment_key not in verification_map:
                continue  # Skip non-verified examples

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
    # Up to 3 random samples per (qid, label, sector) for display from verified examples
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
html { background: #000000; }
body { font-family: "Inter", "Segoe UI", system-ui, sans-serif; margin: 0; padding: 10px; background: #000000; color: #e0e0e0; min-height: 100vh; font-size: 12px; }
h1 { text-align: center; color: #e0e0e0; font-size: 1.2em; margin: 0.3em 0; font-weight: 600; }
.back-link { text-align: center; margin-bottom: 12px; font-size: 0.75em; }
.back-link a { color: #6c9bcf; text-decoration: none; }
.back-link a:hover { text-decoration: underline; }
.question-section { max-width: 1400px; margin: 0 auto 8px; background: #1a1a1a; border-radius: 8px; box-shadow: 0 1px 3px rgba(255,255,255,0.05); }
.question-section summary { font-size: 13px; font-weight: 600; color: #b0b0b0; padding: 10px 15px; cursor: pointer; list-style: none; border-bottom: 1px solid transparent; }
.question-section summary::-webkit-details-marker { display: none; }
.question-section summary::before { content: "▶ "; color: #6c9bcf; font-size: 9px; }
.question-section[open] summary::before { content: "▼ "; }
.question-section[open] summary { border-bottom-color: #3a3a3a; }
.question-content { padding: 14px; padding-top: 8px; }
.category-group { margin-bottom: 12px; }
.category-title { font-weight: 600; font-size: 10px; margin-bottom: 6px; padding: 5px 9px; border-radius: 12px; display: inline-block; color: #e0e0e0; background: #2a2a2a; }
.sectors-row { display: grid; grid-template-columns: repeat(9, 1fr); gap: 6px; }
.sector-column { display: flex; flex-direction: column; min-width: 0; }
.sector-header { font-weight: 600; font-size: 8px; color: #888888; margin-bottom: 3px; text-transform: uppercase; }
.example-comment { padding: 5px 7px; border-radius: 6px; font-size: 9px; line-height: 1.35; white-space: pre-wrap; word-break: break-word; border-left: 2px solid #4a6d7c; background: #252525; color: #e0e0e0; position: relative; }
.confidence-badge { position: absolute; top: 2px; right: 2px; font-size: 7px; color: #888888; background: #1a1a1a; padding: 1px 4px; border-radius: 3px; font-weight: 600; }
.prompt-dropdown { margin-bottom: 12px; }
.prompt-dropdown summary { cursor: pointer; color: #6c9bcf; font-size: 10px; user-select: none; }
.prompt-dropdown summary:hover { text-decoration: underline; }
.prompt-box { margin-top: 8px; padding: 9px; border-radius: 6px; background: #252525; border: 1px solid #3a3a3a; font-size: 9.5px; }
.prompt-box .prompt-text { color: #e0e0e0; line-height: 1.45; margin-bottom: 7px; }
.prompt-box .choices-label { color: #a0a0a0; font-weight: 600; margin-bottom: 3px; font-size: 9.5px; }
.prompt-box .choices-list { color: #b0b0b0; list-style: none; padding-left: 0; font-size: 9.5px; }
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
                    # Check verification status for border color and get confidence
                    border_style = ""
                    confidence_badge = ""
                    if text:
                        comment_key = text[:200].strip().lower()
                        if comment_key in verification_map:
                            vllm, reasoning, match = verification_map[comment_key]
                            if match:
                                border_style = ' style="border-left: 4px solid #8fcc8f;"'  # Green = agree
                            else:
                                border_style = ' style="border-left: 4px solid #ff9aa8;"'  # Red = disagree

                        # Get confidence for this question
                        conf_key = (comment_key, qid)
                        if conf_key in confidence_map:
                            conf_val = confidence_map[conf_key]
                            confidence_badge = f'<span class="confidence-badge">{conf_val:.2f}</span>'

                    html_parts.append(f'<div class="sector-column"><div class="sector-header">{header}</div>')
                    html_parts.append(f'<div class="example-comment"{border_style}>{confidence_badge}{html_escape(text) if text else "—"}</div></div>\n')
            html_parts.append("</div></div>\n")
        html_parts.append("</div></details>\n")

    # Against recheck (2nd pass) examples: comments labelled "against" (1st pass) and their recheck label
    if recheck_samples:
        html_parts.append('<details class="question-section"><summary>Against recheck (2nd pass)</summary>\n')
        html_parts.append('<div class="question-content">\n')
        html_parts.append('<p style="font-size: 10px; color: #7f8c8d; margin-bottom: 10px;">Comments that were labelled &quot;against&quot; in Author stance (1st pass) and re-labelled in a stringent second LLM pass.</p>\n')
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
                    # Check verification status for border color and get confidence
                    border_style = ""
                    confidence_badge = ""
                    if text:
                        comment_key = text[:200].strip().lower()
                        if comment_key in verification_map:
                            vllm, reasoning, match = verification_map[comment_key]
                            if match:
                                border_style = ' style="border-left: 4px solid #8fcc8f;"'  # Green = agree
                            else:
                                border_style = ' style="border-left: 4px solid #ff9aa8;"'  # Red = disagree

                        # Get confidence for this question
                        conf_key = (comment_key, qid)
                        if conf_key in confidence_map:
                            conf_val = confidence_map[conf_key]
                            confidence_badge = f'<span class="confidence-badge">{conf_val:.2f}</span>'

                    html_parts.append(f'<div class="sector-column"><div class="sector-header">{header}</div>')
                    html_parts.append(f'<div class="example-comment"{border_style}>{confidence_badge}{html_escape(text) if text else "—"}</div></div>\n')
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
