"""
temp.py - Comprehensive visualization experiments with all dashboard data
"""
import json

# Load data
with open("paper4data/00_verification_results.json", "r", encoding="utf-8") as f:
    vr = json.load(f)

with open("paper4data/00_verification_samples.json", "r", encoding="utf-8") as f:
    vs = json.load(f)

# ═══════════════════════════════════════════════════════════════════
# Build examples HTML from verification samples
# ═══════════════════════════════════════════════════════════════════
examples_html = ""
question_order = [
    ("norms", "1.1_gate", "Norm signal present (gate)"),
    ("norms", "1.1.1_stance", "Author stance"),
    ("norms", "1.2.1_descriptive", "Descriptive norm"),
    ("norms", "1.2.2_injunctive", "Injunctive norm"),
    ("norms", "1.3.1_reference_group", "Reference group"),
    ("norms", "1.3.1b_perceived_reference_stance", "Perceived reference stance"),
    ("norms", "1.3.3_second_order", "Second-order normative belief"),
]

# Add a few survey questions too
survey_qs = [
    ("survey", "diet 4", "Plant-based eating - social prompting"),
    ("survey", "diet 6", "Food choice - animal welfare"),
    ("survey", "ev 2", "EVs - purchase cost"),
    ("survey", "solar 1", "Home solar - cost savings"),
]
question_order += survey_qs

for task_type, qid, label in question_order:
    samples = vs.get(task_type, {}).get(qid, [])
    if not samples:
        continue
    # Get accuracy for this question
    task_results = vr.get("by_task", {}).get(task_type, {}).get(qid, {})
    acc = task_results.get("accuracy", 0)
    acc_color = "#8fcc8f" if acc >= 0.85 else "#ffb87a" if acc >= 0.70 else "#ff9aa8"

    # Separate matches and mismatches
    matches = [s for s in samples if str(s.get("vllm_label","")).strip().lower() == str(s.get("reasoning_label","")).strip().lower()]
    mismatches = [s for s in samples if str(s.get("vllm_label","")).strip().lower() != str(s.get("reasoning_label","")).strip().lower()]

    examples_html += f'''<details class="ex-section" {"open" if acc < 0.70 else ""}>
<summary><span style="color:{acc_color};font-weight:700;">{acc*100:.0f}%</span> {label} <span style="color:#4a6a8a;font-size:0.8em;">({qid})</span></summary>
<div class="ex-content">
<div style="margin-bottom:10px;color:#6a8caf;font-size:0.8em;">{len(matches)} matches, {len(mismatches)} mismatches out of {len(samples)} samples</div>
'''
    # Show up to 3 mismatches
    if mismatches:
        examples_html += '<div class="ex-group-title" style="color:#ff9aa8;">Disagreements</div>'
        for s in mismatches[:4]:
            comment = str(s.get("comment",""))[:250]
            if len(str(s.get("comment",""))) > 250: comment += "..."
            comment = comment.replace("<","&lt;").replace(">","&gt;").replace("&","&amp;")
            examples_html += f'''<div class="ex-card mismatch">
<div class="ex-comment">{comment}</div>
<div class="ex-labels"><span class="ex-tag">Fast: <b>{s.get("vllm_label","")}</b></span>
<span class="ex-tag reason">Reasoning: <b>{s.get("reasoning_label","")}</b></span>
<span class="ex-sector">{s.get("sector","")}</span></div></div>
'''

    # Show 2 matches
    if matches:
        examples_html += '<div class="ex-group-title" style="color:#8fcc8f;">Agreements</div>'
        for s in matches[:2]:
            comment = str(s.get("comment",""))[:250]
            if len(str(s.get("comment",""))) > 250: comment += "..."
            comment = comment.replace("<","&lt;").replace(">","&gt;").replace("&","&amp;")
            examples_html += f'''<div class="ex-card match">
<div class="ex-comment">{comment}</div>
<div class="ex-labels"><span class="ex-tag">Both: <b>{s.get("vllm_label","")}</b></span>
<span class="ex-sector">{s.get("sector","")}</span></div></div>
'''

    examples_html += '</div></details>\n'

# ═══════════════════════════════════════════════════════════════════
# Build verification data for JS
# ═══════════════════════════════════════════════════════════════════
# Accuracy by question sorted
acc_data = []
for task_type in ["norms", "survey"]:
    for qid, m in vr["by_task"][task_type].items():
        acc_data.append({"q": qid[:35], "acc": m["accuracy"], "type": task_type})
acc_data.sort(key=lambda x: x["acc"])
acc_questions = json.dumps([d["q"] for d in acc_data])
acc_values = json.dumps([d["acc"] for d in acc_data])
acc_colors = json.dumps(["#ff9aa8" if d["acc"]<0.70 else "#ffb87a" if d["acc"]<0.85 else "#8fcc8f" for d in acc_data])

# Estimation errors - top 30
est_data = []
for task_type in ["norms", "survey"]:
    for qid, m in vr["by_task"][task_type].items():
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
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.full{grid-column:1/-1}
.gauge-clip{height:220px;overflow:hidden;position:relative}
.bubble-chart{width:100%;height:300px}
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
.ex-section summary{cursor:pointer;padding:8px 12px;background:#12203a;border-radius:6px;font-size:0.9em;color:#b0c4d8;list-style:none}
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
</style>
</head>
<body>
<h1>SOCIAL NORMS IN CLIMATE DISCUSSIONS</h1>
<p class="subtitle">Reddit analysis across Food, Transport &amp; Housing &mdash; 9,000 comments &mdash; LLM-labeled with verification</p>

<div class="tabs">
<button class="tab-btn active" onclick="showTab('norms')">Norms</button>
<button class="tab-btn" onclick="showTab('author-stance')">Author Stance</button>
<span class="tab-sep"></span>
<button class="tab-btn" onclick="showTab('survey')">Factors &amp; Barriers</button>
<span class="tab-sep"></span>
<button class="tab-btn" onclick="showTab('verification')">Verification</button>
<button class="tab-btn" onclick="showTab('examples')">Examples</button>
</div>

<!-- TAB 1: NORMS (presence + dimensions + reference groups) -->
<div class="tab-content active" id="tab-norms">
<div class="sec">
<div class="sec-header">
<div class="sec-title">Does the comment reference a social norm?</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Gate question (1.1): Each comment is first classified as containing a social norm or not. A social norm is any reference to what people do (descriptive) or should do (injunctive) regarding climate actions. Food discussions contain norms most often (40%), while transport/housing are lower.</div>
<div class="gauge-clip"><div id="gauge-chart"></div></div>
<div style="text-align:center;margin-top:6px">
<span style="color:#5ab4ac;font-size:0.85em">&#9679; Food 40.2%</span> &nbsp;
<span style="color:#af8dc3;font-size:0.85em">&#9679; Transport 11.6%</span> &nbsp;
<span style="color:#f4a460;font-size:0.85em">&#9679; Housing 16.2%</span>
</div>
</div>
<div class="sec" style="margin-top:16px">
<div class="sec-header">
<div class="sec-title">Norm Classification Flow</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Each comment is classified across 4 norm dimensions. Select a sector to trace its 3,000 comments through each dimension. <b>Descriptive</b>: what people do. <b>Injunctive</b>: what people should do. <b>Second-order</b>: beliefs about others' beliefs. <b>Perceived stance</b>: what reference groups think.</div>
<div class="sankey-toggles">
<button class="sankey-toggle active" data-sec="Food" onclick="window._sankeySetSector('Food')">Food</button>
<button class="sankey-toggle" data-sec="Transport" onclick="window._sankeySetSector('Transport')">Transport</button>
<button class="sankey-toggle" data-sec="Housing" onclick="window._sankeySetSector('Housing')">Housing</button>
<button class="sankey-toggle" data-sec="all" onclick="window._sankeySetSector('all')">All Sectors</button>
</div>
<div id="sankey-norms"></div>
</div>
<div class="sec" style="margin-top:16px">
<div class="sec-header">
<div class="sec-title">Reference Groups Mentioned</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">Which social groups does the speaker reference when discussing climate actions? Shows the count of mentions per reference group category across sectors. Size = frequency of mention.</div>
<div id="bubble-pack" class="bubble-chart"></div>
<div class="legend" id="bubble-legend"></div>
</div>
</div>

<!-- TAB 3: AUTHOR STANCE -->
<div class="tab-content" id="tab-author-stance">
<div class="sec">
<div class="sec-header">
<div class="sec-title">Author Stance Toward Climate Action</div>
<button class="info-btn" onclick="toggleInfo(this)">?</button>
</div>
<div class="info-popup">How does the comment author position themselves toward the climate action (veganism, EVs, solar)? Pro = supportive, Against = opposed, Against particular but pro = criticizes specifics but supports overall, Neither/Mixed = neutral or ambivalent, Pro but lack of options = wants to but faces barriers.</div>
<div id="treemap-stance-main"></div>
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

<!-- TAB 5: VERIFICATION -->
<div class="tab-content" id="tab-verification">
<div class="sec" style="padding:10px 16px">
<div class="sec-header"><div class="sec-title" style="font-size:0.85em">Model Verification: Fast (Mistral-7B) vs Reasoning (GPT-oss-20B)</div><button class="info-btn" onclick="toggleInfo(this)">?</button></div>
<div class="info-popup">50 samples per question re-labeled with a slower reasoning model. Accuracy = agreement rate. Cohen's kappa accounts for chance agreement. Confidence = model's logprob-derived certainty.</div>
</div>
<div class="stat-grid">
STAT_BOXES
</div>
<div class="grid2">
<div class="sec"><div class="sec-title" style="font-size:0.85em;margin-bottom:8px">Accuracy by Question</div><div id="chart-acc"></div></div>
<div class="sec"><div class="sec-title" style="font-size:0.85em;margin-bottom:8px">Top Estimation Errors</div><div id="chart-est"></div></div>
<div class="sec"><div class="sec-title" style="font-size:0.85em;margin-bottom:8px">Confidence vs Mismatch</div><div id="chart-conf-box"></div><div id="chart-conf-scatter"></div></div>
<div class="sec"><div class="sec-title" style="font-size:0.85em;margin-bottom:8px">Accuracy (Conf &gt; 0.9)</div><div id="chart-hc-acc"></div></div>
</div>
</div>

<!-- TAB 6: EXAMPLES -->
<div class="tab-content" id="tab-examples">
<div style="margin-bottom:12px;color:#6a8caf;font-size:0.82em">
Verification examples from 50 samples per question. <span style="color:#8fcc8f">Green</span> = models agree. <span style="color:#ff9aa8">Red</span> = models disagree.
Lowest-accuracy questions are expanded by default.
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
    if(id==='survey') setTimeout(function(){if(window._animateSurvey)window._animateSurvey();},200);
    if(id==='verification') setTimeout(function(){if(window._animateVerification)window._animateVerification();},200);
}
function toggleInfo(btn){var p=btn.parentElement.nextElementSibling;if(p&&p.classList.contains('info-popup'))p.classList.toggle('show');}
var PC={displayModeBar:false,responsive:true};
var DB='rgba(0,0,0,0)';
var DL={paper_bgcolor:DB,plot_bgcolor:DB,font:{color:'#b0c4d8',family:'Inter,sans-serif',size:10}};

// ═══════════════ GAUGE (animated from 0, replayable) ═══════════════
(function(){
    var f=40.2,t=11.6,h=16.2,avg=(f+t+h)/3,vis=f+t+h;
    Plotly.newPlot('gauge-chart',[{
        values:[0.01,0.01,0.01,0.03],labels:['Food','Transport','Housing',''],
        marker:{colors:['#5ab4ac','#af8dc3','#f4a460','rgba(0,0,0,0)'],line:{color:'#12203a',width:4}},
        hole:0.65,direction:'clockwise',sort:false,rotation:270,
        textposition:'none',type:'pie',showlegend:false,
        hovertemplate:'<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    }],Object.assign({},DL,{height:440,margin:{t:5,b:5,l:5,r:5},
        annotations:[
            {text:'<b>0.0%</b>',x:0.5,y:0.55,font:{size:46,color:'#fff'},showarrow:false},
            {text:'Avg Norm Signal',x:0.5,y:0.42,font:{size:12,color:'#6a8caf'},showarrow:false},
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

// ═══════════════ TREEMAP - AUTHOR STANCE ═══════════════
(function(){
    // Stance colors: green(pro) -> muted green -> neutral -> orange -> red(against)
    var sc={pro:'#8fcc8f',plo:'#7bb8a0',mix:'#b8b8a0',abp:'#e0a878',agt:'#ff9aa8'};
    // Sector parent backgrounds (dark muted)
    var sp={Food:'#1a3a4a',Transport:'#2a1a3a',Housing:'#3a2a1a'};
    // Data: author stance per sector (from dashboard, 3000/sector)
    var d={Food:{pro:735,against:252,abp:83,'neither/mixed':1404,plo:526},
           Transport:{pro:372,against:274,abp:66,'neither/mixed':1754,plo:534},
           Housing:{pro:531,against:243,abp:69,'neither/mixed':1555,plo:602}};
    var labels=['All','Food','Transport','Housing'];
    var parents=['','All','All','All'];
    var values=[0,0,0,0];
    var colors=['#0f1d33',sp.Food,sp.Transport,sp.Housing];
    var texts=['','','',''];
    var tcolors=['#fff','#fff','#fff','#fff']; // text colors per node
    var stances=[['pro','Pro',sc.pro],['plo','Pro but lack of options',sc.plo],['neither/mixed','Neither/Mixed',sc.mix],['abp','Against particular but pro',sc.abp],['against','Against',sc.agt]];
    ['Food','Transport','Housing'].forEach(function(sec){
        var total=0;for(var k in d[sec])total+=d[sec][k];
        stances.forEach(function(st){
            var v=d[sec][st[0]];
            var pct=Math.round(v/total*100);
            labels.push(sec+': '+st[1]);
            parents.push(sec);
            values.push(v);
            colors.push(st[2]);
            texts.push('<b>'+v+'</b><br>'+pct+'%');
            tcolors.push('#0a1628'); // dark text on light backgrounds
        });
    });
    Plotly.newPlot('treemap-stance-main',[{
        type:'treemap',branchvalues:'remainder',
        labels:labels,parents:parents,values:values,
        text:texts,textinfo:'label+text',
        marker:{colors:colors,line:{color:'#0a1628',width:3}},
        textfont:{color:tcolors,size:12},pathbar:{visible:false},
    }],Object.assign({},DL,{height:420,margin:{t:5,b:5,l:5,r:5}}),PC);
})();

// ═══════════════ BUBBLES ═══════════════
var bubbleData={
    categories:['family','partner/spouse','friends','coworkers','neighbors','local community','political tribe','online community','other reddit user'],
    colors:['#a8b8c2','#c49fc4','#8fbfd9','#c692c6','#7cadc6','#7cc2b8','#a8b1b8','#b8b8b8','#ffa8c2'],
    Food:[140,60,290,112,13,108,1,24,89],Transport:[36,6,26,335,24,65,5,8,97],Housing:[28,3,26,334,39,104,3,16,98]
};
function initBubbles(){
    var el=document.getElementById('bubble-pack');if(!el)return;el.innerHTML='';
    var w=el.clientWidth||900,h=320;
    var svg=d3.select('#bubble-pack').append('svg').attr('viewBox','0 0 '+w+' '+h);
    var sectors=['Food','Transport','Housing'],cw=w/3,nodes=[];
    sectors.forEach(function(sec,si){
        var cx=cw*si+cw/2;
        svg.append('text').attr('x',cx).attr('y',18).attr('class','bubble-sector-label').attr('text-anchor','middle').text(sec.toUpperCase());
        bubbleData.categories.forEach(function(cat,ci){
            var val=bubbleData[sec][ci];if(val===0)return;
            nodes.push({category:cat,value:val,r:Math.sqrt(val)*2.2,color:bubbleData.colors[ci],cx:cx,cy:h/2+5,x:cx+(Math.random()-0.5)*60,y:h/2+5+(Math.random()-0.5)*60});
        });
    });
    svg.selectAll('.sep').data([1,2]).enter().append('line')
        .attr('x1',function(d){return cw*d}).attr('x2',function(d){return cw*d})
        .attr('y1',28).attr('y2',h-5).attr('stroke','#1a3050').attr('stroke-width',1).attr('stroke-dasharray','4,4');
    var g=svg.selectAll('.b').data(nodes).enter().append('g');
    g.append('circle').attr('cx',function(d){return d.x}).attr('cy',function(d){return d.y}).attr('r',0)
        .attr('fill',function(d){return d.color}).attr('fill-opacity',0.85).attr('stroke',function(d){return d.color}).attr('stroke-opacity',0.3).attr('stroke-width',2)
        .transition().duration(800).delay(function(d,i){return i*30}).attr('r',function(d){return d.r});
    g.append('text').attr('x',function(d){return d.x}).attr('y',function(d){return d.y-2}).attr('class','bubble-label')
        .attr('font-size',function(d){return Math.max(8,Math.min(d.r*0.6,13))+'px'}).attr('opacity',0)
        .text(function(d){return d.value>=8?d.category:''}).transition().duration(600).delay(function(d,i){return 800+i*30}).attr('opacity',1);
    g.append('text').attr('x',function(d){return d.x}).attr('y',function(d){return d.y+(d.r>20?12:8)}).attr('class','bubble-label')
        .attr('font-size',function(d){return Math.max(9,Math.min(d.r*0.55,14))+'px'}).attr('font-weight','700').attr('opacity',0)
        .text(function(d){return d.value}).transition().duration(600).delay(function(d,i){return 800+i*30}).attr('opacity',1);
    d3.forceSimulation(nodes).force('x',d3.forceX(function(d){return d.cx}).strength(0.1))
        .force('y',d3.forceY(h/2+10).strength(0.08)).force('collide',d3.forceCollide(function(d){return d.r+2}).strength(0.95))
        .force('charge',d3.forceManyBody().strength(-1)).alpha(0.8)
        .on('tick',function(){
            g.select('circle').attr('cx',function(d){return d.x}).attr('cy',function(d){return d.y});
            g.selectAll('text').attr('x',function(d){return d.x});
            g.select('text:nth-child(2)').attr('y',function(d){return d.y-2});
            g.select('text:nth-child(3)').attr('y',function(d){return d.y+(d.r>20?12:8)});
        });
    var le=document.getElementById('bubble-legend');le.innerHTML='';
    bubbleData.categories.forEach(function(c,i){le.innerHTML+='<span class="legend-item"><span class="legend-dot" style="background:'+bubbleData.colors[i]+'"></span>'+c+'</span>';});
}

// ═══════════════ NORMS SANKEY (sectors → dimensions → categories, per-sector links) ═══════════════
(function(){
    // Per-sector data: [present/strong/pro, absent/weak/against, unclear/none/mixed]
    var SD={
        Food:     {desc:[874,743,1383],  inj:[288,2526,186],  sec:[377,628,1995], perc:[467,152,2381]},
        Transport:{desc:[419,2176,405],  inj:[150,2805,45],   sec:[319,480,2201], perc:[888,350,1762]},
        Housing:  {desc:[395,1957,648],  inj:[206,2735,59],   sec:[362,522,2116], perc:[803,368,1829]}
    };
    var secs=['Food','Transport','Housing'],dims=['desc','inj','sec','perc'];
    // 19 nodes: 0-2 sectors, 3-6 dimensions, 7-18 categories
    var baseLabels=['Food<br>3,000','Transport<br>3,000','Housing<br>3,000',
        'Descriptive<br>Norm','Injunctive<br>Norm','Second-Order<br>Belief','Perceived<br>Stance'];
    var catNames=[['Present','Absent','Unclear'],['Present','Absent','Unclear'],
        ['Strong','Weak','None'],['Pro','Against','Mixed']];
    // Aggregate category totals
    var catTotals=[];
    dims.forEach(function(d,di){
        [0,1,2].forEach(function(ci){
            var t=0;secs.forEach(function(s){t+=SD[s][d][ci]});
            catTotals.push({name:catNames[di][ci],total:t});
        });
    });
    var catLabelsAll=catTotals.map(function(c){return c.name+'<br>'+c.total.toLocaleString()});
    // Per-sector category labels
    var catLabelsSec={};
    secs.forEach(function(s){
        catLabelsSec[s]=[];
        dims.forEach(function(d,di){
            [0,1,2].forEach(function(ci){
                catLabelsSec[s].push(catNames[di][ci]+'<br>'+SD[s][d][ci].toLocaleString());
            });
        });
    });
    var nl=baseLabels.concat(catLabelsAll);
    var nc=['#5ab4ac','#af8dc3','#f4a460',
        '#7caed6','#8fbfd9','#b8a8c8','#8fcc8f',
        '#7caed6','#566a7a','#8a9aa8',
        '#8fbfd9','#566a7a','#8a9aa8',
        '#ffb0a0','#8fbfd9','#8a9aa8',
        '#8fcc8f','#ff9aa8','#ffe87a'];
    // Build links: 48 total, organized by sector (0-15 Food, 16-31 Transport, 32-47 Housing)
    var src=[],tgt=[],val=[];
    secs.forEach(function(s,si){
        // 4 sector→dimension links
        dims.forEach(function(d,di){src.push(si);tgt.push(3+di);val.push(3000);});
        // 12 dimension→category links (per sector)
        dims.forEach(function(d,di){
            SD[s][d].forEach(function(v,ci){src.push(3+di);tgt.push(7+di*3+ci);val.push(v);});
        });
    });
    // Color helpers
    var SC={Food:'90,180,172',Transport:'175,141,195',Housing:'244,164,96'};
    function getLinkColors(active){
        var c=[];
        secs.forEach(function(s){
            var bright='rgba('+SC[s]+',0.45)',dim='rgba('+SC[s]+',0.06)';
            var use=(active==='all'||active===s)?bright:dim;
            for(var i=0;i<16;i++)c.push(use);
        });
        return c;
    }
    // Default: Food highlighted
    Plotly.newPlot('sankey-norms',[{type:'sankey',orientation:'h',
        node:{label:nl,color:nc,pad:20,thickness:30,line:{color:'#0a1628',width:1}},
        link:{source:src,target:tgt,value:val,color:getLinkColors('Food')}
    }],Object.assign({},DL,{height:480,margin:{t:10,b:10,l:10,r:10}}),PC);
    // Toggle sector
    window._sankeySetSector=function(sec){
        var newColors=getLinkColors(sec);
        var newLabels;
        if(sec==='all'){
            newLabels=baseLabels.concat(catLabelsAll);
        }else{
            // Update dimension labels to show per-sector total
            var dimLabels=['Descriptive<br>Norm','Injunctive<br>Norm','Second-Order<br>Belief','Perceived<br>Stance'];
            var secLabel=[sec+'<br>3,000'];
            var otherSecs=secs.filter(function(s){return s!==sec});
            var sLabels=secs.map(function(s){return s=== sec?(s+'<br>3,000'):(s+'<br><span style="opacity:0.3">3,000</span>');});
            newLabels=sLabels.concat(dimLabels).concat(catLabelsSec[sec]);
        }
        Plotly.restyle('sankey-norms',{'link.color':[newColors],'node.label':[newLabels]},0);
        document.querySelectorAll('.sankey-toggle').forEach(function(b){
            b.classList.toggle('active',b.dataset.sec===sec);
        });
    };
})();
window._animateNorms=function(){
    // Re-trigger default sector selection
    if(window._sankeySetSector) window._sankeySetSector('Food');
};

// ═══════════════ SURVEY RADIALS (D3 sequential spoke animation) ═══════════════
// D3.arc: 0 = 12 o'clock, angles go clockwise
// To place labels with Math.cos/sin, convert: trigAngle = d3Angle - PI/2
function drawRadial(svgId, labels, values, baseColor, maxR, delay){
    var svg=d3.select('#'+svgId);
    var cx=230,cy=210,outerR=130;
    var n=labels.length, sliceAngle=2*Math.PI/n;
    var gapFrac=0.15; // fraction of slice that is gap
    var spokeAngle=sliceAngle*(1-gapFrac);
    var bc=baseColor;
    var arc=d3.arc();
    // Grid circles
    [0.25,0.5,0.75,1.0].forEach(function(s){
        svg.append('circle').attr('cx',cx).attr('cy',cy).attr('r',outerR*s)
            .attr('fill','none').attr('stroke','#1a2a40').attr('stroke-width',0.5);
    });
    // Grid tick labels (at top)
    [0.25,0.5,0.75,1.0].forEach(function(s){
        svg.append('text').attr('x',cx+3).attr('y',cy-outerR*s-1)
            .attr('fill','#3a5a7a').attr('font-size','8px').attr('font-family','Inter,sans-serif')
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
        // Tooltip on hover
        g.append('title').text(labels[i]+': '+v.toFixed(1)+'%');
        // Hover highlight
        g.on('mouseover',function(){d3.select(this).select('path').attr('stroke','#fff').attr('stroke-width',2).attr('stroke-opacity',1);})
         .on('mouseout',function(){d3.select(this).select('path').attr('stroke',bc).attr('stroke-width',1).attr('stroke-opacity',0.6);});
        spokes.push({g:g,path:path,label:labels[i],value:v,startD3:startD3,endD3:endD3,r:r,trigMid:trigMid,alpha:alpha});
    });
    // Labels at spoke centers
    spokes.forEach(function(s,i){
        var lx=cx+Math.cos(s.trigMid)*(outerR+16);
        var ly=cy+Math.sin(s.trigMid)*(outerR+16);
        var anchor=Math.abs(Math.cos(s.trigMid))<0.15?'middle':(Math.cos(s.trigMid)>0?'start':'end');
        svg.append('text').attr('x',lx).attr('y',ly+4)
            .attr('text-anchor',anchor).attr('fill','#6a8caf')
            .attr('font-size','9px').attr('font-family','Inter,sans-serif')
            .attr('opacity',0).attr('class','rl-'+svgId+'-'+i)
            .text(s.label);
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
            // Value label - pushed toward outer edge of spoke, skip tiny ones
            var labelR=Math.max(s.r*0.72, 18);
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
drawRadial('radial-food',
    ['health','convenience','cost','social prompting','barriers','identity','animal welfare','climate','alt. meat','reduce red meat','dairy alt.','lab-grown','taste'],
    [5.2,9.6,2.8,12.2,5.2,4.4,20.0,6.0,6.2,3.8,4.0,0.0,9.6],'#5ab4ac',25,0);
drawRadial('radial-transport',
    ['environment','purchase cost','operating cost','infrastructure','driving exp.','reliability'],
    [3.0,7.8,4.8,4.4,6.8,7.4],'#af8dc3',10,0);
drawRadial('radial-housing',
    ['cost savings','confidence','env. benefit','obsolescence','payback','climate','installer trust','support info','maintenance','performance'],
    [11.6,8.8,14.0,4.6,7.0,1.8,5.0,1.0,0.6,6.6],'#f4a460',15,0);
window._animateSurvey=function(){
    // Reset and replay - clear dynamic elements, redraw
    ['radial-food','radial-transport','radial-housing'].forEach(function(id){
        var svg=document.getElementById(id);
        svg.innerHTML='';
    });
    drawRadial('radial-food',
        ['health','convenience','cost','social prompting','barriers','identity','animal welfare','climate','alt. meat','reduce red meat','dairy alt.','lab-grown','taste'],
        [5.2,9.6,2.8,12.2,5.2,4.4,20.0,6.0,6.2,3.8,4.0,0.0,9.6],'#5ab4ac',25,0);
    drawRadial('radial-transport',
        ['environment','purchase cost','operating cost','infrastructure','driving exp.','reliability'],
        [3.0,7.8,4.8,4.4,6.8,7.4],'#af8dc3',10,0);
    drawRadial('radial-housing',
        ['cost savings','confidence','env. benefit','obsolescence','payback','climate','installer trust','support info','maintenance','performance'],
        [11.6,8.8,14.0,4.6,7.0,1.8,5.0,1.0,0.6,6.6],'#f4a460',15,0);
    window['_animRadial_radial-food']();
    window['_animRadial_radial-transport']();
    window['_animRadial_radial-housing']();
};

// ═══════════════ VERIFICATION CHARTS (animated) ═══════════════
// Accuracy by question - start at 0
Plotly.newPlot('chart-acc',[{
    y:ACC_QUESTIONS, x:ACC_VALUES.map(function(){return 0}), type:'bar', orientation:'h',
    marker:{color:ACC_COLORS,line:{width:0}},
    text:ACC_VALUES.map(function(){return ''}),
    textposition:'outside',textfont:{size:8,color:'#b0c4d8'},
    hovertemplate:'%{y}<br>Accuracy: %{x:.1%}<extra></extra>'
}],Object.assign({},DL,{
    height:500,margin:{l:200,t:5,b:30,r:40},
    xaxis:{range:[0,1.08],tickformat:'.0%',gridcolor:'#1a2a40',tickfont:{size:8,color:'#4a6a8a'}},
    yaxis:{tickfont:{size:8,color:'#b0c4d8'}},showlegend:false
}),PC);

// Estimation errors - start at 0
Plotly.newPlot('chart-est',[{
    y:EST_LABELS, x:EST_VALUES.map(function(){return 0}), type:'bar', orientation:'h',
    marker:{color:EST_COLORS,line:{width:0}},
    text:EST_VALUES.map(function(){return ''}),
    textposition:'outside',textfont:{size:7,color:'#b0c4d8'},showlegend:false,
    hovertemplate:'%{y}<br>Error: %{x:+.1f}pp<extra></extra>'
}],Object.assign({},DL,{
    height:500,margin:{l:180,t:20,b:30,r:40},
    xaxis:{gridcolor:'#1a2a40',zeroline:true,zerolinecolor:'#4a6a8a',zerolinewidth:1.5,tickfont:{size:8,color:'#4a6a8a'}},
    yaxis:{tickfont:{size:7,color:'#b0c4d8'}},showlegend:false,
    annotations:[
        {x:-25,y:1.02,xref:'x',yref:'paper',text:'\u2190 Underestimation',showarrow:false,font:{size:9,color:'#5ab4ac'}},
        {x:25,y:1.02,xref:'x',yref:'paper',text:'Overestimation \u2192',showarrow:false,font:{size:9,color:'#c06070'}}
    ]
}),PC);
// Confidence vs Mismatch - box plot
Plotly.newPlot('chart-conf-box',[{
    x:CONF_BIN_LABELS, y:CONF_BIN_MISMATCH.map(function(){return 0}), type:'bar',
    marker:{color:['#8fcc8f','#ffb87a','#ff9aa8','#ffb87a','#8fcc8f'],line:{width:0}},
    text:CONF_BIN_MISMATCH.map(function(){return ''}),textposition:'outside',textfont:{size:9,color:'#b0c4d8'},
    hovertemplate:'Confidence: %{x}<br>Mismatch: %{y:.1f}%<extra></extra>'
}],Object.assign({},DL,{
    height:220,margin:{l:50,t:5,b:35,r:15},
    xaxis:{title:{text:'Confidence',font:{size:9,color:'#6a8caf'}},tickfont:{size:8,color:'#6a8caf'}},
    yaxis:{title:{text:'Mismatch (%)',font:{size:9,color:'#6a8caf'}},tickfont:{size:8,color:'#4a6a8a'},gridcolor:'#1a2a40',range:[0,50]},
    showlegend:false
}),PC);

// Confidence vs Mismatch - scatter
Plotly.newPlot('chart-conf-scatter',[{
    x:SCATTER_X.map(function(){return 0.5}), y:SCATTER_Y.map(function(){return 0}),
    text:SCATTER_LABELS, mode:'markers',type:'scatter',
    marker:{color:'#5ab4ac',size:8,opacity:0.7,line:{color:'#0a1628',width:1}},
    hovertemplate:'<b>%{text}</b><br>Avg Conf: %{x:.3f}<br>Mismatch: %{y:.1f}%<extra></extra>'
}],Object.assign({},DL,{
    height:220,margin:{l:50,t:5,b:35,r:15},
    xaxis:{title:{text:'Avg Confidence',font:{size:9,color:'#6a8caf'}},tickfont:{size:8,color:'#4a6a8a'},gridcolor:'#1a2a40',range:[0.5,1]},
    yaxis:{title:{text:'Mismatch (%)',font:{size:9,color:'#6a8caf'}},tickfont:{size:8,color:'#4a6a8a'},gridcolor:'#1a2a40'},
    showlegend:false
}),PC);

// High-conf accuracy bar chart
Plotly.newPlot('chart-hc-acc',[{
    y:HC_QUESTIONS, x:HC_VALUES.map(function(){return 0}), type:'bar', orientation:'h',
    marker:{color:HC_COLORS,line:{width:0}},
    text:HC_VALUES.map(function(){return ''}),
    textposition:'outside',textfont:{size:7,color:'#b0c4d8'},
    hovertemplate:'%{y}<br>Accuracy (Conf>0.9): %{x:.1%}<extra></extra>'
}],Object.assign({},DL,{
    height:500,margin:{l:200,t:5,b:30,r:40},
    xaxis:{range:[0,1.08],tickformat:'.0%',gridcolor:'#1a2a40',tickfont:{size:8,color:'#4a6a8a'}},
    yaxis:{tickfont:{size:7,color:'#b0c4d8'}},showlegend:false
}),PC);

// Animate verification charts when tab shown - replays every time
window._animateVerification=function(){
    // Reset to 0 first
    Plotly.restyle('chart-acc',{x:[ACC_VALUES.map(function(){return 0})],text:[ACC_VALUES.map(function(){return ''})]},0);
    Plotly.restyle('chart-est',{x:[EST_VALUES.map(function(){return 0})],text:[EST_VALUES.map(function(){return ''})]},0);
    Plotly.restyle('chart-conf-box',{y:[CONF_BIN_MISMATCH.map(function(){return 0})],text:[CONF_BIN_MISMATCH.map(function(){return ''})]},0);
    Plotly.restyle('chart-conf-scatter',{x:[SCATTER_X.map(function(){return 0.5})],y:[SCATTER_Y.map(function(){return 0})]},0);
    Plotly.restyle('chart-hc-acc',{x:[HC_VALUES.map(function(){return 0})],text:[HC_VALUES.map(function(){return ''})]},0);
    setTimeout(function(){
        Plotly.animate('chart-acc',{data:[{x:ACC_VALUES,text:ACC_VALUES.map(function(v){return (v*100).toFixed(0)+'%'})}]},{
            transition:{duration:900,easing:'cubic-out'},frame:{duration:900}});
        Plotly.animate('chart-est',{data:[{x:EST_VALUES,text:EST_VALUES.map(function(v){return (v>0?'+':'')+v+'pp'})}]},{
            transition:{duration:900,easing:'cubic-out'},frame:{duration:900}});
        Plotly.animate('chart-conf-box',{data:[{y:CONF_BIN_MISMATCH,text:CONF_BIN_MISMATCH.map(function(v){return v.toFixed(1)+'%'})}]},{
            transition:{duration:800,easing:'cubic-out'},frame:{duration:800}});
        Plotly.animate('chart-conf-scatter',{data:[{x:SCATTER_X,y:SCATTER_Y}]},{
            transition:{duration:1000,easing:'cubic-out'},frame:{duration:1000}});
        Plotly.animate('chart-hc-acc',{data:[{x:HC_VALUES,text:HC_VALUES.map(function(v){return (v*100).toFixed(0)+'%'})}]},{
            transition:{duration:900,easing:'cubic-out'},frame:{duration:900}});
    },50);
};

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

with open("temp.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated temp.html - 6 tabs:")
print(f"  1. Norm Presence (gauge)")
print(f"  2. Norm Dimensions (4 stacked bars + reference group bubbles)")
print(f"  3. Author Stance (treemap)")
print(f"  4. Factors & Barriers (3 radial charts)")
print(f"  5. Verification (stat boxes + 4 charts)")
print(f"  6. Examples ({len(question_order)} questions)")
