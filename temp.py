"""
temp.py - Flipped semi-circular gauge chart for norm signal present across sectors
"""
import json
import plotly.graph_objects as go

# Load norms labels data to calculate norm signal present percentages
with open("paper4data/norms_labels.json", "r", encoding="utf-8") as f:
    norms_data = json.load(f)

# Calculate norm signal present for each sector
sector_stats = {}
for sector in ["food", "transport", "housing"]:
    if sector in norms_data:
        comments = norms_data[sector]
        total = len(comments)
        # Count comments with gate=1 (norm signal present)
        gate_present = sum(1 for c in comments if c.get("answers", {}).get("1.1_gate") == "1")
        pct = (gate_present / total * 100) if total > 0 else 0
        sector_stats[sector] = {
            "total": total,
            "gate_present": gate_present,
            "percentage": pct
        }

# Print statistics
print("\nNorm Signal Present by Sector:")
print("-" * 60)
for sector, stats in sector_stats.items():
    print(f"{sector:12} {stats['percentage']:5.1f}% ({stats['gate_present']}/{stats['total']})")
print("-" * 60)

# Calculate overall average
all_sectors_pct = [stats['percentage'] for stats in sector_stats.values()]
avg_pct = sum(all_sectors_pct) / len(all_sectors_pct)
print(f"Average:     {avg_pct:5.1f}%\n")

# Create flipped semi-circular gauge chart
fig = go.Figure()

# Colors for each sector
colors = {
    'food': '#5ab4ac',      # Teal
    'transport': '#af8dc3', # Purple
    'housing': '#f46d43'    # Orange
}

# Values for the gauge (showing each sector)
values = [
    sector_stats['food']['percentage'],
    sector_stats['transport']['percentage'],
    sector_stats['housing']['percentage'],
    100  # Bottom half (hidden)
]

labels = ['Food', 'Transport', 'Housing', '']

# Create semi-circle gauge (flipped - arc on top)
fig.add_trace(go.Pie(
    values=values,
    labels=labels,
    marker=dict(
        colors=[colors['food'], colors['transport'], colors['housing'], 'rgba(255,255,255,0)'],
        line=dict(color='white', width=4)
    ),
    hole=0.70,
    direction='clockwise',
    sort=False,
    rotation=270,  # Rotated to put arc on top
    textposition='none',
    hovertemplate='<b>%{label}</b><br>%{value:.1f}% norm signal present<extra></extra>',
    showlegend=False
))

# Add center text showing average percentage
fig.add_annotation(
    text=f'<b>{avg_pct:.1f}%</b>',
    x=0.5, y=0.55,
    font=dict(size=52, color='#1a1a1a', family='Arial'),
    showarrow=False
)

fig.add_annotation(
    text='Avg. Norm Signal',
    x=0.5, y=0.40,
    font=dict(size=14, color='#666666', family='Arial'),
    showarrow=False
)

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Norm Signal Present</b>',
        x=0.5,
        xanchor='center',
        font=dict(size=22, color='#1a1a1a', family='Arial')
    ),
    height=450,
    margin=dict(t=80, b=80, l=20, r=20),
    paper_bgcolor='white',
    plot_bgcolor='white',
)

# Add legend annotations at the bottom
legend_y = 0.12
fig.add_annotation(
    text='●',
    x=0.25, y=legend_y,
    font=dict(size=24, color=colors['food']),
    showarrow=False
)
fig.add_annotation(
    text=f'Food ({sector_stats["food"]["percentage"]:.1f}%)',
    x=0.29, y=legend_y,
    font=dict(size=13, color='#1a1a1a', family='Arial'),
    showarrow=False,
    xanchor='left'
)

fig.add_annotation(
    text='●',
    x=0.45, y=legend_y,
    font=dict(size=24, color=colors['transport']),
    showarrow=False
)
fig.add_annotation(
    text=f'Transport ({sector_stats["transport"]["percentage"]:.1f}%)',
    x=0.49, y=legend_y,
    font=dict(size=13, color='#1a1a1a', family='Arial'),
    showarrow=False,
    xanchor='left'
)

fig.add_annotation(
    text='●',
    x=0.70, y=legend_y,
    font=dict(size=24, color=colors['housing']),
    showarrow=False
)
fig.add_annotation(
    text=f'Housing ({sector_stats["housing"]["percentage"]:.1f}%)',
    x=0.74, y=legend_y,
    font=dict(size=13, color='#1a1a1a', family='Arial'),
    showarrow=False,
    xanchor='left'
)

# Save to HTML
fig.write_html("temp.html", config={'displayModeBar': False})
print("Saved to temp.html")
