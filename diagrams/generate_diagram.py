"""
Generates diagrams/pipeline_architecture.png — a visual of the five-agent
recursive pipeline and its feedback loop. Run this once if you ever need to
regenerate the image (e.g. after changing the architecture):

    python diagrams/generate_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")

BOX_W, BOX_H = 6.4, 1.1
CENTER_X = 5

colors = {
    "data": "#e8e8e8",
    "agent2": "#bfe3d4",
    "agent1": "#c9d9f2",
    "agent5": "#f2e0b8",
    "agent3": "#f0c3c3",
    "agent4": "#d8c3ea",
    "done": "#e8e8e8",
}

boxes = [
    ("Dataset\n(Adult Census Income CSV)", 12.0, colors["data"], None),
    ("Agent 2 — Feature Engineering\nImputation, encoding, scaling,\ninteractions, aggregates", 10.3, colors["agent2"], None),
    ("Agent 1 — Model Benchmarking\nFits 9-12 algorithms with\nstratified k-fold CV", 8.6, colors["agent1"], None),
    ("Agent 5 — Hyperparameter Tuning\nRandomizedSearchCV on\ntop-N models", 6.9, colors["agent5"], None),
    ("Agent 3 — Results Aggregation\nMerges outputs, finds patterns,\ncompares to prior pass", 5.2, colors["agent3"], None),
    ("Agent 4 — Reporting\nWrites report.md +\nnext_pass_config.json", 3.5, colors["agent4"], None),
    ("Pass complete\nMetrics logged to\nmetrics/pass_comparison.csv", 1.8, colors["done"], None),
]

for label, y, color, _ in boxes:
    box = FancyBboxPatch(
        (CENTER_X - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.4, edgecolor="#333333", facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(CENTER_X, y, label, ha="center", va="center", fontsize=10.5,
             fontweight="medium", zorder=4, linespacing=1.4)

# Straight arrows connecting each stage vertically
for i in range(len(boxes) - 1):
    y_start = boxes[i][1] - BOX_H / 2
    y_end = boxes[i + 1][1] + BOX_H / 2
    arrow = FancyArrowPatch(
        (CENTER_X, y_start), (CENTER_X, y_end),
        arrowstyle="-|>", mutation_scale=18, linewidth=1.6,
        color="#333333", zorder=2,
    )
    ax.add_patch(arrow)

# Feedback loop arrow: Agent 4 -> back up to Agent 1 (curving around the right side)
loop_x = 9.1
p1 = (CENTER_X + BOX_W / 2, 3.5)   # right edge of Agent 4 box
p2 = (loop_x, 3.5)
p3 = (loop_x, 8.6)
p4 = (CENTER_X + BOX_W / 2, 8.6)   # right edge of Agent 1 box

path_data = [
    (MplPath.MOVETO, p1),
    (MplPath.LINETO, p2),
    (MplPath.LINETO, p3),
    (MplPath.LINETO, p4),
]
codes, verts = zip(*path_data)
loop_path = MplPath(verts, codes)
loop_patch = mpatches.PathPatch(
    loop_path, facecolor="none", edgecolor="#a83232",
    linewidth=1.8, linestyle="--", zorder=2,
)
ax.add_patch(loop_patch)

arrow_head = FancyArrowPatch(
    p4, (CENTER_X + BOX_W / 2 + 0.05, 8.6),
    arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
    color="#a83232", zorder=2,
)
ax.add_patch(arrow_head)

ax.text(loop_x + 0.25, 6.05, "next_pass_config.json\nfeeds back into Agent 1\n(model list, transforms,\ntuned hyperparameters)",
        ha="left", va="center", fontsize=8.7, color="#a83232", style="italic", linespacing=1.4)

# Loop label at top
ax.text(CENTER_X, 12.9, "Recursive Self-Improving (RSI) Data Science Pipeline",
        ha="center", va="center", fontsize=13.5, fontweight="bold")
ax.text(CENTER_X, 0.6, "↻ Repeat for passes 2, 3, ... — each pass carries forward what the last pass learned",
        ha="center", va="center", fontsize=9.5, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("diagrams/pipeline_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("Saved diagrams/pipeline_architecture.png")
