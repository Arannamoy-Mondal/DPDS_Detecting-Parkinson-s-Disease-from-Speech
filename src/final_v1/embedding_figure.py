"""Schematic of how a frozen encoder turns a recording into one embedding (figure for report v1.3). Numbers are the real ones used."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FIG = WORKSPACE / "reports/FINAL_V1/study_report/figures"; FIG.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(-0.15, 10.15); ax.set_ylim(0, 3.6)
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#6B7280", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
def arrow(x0, x1, y=1.8):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color="#6B7280"))
box(0.0, 1.2, 1.6, 1.2, "Recording\n(16 kHz waveform)\nmedian 10.6 s\n(3 s to 250 s)", "#F3F4F6")
box(1.9, 1.2, 1.6, 1.2, "Cut into\n10-second windows\n(last one aligned\nto the end)", "#E8EFFA")
box(3.8, 1.2, 2.2, 1.2, "Frozen encoder\n(CNN + 24 transformer\nlayers, ~315M weights,\nnever trained here)", "#E8F3EC")
box(6.5, 1.2, 1.6, 1.2, "Average over time\n499 vectors x 1,024\n-> 1,024 numbers\nper window", "#FFF7E6")
box(8.4, 1.2, 1.5, 1.2, "Average over\nwindows\n-> one 1,024-\nnumber embedding", "#FFF7E6")
for a, b in [(1.6, 1.9), (3.5, 3.8), (6.0, 6.5), (8.1, 8.4)]:
    arrow(a, b)
ax.text(5.0, 0.6, "The arithmetic is only averaging: no learned parameter is fitted in this step.", ha="center", fontsize=8.5, color="#374151")
fig.savefig(FIG / "embedding_pipeline.pdf", bbox_inches="tight"); plt.close(fig)
print("embedding figure written")
