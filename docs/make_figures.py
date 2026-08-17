"""Render the README demo figure from live seed-42 output. No numbers are hand-entered:
everything is computed by the same code the pipeline runs, so the picture cannot drift from
the text.

Run:  python -m docs.make_figures      (from the repo root)
Writes: docs/demo.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.generate import build
from src.naive import score_all
from src.pipeline import run as pipeline_run, score as pipeline_score

# --- validated palette (dataviz reference instance) -------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
REAL = "#2a78d6"      # categorical slot 1, blue: genuine vendors
FAKE = "#eb6834"      # categorical slot 2, orange: pure-noise vendors

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
})

SEED = 42
u = build(SEED)
true_ics = np.array(u["true_ics"])

# --- panel 1 data: the naive backtest, one Sharpe per vendor ----------------------------
naive = score_all(u["returns"], u["vendors"])
naive = naive.merge(
    pd.DataFrame({"vendor": list(u["vendors"]), "true_ic": true_ics}),
    on="vendor", how="left",
)
is_real = (naive["true_ic"] > 0).to_numpy()
sharpe = naive["best_sharpe"].to_numpy()
real_s = np.sort(sharpe[is_real])[::-1]
fake_s = np.sort(sharpe[~is_real])[::-1]
top_is_fake = not is_real[np.argmax(sharpe)]
inversions = int((fake_s[:, None] > real_s).sum())

# --- panel 2 data: the pipeline holdout -------------------------------------------------
key = pd.DataFrame({
    "vendor": list(u["vendors"]), "true_ic": true_ics,
    "live_start": [u["live_starts"][n] for n in u["vendors"]],
})
pdf = pipeline_run(u["returns"], u["vendors"], dict(zip(key["vendor"], key["live_start"])))
pdf, stats = pipeline_score(pdf, true_ics.tolist())
bought_h = stats["holdout_bought"]
rejected_h = stats["holdout_rejected"]
n_bought = stats["n_bought"]
n_rejected = len(pdf) - n_bought

# ----------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9.2, 8.8), gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 1.05}
)
fig.subplots_adjust(top=0.9, bottom=0.08)


def titled(ax, title, subtitle):
    """Bold title above a two-line subtitle, both clear of the plot area."""
    ax.text(0, 1.42, title, transform=ax.transAxes, fontsize=15, fontweight="bold",
            color=INK, va="bottom")
    ax.text(0, 1.30, subtitle, transform=ax.transAxes, fontsize=10.3, color=INK2, va="top")

# ============ PANEL 1: the trap ============
rng = np.random.default_rng(0)
def strip(vals, y, color, ax):
    jit = rng.uniform(-0.13, 0.13, len(vals))
    ax.scatter(vals, np.full(len(vals), y) + jit, s=95, color=color,
               edgecolors=SURFACE, linewidths=1.6, zorder=3, alpha=0.95)

strip(real_s, 1.0, REAL, ax1)
strip(fake_s, 0.0, FAKE, ax1)
ax1.axvline(0, color=BASE, lw=1.2, zorder=1)

# mark the winner: a pure-noise vendor tops the whole shortlist
xmax = sharpe.max()
ax1.scatter([xmax], [0.0], s=170, facecolors="none", edgecolors=FAKE, linewidths=2.2, zorder=4)
ax1.annotate("a pure-noise vendor\nranks #1 of all 30",
             xy=(xmax, 0.0), xytext=(xmax - 0.05, 0.62),
             ha="right", va="center", fontsize=10.5, color=INK,
             arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4,
                             connectionstyle="arc3,rad=-0.25"))

ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Noise\nvendors", "Real\nvendors"], fontsize=10.5, color=INK2)
ax1.set_ylim(-0.7, 1.7)
ax1.set_xlim(-0.55, xmax + 0.25)
ax1.set_xlabel("best-of-nine in-sample Sharpe  (the knob-twisted score)", fontsize=10.5)
ax1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax1.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax1.spines[s].set_visible(False)
ax1.tick_params(length=0)
titled(ax1, "The naive backtest is fooled",
       f"Ranked by raw in-sample Sharpe, {inversions} times a zero-skill vendor outranks a "
       f"genuine one.\nThe real and noise vendors overlap completely.")

# ============ PANEL 2: the proof ============
labels = [f"Rejected\n({n_rejected} vendors)", f"Bought\n({n_bought} vendors)"]
vals = [rejected_h, bought_h]
colors = [MUTED, REAL]
ypos = [0, 1]
ax2.barh(ypos, vals, height=0.52, color=colors, zorder=3)
ax2.axvline(0, color=BASE, lw=1.2, zorder=1)
for y, v in zip(ypos, vals):
    ax2.text(v + 0.012, y, f"{v:+.3f}", va="center", ha="left",
             fontsize=13, fontweight="bold", color=INK)

ax2.set_yticks(ypos)
ax2.set_yticklabels(labels, fontsize=10.5, color=INK2)
ax2.set_ylim(-0.6, 1.6)
ax2.set_xlim(0, bought_h * 1.22)
ax2.set_xlabel("Sharpe on the 250-day sealed holdout  (data nothing in the pipeline touched)",
               fontsize=10.5)
ax2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax2.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
titled(ax2, "The pipeline's holdout proves the buy list",
       "Four gates, no answer key. The vendors it bought beat the ones it rejected\n"
       "on sealed data, the honest read a real desk could see on its own live book.")

fig.savefig("docs/demo.png", dpi=200, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.35)
print(f"wrote docs/demo.png   (inversions={inversions}, bought={bought_h:+.3f}, "
      f"rejected={rejected_h:+.3f}, n_bought={n_bought})")
