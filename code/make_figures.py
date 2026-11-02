"""
Publication figures for "Tune Before You Clean"
===============================================
The numbers below are the dataset-level aggregates printed in summary_v3.md
(Stage 3, 40 datasets x 5 seeds), summary_v4.md (Stage 4) and summary_v5.md
(Stage 5). Each block names its source table so it can be checked line by line.
Output: fig1..fig4 as PDF (vector, for the journal) and PNG (300 dpi) in ./figures
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Palette: grey first, one accent; categorical slots 1-3 of the validated reference palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
GREY, GREY_LIGHT = "#8a8880", "#c9c7bf"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
FAMS = ["LogReg", "KNN", "RF", "HistGB"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def zero_line(ax):
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1: detection (summary_v3.md, section B)
scorers = ["IQR", "MAD", "Robust Mahal.", "LOF", "Isolation Forest", "Data-OOB", "KNN-Shapley", "Self-confidence"]
cols = ["Gross\nfeature\nerrors 10%", "Mild\nfeature\nerrors 10%", "Mixed 10%:\nfeature\nrows",
        "Symmetric\nlabel noise\n20%", "Asymmetric\nlabel noise\n20%", "Mixed 10%:\nlabel\nrows"]
auroc = np.array([
    [0.958, 0.886, 0.957, 0.499, 0.529, 0.447],
    [0.923, 0.839, 0.923, 0.501, 0.540, 0.451],
    [0.991, 0.969, 0.991, 0.498, 0.551, 0.444],
    [0.992, 0.984, 0.992, 0.499, 0.520, 0.443],
    [0.996, 0.980, 0.996, 0.500, 0.554, 0.443],
    [0.729, 0.722, 0.649, 0.898, 0.779, 0.913],
    [0.794, 0.783, 0.730, 0.884, 0.743, 0.879],
    [0.732, 0.725, 0.640, 0.902, 0.781, 0.917],
])
fig, ax = plt.subplots(figsize=(6.2, 3.9))
cmap = mpl.colors.LinearSegmentedColormap.from_list("blue", ["#f4f7fb", "#a9c8ee", BLUE, "#123f78"])
im = ax.imshow(auroc, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
for i in range(auroc.shape[0]):
    for j in range(auroc.shape[1]):
        v = auroc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.82 else INK)
ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
ax.set_yticks(range(len(scorers)), scorers)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(4.5, color="white", lw=2.5)
ax.axvline(2.5, color="white", lw=2.5)
ax.text(-0.30, 2.0, "Classical", rotation=90, ha="right", va="center", transform=ax.get_yaxis_transform(), color=INK2, fontsize=8)
ax.text(-0.30, 6.0, "Target-aware", rotation=90, ha="right", va="center", transform=ax.get_yaxis_transform(), color=INK2, fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("AUROC for the corrupted rows (0.5 = chance)", fontsize=7.5)
cb.outline.set_visible(False)
ax.set_title("Classical detectors find feature errors and miss label errors;\ntarget-aware scorers do the reverse",
             loc="left", fontsize=8.5, color=INK)
save(fig, "fig1_detection")

# ---------------------------------------------------------------- Figure 2: mechanism (summary_v3.md, section C, means)
names = ["IQR", "MAD", "Robust Mahal.", "LOF", "Isolation Forest", "Data-OOB", "KNN-Shapley", "Self-confidence"]
hard = [1.10, 1.07, 1.09, 1.52, 1.20, 3.97, 3.80, 4.34]
minority = [1.19, 1.35, 1.45, 0.95, 1.35, 3.19, 4.06, 3.54]
fig, ax = plt.subplots(figsize=(4.8, 3.5))
ax.axhline(1, color=GREY_LIGHT, lw=0.8, ls="--", zorder=1)
ax.axvline(1, color=GREY_LIGHT, lw=0.8, ls="--", zorder=1)
for k, (x, y, n) in enumerate(zip(hard, minority, names)):
    ax.scatter(x, y, s=46, color=BLUE if k >= 5 else GREY, edgecolor="white", lw=1.2, zorder=3)
# classical cluster: labels stacked in open space to the right, joined by thin leader lines
label_pos = {"Robust Mahal.": (2.05, 2.15), "MAD": (2.05, 1.85), "Isolation Forest": (2.05, 1.55),
             "IQR": (2.05, 1.25), "LOF": (2.05, 0.85)}
for x, y, n in zip(hard, minority, names):
    if n in label_pos:
        tx, ty = label_pos[n]
        ax.annotate(n, (x, y), xytext=(tx, ty), textcoords="data", fontsize=7.2, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=GREY_LIGHT, lw=0.6, shrinkA=0, shrinkB=3))
offsets = {"Data-OOB": (7, -4), "KNN-Shapley": (-70, 4), "Self-confidence": (-38, 8)}
for x, y, n in zip(hard, minority, names):
    if n in offsets:
        ax.annotate(n, (x, y), xytext=offsets[n], textcoords="offset points", fontsize=7.2, color=INK)
ax.set_xlim(0.7, 4.8)
ax.set_ylim(0.6, 4.6)
ax.set_xlabel("Hardness of flagged rows ÷ average hardness (kDN)")
ax.set_ylabel("Smallest class among flagged ÷ its share")
ax.text(0.73, 0.68, "1 = no bias", color=GREY, fontsize=7)
ax.scatter([], [], color=GREY, s=30, label="Classical")
ax.scatter([], [], color=BLUE, s=30, label="Target-aware")
ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.97), fontsize=7.5, handletextpad=0.2)
ax.set_title("On clean data, target-aware scorers flag hard,\nminority-class rows (top 5%, 40 datasets)", loc="left", fontsize=8.5)
save(fig, "fig2_mechanism")

# ---------------------------------------------------------------- Figure 4: oracle vs detector (summary_v5.md, section B)
panels = {
    "Clean data\n(remove top 10%)": {"Isolation Forest": [-0.52, -1.29, -1.12, -1.15],
                                     "Data-OOB": [-0.80, -1.33, -2.15, -2.76]},
    "Gross feature errors 10%": {"Oracle": [5.41, -0.07, 0.08, 0.16],
                                 "Isolation Forest (right-sized)": [5.17, 0.02, 0.09, 0.11]},
    "Symmetric label noise 20%": {"Oracle": [4.50, 2.84, 1.88, 4.87],
                                  "Data-OOB (right-sized)": [2.80, 1.62, 0.49, 2.07]},
    "Asymmetric label noise 20%": {"Oracle": [8.58, 6.93, 9.19, 9.77],
                                   "Data-OOB (right-sized)": [0.39, -0.42, -0.55, -0.92]},
}
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
x = np.arange(4)
w = 0.38
for ax, (title, series) in zip(axes.ravel(), panels.items()):
    (n1, v1), (n2, v2) = series.items()
    c1, c2 = (GREY_LIGHT, BLUE) if "Oracle" in n1 else (GREY, BLUE)
    ax.bar(x - w / 2 - 0.01, v1, w, color=c1, label=n1, zorder=2)
    ax.bar(x + w / 2 + 0.01, v2, w, color=c2, label=n2, zorder=2)
    zero_line(ax)
    ax.set_xticks(x, FAMS)
    ax.set_title(title.replace("\n", " "), fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="upper left", handlelength=1.0)
    ax.tick_params(axis="x", length=0)
for ax in axes[:, 0]:
    ax.set_ylabel("Gain in balanced accuracy (points)")
axes[0, 0].set_ylim(-3.2, 12.5)
fig.suptitle("Perfect removal of feature errors does not help trees or KNN; label errors hurt every model,\n"
             "and real detectors miss almost all asymmetric ones", x=0.01, ha="left", fontsize=8.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.98))
save(fig, "fig4_oracle")

# ---------------------------------------------------------------- Figure 3: tune before you clean (summary_v4.md, section A)
opts = ["Default + remove IF 10%", "Tuned, no removal", "Tuned + remove IF 10%"]
clean = {"LogReg": [-0.52, 0.70, -0.02], "KNN": [-1.29, 0.98, -0.45],
         "RF": [-1.12, 0.45, -0.37], "HistGB": [-1.15, -0.36, -0.96]}
dirty = {"LogReg": [2.21, 0.49, 3.17], "KNN": [-0.45, 1.07, 0.75],
         "RF": [-0.23, 0.75, 0.55], "HistGB": [-0.45, 0.85, 0.05]}
colors = [GREY, BLUE, ORANGE]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
w = 0.26
for ax, (title, data) in zip(axes, [("Clean data", clean),
                                    ("Corrupted data (mean of 5 conditions)", dirty)]):
    for k, (opt, col) in enumerate(zip(opts, colors)):
        vals = [data[f][k] for f in FAMS]
        ax.bar(np.arange(4) + (k - 1) * (w + 0.02), vals, w, color=col, label=opt, zorder=2)
    zero_line(ax)
    ax.set_xticks(np.arange(4), FAMS)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, loc="left", fontsize=8.5)
axes[0].set_ylabel("Gain over default model (points)")
axes[1].legend(fontsize=7, loc="upper right")
fig.suptitle("Under corruption, tuning alone helps every family; removal adds value only for logistic regression",
             x=0.01, ha="left", fontsize=8.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, "fig3_tuning")
