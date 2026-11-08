"""Draws the figures (results/fig*.png) from the tables in results/ (no data, no models needed).
  python src/make_figures.py"""
import sys                                                     # paths
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays, tables
import matplotlib                                              # plotting library
matplotlib.use("Agg")                                          # draw to files, no screen needed
import matplotlib.pyplot as plt                                # plotting interface

ROOT = Path(__file__).resolve().parent.parent                  # the SUBMISSION folder
RES = FIG = ROOT / "results"                                  # tables are read from, and figures written to, results/
FIG.mkdir(exist_ok=True)                                       # make sure the output folder exists
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"   # validated categorical slots 1-2, text and grid tones
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150})   # quiet axes, light grid


def save(fig, name):                                           # write one figure and close it
    fig.tight_layout(); fig.savefig(FIG / name); plt.close(fig); print("wrote", FIG / name)


# 1. dev CV vs public leaderboard for every shipped version
T = pd.read_csv(RES / "cv_vs_leaderboard.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(len(T))
ax.plot(x, T.dev_cv_auc, "o-", color=BLUE, lw=2, ms=8, label="dev cross-validation AUC")   # what the decisions were based on
ax.plot(x, T.public_lb_auc, "s--", color=ORANGE, lw=2, ms=8, label="public leaderboard AUC (never used to decide)")   # what the leaderboard reported afterwards
for i, r in T.iterrows():
    ax.annotate(f"{r.lb_minus_cv:+.5f}", (i, max(r.dev_cv_auc, r.public_lb_auc)), textcoords="offset points", xytext=(0, 12), ha="center", color=MUTED, fontsize=8)
ax.set_ylim(None, 0.8655); ax.set_xticks(x, T.ship); ax.set_ylabel("AUC"); ax.set_title("Cross-validation predicted the leaderboard (labels: LB minus CV)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left"); save(fig, "fig1_cv_vs_leaderboard.png")

# 2. paired fold-by-fold gain of v6 over v5 (the ship gate)
P = pd.read_csv(RES / "cv_paired_v6_vs_v5.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for j, (role, col) in enumerate([("dev (decisions)", BLUE), ("fresh (confirmation only)", ORANGE)]):
    for ms, mk in ((0, "o"), (1, "^")):
        q = P[(P.split_role == role) & (P.model_seed == ms)]
        xx = j + (ms - 0.5) * 0.3 + np.linspace(-0.08, 0.08, len(q))   # spread the folds sideways so every dot is visible
        ax.scatter(xx, q.delta_v6_minus_v5, s=40, color=col, marker=mk, edgecolor="#fcfcfb", linewidth=1, label=f"{role.split(' ')[0]}, model seed {ms}")
ax.axhline(0, color=INK, lw=1); ax.axhline(0.00105, color=MUTED, lw=1, ls=":")   # zero line and the pre-registered mean-gain threshold
ax.text(1.45, 0.00115, "gate: mean >= +0.00105", color=MUTED, fontsize=8, ha="right")
ax.set_xticks([0, 1], ["dev: 15 folds", "fresh: 10 folds"]); ax.set_ylabel("AUC(v6) - AUC(v5), same fold"); ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.0035, None)                                     # room below zero so the legend does not cover the gate line
ax.set_title("Every one of 50 paired folds favoured v6", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center")
save(fig, "fig2_ship_gate_fold_deltas.png")

# 3. per-slice AUC, v5 vs v6 (dev 3x5 OOF, model seed 0)
S = pd.read_csv(RES / "slice_aucs_v5_v6.csv").iloc[::-1]       # bottom-to-top so the first slice is on top
fig, ax = plt.subplots(figsize=(7.2, 6.6)); yy = np.arange(len(S))
ax.hlines(yy, S.dev_v5, S.dev_v6, color=GRID, lw=3)            # connector shows the change
ax.scatter(S.dev_v5, yy, color=BLUE, s=36, zorder=3, label="v5 (previous ship)"); ax.scatter(S.dev_v6, yy, color=ORANGE, s=36, zorder=3, label="v6 (final)")
ax.set_yticks(yy, [f"{s}  (n={n})" for s, n in zip(S.slice, S.n_series)], fontsize=8); ax.set_xlabel("AUC within slice (dev 3x5 out-of-fold, mean over repeats)")
ax.set_title("Where the model is weak: per-slice AUC", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "fig3_slice_aucs.png")

# 4. identified vs unexplained positives
C = pd.read_csv(RES / "ceiling_decomposition.csv"); C = C[C.source.str.contains("dev")]
fig, ax = plt.subplots(figsize=(6.4, 3.4)); labs = ["v3", "v5", "v6"]; x = np.arange(3); w = 0.36
ident = [C[(C.model == m) & C.group.str.startswith("identified")].auc_vs_all_dev_negatives.iloc[0] for m in labs]
unex = [C[(C.model == m) & C.group.str.startswith("unexplained")].auc_vs_all_dev_negatives.iloc[0] for m in labs]
ax.bar(x - w / 2 - 0.01, ident, w, color=BLUE, label="identified positives (32%)"); ax.bar(x + w / 2 + 0.01, unex, w, color=ORANGE, label="unexplained positives (68%)")
for i in range(3):
    ax.text(x[i] - w / 2, ident[i] + 0.004, f"{ident[i]:.3f}", ha="center", fontsize=8, color=INK); ax.text(x[i] + w / 2, unex[i] + 0.004, f"{unex[i]:.3f}", ha="center", fontsize=8, color=INK)
ax.axhline(0.816, color=MUTED, ls=":", lw=1); ax.text(2.55, 0.819, "unexplained AUC needed for 0.87 overall", fontsize=7, color=MUTED, ha="right")
ax.set_ylim(0.7, 1.07); ax.set_xticks(x, labs); ax.set_ylabel("AUC vs all dev negatives")
ax.set_title("All remaining error is in the weak-effect positives", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
save(fig, "fig4_identified_vs_unexplained.png")

# 5. simulated-null calibration layer: spread of 'z-scores' on real negatives (should be 1)
D = pd.read_csv(RES / "null_calibration_diagnostic_v8.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.8)); x = np.arange(len(D))
ax.scatter(x, D.old_z_sd_median, color=ORANGE, s=36, label="shipped simulated-null z (in v6)")
ax.scatter(x, D.new_pz_sd_median, color=BLUE, s=36, label="empirical-null z (research, not shipped)")
ax.axhline(1, color=INK, lw=1); ax.set_xticks(x, D.stratum, rotation=70, fontsize=7); ax.set_ylabel("median SD over statistics\n(held-out real negatives)")
ax.set_title("The shipped null-calibration layer is mis-scaled on real data", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8)
save(fig, "fig5_null_calibration.png")
