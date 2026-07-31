#!/usr/bin/env python
"""Combined 2x2 counts figure: CM(alpha=1) vs RCM(alpha*) vs MWPM.

Top row  (a,b): sub-threshold weight (d-1)/2  -- a: d=5 weight-2, b: d=7 weight-3.
Bottom row (c,d): threshold weight (d+1)/2    -- c: d=5 weight-3, d: d=7 weight-4.

Reads the two caches produced by the notebooks:
  data/cm_lowweight_counts_bestalpha_d5_d7.npz  (sub-threshold, from reg_cm_bestalpha_figures.ipynb)
  data/cm_threshold_counts_d5_d7.npz            (threshold,     from cm_vs_mwpm_confusion.ipynb)
Output: plots/figures/cm_rcm_counts_sub_and_threshold_d5_d7.{png,pdf}
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "plots", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 120, "mathtext.fontset": "cm"})

sub = dict(np.load(os.path.join(ROOT, "data", "cm_lowweight_counts_bestalpha_d5_d7.npz")))
thr = dict(np.load(os.path.join(ROOT, "data", "cm_threshold_counts_d5_d7.npz")))

C_CM, C_RCM, C_MW = "#e67e22", "#009E73", "#2e86de"
fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)

# (a) d=5, weight-2, sub-threshold (exhaustive; no error bars)
ax = axs[0, 0]
ax.plot(sub["P5"], sub["cm5_1"], "s-", color=C_CM)
ax.plot(sub["P5"], sub["cm5_a"], "D-", color=C_RCM)
ax.plot(sub["P5"], sub["mw5"], "o--", color=C_MW)
ax.set_ylabel("confusing configurations")
ax.set_title(r"$d=5$: weight 2")

# (b) d=7, weight-3, sub-threshold (sampled, x10^3)
ax = axs[0, 1]
ax.errorbar(sub["P7"], sub["cm7_1"]/1e3, yerr=sub["cm7_1e"]/1e3, marker="s", ls="-", color=C_CM, capsize=3, elinewidth=1.0)
ax.errorbar(sub["P7"], sub["cm7_a"]/1e3, yerr=sub["cm7_ae"]/1e3, marker="D", ls="-", color=C_RCM, capsize=3, elinewidth=1.0)
ax.plot(sub["P7"], sub["mw7"]/1e3, "o--", color=C_MW)
ax.set_ylabel(r"est. confusing configs ($\times 10^{3}$)")
ax.set_title(r"$d=7$: weight 3")

# (c) d=5, weight-3, threshold (sampled, per 1e6)
ax = axs[1, 0]
ax.errorbar(thr["P"], thr["cm5"], yerr=thr["cm_e5"], marker="s", ls="-", color=C_CM, capsize=3, elinewidth=1.0)
ax.errorbar(thr["P"], thr["rcm5"], yerr=thr["rcm_e5"], marker="D", ls="-", color=C_RCM, capsize=3, elinewidth=1.0)
ax.errorbar(thr["P"], thr["mw5"], yerr=thr["mw_e5"], marker="o", ls="--", color=C_MW, capsize=3, elinewidth=1.0)
ax.set_ylabel(r"failures per $10^{6}$ configs")
ax.set_title(r"$d=5$: weight 3")

# (d) d=7, weight-4, threshold (sampled, per 1e6)
ax = axs[1, 1]
ax.errorbar(thr["P"], thr["cm7"], yerr=thr["cm_e7"], marker="s", ls="-", color=C_CM, capsize=3, elinewidth=1.0)
ax.errorbar(thr["P"], thr["rcm7"], yerr=thr["rcm_e7"], marker="D", ls="-", color=C_RCM, capsize=3, elinewidth=1.0)
ax.errorbar(thr["P"], thr["mw7"], yerr=thr["mw_e7"], marker="o", ls="--", color=C_MW, capsize=3, elinewidth=1.0)
ax.set_ylabel(r"failures per $10^{6}$ configs")
ax.set_title(r"$d=7$: weight 4")

for ax in axs.flat:
    ax.set_xscale("log")
    ax.margins(y=0.12)
    hi = ax.get_ylim()[1]
    ax.set_ylim(-0.035 * hi, hi)          # small negative floor so MWPM=0 markers stay visible
for ax in axs[1, :]:
    ax.set_xlabel("Physical error rate $p$")
for ax, lab in zip(axs.flat, "abcd"):
    ax.text(-0.17, 1.03, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold")

handles = [Line2D([0], [0], marker="s", ls="-", color=C_CM, label=r"CM ($\alpha=1$)"),
           Line2D([0], [0], marker="D", ls="-", color=C_RCM, label=r"RCM ($\alpha^{*}$)"),
           Line2D([0], [0], marker="o", ls="--", color=C_MW, label="MWPM")]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=12)

for ext in ("png", "pdf"):
    out = os.path.join(FIG, f"cm_rcm_counts_sub_and_threshold_d5_d7.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)
