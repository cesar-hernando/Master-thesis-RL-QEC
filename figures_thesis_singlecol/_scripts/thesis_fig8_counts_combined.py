#!/usr/bin/env python
"""Thesis fig 8: combined 2x2 counts, CM(alpha=1) vs RCM(alpha*) vs MWPM, with the NEW alpha*.

Top row  (a,b): sub-threshold weight (d-1)/2  -- a: d=5 weight-2 (exact), b: d=7 weight-3 (est.).
Bottom row (c,d): threshold weight (d+1)/2    -- c: d=5 weight-3, d: d=7 weight-4 (per 1e6).
Reads data/cm_counts_fig8_newalpha.npz (recompute_fig8_counts.py).  TeX look + grid.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import thesis_style as ts

z = dict(np.load(os.path.join(ts.ROOT, "data", "cm_counts_fig8_newalpha.npz")))
P = z["P"]
C_CM, C_RCM, C_MW = "#D55E00", "#009E73", "#0072B2"

ts.set_style()
fig, axs = plt.subplots(2, 2, figsize=ts.figsize(2, 2, panel_ratio=0.82), constrained_layout=True)


def panel(ax, mw, mwe, cm, cme, rcm, rcme, ylabel, title):
    ax.errorbar(P, cm, yerr=cme, marker="s", ls="-", color=C_CM, ms=4.5, lw=1.3, capsize=2.2, elinewidth=0.9)
    ax.errorbar(P, rcm, yerr=rcme, marker="D", ls="-", color=C_RCM, ms=4.5, lw=1.3, capsize=2.2, elinewidth=0.9)
    ax.errorbar(P, mw, yerr=mwe, marker="o", ls="--", color=C_MW, ms=4.5, lw=1.3, capsize=2.2, elinewidth=0.9)
    ax.set_ylabel(ylabel); ax.set_title(title)


z0 = np.zeros_like(P)
# (a) d5 weight-2 sub-threshold (exact; no error bars)
panel(axs[0, 0], z["sub_d5_mw"], z0, z["sub_d5_cm1"], z0, z["sub_d5_astar"], z0,
      "Confusing configurations", r"$d=5$: weight 2")
# (b) d7 weight-3 sub-threshold (estimated, x10^3)
panel(axs[0, 1], z["sub_d7_mw"] / 1e3, z0, z["sub_d7_cm1"] / 1e3, z["sub_d7_cm1e"] / 1e3,
      z["sub_d7_astar"] / 1e3, z["sub_d7_astare"] / 1e3,
      r"Estimated confusing configs ($\times 10^{3}$)", r"$d=7$: weight 3")
# (c) d5 weight-3 threshold (per 1e6)
panel(axs[1, 0], z["thr_d5_mw"], z["thr_d5_mwe"], z["thr_d5_cm1"], z["thr_d5_cm1e"],
      z["thr_d5_astar"], z["thr_d5_astare"], r"Failures per $10^{6}$ configs", r"$d=5$: weight 3")
# (d) d7 weight-4 threshold (per 1e6)
panel(axs[1, 1], z["thr_d7_mw"], z["thr_d7_mwe"], z["thr_d7_cm1"], z["thr_d7_cm1e"],
      z["thr_d7_astar"], z["thr_d7_astare"], r"Failures per $10^{6}$ configs", r"$d=7$: weight 4")

for ax in axs.flat:
    ax.set_xscale("log")
    ax.margins(y=0.12)
    hi = ax.get_ylim()[1]
    ax.set_ylim(-0.035 * hi, hi)
for ax in axs[1, :]:
    ax.set_xlabel(r"Physical error rate $p$")
for ax, lab in zip(axs.flat, "abcd"):
    ts.panel(ax, lab, x=-0.22, y=1.03)

handles = [Line2D([0], [0], marker="s", ls="-", color=C_CM, label=r"CM ($\alpha=1$)"),
           Line2D([0], [0], marker="D", ls="-", color=C_RCM, label=r"RCM ($\alpha^{*}$)"),
           Line2D([0], [0], marker="o", ls="--", color=C_MW, label="MWPM")]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=True, framealpha=1.0,
           edgecolor="0.7", fontsize=ts.BASE)
ts.save(fig, "cm_rcm_counts_sub_and_threshold_d5_d7")
