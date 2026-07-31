#!/usr/bin/env python
"""CM (alpha=1) vs MWPM logical error rate, distances 5 and 7 (overlaid).

Left : absolute LER p_{L,d} for MWPM (circles) and CM, alpha=1 (squares) vs p (log-log).
Right: ratio p_{L,d}^CM / p_{L,d}^MWPM vs p (crossing 1 = hard CM falls behind MWPM).

p_{L,d}: logical error rate for a distance-d memory experiment decoding d rounds.
Data: data/reg_cm_alpha_scan_d3d5d7.csv. The two lowest-p d=7 points (1e-4, 2e-4) are
dropped (low statistics). Style matches final_plots/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "reg_cm_alpha_scan_d3d5d7.csv")
OUT = os.path.join(ROOT, "final_plots")
DISTANCES = [3, 5, 7]
DROP = {7: [1e-4, 2e-4]}

DCOL = {3: "#0072B2", 5: "#D55E00", 7: "#009E73"}      # colour = distance
BASE = "0.55"

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.9, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.9,
    "lines.linewidth": 1.8, "mathtext.fontset": "dejavusans",
})

PLR = r"p_{\mathrm{L},d}"
df = pd.read_csv(CSV)


def series(d):
    sub = df[df.distance == d]
    rows = []
    for p in sorted(sub.p.unique()):
        if any(abs(p - x) <= 1e-9 for x in DROP.get(d, [])):
            continue
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        cm = s[(s.decoder == "cm") & (np.abs(s.alpha - 1.0) < 1e-9)].iloc[0]
        r = cm.ler / mw.ler
        rse = r * np.sqrt((cm.ler_std / cm.ler) ** 2 + (mw.ler_std / mw.ler) ** 2)
        rows.append((p, mw.ler, mw.ler_std, cm.ler, cm.ler_std, r, rse))
    return np.array(rows).T


fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.1))

for d in DISTANCES:
    c = DCOL[d]
    p, ml, ms, cl, cs, r, rse = series(d)

    # ---- left: absolute LER (MWPM = dashed circle, CM = solid square) ----
    axL.errorbar(p, ml, yerr=ms, marker="o", ls=(0, (4, 3)), color=c, ms=5.5, mec="white",
                 mew=0.7, capsize=2.2, elinewidth=0.9, lw=1.6, zorder=4,
                 label=rf"MWPM, $d{{=}}{d}$")
    axL.errorbar(p, cl, yerr=cs, marker="s", ls="-", color=c, ms=5, mec="white",
                 mew=0.7, capsize=2.2, elinewidth=0.9, lw=1.6, zorder=4,
                 label=rf"CM, $d{{=}}{d}$")

    # ---- right: ratio ----
    axR.errorbar(p, r, yerr=rse, marker="s", color=c, ms=5, mec="white", mew=0.6,
                 capsize=2.2, elinewidth=0.9, zorder=3, label=rf"$d = {d}$")

axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel(r"Physical error rate $p$")
axL.set_ylabel(rf"Logical error rate $\,{PLR}$")
axL.legend(loc="upper left")

axR.axhline(1.0, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=1)
axR.set_xscale("log")
axR.set_xlabel(r"Physical error rate $p$")
axR.set_ylabel(rf"${PLR}^{{\mathrm{{CM}}}}\,/\,{PLR}^{{\mathrm{{MWPM}}}}$")
axR.legend(loc="upper center", ncol=2)

for ax, lab in zip((axL, axR), "ab"):
    ax.text(-0.16, 1.02, f"({lab})", transform=ax.transAxes, fontsize=12, fontweight="bold")

fig.tight_layout(w_pad=2.4)
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    out = os.path.join(OUT, f"cm_vs_mwpm_ler_d5_d7.{ext}")
    fig.savefig(out, dpi=300)
    print(f"saved {out}")
