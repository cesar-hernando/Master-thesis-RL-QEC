#!/usr/bin/env python
"""Thesis fig 1: CM (alpha=1) vs MWPM logical error rate (overlaid distances).

Left : absolute LER p_{L,d} for MWPM (dashed circle) and CM alpha=1 (solid square) vs p.
Right: ratio p_{L,d}^CM / p_{L,d}^MWPM vs p.
Two variants: core (d=3,5,7) and extended (+d=9,11).  New reg_cm_alpha_scan data, TeX look + grid.
Left-panel legend keeps the "d=" aligned by putting it first in each label.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

ts.set_style()
df = ts.load_scan()
PLR = r"p_{\mathrm{L},d}"


def series(d):
    sub = df[df.distance == d]
    rows = []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        cm = s[(s.decoder == "cm") & (np.abs(s.alpha - 1.0) < 1e-9)].iloc[0]
        r = cm.ler / mw.ler
        rse = r * np.sqrt((cm.ler_std / cm.ler) ** 2 + (mw.ler_std / mw.ler) ** 2)
        rows.append((p, mw.ler, mw.ler_std, cm.ler, cm.ler_std, r, rse))
    return np.array(rows).T


def make(distances, fname):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=ts.figsize(1, 2, panel_ratio=0.98))
    Lh = {}   # per-distance (handle, label) pairs for the left panel
    for d in distances:
        c = ts.DCOL[d]
        p, ml, ms, cl, cs, r, rse = series(d)
        # left: MWPM dashed circle, CM solid square  (label starts with d= for alignment)
        h_mw = axL.errorbar(p, ml, yerr=ms, marker="o", ls=(0, (4, 3)), color=c, ms=5.5, mec="white",
                            mew=0.7, capsize=2.2, elinewidth=0.9, lw=1.6, zorder=4)
        h_cm = axL.errorbar(p, cl, yerr=cs, marker="s", ls="-", color=c, ms=5, mec="white",
                            mew=0.7, capsize=2.2, elinewidth=0.9, lw=1.6, zorder=4)
        Lh[d] = [(h_mw, rf"$d={d}$, MWPM"), (h_cm, rf"$d={d}$, CM")]
        # right: ratio
        axR.errorbar(p, r, yerr=rse, marker="s", color=c, ms=5, mec="white", mew=0.6,
                     capsize=2.2, elinewidth=0.9, zorder=3, label=rf"$d={d}$")

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"Physical error rate $p$")
    axL.set_ylabel(rf"Logical error rate $\,{PLR}$")
    # split legend: small d (high curves) -> upper left; large d (low curves) -> lower right
    small = [d for d in distances if d <= 5]
    large = [d for d in distances if d > 5]
    hs = [(h, l) for d in small for (h, l) in Lh[d]]
    hl = [(h, l) for d in large for (h, l) in Lh[d]]
    if hs:
        leg1 = ts.legend(axL, [h for h, _ in hs], [l for _, l in hs], loc="upper left",
                         ncol=1, labelspacing=0.3, handlelength=1.9)
        axL.add_artist(leg1)
    if hl:
        ts.legend(axL, [h for h, _ in hl], [l for _, l in hl], loc="lower right",
                  ncol=1, labelspacing=0.3, handlelength=1.9)

    axR.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axR.set_xscale("log")
    axR.set_xlabel(r"Physical error rate $p$")
    axR.set_ylabel(rf"${PLR}^{{\mathrm{{CM}}}}\,/\,{PLR}^{{\mathrm{{MWPM}}}}$")
    ts.legend(axR, loc="upper center", ncol=3)

    for ax, lab in zip((axL, axR), "ab"):
        ts.panel(ax, lab, x=-0.20)
    fig.tight_layout(w_pad=1.2)
    ts.save(fig, fname)


for dists, suffix in ts.VARIANTS:
    make(dists, "cm_vs_mwpm_ler_d5_d7" + suffix)
