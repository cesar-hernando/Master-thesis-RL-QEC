#!/usr/bin/env python
"""Thesis fig 2b: LER(alpha)/LER(MWPM) vs p, one marker-series per alpha (colour = alpha).

Ratio uses error COUNTS (common-random-numbers), per distance subplot.
Two variants: core (d=3,5,7) and extended (+d=9,11).  New scan data, TeX look + grid.
"""
import os
import string
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import thesis_style as ts

df = ts.load_scan()
ALPHAS = ts.GRID_ALPHAS + [1.0]


def counts(d, a, p):
    s = df[(df.distance == d) & (np.abs(df.p - p) < 1e-12)]
    mw = s[s.decoder == "mwpm"].iloc[0].errors
    cm = s[(s.decoder == "cm") & (np.abs(s.alpha - a) < 1e-9)]
    if len(cm) == 0 or mw == 0:
        return None
    return cm.iloc[0].errors / mw


def make(distances, fname, nrows=1):
    norm = Normalize(vmin=0.1, vmax=1.0)
    cmap = mpl.colors.ListedColormap(plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 256)))
    ncols = int(np.ceil(len(distances) / nrows))
    gk = {"hspace": 0.42} if nrows > 1 else {}
    with mpl.rc_context({"font.size": 12, "axes.labelsize": 14.5, "axes.titlesize": 14.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.8 * nrows),
                                 sharey=True, squeeze=False, gridspec_kw=gk)
        flat = axes.flat
        used = []
        for k, d in enumerate(distances):
            ax = axes[k // ncols, k % ncols]; used.append(ax)
            ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
            ps_all = sorted(df[df.distance == d].p.unique())
            for a in ALPHAS:
                xs, ys = [], []
                for p in ps_all:
                    r = counts(d, a, p)
                    if r is not None:
                        xs.append(p); ys.append(r)
                ax.plot(xs, ys, marker="o", ms=4.5, ls="none", color=cmap(norm(a)), zorder=2)
            ax.set_xscale("log")
            ax.set_title(ts.DLABEL[d])
            ax.set_xlim(6e-5, 1.3e-2)
        # x-labels on the lowest visible axis of each column; y-labels on left column
        for k, d in enumerate(distances):
            r, c = k // ncols, k % ncols
            if r == nrows - 1 or (k + ncols) >= len(distances):
                axes[r, c].set_xlabel(r"Physical error rate $p$")
            if c == 0:
                axes[r, c].set_ylabel(r"$p_{L,d}^{\alpha\mathrm{-CM}}\,/\,p_{L,d}^{\mathrm{MWPM}}$")
        labs = string.ascii_lowercase                      # panel labels (a),(b),...
        for i, d in enumerate(distances):
            r, c = i // ncols, i % ncols
            axes[r, c].text(-0.16 if c == 0 else -0.08, 1.09, f"({labs[i]})",
                            transform=axes[r, c].transAxes, fontsize=12, fontweight="bold")
        for ax in flat:                                    # hide unused slots
            if ax not in used:
                ax.set_visible(False)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=used, location="bottom", pad=0.24 if nrows == 1 else 0.13,
                          shrink=0.55, aspect=55, fraction=0.05)
        cb.set_label(r"Reweighting strength $\alpha$")
        cb.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        ts.save(fig, fname)


ts.set_style()
for dists, suffix in ts.VARIANTS:
    make(dists, "fig2b_ler_ratio_vs_p_byalpha" + suffix, nrows=1 if len(dists) <= 3 else 2)
