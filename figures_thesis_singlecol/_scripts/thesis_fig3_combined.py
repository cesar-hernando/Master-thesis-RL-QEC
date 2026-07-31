#!/usr/bin/env python
"""Thesis figs 3a / 3b (formerly the combined 2xN figure), now SPLIT into two figures:

  ler_ratio_vs_p_bydistance   : LER(alpha)/LER(MWPM), alpha* (square, solid) & CM (circle, dashed)
  best_alpha_vs_p_bydistance  : optimal alpha*(p)

Each figure lays the distances out in TWO rows.  Ratios use error COUNTS (CRN).  Variants:
core / +d9 / +d9,d11.  New scan data, TeX look + grid, opaque boxed legends.
"""
import os
import string
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

df = ts.load_scan()


def ratio_series(d, which):
    sub = df[df.distance == d]
    xs, ys, es = [], [], []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        row = ts.best_row(s) if which == "best" else \
            s[(s.decoder == "cm") & (np.abs(s.alpha - 1.0) < 1e-9)].iloc[0]
        r = row.errors / mw.errors
        se = r * np.sqrt(1.0 / max(row.errors, 1) + 1.0 / max(mw.errors, 1))
        xs.append(p); ys.append(r); es.append(se)
    return np.array(xs), np.array(ys), np.array(es)


def alpha_series(d):
    sub = df[df.distance == d]
    xs, ys = [], []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        xs.append(p); ys.append(ts.best_row(s).alpha)
    return np.array(xs), np.array(ys)


def _finish(fig, axes, distances, ncols, nrows, ylabel):
    for k, d in enumerate(distances):
        r, c = k // ncols, k % ncols
        if r == nrows - 1 or (k + ncols) >= len(distances):
            axes[r, c].set_xlabel(r"Physical error rate $p$")
        if c == 0:
            axes[r, c].set_ylabel(ylabel)
    used = [axes[k // ncols, k % ncols] for k in range(len(distances))]
    labs = string.ascii_lowercase
    for i, ax in enumerate(used):
        ts.panel(ax, labs[i], x=-0.22, y=1.04)
    for ax in axes.flat:
        if ax not in used:
            ax.set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=1.0)


def make_ler(distances, fname):
    ncols = int(np.ceil(len(distances) / 2)); nrows = 2
    with mpl.rc_context({}):
        fig, axes = plt.subplots(nrows, ncols, figsize=ts.figsize(nrows, ncols, panel_ratio=0.95),
                                 sharey=True, squeeze=False)
        for k, d in enumerate(distances):
            ax = axes[k // ncols, k % ncols]; c = ts.DCOL[d]
            ax.axhline(1.0, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
            p, r, e = ratio_series(d, "best")
            ax.errorbar(p, r, yerr=e, marker="s", color=c, ls="-", mec="white", mew=0.6,
                        capsize=2.5, elinewidth=1.0, zorder=4, label=r"$\alpha^{*}$")
            p1, r1, e1 = ratio_series(d, 1.0)
            ax.errorbar(p1, r1, yerr=e1, marker="o", color=c, ls=(0, (2, 2)), lw=1.2, ms=5,
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.0, zorder=3, label=r"CM")
            ax.set_title(ts.DLABEL[d]); ax.set_xscale("log"); ax.set_xlim(6e-5, 1.3e-2)
            ts.legend(ax, loc="best")
        _finish(fig, axes, distances, ncols, nrows, r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
        ts.save(fig, fname)


def make_alpha(distances, fname):
    ncols = int(np.ceil(len(distances) / 2)); nrows = 2
    with mpl.rc_context({}):
        fig, axes = plt.subplots(nrows, ncols, figsize=ts.figsize(nrows, ncols, panel_ratio=0.95),
                                 sharey=True, squeeze=False)
        for k, d in enumerate(distances):
            ax = axes[k // ncols, k % ncols]; c = ts.DCOL[d]
            pb, ab = alpha_series(d)
            ax.plot(pb, ab, marker="s", color=c, ls="-", mec="white", mew=0.6, lw=1.4, ms=4.5)
            ax.set_title(ts.DLABEL[d]); ax.set_xscale("log"); ax.set_xlim(6e-5, 1.3e-2)
            ax.set_ylim(0, 1.0)
        _finish(fig, axes, distances, ncols, nrows, r"Optimal $\alpha^{*}$")
        ts.save(fig, fname)


ts.set_style()
for dists, suffix in ts.VARIANTS:
    make_ler(dists, "ler_ratio_vs_p_bydistance" + suffix)
    make_alpha(dists, "best_alpha_vs_p_bydistance" + suffix)
