#!/usr/bin/env python
"""Thesis comparison figure: logical error rate of Regularized CM (alpha*), belief-matching,
Tesseract and Neural CM, EACH normalised by its own dataset's MWPM baseline (so the shots /
conditions cancel).  Panels d=5 and d=7.  Two options: with / without NCM.

Every dataset carries its own MWPM: scan (mwpm rows), belief-matching (bp_iters='mwpm'),
Tesseract (mwpm_mean), Neural CM (static_mean).  The MWPM baseline is the dashed line at 1.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

D = os.path.join(ts.ROOT, "data")
scan = ts.load_scan()

# method -> (colour, marker, label)
M = {
    "regcm": ("#0072B2", "s", "Regularized CM"),
    "cm":    ("#CC79A7", "o", "CM"),
    "bm":    ("#009E73", "^", "Belief-matching"),
    "tess":  ("#D55E00", "D", "Tesseract"),
    "ncm":   ("#785EF0", "v", "Neural CM"),
}


def _ratio(mean_dec, std_dec, mean_mw, std_mw):
    r = mean_dec / mean_mw
    se = r * np.sqrt((std_dec / mean_dec) ** 2 + (std_mw / mean_mw) ** 2)
    return r, se


def scan_ratio(d, which="best"):
    sub = scan[scan.distance == d]; xs, ys, es = [], [], []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]; mw = s[s.decoder == "mwpm"].iloc[0]
        row = ts.best_row(s) if which == "best" else \
            s[(s.decoder == "cm") & (np.abs(s.alpha - 1.0) < 1e-9)].iloc[0]
        r, se = _ratio(row.ler, row.ler_std, mw.ler, mw.ler_std)
        xs.append(p); ys.append(r); es.append(se)
    return np.array(xs), np.array(ys), np.array(es)


def bm_ratio(d):
    df = pd.read_csv(os.path.join(D, f"ler_vs_p_beliefmatching_d{d}_extended.csv"))
    b = df[df.bp_iters == "5"].set_index("p"); m = df[df.bp_iters == "mwpm"].set_index("p")
    xs, ys, es = [], [], []
    for p in sorted(b.index):
        r, se = _ratio(b.ler_mean[p], b.ler_std[p], m.ler_mean[p], m.ler_std[p])
        xs.append(p); ys.append(r); es.append(se)
    return np.array(xs), np.array(ys), np.array(es)


def tess_ratio():
    df = pd.concat([pd.read_csv(os.path.join(D, "ler_vs_p_tesseract_d5.csv")),
                    pd.read_csv(os.path.join(D, "ler_vs_p_tesseract_d5_highp.csv"))]).sort_values("p")
    r, se = _ratio(df.tess_mean.to_numpy(), df.tess_std.to_numpy(),
                   df.mwpm_mean.to_numpy(), df.mwpm_std.to_numpy())
    return df.p.to_numpy(), r, se


def ncm_ratio():
    df = pd.read_csv(os.path.join(D, "ler_vs_p_slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.csv")).sort_values("p")
    r, se = _ratio(df.neural_mean.to_numpy(), df.neural_std.to_numpy(),
                   df.static_mean.to_numpy(), df.static_std.to_numpy())
    return df.p.to_numpy(), r, se


def plot_method(ax, key, x, y, e):
    c, mk, lab = M[key]
    ax.errorbar(x, y, yerr=e, marker=mk, ls="-", color=c, ms=4.5, mec="white", mew=0.6,
                capsize=2.0, elinewidth=0.8, lw=1.3, label=lab)


def make(with_ncm):
    fig, (a5, a7) = plt.subplots(1, 2, figsize=ts.figsize(1, 2, panel_ratio=1.0, extra_h=0.7))
    for ax in (a5, a7):
        ax.axhline(1.0, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1, label="MWPM")
    # d=5
    plot_method(a5, "regcm", *scan_ratio(5, "best"))
    plot_method(a5, "cm", *scan_ratio(5, "cm1"))
    plot_method(a5, "bm", *bm_ratio(5))
    plot_method(a5, "tess", *tess_ratio())
    if with_ncm:
        plot_method(a5, "ncm", *ncm_ratio())
    a5.set_title(r"$d=5$")
    # d=7
    plot_method(a7, "regcm", *scan_ratio(7, "best"))
    plot_method(a7, "cm", *scan_ratio(7, "cm1"))
    plot_method(a7, "bm", *bm_ratio(7))
    a7.set_title(r"$d=7$")

    for ax, lab in zip((a5, a7), "ab"):
        ax.set_xscale("log")
        ax.set_xlabel(r"Physical error rate $p$")
        ax.set_ylabel(r"$p_{\mathrm{L}}\,/\,p_{\mathrm{L}}^{\mathrm{MWPM}}$")
        ts.panel(ax, lab, x=-0.20)
    h, l = a5.get_legend_handles_labels()      # a5 carries all methods (+ MWPM line)
    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(top=0.80)
    leg = fig.legend(h, l, loc="upper center", ncol=3, frameon=True, framealpha=1.0,
                     edgecolor="0.7", fontsize=ts.BASE, bbox_to_anchor=(0.5, 1.0))
    leg.get_frame().set_linewidth(0.8)
    ts.save(fig, "decoder_comparison_ler_d5_d7" + ("" if with_ncm else "_no_ncm"))


ts.set_style()
make(with_ncm=True)
make(with_ncm=False)
