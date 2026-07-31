#!/usr/bin/env python
"""Thesis figs 6a / 6b (+ 7): the fitted LER-ratio and the fitted optimal-alpha, now SPLIT into
two separate figures (like fig 3), each with the distances in TWO rows:

  ler_ratio_fits_bydistance  : p_L^{a-CM}/p_L^{MWPM} (alpha*-RCM circles) and p_L^{CM}/p_L^{MWPM}
                               (CM squares), each with its fitted law (solid).
  best_alpha_fits_bydistance : optimal alpha*(p) with saturating fit alpha0/(1+(p0/p)^k).

Fitted coefficients go to a separate table figure (ratio_fits_coefficients).  Variants:
core / +d9 / +d9,d11.  New scan data, TeX look + grid.
"""
import os
import string
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

import thesis_style as ts

df = ts.load_scan()
C_DAMPED = "#0072B2"   # alpha*-RCM
C_HARD = "#D55E00"     # hard CM
PL = r"p_{\mathrm{L}}"
Y_RATIO = rf"${PL}\,/\,{PL}^{{\mathrm{{MWPM}}}}$"


def extract(d):
    sub = df[df.distance == d]
    rows = []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        cm1 = s[(s.decoder == "cm") & (np.abs(s.alpha - 1.0) < 1e-9)].iloc[0]
        best = ts.best_row(s)

        def ratio(x):
            r = x.ler / mw.ler
            se = r * np.sqrt((x.ler_std / x.ler) ** 2 + (mw.ler_std / mw.ler) ** 2)
            return r, se
        rb, sb = ratio(best); rc, sc = ratio(cm1)
        rows.append((p, rb, sb, rc, sc, best.alpha))
    a = np.array(rows)
    return dict(p=a[:, 0], r_best=a[:, 1], se_best=a[:, 2], r_cm=a[:, 3], se_cm=a[:, 4], ab=a[:, 5])


def f_best(p, a, b):
    return 1.0 + a + b * p


def f_cm(p, a, b, c):
    return 1.0 + a + b * p + c * np.log(1.0 / p) / p


def f_sat(p, L, p0, m):
    return L / (1.0 + (p0 / p) ** m)


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ssr = np.sum((y - yhat) ** 2); sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - ssr / sst if sst > 0 else float("nan")


def finish(fig, axes, distances, ncols, nrows, ylabel):
    for k in range(len(distances)):
        r, c = k // ncols, k % ncols
        if r == nrows - 1 or (k + ncols) >= len(distances):
            axes[r, c].set_xlabel(r"Physical error rate $p$")
        if c == 0:
            axes[r, c].set_ylabel(ylabel)
    used = [axes[k // ncols, k % ncols] for k in range(len(distances))]
    for i, ax in enumerate(used):
        ts.panel(ax, string.ascii_lowercase[i], x=-0.24, y=1.04)
    for ax in axes.flat:
        if ax not in used:
            ax.set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=1.1)


def make(distances, suffix):
    nrows, ncols = 2, int(np.ceil(len(distances) / 2))
    ptop, pbot = {}, {}
    with mpl.rc_context({"legend.fontsize": ts.BASE - 1.5}):   # dense legend -> a touch smaller
        figA, axesA = plt.subplots(nrows, ncols, figsize=ts.figsize(nrows, ncols, panel_ratio=0.95),
                                   squeeze=False)
        figB, axesB = plt.subplots(nrows, ncols, figsize=ts.figsize(nrows, ncols, panel_ratio=0.95),
                                   sharey=True, squeeze=False)
        for k, d in enumerate(distances):
            axA = axesA[k // ncols, k % ncols]; axB = axesB[k // ncols, k % ncols]
            t = extract(d); p = t["p"]; pg = np.geomspace(p.min(), p.max(), 300)

            # ---- LER ratio fits ----
            axA.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
            axA.errorbar(p, t["r_best"], yerr=t["se_best"], fmt="o", color=C_DAMPED, ms=5,
                         mec="white", mew=0.6, capsize=2.2, elinewidth=0.9, zorder=4)
            axA.errorbar(p, t["r_cm"], yerr=t["se_cm"], fmt="s", color=C_HARD, ms=5,
                         mec="white", mew=0.6, capsize=2.2, elinewidth=0.9, zorder=4)
            pb, _ = curve_fit(f_best, p, t["r_best"], sigma=t["se_best"], absolute_sigma=True, p0=[-0.4, 30.0])
            axA.plot(pg, f_best(pg, *pb), "-", color=C_DAMPED, lw=1.5, zorder=3)
            try:
                pc, _ = curve_fit(f_cm, p, t["r_cm"], sigma=t["se_cm"], absolute_sigma=True,
                                  p0=[-0.4, 30.0, 5e-6], maxfev=40000)
            except RuntimeError:
                pc = [np.nan, np.nan, np.nan]
            axA.plot(pg, f_cm(pg, *pc), "-", color=C_HARD, lw=1.5, zorder=3)
            ptop[d] = (1 + pb[0], pb[1], pc[2], r2(t["r_best"], f_best(p, *pb)), r2(t["r_cm"], f_cm(p, *pc)))
            axA.set_xscale("log"); axA.set_xlim(6e-5, 1.4e-2); axA.set_title(ts.DLABEL[d])

            # ---- alpha*(p) saturating fit ----
            axB.plot(p, t["ab"], "o", color=C_DAMPED, ms=5.5, mec="white", mew=0.6, zorder=4)
            ps, _ = curve_fit(f_sat, p, t["ab"], p0=[0.9, 7e-4, 1.0],
                              bounds=([0.3, 1e-5, 0.2], [1.0, 1e-1, 4.0]), maxfev=40000)
            axB.plot(pg, f_sat(pg, *ps), "-", color=C_DAMPED, lw=1.5, zorder=3)
            axB.axhline(ps[0], color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
            pbot[d] = (ps[0], ps[1], ps[2], r2(t["ab"], f_sat(p, *ps)))
            axB.set_xscale("log"); axB.set_ylim(0, 1.02); axB.set_xlim(6e-5, 1.4e-2); axB.set_title(ts.DLABEL[d])

        h_top = [Line2D([], [], marker="o", ls="none", color=C_DAMPED, mec="white", mew=0.6, ms=6),
                 Line2D([], [], marker="s", ls="none", color=C_HARD, mec="white", mew=0.6, ms=6),
                 Line2D([], [], color=C_DAMPED, ls="-", lw=1.6),
                 Line2D([], [], color=C_HARD, ls="-", lw=1.6)]
        ts.legend(axesA[0, 0], h_top, [r"$\alpha^{*}$-RCM", r"CM", r"fit $1{+}a{+}bp$",
                  r"fit $1{+}a{+}bp{+}c\,\ln(1/p)/p$"], loc="best", labelspacing=0.4, handlelength=1.7)
        h_bot = [Line2D([], [], marker="o", ls="none", color=C_DAMPED, mec="white", mew=0.6, ms=6),
                 Line2D([], [], color=C_DAMPED, ls="-", lw=1.6)]
        ts.legend(axesB[0, 0], h_bot, [r"swept $\alpha^{*}$", r"fit $\alpha_0/(1+(p_0/p)^k)$"],
                  loc="best", labelspacing=0.4, handlelength=1.7)
        finish(figA, axesA, distances, ncols, nrows, Y_RATIO)
        finish(figB, axesB, distances, ncols, nrows, r"Optimal reweighting strength $\alpha^{*}$")
        ts.save(figA, "ler_ratio_fits_bydistance" + suffix)
        ts.save(figB, "best_alpha_fits_bydistance" + suffix)

    # ---------------- coefficients table ----------------
    figt, axt = plt.subplots(2, 1, figsize=(ts.TEXTWIDTH, 1.5 + 0.55 * len(distances)))
    figt.subplots_adjust(hspace=0.75, top=0.90, bottom=0.05, left=0.04, right=0.96)
    for a in axt:
        a.axis("off")
    axt[0].set_title(r"LER-ratio fits:   "
                     r"$p_{\mathrm{L}}^{\alpha\mathrm{-CM}}/p_{\mathrm{L}}^{\mathrm{MWPM}} = 1{+}a{+}bp$"
                     r"     and     "
                     r"$p_{\mathrm{L}}^{\mathrm{CM}}/p_{\mathrm{L}}^{\mathrm{MWPM}} = 1{+}a{+}bp{+}c\,\ln(1/p)/p$",
                     fontsize=ts.BASE - 1, pad=18)
    col_top = [r"$d$", r"$1+a$", r"$b$", r"$c\;(10^{-6})$", r"$R^2_{\alpha}$", r"$R^2_{\mathrm{CM}}$"]
    cells_top = [[f"{d}", f"{ptop[d][0]:.2f}", f"{ptop[d][1]:.0f}", f"{ptop[d][2]*1e6:.1f}",
                  f"{ptop[d][3]:.3f}", f"{ptop[d][4]:.3f}"] for d in distances]
    tb0 = axt[0].table(cellText=cells_top, colLabels=col_top, loc="center", cellLoc="center",
                       bbox=[0.0, 0.0, 1.0, 0.82])
    tb0.auto_set_font_size(False); tb0.set_fontsize(ts.BASE - 1)
    axt[1].set_title(r"Optimal reweighting strength fit:   $\alpha^{*}(p)=\alpha_0\,/\,(1+(p_0/p)^k)$",
                     fontsize=ts.BASE - 1, pad=18)
    col_bot = [r"$d$", r"$\alpha_0$", r"$p_0\;(10^{-4})$", r"$k$", r"$R^2$"]
    cells_bot = [[f"{d}", f"{pbot[d][0]:.2f}", f"{pbot[d][1]*1e4:.0f}", f"{pbot[d][2]:.1f}",
                  f"{pbot[d][3]:.3f}"] for d in distances]
    tb1 = axt[1].table(cellText=cells_bot, colLabels=col_bot, loc="center", cellLoc="center",
                       bbox=[0.0, 0.0, 1.0, 0.82])
    tb1.auto_set_font_size(False); tb1.set_fontsize(ts.BASE - 1)
    ts.save(figt, "ratio_fits_coefficients" + suffix)


ts.set_style()
for dists, suffix in ts.VARIANTS:
    make(dists, suffix)
