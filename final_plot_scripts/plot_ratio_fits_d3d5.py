#!/usr/bin/env python
"""Paper figure: damped vs hard correlated matching relative to MWPM, d = 3 and 5.

Uses the new reg_cm_alpha_scan data (data/reg_cm_alpha_scan_d3d5.csv).

Panels (a,b): logical-error-rate ratios to MWPM with fitted laws
    alpha-CM (best alpha):  p_L^{a-CM}/p_L^{MWPM} = 1 + a + b p
    hard  CM (alpha = 1) :  p_L^{CM}  /p_L^{MWPM} = 1 + a + b p + h ln(1/p)/p
Panels (c,d): optimal reweighting strength alpha*(p) with the saturating law
    L / (1 + (p0/p)^m).

Style matches final_plots/ (closed box, inward ticks, Wong colours, no grid).
Fit statistics are printed to stdout for the caption.
Output: final_plots/ratio_fits_d3_d5.pdf (vector) and .png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "reg_cm_alpha_scan_d3d5d7.csv")
OUT = os.path.join(ROOT, "final_plots")
DISTANCES = (3, 5, 7)

C_DAMPED = "#0072B2"   # alpha-CM (best alpha)
C_HARD = "#D55E00"     # hard CM (alpha = 1)
BASE = "0.55"          # MWPM baseline

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
    "legend.frameon": False, "legend.fontsize": 7.8, "legend.handlelength": 1.9,
    "lines.linewidth": 1.8, "mathtext.fontset": "dejavusans",
})

# p_L notation building blocks
PL = r"p_{\mathrm{L}}"
Y_RATIO = rf"${PL}\,/\,{PL}^{{\mathrm{{MWPM}}}}$"
LAB_ACM = rf"${PL}^{{\alpha\mathrm{{-CM}}}}/{PL}^{{\mathrm{{MWPM}}}}$"
LAB_CM = rf"${PL}^{{\mathrm{{CM}}}}/{PL}^{{\mathrm{{MWPM}}}}$"

df = pd.read_csv(CSV)


def extract(d):
    sub = df[df.distance == d]
    rows = []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        cm = s[s.decoder == "cm"]
        cm1 = cm[cm.alpha == 1.0].iloc[0]
        best = cm[cm.alpha < 1.0].loc[cm[cm.alpha < 1.0].ler.idxmin()]

        def ratio(x):
            r = x.ler / mw.ler
            se = r * np.sqrt((x.ler_std / x.ler) ** 2 + (mw.ler_std / mw.ler) ** 2)
            return r, se
        rb, sb = ratio(best)
        rc, sc = ratio(cm1)
        rows.append(dict(p=p, r_best=rb, se_best=sb, r_cm=rc, se_cm=sc,
                         alpha_best=best.alpha))
    return pd.DataFrame(rows)


def f_best(p, a, b):
    return 1.0 + a + b * p


def f_cm(p, a, b, c):
    return 1.0 + a + b * p + c * np.log(1.0 / p) / p


def f_sat(p, L, p0, m):
    return L / (1.0 + (p0 / p) ** m)


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, len(DISTANCES), figsize=(3.7 * len(DISTANCES), 5.6))
pg = np.logspace(-4.15, -1.95, 400)

ptop, pbot = {}, {}   # fitted params per distance, for the shared legends

for col, d in enumerate(DISTANCES):
    t = extract(d)
    p = t.p.to_numpy()

    # ------------------------------ ratios (a, b) -----------------------------------
    ax = axes[0, col]
    ax.axhline(1.0, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(t.p, t.r_best, yerr=t.se_best, fmt="o", color=C_DAMPED, ms=5,
                mec="white", mew=0.6, capsize=2.2, elinewidth=0.9, zorder=4)
    ax.errorbar(t.p, t.r_cm, yerr=t.se_cm, fmt="s", color=C_HARD, ms=5,
                mec="white", mew=0.6, capsize=2.2, elinewidth=0.9, zorder=4)

    pb, cb = curve_fit(f_best, p, t.r_best, sigma=t.se_best, absolute_sigma=True,
                       p0=[-0.4, 30.0])
    eb = np.sqrt(np.diag(cb))
    ax.plot(pg, f_best(pg, *pb), "-", color=C_DAMPED, lw=1.5, zorder=3)
    pc, cc = curve_fit(f_cm, p, t.r_cm, sigma=t.se_cm, absolute_sigma=True,
                       p0=[-0.4, 30.0, 5e-6], maxfev=20000)
    ec = np.sqrt(np.diag(cc))
    ax.plot(pg, f_cm(pg, *pc), "-", color=C_HARD, lw=1.5, zorder=3)
    ptop[d] = (1 + pb[0], pb[1], pc[2],
               r2(t.r_best, f_best(p, *pb)), r2(t.r_cm, f_cm(p, *pc)))

    chi_b = np.sum(((t.r_best - f_best(p, *pb)) / t.se_best) ** 2) / (len(p) - 2)
    chi_c = np.sum(((t.r_cm - f_cm(p, *pc)) / t.se_cm) ** 2) / (len(p) - 3)
    print(f"d={d}  a-CM : 1+a = {1+pb[0]:.3f}  b = {pb[1]:.2f} +- {eb[1]:.2f}"
          f"   chi2/dof = {chi_b:.2f}")
    print(f"d={d}  CM   : c = {pc[2]:.2e} +- {ec[2]:.1e}   chi2/dof = {chi_c:.2f}")
    try:
        pstar = brentq(lambda q: pc[0] + pc[1] * q + pc[2] * np.log(1 / q) / q, 1e-6, 5e-3)
        print(f"d={d}  fitted CM = MWPM crossover p* = {pstar:.2e}")
    except ValueError:
        print(f"d={d}  crossover not bracketed in [1e-6, 5e-3]")

    lo = min(t.r_best.min(), t.r_cm.min())
    hi = max(t.r_best.max(), t.r_cm.max())
    ax.set_xscale("log")
    ax.set_ylim(lo - 0.06, max(hi, 1.02) + 0.08)
    ax.set_xlim(7e-5, 1.4e-2)
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(Y_RATIO if col == 0 else "")
    ax.set_title(rf"$d = {d}$", pad=4)

    # --------------------------- alpha*(p) (c, d) -----------------------------------
    ax = axes[1, col]
    ax.plot(t.p, t.alpha_best, "o", color=C_DAMPED, ms=5.5, mec="white", mew=0.6, zorder=4)
    ps, cs = curve_fit(f_sat, p, t.alpha_best, p0=[0.8, 7e-4, 1.0],
                       bounds=([0.3, 1e-5, 0.2], [1.0, 1e-1, 4.0]), maxfev=20000)
    es = np.sqrt(np.diag(cs))
    ax.plot(pg, f_sat(pg, *ps), "-", color=C_DAMPED, lw=1.5, zorder=3)
    ax.axhline(ps[0], color=BASE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    pbot[d] = (ps[0], ps[1], ps[2], r2(t.alpha_best, f_sat(p, *ps)))
    print(f"d={d}  alpha*: alpha0 = {ps[0]:.3f}  p0 = {ps[1]:.2e}  k = {ps[2]:.2f} +- {es[2]:.2f}\n")

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(7e-5, 1.4e-2)
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"Optimal reweighting strength $\alpha^{*}$" if col == 0 else "")

for ax, lab in zip(axes.flat, "abcdefgh"):
    ax.text(-0.24 if ax in axes[:, 0] else -0.12, 1.04, f"({lab})",
            transform=ax.transAxes, fontsize=11, fontweight="bold")

# =========================== in-figure legends (descriptive only) ====================
# The numeric fitted-coefficient values live in a separate PDF (below); the panels carry
# only the marker / fit-form key, placed over empty regions of the d=7 (top) and
# d=3 (bottom) panels.
FRAME = dict(frameon=True, framealpha=0.92, edgecolor="0.8", facecolor="white", fancybox=False)

h_top = [Line2D([], [], marker="o", ls="none", color=C_DAMPED, mec="white", mew=0.6, ms=6),
         Line2D([], [], marker="s", ls="none", color=C_HARD, mec="white", mew=0.6, ms=6),
         Line2D([], [], color=C_DAMPED, ls="-", lw=1.6),
         Line2D([], [], color=C_HARD, ls="-", lw=1.6)]
l_top = [r"$\alpha^{*}$-RCM", r"CM",
         r"fit $1{+}a{+}bp$", r"fit $1{+}a{+}bp{+}c\,\ln(1/p)/p$"]
axes[0, 0].legend(h_top, l_top, loc="upper center", ncol=2, fontsize=7.4, labelspacing=0.4,
                  columnspacing=1.2, handlelength=1.7, **FRAME)

h_bot = [Line2D([], [], marker="o", ls="none", color=C_DAMPED, mec="white", mew=0.6, ms=6),
         Line2D([], [], color=C_DAMPED, ls="-", lw=1.6)]
l_bot = [r"swept $\alpha^{*}$", r"fit $\alpha_0/(1+(p_0/p)^k)$"]
axes[1, 0].legend(h_bot, l_bot, loc="lower right", fontsize=8.2, labelspacing=0.4,
                  handlelength=1.7, **FRAME)

fig.tight_layout(h_pad=1.6, w_pad=1.8)

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    out = os.path.join(OUT, f"ratio_fits_d3_d5_d7.{ext}")
    fig.savefig(out, dpi=300)
    print(f"saved {out}")

# ======================= separate PDF: fitted coefficient values =====================
figt, axt = plt.subplots(2, 1, figsize=(7.4, 3.6))
for a in axt:
    a.axis("off")

axt[0].set_title(r"LER-ratio fits:   "
                 r"$p_{\mathrm{L}}^{\alpha\mathrm{-CM}}/p_{\mathrm{L}}^{\mathrm{MWPM}} = 1+a+bp$"
                 r"     and     "
                 r"$p_{\mathrm{L}}^{\mathrm{CM}}/p_{\mathrm{L}}^{\mathrm{MWPM}} = 1+a+bp+c\,\ln(1/p)/p$",
                 fontsize=8.5, pad=8)
col_top = [r"$d$", r"$1+a$", r"$b$", r"$c\;(10^{-6})$", r"$R^2_{\alpha}$", r"$R^2_{\mathrm{CM}}$"]
cells_top = [[f"{dd}", f"{ptop[dd][0]:.2f}", f"{ptop[dd][1]:.0f}", f"{ptop[dd][2]*1e6:.1f}",
              f"{ptop[dd][3]:.3f}", f"{ptop[dd][4]:.3f}"] for dd in DISTANCES]
tb0 = axt[0].table(cellText=cells_top, colLabels=col_top, loc="center", cellLoc="center")
tb0.auto_set_font_size(False); tb0.set_fontsize(9.5); tb0.scale(1, 1.55)

axt[1].set_title(r"Optimal reweighting strength fit:   $\alpha^{*}(p) = \alpha_0\,/\,(1+(p_0/p)^k)$",
                 fontsize=8.5, pad=8)
col_bot = [r"$d$", r"$\alpha_0$", r"$p_0\;(10^{-4})$", r"$k$", r"$R^2$"]
cells_bot = [[f"{dd}", f"{pbot[dd][0]:.2f}", f"{pbot[dd][1]*1e4:.0f}", f"{pbot[dd][2]:.1f}",
              f"{pbot[dd][3]:.3f}"] for dd in DISTANCES]
tb1 = axt[1].table(cellText=cells_bot, colLabels=col_bot, loc="center", cellLoc="center")
tb1.auto_set_font_size(False); tb1.set_fontsize(9.5); tb1.scale(1, 1.55)

figt.tight_layout(h_pad=2.5)
for ext in ("pdf", "png"):
    out = os.path.join(OUT, f"ratio_fits_d3_d5_d7_coefficients.{ext}")
    figt.savefig(out, dpi=300)
    print(f"saved {out}")
