#!/usr/bin/env python
"""Paper figure: damped vs hard correlated matching relative to MWPM, d = 5 and 7.

Panels (a,b): LER ratios to MWPM with fitted laws
    damped (best-alpha):  R(p) = 1 + a + b p                      (benefit sector)
    hard   (alpha = 1) :  R(p) = 1 + a + b p + h ln(1/p)/p        (+ distance-loss channel)
Panels (c,d): optimal damping alpha*(p) with the saturating law L / (1 + (p0/p)^m).

Data: data/alpha_sweep_extended_fixed_MWPM.csv (+ d7 low-p extension).
Output: plots/ratio_fits_d5_d7.pdf (vector, for the paper) and .png (preview).
Fit statistics (chi2/dof etc.) are printed to stdout for the caption.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- palette (validated reference instance; color follows the entity) --------------
C_DAMPED = "#2a78d6"     # categorical slot 1 (blue):  damped CM and its alpha* knob
C_HARD = "#e34948"       # categorical slot 6 (red):   hard CM (alpha = 1)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.solid_capstyle": "round",
})

df = pd.concat([
    pd.read_csv(os.path.join(ROOT, "data", "alpha_sweep_extended_fixed_MWPM.csv")),
    pd.read_csv(os.path.join(ROOT, "data", "alpha_sweep_extended_fixed_MWPM_d7_lowp.csv")),
], ignore_index=True)


def extract(d):
    sub = df[df.distance == d]
    rows = []
    for p in sorted(sub.p.unique()):
        s = sub[sub.p == p]
        mw = s[s.decoder == "mwpm"].iloc[0]
        cm = s[s.decoder == "cm"]
        cm1 = cm[cm.alpha == 1.0].iloc[0]
        best = cm.loc[cm.ler.idxmin()]
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


def f_cm(p, a, b, h):
    return 1.0 + a + b * p + h * np.log(1.0 / p) / p


def f_sat(p, L, p0, m):
    return L / (1.0 + (p0 / p) ** m)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.3))
pg = np.logspace(-4.15, -1.95, 400)

for col, d in enumerate((5, 7)):
    t = extract(d)
    p = t.p.to_numpy()

    # ------------------------------ ratios (a, b) -----------------------------------
    ax = axes[0, col]
    ax.axhline(1.0, color=BASE, lw=0.9, zorder=1)
    ax.text(1.15e-2, 1.0, "MWPM", color=MUTED, fontsize=7, va="center", ha="left")

    ax.errorbar(t.p, t.r_best, yerr=t.se_best, fmt="o", color=C_DAMPED, ms=4,
                mew=0, capsize=1.8, elinewidth=0.8, zorder=4,
                label=r"damped CM (best $\alpha$)")
    ax.errorbar(t.p, t.r_cm, yerr=t.se_cm, fmt="s", color=C_HARD, ms=4,
                mew=0, capsize=1.8, elinewidth=0.8, zorder=4,
                label=r"hard CM ($\alpha=1$)")

    pb, cb = curve_fit(f_best, p, t.r_best, sigma=t.se_best, absolute_sigma=True,
                       p0=[-0.4, 30.0])
    eb = np.sqrt(np.diag(cb))
    ax.plot(pg, f_best(pg, *pb), "-", color=C_DAMPED, lw=1.4, zorder=3,
            label=rf"$1+a+bp$;  $1{{+}}a={1+pb[0]:.2f}$, $b={pb[1]:.0f}$")

    pc, cc = curve_fit(f_cm, p, t.r_cm, sigma=t.se_cm, absolute_sigma=True,
                       p0=[-0.4, 30.0, 5e-6], maxfev=20000)
    ec = np.sqrt(np.diag(cc))
    ax.plot(pg, f_cm(pg, *pc), "--", color=C_HARD, lw=1.4, zorder=3,
            label=rf"$1+a+bp+h\,\frac{{\ln(1/p)}}{{p}}$;  $h={pc[2]*1e6:.1f}{{\times}}10^{{-6}}$")

    chi_b = np.sum(((t.r_best - f_best(p, *pb)) / t.se_best) ** 2) / (len(p) - 2)
    chi_c = np.sum(((t.r_cm - f_cm(p, *pc)) / t.se_cm) ** 2) / (len(p) - 3)
    print(f"d={d}  damped: a = {pb[0]:.4f} +- {eb[0]:.4f}  b = {pb[1]:.2f} +- {eb[1]:.2f}"
          f"   plateau 1+a = {1+pb[0]:.3f}   chi2/dof = {chi_b:.2f}")
    print(f"d={d}  hard  : a = {pc[0]:.4f} +- {ec[0]:.4f}  b = {pc[1]:.2f} +- {ec[1]:.2f}"
          f"  h = {pc[2]:.2e} +- {ec[2]:.1e}   chi2/dof = {chi_c:.2f}")

    # crossover of the fitted hard-CM law with MWPM (R = 1)
    try:
        pstar = brentq(lambda q: pc[0] + pc[1] * q + pc[2] * np.log(1 / q) / q,
                       1e-6, 5e-3)
        print(f"d={d}  fitted hard-CM = MWPM crossover p* = {pstar:.2e}")
        if p.min() <= pstar <= p.max():
            ax.plot([pstar], [1.0], marker="v", color=C_HARD, ms=5, zorder=5,
                    clip_on=False)
            ax.annotate(rf"$p^{{*}}\!\approx\!{pstar*1e4:.1f}{{\times}}10^{{-4}}$",
                        (pstar, 1.0), xytext=(-4, -7), textcoords="offset points",
                        ha="right", va="top", fontsize=7, color=INK2)
    except ValueError:
        print(f"d={d}  crossover not bracketed in [1e-6, 5e-3]")

    lo = min(t.r_best.min(), t.r_cm.min())
    hi = max(t.r_best.max(), t.r_cm.max())
    ax.set_xscale("log")
    # keep the MWPM reference line clear of the axis top edge
    ax.set_ylim(lo - 0.10, max(hi + 0.22, 1.08))
    ax.set_xlabel(r"physical error rate $p$")
    ax.set_ylabel("LER / LER(MWPM)" if col == 0 else "")
    ax.set_title(rf"$d = {d}$", pad=4)
    leg = ax.legend(loc="upper center", handlelength=1.8,
                    borderaxespad=0.2, labelspacing=0.35,
                    frameon=True, facecolor="white", edgecolor="none",
                    framealpha=0.9)
    despine(ax)

    # --------------------------- alpha*(p) (c, d) -----------------------------------
    ax = axes[1, col]
    ax.plot(t.p, t.alpha_best, "o", color=C_DAMPED, ms=4.5, mew=0, zorder=4,
            label=r"swept $\alpha^{*}$")
    ps, cs = curve_fit(f_sat, p, t.alpha_best, p0=[0.8, 7e-4, 1.0],
                       bounds=([0.3, 1e-5, 0.2], [1.2, 1e-1, 4.0]), maxfev=20000)
    es = np.sqrt(np.diag(cs))
    ax.plot(pg, f_sat(pg, *ps), "-", color=C_DAMPED, lw=1.4, zorder=3,
            label=(rf"$L/(1+(p_0/p)^m)$;"
                   rf"  $L={ps[0]:.2f}$, $p_0={ps[1]*1e4:.0f}{{\times}}10^{{-4}}$, $m={ps[2]:.1f}$"))
    ax.axhline(ps[0], color=BASE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    print(f"d={d}  alpha*: L = {ps[0]:.3f} +- {es[0]:.3f}  p0 = {ps[1]:.2e} +- {es[1]:.1e}"
          f"  m = {ps[2]:.2f} +- {es[2]:.2f}")
    print()

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"physical error rate $p$")
    ax.set_ylabel(r"optimal damping $\alpha^{*}$" if col == 0 else "")
    ax.legend(loc="upper left", handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.35,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    despine(ax)

# panel labels
for ax, lab in zip(axes.flat, "abcd"):
    ax.text(-0.16 if ax in axes[:, 0] else -0.10, 1.04, f"({lab})",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)

fig.tight_layout(h_pad=1.4, w_pad=1.6)
os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "plots", f"ratio_fits_d5_d7.{ext}")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"saved {out}")
