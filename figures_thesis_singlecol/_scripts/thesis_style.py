#!/usr/bin/env python
"""Shared style + data helpers for the thesis figure set (figures_thesis/).

LaTeX (usetex) is not installed on this machine, so the "TeX look" is emulated with
Computer Modern math (mathtext.fontset='cm') + a serif text font (cmr10, shipped with
matplotlib). Grid lines are on by default. Swap `text.usetex` to True in set_style if a
LaTeX toolchain becomes available.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# this copy lives in figures_thesis_singlecol/_scripts, so the repo root is THREE levels up,
# and the single-column figures are written to figures_thesis_singlecol/{pdf,png}.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figures_thesis_singlecol")
COMBINED_CSV = os.path.join(ROOT, "data", "reg_cm_alpha_scan_new_combined.csv")

# Single-column text width of the thesis (inches).  Every figure is authored at this width (or a
# fraction of it) and included in LaTeX at \textwidth with NO scaling, so the point sizes set below
# are exactly what appears on the page and are identical across figures.
TEXTWIDTH = 5.0
BASE = 9.0        # single uniform font size (pt) shared by every figure


def figsize(nrows=1, ncols=1, panel_ratio=0.92, w_frac=1.0, extra_h=0.0):
    """Figure size that fills (a fraction of) the text width.  `panel_ratio` = panel height/width;
    `extra_h` adds inches for a shared colorbar/legend row."""
    w = TEXTWIDTH * w_frac
    h = (w / ncols) * panel_ratio * nrows + extra_h
    return (w, h)

# distance colours (Wong/Tol colour-blind-safe, no black) + markers, extended to d = 9, 11
DCOL = {3: "#0072B2", 5: "#D55E00", 7: "#009E73", 9: "#CC79A7", 11: "#785EF0"}
DMARK = {3: "o", 5: "s", 7: "^", 9: "D", 11: "v"}
DLABEL = {d: rf"$d={d}$" for d in (3, 5, 7, 9, 11)}
CORE = [3, 5, 7]
EXT9 = [3, 5, 7, 9]        # with d=9 but not d=11
EXT = [3, 5, 7, 9, 11]     # with d=9 and d=11

# (distances, filename-suffix) variants produced for every multi-distance figure
VARIANTS = [(CORE, ""), (EXT9, "_with_d9"), (EXT, "_with_d9_d11")]
GRID_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

USETEX = False   # LaTeX toolchain not available; emulate with Computer Modern


def set_style(base=BASE):
    """Single-column thesis style: serif (Computer Modern), ONE uniform font size everywhere, thin
    axes/ticks, light grid.  Figures are authored at ts.TEXTWIDTH and included at \\textwidth with no
    scaling, so `base` pt is the true on-page size and is the same in every figure."""
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 400, "savefig.bbox": "tight",
        "text.usetex": USETEX,
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        # one uniform size for text, labels, ticks, titles and legends
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base, "ytick.labelsize": base,
        "legend.fontsize": base, "figure.titlesize": base,
        # thin lines / short outward ticks (single-column look)
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.spines.top": True, "axes.spines.right": True,
        "legend.frameon": False,
        "lines.linewidth": 1.2, "lines.markersize": 4.2,
        "errorbar.capsize": 2.0,
        # light thin grid
        "axes.grid": True, "axes.grid.which": "major",
        "grid.color": "0.85", "grid.linewidth": 0.4, "grid.alpha": 0.7, "grid.linestyle": "-",
    })


# panel label (a),(b),... in a consistent size/weight across every figure
def panel(ax, letter, x=-0.17, y=1.03):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=BASE + 1, fontweight="bold")


# boxed legend on a fully-opaque white background so it never shows grid lines / data through it
LEGKW = dict(frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.7", fancybox=False)


def legend(ax, *args, **kw):
    kw = {**LEGKW, **kw}
    leg = ax.legend(*args, **kw)
    if leg is not None:
        leg.set_zorder(20)
        leg.get_frame().set_linewidth(0.8)
    return leg


def save(fig, name, close=True):
    for ext in ("pdf", "png"):
        d = os.path.join(OUT, ext); os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"saved {os.path.basename(OUT)}/{{pdf,png}}/{name}")


def load_scan():
    """Return df with the new combined scan (distances 3,5,7,9,11)."""
    return pd.read_csv(COMBINED_CSV)


def best_row(sub_p):
    """Row of the best damped alpha (min LER over alpha<1) for a single-p slice."""
    cm = sub_p[(sub_p.decoder == "cm") & (sub_p.alpha < 1.0)]
    return cm.loc[cm.ler.idxmin()]


if __name__ == "__main__":
    # quick style smoke-test
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    set_style()
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.logspace(-4, -2, 20)
    ax.plot(x, x ** 0.5, marker="o", label=r"$\alpha^{*}$-RCM")
    ax.plot(x, x ** 0.4, marker="s", label=r"CM")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"$p_{\mathrm{L}}/p_{\mathrm{L}}^{\mathrm{MWPM}}$")
    ax.legend()
    save(fig, "_style_smoketest")
    print("distances in CSV:", sorted(load_scan().distance.unique()))
