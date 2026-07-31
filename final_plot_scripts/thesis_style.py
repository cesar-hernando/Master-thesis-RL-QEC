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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_thesis")
COMBINED_CSV = os.path.join(ROOT, "data", "reg_cm_alpha_scan_new_combined.csv")

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


def set_style(base=10.0):
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "text.usetex": USETEX,
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "font.size": base, "axes.labelsize": base + 1, "axes.titlesize": base + 1,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        # grid
        "axes.grid": True, "axes.grid.which": "major",
        "grid.color": "0.8", "grid.linewidth": 0.6, "grid.alpha": 0.6, "grid.linestyle": "-",
    })


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
    print(f"saved figures_thesis/{{pdf,png}}/{name}")


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
