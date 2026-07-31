#!/usr/bin/env python
"""Publication figures for the Regularized-CM alpha scan (distances 3 and 5).

Reads the per-(d,p) reg_cm_alpha_scan CSVs, writes a single combined CSV, and produces:

  fig1  LER(alpha_best)/LER(MWPM) vs p          -- combined (one axes) and 1x2 subplots
  fig2  LER(alpha)/LER(MWPM) vs alpha, per p    -- 1x2 subplots (d=3 | d=5)
  fig3  best alpha vs p                         -- combined (one axes) and 1x2 subplots

Each figure is saved as both .pdf (vector, for LaTeX) and .png (300 dpi).
"""
import csv
import glob
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_GLOB = os.path.join(ROOT, "data", "reg_cm_alpha_scan", "*d{d}*.csv")
COMBINED_CSV = os.path.join(ROOT, "data", "reg_cm_alpha_scan_d3d5d7.csv")
OUT = os.path.join(ROOT, "final_plots")
DISTANCES = [3, 5, 7]
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # damped alphas on the scan grid

# distance colours (Wong colour-blind-safe) and per-distance marker
DCOL = {3: "#0072B2", 5: "#D55E00", 7: "#009E73"}
DMARK = {3: "o", 5: "s", 7: "^"}
DLABEL = {3: r"$d=3$", 5: r"$d=5$", 7: r"$d=7$"}
PCMAP = plt.get_cmap("viridis")


# =============================================================================
# Load
# =============================================================================
def load():
    """data[d] = dict(p -> dict(mwpm=err, mwpm_ler, shots, cm={alpha: (err, ler)}))."""
    data = {}
    all_rows = []
    for d in DISTANCES:
        data[d] = {}
        for f in sorted(glob.glob(SCAN_GLOB.format(d=d))):
            rows = list(csv.DictReader(open(f)))
            all_rows += rows
            for r in rows:
                p = float(r["p"])
                rec = data[d].setdefault(p, dict(shots=int(r["shots"]), cm={}))
                if r["decoder"] == "mwpm":
                    rec["mwpm"] = int(r["errors"])
                    rec["mwpm_ler"] = float(r["ler"])
                elif r["decoder"] == "cm" and r["alpha"]:
                    rec["cm"][round(float(r["alpha"]), 4)] = (int(r["errors"]), float(r["ler"]))
    return data, all_rows


def write_combined(all_rows):
    fields = ["distance", "rounds", "p", "decoder", "alpha", "shots", "errors",
              "ler", "ler_std", "decode_seconds", "us_per_shot", "is_best", "best_at_endpoint"]
    with open(COMBINED_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: (int(r["distance"]), float(r["p"]),
                                                 r["decoder"], r["alpha"] or "")):
            w.writerow(r)
    print(f"wrote {COMBINED_CSV}  ({len(all_rows)} rows)")


# =============================================================================
# Derived quantities
# =============================================================================
def best_alpha(rec):
    """Best damped alpha (argmin LER over the 0.1..0.9 grid) and its (err, ler)."""
    damped = {a: rec["cm"][a] for a in GRID if a in rec["cm"]}
    a = min(damped, key=lambda a: damped[a][1])
    return a, damped[a]


def ratio_err(e_num, e_den):
    """Relative error of the count ratio (independent-Poisson upper bound; CRN makes the
    real error smaller since numerator and denominator share the same shots)."""
    if e_num == 0 or e_den == 0:
        return 0.0
    return (1.0 / e_num + 1.0 / e_den) ** 0.5


def optimum_band(rec):
    """[alpha_lo, alpha_hi] of damped alphas whose LER is within ~1 sigma of the minimum
    (shows how broad/flat the optimum is)."""
    damped = {a: rec["cm"][a] for a in GRID if a in rec["cm"]}
    shots = rec["shots"]
    a0, (e0, l0) = best_alpha(rec)
    se0 = (l0 * (1 - l0) / shots) ** 0.5 if shots else 0.0
    compat = []
    for a, (e, l) in damped.items():
        se = (l * (1 - l) / shots) ** 0.5 if shots else 0.0
        if l - l0 <= (se ** 2 + se0 ** 2) ** 0.5:
            compat.append(a)
    return (min(compat), max(compat)) if compat else (a0, a0)


# =============================================================================
# Style
# =============================================================================
def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.9, "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.spines.top": True, "axes.spines.right": True,   # closed box
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}.pdf/.png")


# =============================================================================
# Figure 1 -- LER(alpha_best)/LER(MWPM) vs p
# =============================================================================
def series_ratio_vs_p(data, d, alpha="best"):
    ps, r, err = [], [], []
    for p in sorted(data[d]):
        rec = data[d][p]
        e_den = rec["mwpm"]
        if alpha == "best":
            _, (e_num, _) = best_alpha(rec)
        else:
            e_num = rec["cm"][alpha][0]
        ps.append(p)
        rr = e_num / e_den
        r.append(rr)
        err.append(rr * ratio_err(e_num, e_den))
    return np.array(ps), np.array(r), np.array(err)


def fig1(data):
    # --- combined (single axes) ---
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
    for d in DISTANCES:
        p, r, e = series_ratio_vs_p(data, d, "best")
        ax.errorbar(p, r, yerr=e, marker=DMARK[d], color=DCOL[d], mfc=DCOL[d], mec="white",
                    mew=0.6, capsize=2.5, elinewidth=1.0, label=DLABEL[d] + r", $\alpha^{*}$",
                    zorder=3)
        p1, r1, e1 = series_ratio_vs_p(data, d, 1.0)
        ax.errorbar(p1, r1, yerr=e1, marker=DMARK[d], color=DCOL[d], ls=(0, (2, 2)), lw=1.2,
                    ms=5, mfc="none", mec=DCOL[d], capsize=2.5, elinewidth=1.0,
                    label=DLABEL[d] + r", CM ($\alpha=1$)", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
    ax.set_xlim(8e-5, 1.25e-2)
    ax.legend(loc="upper center", ncol=2, handlelength=2.2, columnspacing=1.2,
              bbox_to_anchor=(0.52, 1.0))
    fig.tight_layout()
    save(fig, "fig1_ler_ratio_vs_p_combined")

    # --- 1x2 subplots ---
    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(4.1 * len(DISTANCES), 3.7), sharey=True)
    for ax, d in zip(axes, DISTANCES):
        ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)))
        p, r, e = series_ratio_vs_p(data, d, "best")
        ax.errorbar(p, r, yerr=e, marker=DMARK[d], color=DCOL[d], mec="white", mew=0.6,
                    capsize=2.5, elinewidth=1.0, label=r"$\alpha^{*}$")
        p1, r1, e1 = series_ratio_vs_p(data, d, 1.0)
        ax.errorbar(p1, r1, yerr=e1, marker=DMARK[d], color=DCOL[d], ls=(0, (2, 2)), lw=1.2,
                    ms=5, mfc="none", mec=DCOL[d], capsize=2.5, elinewidth=1.0,
                    label=r"CM ($\alpha=1$)")
        ax.set_xscale("log")
        ax.set_xlabel(r"Physical error rate $p$")
        ax.set_title(DLABEL[d])
        ax.set_xlim(8e-5, 1.25e-2)
        ax.legend(loc="lower left" if d == 3 else "upper right")
    axes[0].set_ylabel(r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
    fig.tight_layout()
    save(fig, "fig1_ler_ratio_vs_p_subplots")


# =============================================================================
# Figure 2 -- LER(alpha)/LER(MWPM) vs alpha, one curve per p
# =============================================================================
def fig2(data):
    alphas_full = GRID + [1.0]
    pall = sorted(set(p for d in DISTANCES for p in data[d]))
    norm = LogNorm(vmin=min(pall), vmax=max(pall))
    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(4.3 * len(DISTANCES), 3.9), sharey=True)
    for ax, d in zip(axes, DISTANCES):
        ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.axvline(1.0, color="0.8", lw=0.8, zorder=1)
        for p in sorted(data[d]):
            rec = data[d][p]
            den = rec["mwpm"]
            xs = [a for a in alphas_full if a in rec["cm"]]
            ys = [rec["cm"][a][0] / den for a in xs]
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=PCMAP(norm(p)), zorder=2)
        ax.set_xlabel(r"Reweighting strength $\alpha$")
        ax.set_title(DLABEL[d])
        ax.set_xlim(0.05, 1.05)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    axes[0].set_ylabel(r"$\mathrm{LER}(\alpha)\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
    sm = ScalarMappable(norm=norm, cmap=PCMAP)
    cb = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.045)
    cb.set_label(r"Physical error rate $p$")
    save(fig, "fig2_ler_ratio_vs_alpha")


# =============================================================================
# Figure 2b -- p on x, one curve per alpha (colour = alpha)
# =============================================================================
def fig2b(data):
    alphas_full = GRID + [1.0]
    norm = Normalize(vmin=0.1, vmax=1.0)
    # truncated plasma: purple -> magenta -> orange, dropping the bright yellow end
    cmap = mpl.colors.ListedColormap(plt.get_cmap("plasma")(np.linspace(0.0, 0.85, 256)))
    with mpl.rc_context({"font.size": 12, "axes.labelsize": 14.5, "axes.titlesize": 14.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, axes = plt.subplots(1, len(DISTANCES), figsize=(4.3 * len(DISTANCES), 3.9),
                                 sharey=True)
        for ax, d in zip(axes, DISTANCES):
            ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
            for a in alphas_full:
                ps = sorted(pp for pp in data[d] if a in data[d][pp]["cm"])
                ys = [data[d][pp]["cm"][a][0] / data[d][pp]["mwpm"] for pp in ps]
                ax.plot(ps, ys, marker="o", ms=4, ls="none", color=cmap(norm(a)), zorder=2)
            ax.set_xscale("log")
            ax.set_xlabel(r"Physical error rate $p$")
            ax.set_title(DLABEL[d])
            ax.set_xlim(8e-5, 1.25e-2)
        axes[0].set_ylabel(r"$\mathrm{LER}(\alpha)\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, location="bottom", pad=0.26, shrink=0.6, aspect=55,
                          fraction=0.05)
        cb.set_label(r"Reweighting strength $\alpha$")
        cb.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        save(fig, "fig2b_ler_ratio_vs_p_byalpha")


# =============================================================================
# Figure 3 -- best alpha vs p
# =============================================================================
def series_bestalpha_vs_p(data, d):
    ps, ab = [], []
    for p in sorted(data[d]):
        a, _ = best_alpha(data[d][p])
        ps.append(p); ab.append(a)
    return np.array(ps), np.array(ab)


def fig3(data):
    # --- combined ---
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for d in DISTANCES:
        p, a = series_bestalpha_vs_p(data, d)
        ax.plot(p, a, marker=DMARK[d], color=DCOL[d], mec="white", mew=0.6, lw=1.8,
                label=DLABEL[d])
    ax.set_xscale("log")
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"Optimal reweighting strength $\alpha^{*}$")
    ax.set_xlim(8e-5, 1.25e-2)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_best_alpha_vs_p_combined")

    # --- 1x2 subplots ---
    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(4.1 * len(DISTANCES), 3.7), sharey=True)
    for ax, d in zip(axes, DISTANCES):
        p, a = series_bestalpha_vs_p(data, d)
        ax.plot(p, a, marker=DMARK[d], color=DCOL[d], mec="white", mew=0.6, lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Physical error rate $p$")
        ax.set_title(DLABEL[d])
        ax.set_xlim(8e-5, 1.25e-2)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel(r"Optimal reweighting strength $\alpha^{*}$")
    fig.tight_layout()
    save(fig, "fig3_best_alpha_vs_p_subplots")


# =============================================================================
# Combined 2x3 -- top: LER ratio (fig1);  bottom: best alpha (fig3)
# =============================================================================
def fig_combined(data):
    ncol = len(DISTANCES)
    with mpl.rc_context({"font.size": 12, "axes.labelsize": 14.5, "axes.titlesize": 14.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12}):
        fig, axes = plt.subplots(2, ncol, figsize=(4.1 * ncol, 6.8),
                                 sharex=True, sharey="row")
        for j, d in enumerate(DISTANCES):
            # --- top row: LER(alpha)/LER(MWPM) ---
            ax = axes[0, j]
            ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)))
            # alpha* = square (solid); CM(alpha=1) = filled circle (dashed); colour = distance
            p, r, e = series_ratio_vs_p(data, d, "best")
            ax.errorbar(p, r, yerr=e, marker="s", color=DCOL[d], mec="white", mew=0.6,
                        capsize=2.5, elinewidth=1.0, label=r"$\alpha^{*}$")
            p1, r1, e1 = series_ratio_vs_p(data, d, 1.0)
            ax.errorbar(p1, r1, yerr=e1, marker="o", color=DCOL[d], ls=(0, (2, 2)), lw=1.2,
                        ms=5, mec="white", mew=0.6, capsize=2.5, elinewidth=1.0,
                        label=r"CM ($\alpha=1$)")
            ax.set_title(DLABEL[d])
            ax.set_xlim(8e-5, 1.25e-2)
            ax.legend(loc="lower right")

            # --- bottom row: optimal alpha* (square) ---
            ax = axes[1, j]
            pb, ab = series_bestalpha_vs_p(data, d)
            ax.plot(pb, ab, marker="s", color=DCOL[d], mec="white", mew=0.6, lw=1.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"Physical error rate $p$")
            ax.set_xlim(8e-5, 1.25e-2)
            ax.set_ylim(0, 1.0)

        axes[0, 0].set_ylabel(r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
        axes[1, 0].set_ylabel(r"Optimal $\alpha^{*}$")
        for ax, lab in zip(axes.flat, "abcdef"):
            ax.text(-0.13 if ax in axes[:, 0] else -0.07, 1.03, f"({lab})",
                    transform=ax.transAxes, fontsize=13, fontweight="bold")
        fig.tight_layout()
        save(fig, "fig_combined_ratio_bestalpha")


def main():
    os.makedirs(OUT, exist_ok=True)
    set_style()
    data, all_rows = load()
    write_combined(all_rows)
    fig1(data)
    fig_combined(data)
    fig2(data)
    fig2b(data)
    fig3(data)
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
