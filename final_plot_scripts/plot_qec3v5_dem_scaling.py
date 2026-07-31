#!/usr/bin/env python
"""Plot the qec3v5 DEM noise-scaling sweep (cluster output in data/qec3v5_dem_scaling/).

Reads the 27 per-task CSVs (3 DEM models x 9 noise scales s), and produces, in final_plots/:
  qec3v5_dem_scaling_ratio  : LER/LER_MWPM vs s, one panel per DEM, CM(alpha=1) vs RCM(best alpha)
  qec3v5_dem_scaling_alpha  : optimal alpha*(s), the three DEMs overlaid
  qec3v5_dem_scaling_ler    : absolute LER vs s, one panel per DEM, MWPM / CM / RCM

s=1 is the experimentally-calibrated (Sycamore) noise level; s<1 are scaled-down future-device points.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "data", "qec3v5_dem_scaling")
OUT = os.path.join(ROOT, "final_plots")
os.makedirs(OUT, exist_ok=True)

ALPHAS = [round(0.1 * i, 1) for i in range(1, 11)]
DAMPED = [a for a in ALPHAS if a < 1.0]
DEMS = ["analytical", "proj", "pij"]
DLAB = {"analytical": "analytical (Stim)", "proj": "decompose_errors.py", "pij": r"calibrated $p_{ij}$"}
C_MW, C_CM, C_RCM = "#2e86de", "#e67e22", "#009E73"


def load():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(IN_DIR, "task_*.csv"))],
                   ignore_index=True)
    return df.sort_values(["dem", "s"]).reset_index(drop=True)


def prep(df, dem):
    d = df[df.dem == dem].sort_values("s")
    s = d["s"].to_numpy(); n = d["n_shots"].to_numpy()
    mw = d["mwpm_err"].to_numpy(); cm = d["cm_err_a1.0"].to_numpy()
    dmat = np.array([d[f"cm_err_a{a}"].to_numpy() for a in DAMPED])   # (nalpha, npts)
    best_err = dmat.min(axis=0)
    best_a = np.array([DAMPED[i] for i in dmat.argmin(axis=0)])
    return dict(s=s, n=n, mw=mw, cm=cm, best=best_err, ba=best_a)


def se_ratio(en, ed):
    return np.sqrt(1.0 / np.maximum(en, 1) + 1.0 / np.maximum(ed, 1))


def set_style():
    mpl.rcParams.update({
        "font.size": 10, "font.family": "sans-serif", "figure.dpi": 130,
        "axes.linewidth": 0.9, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "mathtext.fontset": "dejavusans", "axes.grid": True,
        "grid.color": "0.85", "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("saved final_plots/" + name + ".{pdf,png}")


def annotate_syc(ax, y=0.02):
    ax.axvline(1.0, color="0.65", lw=1.0, ls=":", zorder=0)
    ax.annotate("Sycamore\n(experimental)", xy=(1.0, y), xycoords=("data", "axes fraction"),
                ha="right", va="bottom", fontsize=8, color="0.4", rotation=0)


def fig_ratio(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, dem in zip(axes, DEMS):
        d = prep(df, dem)
        ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.axvline(1.0, color="0.65", lw=1.0, ls=":", zorder=0)
        r_cm, r_rc = d["cm"] / d["mw"], d["best"] / d["mw"]
        ax.errorbar(d["s"], r_cm, yerr=r_cm * se_ratio(d["cm"], d["mw"]), marker="s", color=C_CM,
                    capsize=2.5, lw=1.5, label=r"CM ($\alpha=1$)")
        ax.errorbar(d["s"], r_rc, yerr=r_rc * se_ratio(d["best"], d["mw"]), marker="D", color=C_RCM,
                    capsize=2.5, lw=1.5, label=r"RCM ($\alpha^{*}$)")
        ax.set_xscale("log"); ax.set_xlabel(r"Noise scale $s$")
        ax.set_title(DLAB[dem]); ax.legend(loc="best")
    axes[0].set_ylabel(r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$")
    axes[0].annotate("Sycamore\n(experimental)", xy=(1.0, 0.02), xycoords=("data", "axes fraction"),
                     ha="right", va="bottom", fontsize=8, color="0.4")
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.08, 1.02, f"({lab})", transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "qec3v5_dem_scaling_ratio")


def fig_alpha(df):
    DCOL = {"analytical": "#e67e22", "proj": "#009E73", "pij": "#2e86de"}
    DMK = {"analytical": "s", "proj": "D", "pij": "o"}
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for dem in DEMS:
        d = prep(df, dem)
        ax.plot(d["s"], d["ba"], marker=DMK[dem], color=DCOL[dem], ms=7, lw=1.6, label=DLAB[dem])
    ax.axvline(1.0, color="0.65", lw=1.0, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Noise scale $s$  ($s\!=\!1$: Sycamore experimental)")
    ax.set_ylabel(r"Optimal reweighting strength $\alpha^{*}$")
    ax.set_title("RCM optimum vs noise scale, across DEM models")
    ax.legend()
    fig.tight_layout()
    save(fig, "qec3v5_dem_scaling_alpha")


def fig_ler(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, dem in zip(axes, DEMS):
        d = prep(df, dem)
        ax.plot(d["s"], d["mw"] / d["n"], "o--", color=C_MW, lw=1.5, label="MWPM")
        ax.plot(d["s"], d["cm"] / d["n"], "s-", color=C_CM, lw=1.5, label=r"CM ($\alpha=1$)")
        ax.plot(d["s"], d["best"] / d["n"], "D-", color=C_RCM, lw=1.5, label=r"RCM ($\alpha^{*}$)")
        ax.axvline(1.0, color="0.65", lw=1.0, ls=":", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Noise scale $s$"); ax.set_title(DLAB[dem]); ax.legend(loc="best")
    axes[0].set_ylabel(r"Logical error rate  $p_{\mathrm{L}}$")
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.08, 1.02, f"({lab})", transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "qec3v5_dem_scaling_ler")


if __name__ == "__main__":
    set_style()
    df = load()
    print(f"loaded {len(df)} rows: {sorted(df.dem.unique())} x {len(df.s.unique())} scales")
    fig_ratio(df)
    fig_alpha(df)
    fig_ler(df)
