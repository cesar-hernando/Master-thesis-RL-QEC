#!/usr/bin/env python
"""Thesis figure: panel (c) of the DEM-scaling ratio plot on its own -- the Spitz-calibrated pij
DEM.  LER/LER_MWPM vs noise scale s for CM(alpha=1) and RCM(best alpha), from the cluster sweep in
data/qec3v5_dem_scaling/.  Legend top-left; the s=1 decade tick is labelled "1" and marked
"Sycamore".  Saved to figures_thesis/{pdf,png}/qec3v5_dem_scaling_ratio_pij.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import thesis_style as ts

IN_DIR = os.path.join(ts.ROOT, "data", "qec3v5_dem_scaling")
BM_CSV = os.path.join(ts.ROOT, "data", "qec3v5_pij_bm_scaling.csv")
DAMPED = [round(0.1 * i, 1) for i in range(1, 10)]
C_CM, C_RCM, C_BM5, C_BM20 = "#D55E00", "#009E73", "#b19cd9", "#785EF0"


def load_pij():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(IN_DIR, "task_*.csv"))],
                   ignore_index=True)
    d = df[df.dem == "pij"].sort_values("s")
    s = d["s"].to_numpy(); mw = d["mwpm_err"].to_numpy(); cm = d["cm_err_a1.0"].to_numpy()
    dmat = np.array([d[f"cm_err_a{a}"].to_numpy() for a in DAMPED])
    return s, mw, cm, dmat.min(axis=0)


def se_ratio(en, ed):
    return np.sqrt(1.0 / np.maximum(en, 1) + 1.0 / np.maximum(ed, 1))


def xfmt(v, pos):
    if np.isclose(v, 1.0):
        return "1"
    e = int(np.round(np.log10(v)))
    return r"$10^{%d}$" % e


s, mw, cm, best = load_pij()
r_cm, r_rc = cm / mw, best / mw

ts.set_style(base=12)
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.axvline(1.0, color="0.6", lw=1.0, ls=":", zorder=1)
ax.errorbar(s, r_cm, yerr=r_cm * se_ratio(cm, mw), marker="s", color=C_CM, lw=1.8, capsize=3,
            label=r"CM ($\alpha=1$)")
ax.errorbar(s, r_rc, yerr=r_rc * se_ratio(best, mw), marker="D", color=C_RCM, lw=1.8, capsize=3,
            label=r"RCM ($\alpha^{*}$)")
# belief-matching (5 and 20 BP iters), where affordable
if os.path.exists(BM_CSV):
    b = pd.read_csv(BM_CSV).sort_values("s")
    bs, bmw = b["s"].to_numpy(), b["mwpm_err"].to_numpy()
    for col, ecol, mk, lab in [(C_BM20, "bm20_err", "v", "BM (20 iters)")]:
        be = b[ecol].to_numpy(); rb = be / bmw
        ax.errorbar(bs, rb, yerr=rb * se_ratio(be, bmw), marker=mk, color=col, lw=1.8, capsize=3,
                    label=lab)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
ax.set_xlabel(r"Noise scale $s$")
ax.set_ylabel(r"$p_{\mathrm{L}}\,/\,p_{\mathrm{L}}^{\mathrm{MWPM}}$")
# "Sycamore" label at s=1 (data x, axes-fraction y), in the empty region below the curves
ax.text(1.0, 0.30, "Sycamore ", rotation=90, ha="right", va="center",
        transform=ax.get_xaxis_transform(), fontsize=11, color="0.4")
ts.legend(ax, loc="upper left")
fig.tight_layout()
ts.save(fig, "qec3v5_dem_scaling_ratio_pij")
