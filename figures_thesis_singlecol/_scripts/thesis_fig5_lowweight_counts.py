#!/usr/bin/env python
"""Thesis fig 5: low-weight (sub-threshold) configurations that confuse CM(alpha=1) but not MWPM.

Left  (a): d=5 weight-2, exhaustive count.
Right (b): d=7 weight-3, sampled estimate (x10^3, Poisson bars).
Data from cache cm_lowweight_counts_d5_d7.npz (DEM-based, independent of the alpha scan); only the
style is the thesis TeX-look + grid.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

z = dict(np.load(os.path.join(ts.ROOT, "data", "cm_lowweight_counts_d5_d7.npz")))
C_CM, C_MW = "#D55E00", "#0072B2"

ts.set_style()
fig, (a5, a7) = plt.subplots(1, 2, figsize=ts.figsize(1, 2, panel_ratio=1.0))

a5.plot(z["P5"], z["cm5"], "s-", color=C_CM, ms=4.5, lw=1.3, label=r"CM ($\alpha=1$)")
a5.plot(z["P5"], z["mw5"], "o--", color=C_MW, ms=4.5, lw=1.3, label="MWPM")
a5.set_title(r"$d=5$: weight-2 configurations")
a5.set_ylabel("Confusing configurations")

a7.errorbar(z["P7"], z["cm7"] / 1e3, yerr=z["cm7e"] / 1e3, marker="s", ls="-", color=C_CM, ms=4.5,
            lw=1.3, capsize=2.5, elinewidth=0.9, label=r"CM ($\alpha=1$)")
a7.plot(z["P7"], z["mw7"] / 1e3, "o--", color=C_MW, ms=4.5, lw=1.3, label="MWPM")
a7.set_title(r"$d=7$: weight-3 configurations")
a7.set_ylabel(r"Estimated confusing configurations ($\times 10^{3}$)")

for ax, lab in zip((a5, a7), "ab"):
    ax.set_xscale("log"); ax.set_xlabel(r"Physical error rate $p$")
    ax.margins(y=0.12)
    hi = ax.get_ylim()[1]
    ax.set_ylim(-0.035 * hi, hi)
    ts.panel(ax, lab, x=-0.20, y=1.04)
ts.legend(a5, loc="best")
ts.legend(a7, loc="best")
fig.tight_layout()
ts.save(fig, "cm_vs_mwpm_lowweight_counts_d5_d7")
