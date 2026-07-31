#!/usr/bin/env python
"""Builds figures_generation/qec3v5_noise_scaling.ipynb.

Presents the noise-scaling study: take Google's data-calibrated Sycamore DEM, scale it uniformly to
simulate a quieter (future) device, and compare MWPM / CM / RCM / belief-matching as the effective
physical error rate drops. Loads the cache produced by
final_plot_scripts/compute_qec3v5_noise_scaling.py.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Extrapolating Google's calibrated noise to lower error rates: does RCM help?

The Sycamore experiments run **near threshold**, where regularized CM (RCM) gives no gain over
CM($\alpha=1$). But RCM's advantage is a **deep sub-threshold** effect. Future devices will operate at
lower physical error rates — so we ask: *if the same device became uniformly quieter, would RCM start
to beat CM on its real (calibrated) noise?*

**Method.** We take Google's **Spitz-calibrated (pairwise $p_{ij}$)** DEM (`pij`, $d=5$, $Z$, $r=5$
&mdash; weights reweighted from observed detector correlations; *not* DGR, *not* the RL-optimized
prior of Sivak et al.) as a realistic device noise **model** and scale every error probability
$p_i\to s\,p_i$ ($s\le1$: a uniformly quieter
device, **correlation structure preserved**). We Monte-Carlo sample synthetic shots from the scaled
DEM and decode with MWPM, CM($\alpha=1$), RCM (best $\alpha$ over a grid), and belief-matching.

**Why this is the right knob.** In a circuit-level DEM each mechanism is $\propto$ the physical rate,
so uniform scaling $\approx$ lowering the physical rate. It reproduces the **broken scale invariance**
exactly: the prior edge weight $\log\frac{1}{s\,p}$ **grows** as $s\to0$, while the CM posterior
$P(e_\mu|e_\nu)=P(\mu,\nu)/P(\nu)$ is a ratio of two $\propto s$ quantities and stays **$O(1)$** — so
CM must eventually over-correct and RCM recover, now on **Sycamore-calibrated** noise.

**Caveats.** (i) Uniform scaling is an idealization (real improvements are non-uniform); (ii) the
shots are simulated from the device-calibrated *model*, not new experimental data; (iii)
belief-matching's per-shot BP cost limits it to the higher-rate points.
""")

code(r"""
%matplotlib inline
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "figures_generation" else os.getcwd()
plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "mathtext.fontset": "cm",
                     "axes.grid": True, "grid.alpha": 0.3})
FIG = os.path.join(ROOT, "plots", "figures"); os.makedirs(FIG, exist_ok=True)
CSV = os.path.join(ROOT, "data", "qec3v5_noise_scaling.csv")

df = pd.read_csv(CSV).sort_values("p_med").reset_index(drop=True)
ALPHAS = [float(c.split("_a")[1]) for c in df.columns if c.startswith("cm_err_a")]
DAMPED = [a for a in ALPHAS if a < 1.0]


def se_ratio(en, ed):
    return np.sqrt(1.0 / np.maximum(en, 1) + 1.0 / np.maximum(ed, 1))


p = df["p_med"].to_numpy()
n = df["n_shots"].to_numpy()
mw = df["mwpm_err"].to_numpy()
cm = df["cm_err_a1.0"].to_numpy()
# best damped alpha per scale (argmin over the grid), and its errors
best_err = np.min([df[f"cm_err_a{a}"].to_numpy() for a in DAMPED], axis=0)
best_alpha = np.array([DAMPED[int(np.argmin([df[f"cm_err_a{a}"].iloc[i] for a in DAMPED]))]
                       for i in range(len(df))])
ler = lambda e: e / n
print(df[["s", "p_med", "n_shots", "mwpm_err", "cm_err_a1.0", "bm_shots", "bm_err"]].to_string(index=False))
print("\nalpha* per scale:", dict(zip(np.round(p, 6), best_alpha)))
""")

md(r"""
## LER and ratio-to-MWPM vs the (scaled) physical error rate

Left: absolute LER. Right: ratio to MWPM — the headline. As the effective rate drops, **CM's ratio
turns back up toward 1** (the over-correction re-emerging deep sub-threshold), while **RCM(best $\alpha$)
stays below it** — the effective-distance recovery, now on Google-calibrated noise. Belief-matching
(where affordable) remains the strongest.
""")

code(r"""
C_MW, C_CM, C_RCM, C_BM = "#2e86de", "#e67e22", "#009E73", "#8e44ad"
has_bm = df["bm_shots"].to_numpy() > 0
bm_p = p[has_bm]; bm_ler = (df["bm_err"].to_numpy()[has_bm] / df["bm_shots"].to_numpy()[has_bm])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))
# ---- left: absolute LER ----
axL.plot(p, ler(mw), "o-", color=C_MW, label="MWPM")
axL.plot(p, ler(cm), "s-", color=C_CM, label=r"CM ($\alpha=1$)")
axL.plot(p, ler(best_err), "D-", color=C_RCM, label=r"RCM (best $\alpha$)")
if has_bm.any():
    axL.plot(bm_p, bm_ler, "^--", color=C_BM, label="belief-matching")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel(r"Effective physical error rate  $s\cdot p$  (scaled DEM median)")
axL.set_ylabel(r"Logical error rate"); axL.legend()
axL.set_title("Calibrated Sycamore noise, scaled")
# ---- right: ratio to MWPM ----
axR.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 3)))
r_cm = cm / mw; r_rcm = best_err / mw
axR.errorbar(p, r_cm, yerr=r_cm * se_ratio(cm, mw), marker="s", color=C_CM, capsize=2.5,
             label=r"CM ($\alpha=1$)")
axR.errorbar(p, r_rcm, yerr=r_rcm * se_ratio(best_err, mw), marker="D", color=C_RCM, capsize=2.5,
             label=r"RCM (best $\alpha$)")
if has_bm.any():
    axR.plot(bm_p, bm_ler / ler(mw)[has_bm], "^--", color=C_BM, label="belief-matching")
axR.set_xscale("log"); axR.set_xlabel(r"Effective physical error rate  $s\cdot p$")
axR.set_ylabel(r"$\mathrm{LER}\,/\,\mathrm{LER}_{\mathrm{MWPM}}$"); axR.legend()
axR.set_title("Correlated matching over-corrects as $p\\to0$")
for ax, lab in zip((axL, axR), "ab"):
    ax.text(-0.14, 1.02, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"qec3v5_noise_scaling_ler.{ext}"), bbox_inches="tight")
plt.show()
""")

md(r"""
## Optimal reweighting strength $\alpha^{*}$ vs the (scaled) rate

The best $\alpha$ should **decrease** as the effective rate drops — matching what we found on
synthetic Stim circuits, but now derived from Google-calibrated device noise. Near the real device
rate (right edge) $\alpha^{*}$ is high (no regularization needed); as the device gets quieter,
$\alpha^{*}$ falls (stronger damping wins).
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.plot(p, best_alpha, "o-", color=C_RCM, ms=7)
ax.set_xscale("log"); ax.set_ylim(0, 1.02)
ax.set_xlabel(r"Effective physical error rate  $s\cdot p$")
ax.set_ylabel(r"Optimal reweighting strength $\alpha^{*}$")
ax.set_title("RCM optimum shifts to stronger damping as the device gets quieter")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"qec3v5_noise_scaling_alpha.{ext}"), bbox_inches="tight")
plt.show()
""")

md(r"""
## Takeaways

* Scaling Google's **calibrated** DEM down is a physically-motivated way to ask what the *same
  device* would look like at future (lower) error rates — and it **reproduces the effective-distance
  loss** we established on synthetic circuits: as $s\to0$, CM($\alpha=1$)'s advantage over MWPM
  erodes (its ratio turns back up), and **RCM (damped $\alpha$) keeps the advantage**.
* The optimal $\alpha^{*}$ **decreases** as the effective rate drops — the RCM prescription learned on
  toy circuits carries over to Sycamore-calibrated noise.
* So while RCM offers **no gain at today's near-threshold** experimental rates, this extrapolation
  says it **would** matter on a quieter future device with the same noise structure — the regime where
  regularization pays off. (Belief-matching remains strongest where its BP cost is affordable; its
  deep-sub-threshold behaviour is left for future work.)
""")

nb["cells"] = C
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qec3v5_noise_scaling.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
