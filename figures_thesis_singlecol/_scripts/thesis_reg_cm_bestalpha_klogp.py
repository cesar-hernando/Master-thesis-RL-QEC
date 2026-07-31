#!/usr/bin/env python
"""Thesis figure: broken scale invariance with the optimal alpha*, and the k*log(1/p)+c fit of the
RCM(alpha*) median posterior edge weight (distance 5).

Median decoding-graph edge weights vs p for d=5:
  - prior edge  log((1-p_mu)/p_mu)         -> grows as log(1/p)
  - CM (alpha=1) posterior edge            -> stays O(1) (the broken scale invariance)
  - RCM (alpha*) posterior edge            -> partially restores the log(1/p) growth
The 5 lowest-p RCM points are fit to k*log(1/p)+c.  alpha*(p) is read from the new combined scan.

New "tex" style + grid; saved to figures_thesis/{pdf,png}/reg_cm_bestalpha_klogp_fit.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

sys.path.insert(0, os.path.join(ts.ROOT, "src"))
import stim            # noqa: E402
import pymatching      # noqa: E402
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords  # noqa: E402
from NeuralCM.decoding_graph import DecodingGraph                                     # noqa: E402

C_PRIOR, C_CM, C_RCM = "#0072B2", "#D55E00", "#009E73"   # thesis blue / orange / green
df = ts.load_scan()


def best_alpha(d, p):
    "Optimal reweighting strength alpha*(p): the alpha < 1 minimising LER in the scan."
    s = df[(df.distance == d) & (np.abs(df.p - p) < 1e-12) & (df.decoder == "cm") & (df.alpha < 1.0)]
    return float(s.loc[s.ler.idxmin()].alpha) if len(s) else None


def build_dem(D, P):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=D,
        after_clifford_depolarization=P, before_measure_flip_probability=P,
        after_reset_flip_probability=P, before_round_data_depolarization=P)
    return decompose_errors_for_stim_surface_code_coords(
        circ.detector_error_model(decompose_errors=False))


def median_prior(g):
    p = np.asarray(g.base_p); p = p[(p > 0) & (p < 1)]
    return np.median(np.log((1.0 - p) / p))


def median_posterior(g, alpha):
    "Median over correlated edge-pairs of w = log((1-p_imp)/p_imp), p_imp the Pearl mix (fork math)."
    occ = np.asarray(g.base_p); corr = np.asarray(g.initial_corr_tracer)
    src, dst = np.asarray(g.line_edge_index)
    Pv, Pmu, Pmv = occ[dst], occ[src], corr                      # P(nu), P(mu), P(mu, nu)
    v = (Pv > 0) & (Pv < 1) & (Pmv > 0)
    pc = Pmv[v] / Pv[v]                                          # P(mu | nu)  (hard evidence)
    pn = np.clip((Pmu[v] - Pmv[v]) / (1.0 - Pv[v]), 0.0, None)   # P(mu | ~nu)
    imp = np.clip(alpha * pc + (1.0 - alpha) * pn, 1e-6, 0.499999)
    return np.median(np.log((1.0 - imp) / imp))


def r2(y, yh):
    yh = np.asarray(yh)
    return 1.0 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---- data: median edge weights over the d=5 scan grid -----------------------------------------
PS = np.array(sorted(df[df.distance == 5].p.unique()))
prior, post1, posta = [], [], []
for P in PS:
    g = DecodingGraph.from_dem(build_dem(5, P))
    a = best_alpha(5, P)
    prior.append(median_prior(g))
    post1.append(median_posterior(g, 1.0))
    posta.append(median_posterior(g, a))
prior, post1, posta = map(np.array, (prior, post1, posta))
Lg = np.log(1.0 / PS)

lo = np.argsort(PS)[:5]                                           # the 5 lowest-p points
(k1, c1), *_ = np.linalg.lstsq(np.vstack([Lg[lo], np.ones(5)]).T, posta[lo], rcond=None)
print(f"RCM low-p fit  k*log(1/p)+c :  k = {k1:.3f}  c = {c1:.3f}  "
      f"R^2 = {r2(posta[lo], k1 * Lg[lo] + c1):.3f}")

# ---- plot -------------------------------------------------------------------------------------
ts.set_style()
fig, ax = plt.subplots(figsize=ts.figsize(1, 1, panel_ratio=0.66))
ax.plot(PS, prior, "o-", color=C_PRIOR, lw=1.4, ms=4.5, alpha=0.6,
        label=r"prior edge  ($\propto\log\frac{1}{p}$)")
ax.plot(PS, post1, "s-", color=C_CM, lw=1.4, ms=4.5, alpha=0.6, label=r"CM ($\alpha=1$)")
ax.plot(PS, posta, "D", color=C_RCM, ms=6, label=r"RCM ($\alpha^{*}$)")
pg = np.geomspace(PS[lo].min(), PS[lo].max(), 50)
ax.plot(pg, k1 * np.log(1.0 / pg) + c1, "-", color=C_RCM, lw=1.8,
        label=rf"fit $k\log\frac{{1}}{{p}}+c$:  $k={k1:.2f}$, $c={c1:.2f}$ ($R^2={r2(posta[lo], k1*Lg[lo]+c1):.2f}$)")
ax.set_xscale("log")
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel(r"median edge weight  $w$")
ax.set_ylim(top=float(prior.max()) + 1.4)                        # headroom for the legend
ts.legend(ax, loc="upper right")
fig.tight_layout()
ts.save(fig, "reg_cm_bestalpha_klogp_fit")
