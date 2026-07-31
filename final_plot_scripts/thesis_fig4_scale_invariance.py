#!/usr/bin/env python
"""Thesis fig 4: broken scale invariance (d=5).  Median decoding-graph edge weight vs p:
prior edge  log((1-p_mu)/p_mu)  (grows ~ log(1/p))  vs  CM(alpha=1) posterior edge
log((1-P(e_mu|e_nu))/P(e_mu|e_nu))  (stays O(1)).

Data is computed from the DEM (independent of the alpha scan); only the style is the thesis
TeX-look + grid.  Single distance, no d9/d11 variants.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

sys.path.insert(0, os.path.join(ts.ROOT, "src"))
import stim  # noqa: E402
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords  # noqa: E402
from NeuralCM.decoding_graph import DecodingGraph  # noqa: E402

C_PRIOR = "#0072B2"
C_CM = "#D55E00"


def dem_of(D, P):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=D,
                               after_clifford_depolarization=P, before_measure_flip_probability=P,
                               after_reset_flip_probability=P, before_round_data_depolarization=P)
    return decompose_errors_for_stim_surface_code_coords(c.detector_error_model(decompose_errors=False))


def wlog(p):
    p = np.asarray(p); p = p[(p > 0) & (p < 1)]
    return np.median(np.log((1.0 - p) / p)) if p.size else np.nan


PS = np.geomspace(1e-4, 1e-2, 9)
prior, post = [], []
for P in PS:
    g = DecodingGraph.from_dem(dem_of(5, P))
    occ = np.asarray(g.base_p); corr = np.asarray(g.initial_corr_tracer); src, dst = np.asarray(g.line_edge_index)
    prior.append(wlog(occ))
    pc = np.array([corr[k] / occ[dst[k]] for k in range(g.n_line_edges) if occ[dst[k]] > 0 and corr[k] > 0])
    post.append(wlog(pc))
prior, post = np.array(prior), np.array(post)

ts.set_style(base=12)
fig, ax = plt.subplots(figsize=(7.4, 5.1))
ax.plot(PS, prior, "o-", color=C_PRIOR, lw=2.0, ms=7,
        label=r"prior edge  $\log\dfrac{1-p_\mu}{p_\mu}$")
ax.plot(PS, post, "s-", color=C_CM, lw=2.0, ms=7,
        label=r"CM posterior edge  $\log\dfrac{1-P(e_\mu|e_\nu)}{P(e_\mu|e_\nu)}$")
ax.set_xscale("log")
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel(r"Median edge weight  $w$")
ts.legend(ax, loc="upper right")
fig.tight_layout()
ts.save(fig, "cm_broken_scale_invariance")
print("prior:", np.round(prior, 2))
print("post :", np.round(post, 2))
