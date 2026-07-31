#!/usr/bin/env python
"""Professional comparison at Google qec3v5, d=5, Z-memory, r=5 (50k experimental shots):
MWPM vs CM(alpha=1) vs RCM(several high alpha) vs belief-matching, ALL decoded on Google's
data-calibrated, cross-validated pij DEMs (odd shots <- even-calibrated DEM, and vice versa).

Belief-matching (Higgott's `beliefmatching`, BP + matching) is the slow part, so its predictions
are cached to data/qec3v5_bm_pij_d5_r05.npz.  Output: figures_thesis/{pdf,png}/google_qec3v5_r5_decoder_comparison.
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
import pymatching  # noqa: E402
from beliefmatching import BeliefMatching  # noqa: E402

BASE = os.path.join(ts.ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")


def _read(name, ndet, nobs):
    fmt = "01" if name.endswith(".01") else "b8"
    return np.asarray(stim.read_shot_data_file(path=os.path.join(BASE, name), format=fmt,
                      num_detectors=ndet, num_observables=nobs))


circ = stim.Circuit.from_file(os.path.join(BASE, "circuit_noisy.stim"))
nd, no = circ.num_detectors, circ.num_observables
dets = _read("detection_events.b8", nd, 0).astype(bool)
act = _read("obs_flips_actual.01", 0, no).astype(np.uint8).reshape(-1)
N = len(act)
dem_odd = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))   # decode ODD
dem_even = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_odd_for_even.dem"))  # decode EVEN
idx = np.arange(N); EVEN = idx % 2 == 0; ODD = ~EVEN


def cv_matching(alpha, corr):
    pred = np.zeros(N, np.uint8)
    for mask, dem in ((ODD, dem_odd), (EVEN, dem_even)):
        m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
        kw = {"enable_correlations": True, "alpha": alpha} if corr else {}
        pred[mask] = np.asarray(m.decode_batch(dets[mask], **kw)).reshape(-1)
    return pred


def cv_belief():
    cache = os.path.join(ts.ROOT, "data", "qec3v5_bm_pij_d5_r05.npz")
    if os.path.exists(cache):
        return np.load(cache)["pred"]
    pred = np.zeros(N, np.uint8)
    for mask, dem in ((ODD, dem_odd), (EVEN, dem_even)):
        bm = BeliefMatching(dem, max_bp_iters=20)
        pred[mask] = np.asarray(bm.decode_batch(dets[mask])).reshape(-1)
    np.savez(cache, pred=pred)
    return pred


def ler(pred):
    e = int(np.sum(pred != act)); p = e / N
    return p, float(np.sqrt(p * (1 - p) / N))


C_MW, C_CM, C_RCM, C_BM = "#0072B2", "#D55E00", "#009E73", "#785EF0"
ALPHAS = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

print("decoding MWPM / CM / RCM ...")
rows = [("MWPM", *ler(cv_matching(1.0, False)), C_MW, "o"),
        (r"CM ($\alpha=1$)", *ler(cv_matching(1.0, True)), C_CM, "s")]
for a in ALPHAS:
    rows.append((rf"RCM ($\alpha={a}$)", *ler(cv_matching(a, True)), C_RCM, "D"))
print("decoding belief-matching (BP+matching, cached) ...")
rows.append(("Belief-matching", *ler(cv_belief()), C_BM, "^"))

ts.set_style(base=11)
fig, ax = plt.subplots(figsize=(7.6, 5.4))
ys = np.arange(len(rows))[::-1]                     # first row at top
for y, (lab, l, se, c, mk) in zip(ys, rows):
    ax.errorbar(l, y, xerr=se, marker=mk, ms=9, color=c, mec="white", mew=0.8,
                capsize=3, elinewidth=1.2, zorder=3)
    ax.annotate(f"{l*100:.2f}\\%" if ts.USETEX else f"{l*100:.2f}%", (l + se, y),
                xytext=(8, 0), textcoords="offset points", va="center", ha="left",
                fontsize=9.5, color="0.25")
# faint guide at the belief-matching value
ax.axvline(rows[-1][1], color=C_BM, lw=1.0, ls=(0, (4, 3)), alpha=0.5, zorder=1)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel(r"Logical error rate  $p_{\mathrm{L}}$  (50\,000 shots)" if ts.USETEX
              else r"Logical error rate  $p_{\mathrm{L}}$  (50 000 shots)")
ax.set_title(r"Google qec3v5 $d=5$, $Z$-memory, $r=5$ (Spitz $p_{ij}$-calibrated DEM)")
ax.grid(axis="x", alpha=0.5); ax.grid(axis="y", visible=False)
ax.margins(x=0.13, y=0.06)
fig.tight_layout()
ts.save(fig, "google_qec3v5_r5_decoder_comparison")
for lab, l, se, c, mk in rows:
    print(f"  {lab:<22} {l*100:.2f}% +/- {se*100:.2f}%")
