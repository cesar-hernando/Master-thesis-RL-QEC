#!/usr/bin/env python
"""Thesis figure: MWPM / CM(alpha=1) / RCM(alpha*) on the 50k Google qec3v5 experimental shots
(d=5, Z, r=5), all fed the Spitz pairwise-calibrated (pij) DEM, cross-validated (odd shots decoded
with the even-calibrated DEM and vice-versa).  RCM uses alpha*=0.8 (the s=1 optimum of the
DEM-scaling sweep).  Saved to figures_thesis/{pdf,png}/qec3v5_pij_decoders.
"""
import json
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

BASE = os.path.join(ts.ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")
CACHE = os.path.join(ts.ROOT, "data", "qec3v5_pij_decoders.json")   # cache the (slow) BM decode
ALPHA_STAR = 0.8


def compute():
    """Decode the 50k experimental shots (cross-validated on pij) with all five decoders."""
    sys.path.insert(0, os.path.join(ts.ROOT, "src"))
    import stim            # noqa: E402
    import pymatching      # noqa: E402
    from beliefmatching import BeliefMatching  # noqa: E402

    circ = stim.Circuit.from_file(os.path.join(BASE, "circuit_noisy.stim"))
    nd, no = circ.num_detectors, circ.num_observables
    dets = np.asarray(stim.read_shot_data_file(path=os.path.join(BASE, "detection_events.b8"),
                      format="b8", num_detectors=nd, num_observables=0)).astype(bool)
    act = np.asarray(stim.read_shot_data_file(path=os.path.join(BASE, "obs_flips_actual.01"),
                     format="01", num_detectors=0, num_observables=no)).astype(np.uint8).reshape(-1)
    pij_odd = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))
    pij_even = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_odd_for_even.dem"))

    def cv(decode_fn):
        N = len(act); idx = np.arange(N); EVEN = idx % 2 == 0; ODD = ~EVEN
        pred = np.zeros(N, np.uint8)
        for mask, dem in ((ODD, pij_odd), (EVEN, pij_even)):
            pred[mask] = np.asarray(decode_fn(dem, dets[mask])).reshape(-1)
        p = float((pred != act).mean())
        return p, float(np.sqrt(p * (1 - p) / N))

    def match_fn(corr, alpha):
        def f(dem, d):
            m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
            kw = {"enable_correlations": True, "alpha": alpha} if corr else {}
            return m.decode_batch(d, **kw)
        return f

    DEC = [("MWPM", "#0072B2", match_fn(False, 1.0)),
           (r"CM ($\alpha=1$)", "#D55E00", match_fn(True, 1.0)),
           (rf"RCM ($\alpha^{{*}}={ALPHA_STAR}$)", "#009E73", match_fn(True, ALPHA_STAR)),
           ("BM (5 iters)", "#b19cd9", lambda dem, d: BeliefMatching(dem, max_bp_iters=5).decode_batch(d)),
           ("BM (20 iters)", "#785EF0", lambda dem, d: BeliefMatching(dem, max_bp_iters=20).decode_batch(d))]
    return [[lab, col, *cv(fn)] for lab, col, fn in DEC]


if os.path.exists(CACHE):
    res = json.load(open(CACHE))
else:
    res = compute()
    json.dump(res, open(CACHE, "w"))
res = [r for r in res if r[0] != "BM (5 iters)"]   # keep MWPM / CM / RCM / BM(20 iters)
for lab, col, p, se in res:
    print(f"{lab:<22} {p*100:.2f}% +/- {se*100:.2f}%")

ts.set_style(base=11)
fig, ax = plt.subplots(figsize=(6.8, 4.7))
x = np.arange(len(res))
for i, (lab, col, p, se) in enumerate(res):
    b = ax.bar(i, p * 100, 0.66, yerr=se * 100, capsize=4, color=col, edgecolor="white",
               linewidth=0.8, error_kw=dict(elinewidth=1.1, ecolor="0.3"))
    ax.bar_label(b, labels=[f"{p*100:.2f}\\%" if ts.USETEX else f"{p*100:.2f}%"],
                 padding=3, fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([lab for lab, *_ in res])
ax.set_ylabel(r"Logical error rate  $p_{\mathrm{L}}$")
ax.set_ylim(0, max(p for _, _, p, _ in res) * 100 * 1.16)
ax.set_title(r"Google Sycamore Data $d=5$, $Z$-memory, $r=5$" + "\n"
             + r"Experimentally calibrated DEM")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
ax.grid(axis="y", alpha=0.5); ax.grid(axis="x", visible=False)
fig.tight_layout()
ts.save(fig, "qec3v5_pij_decoders")
