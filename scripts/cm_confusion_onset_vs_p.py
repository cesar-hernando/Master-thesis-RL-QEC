"""From which p does hard CM start being confused by weight-2 (2-fault) errors?

At each p, rebuild the DEM (weights depend on p), enumerate ALL 2-fault syndromes, and
count how many CM(alpha=1) mis-decodes while MWPM succeeds. Because CM's boosted edge
weight stays ~O(1) while bulk weights grow as log(1/p), the confusing set turns on below
some p and saturates as p -> 0. Also report the summed weight sum p_i p_j (the p^2 LER
channel) so it can be compared with MWPM's leading channel.
"""
import os
import sys
import time

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords

D = 5
PS = [1e-2, 7e-3, 4e-3, 2e-3, 1e-3, 7e-4, 4e-4, 2e-4, 1e-4, 7e-5, 4e-5, 2e-5, 1e-5]


def count_confusing(p):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=D,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = decompose_errors_for_stim_surface_code_coords(
        circ.detector_error_model(decompose_errors=False))
    ND = dem.num_detectors
    mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

    mechs = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= 1
        mechs.append((inst.args_copy()[0], frozenset(dets), obs))
    M = len(mechs)
    probs = np.array([m[0] for m in mechs])
    det_masks = np.zeros((M, ND), np.uint8)
    obs_arr = np.zeros(M, np.uint8)
    for i, (pr, dets, ob) in enumerate(mechs):
        det_masks[i, list(dets)] = 1
        obs_arr[i] = ob

    nfail, fw = 0, 0.0
    for i0 in range(0, M, 400):
        blocks, ii, jj = [], [], []
        for i in range(i0, min(i0 + 400, M)):
            j = np.arange(i + 1, M)
            if j.size:
                blocks.append(det_masks[i] ^ det_masks[i + 1:])
                ii.append(np.full(j.size, i)); jj.append(j)
        if not blocks:
            continue
        shots = np.vstack(blocks)
        ii, jj = np.concatenate(ii), np.concatenate(jj)
        ob = obs_arr[ii] ^ obs_arr[jj]
        pm = mwpm.decode_batch(shots)[:, 0]
        c1 = cm.decode_batch(shots, enable_correlations=True, alpha=1.0)[:, 0]
        sel = (c1 != ob) & (pm == ob)
        nfail += int(sel.sum())
        fw += float((probs[ii] * probs[jj])[sel].sum())
    return M, nfail, fw


print(f"d={D} rotated, Tesseract decomposition\n")
print(f"{'p':>9} {'mechs':>6} {'CM-confused 2-fault pairs':>26} {'sum p_i p_j':>12}")
for p in PS:
    t0 = time.time()
    M, nf, fw = count_confusing(p)
    print(f"{p:>9g} {M:>6} {nf:>26} {fw:>12.3e}   ({time.time()-t0:.0f}s)", flush=True)
