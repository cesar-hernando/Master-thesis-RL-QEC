"""Do weight-2 (2-fault) error configurations confuse CM and/or MWPM, vs p?

For p = 1e-2 ... 1e-4, exhaustively decode every pair of DEM error mechanisms with BOTH
MWPM and CM(alpha=1) and count, INDEPENDENTLY for each decoder, how many 2-fault configs
produce a logical error -- and how many of those involve at least one CNOT (DEPOLARIZE2)
error. d=5: the circuit fault distance is 5, so an optimal/distance-preserving decoder
must correct EVERY 2-fault config (a logical needs >=3 faults). MWPM should therefore show
0; any nonzero CM count is a genuine effective-distance loss.
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
PS = [1e-2, 7e-3, 4e-3, 2e-3, 1e-3, 7e-4, 4e-4, 2e-4, 1e-4]


def build(p):
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
        mechs.append((frozenset(dets), obs))
    M = len(mechs)
    det_masks = np.zeros((M, ND), np.uint8)
    obs_arr = np.zeros(M, np.uint8)
    for i, (dets, ob) in enumerate(mechs):
        det_masks[i, list(dets)] = 1
        obs_arr[i] = ob

    # is-CNOT flag per mechanism (gate = DEPOLARIZE2), via explain
    detmap = {}
    for e in circ.explain_detector_error_model_errors(reduce_to_one_representative_error=True):
        ds = frozenset(t.dem_target.val for t in e.dem_error_terms
                       if t.dem_target.is_relative_detector_id())
        if e.circuit_error_locations:
            gates = {loc.instruction_targets.gate for loc in e.circuit_error_locations}
            detmap[ds] = "DEPOLARIZE2" in gates or detmap.get(ds, False)
    is_cnot = np.array([bool(detmap.get(mechs[i][0], False)) for i in range(M)], bool)
    return mwpm, cm, det_masks, obs_arr, is_cnot, M


def sweep(p):
    mwpm, cm, det_masks, obs_arr, is_cnot, M = build(p)
    cm_fail = cm_fail_cnot = 0          # CM(1) 2-fault failures; with >=1 CNOT constituent
    mw_fail = mw_fail_cnot = 0          # MWPM 2-fault failures
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
        cnot_pair = is_cnot[ii] | is_cnot[jj]
        pm = mwpm.decode_batch(shots)[:, 0]
        c1 = cm.decode_batch(shots, enable_correlations=True, alpha=1.0)[:, 0]
        bc, bm = (c1 != ob), (pm != ob)
        cm_fail += int(bc.sum()); cm_fail_cnot += int((bc & cnot_pair).sum())
        mw_fail += int(bm.sum()); mw_fail_cnot += int((bm & cnot_pair).sum())
    return M, cm_fail, cm_fail_cnot, mw_fail, mw_fail_cnot


print(f"d={D} rotated (Tesseract).  2-fault configs that each decoder MIS-decodes:\n")
print(f"{'p':>9} | {'CM(1) fails':>11} {'(>=1 CNOT)':>10} | {'MWPM fails':>10} {'(>=1 CNOT)':>10}")
for p in PS:
    t0 = time.time()
    M, cf, cfc, mf, mfc = sweep(p)
    print(f"{p:>9g} | {cf:>11} {cfc:>10} | {mf:>10} {mfc:>10}   ({time.time()-t0:.0f}s)", flush=True)
