"""Is CM's effective-distance loss a property of the ROTATED code, or of correlated
matching itself? Run the exhaustive 2-fault enumeration on rotated AND unrotated d=5
surface codes (same circuit-level depolarizing noise) and compare.

A pair where CM(alpha=1) fails but MWPM succeeds is a p^2 logical-error channel at d=5
that MWPM does not have -> a genuine loss of one unit of effective distance. If such
pairs exist on the UNrotated code too, the degradation is intrinsic to correlated
matching (contradicting Fowler's circuit-level conjecture), not an artefact of rotation.
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords

D, P = 5, 2e-4


def build_dem(circ, method):
    if method == "custom":                       # coordinate-aware (rotated only)
        return decompose_errors_for_stim_surface_code_coords(
            circ.detector_error_model(decompose_errors=False))
    return circ.detector_error_model(decompose_errors=True)   # Stim built-in (any code)


def run(code, method):
    circ = stim.Circuit.generated(
        f"surface_code:{code}_memory_z", distance=D, rounds=D,
        after_clifford_depolarization=P, before_measure_flip_probability=P,
        after_reset_flip_probability=P, before_round_data_depolarization=P)
    dem = build_dem(circ, method)
    ND = dem.num_detectors
    mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

    mechs = []                        # (prob, frozenset dets, obs)
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

    # physical gate of each net-syndrome (prefer a CNOT explanation)
    detmap = {}
    for e in circ.explain_detector_error_model_errors(reduce_to_one_representative_error=True):
        ds = frozenset(t.dem_target.val for t in e.dem_error_terms
                       if t.dem_target.is_relative_detector_id())
        if e.circuit_error_locations:
            loc = e.circuit_error_locations[0]
            pauli = "".join(sorted(g.gate_target.pauli_type
                                   for g in loc.flipped_pauli_product))
            detmap.setdefault(ds, set()).add((loc.instruction_targets.gate, pauli, len(pauli)))

    def tag(i):
        opts = detmap.get(mechs[i][1])
        if not opts:
            return ("?", "?", 0)
        return sorted(opts, key=lambda o: (o[0] != "DEPOLARIZE2", o))[0]

    # single-fault sanity (both decoders must be perfect)
    pm = mwpm.decode_batch(det_masks)[:, 0]
    c1 = cm.decode_batch(det_masks, enable_correlations=True, alpha=1.0)[:, 0]
    sf = (int((pm != obs_arr).sum()), int((c1 != obs_arr).sum()))

    # all pairs
    fail, fw = [], 0.0
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
        w = probs[ii] * probs[jj]
        fw += float(w[sel].sum())
        for k in np.flatnonzero(sel):
            fail.append((int(ii[k]), int(jj[k])))

    gates = Counter(tuple(sorted((tag(i)[0], tag(j)[0]))) for i, j in fail)
    both_cnot = sum(n for g, n in gates.items() if g == ("DEPOLARIZE2", "DEPOLARIZE2"))
    weights = Counter(tuple(sorted((tag(i)[2], tag(j)[2]))) for i, j in fail)
    return dict(M=M, ND=ND, sf=sf, nfail=len(fail), fw=fw, both_cnot=both_cnot,
                gates=gates, weights=weights)


for code, method in (("rotated", "custom"), ("rotated", "stim"), ("unrotated", "stim")):
    t0 = time.time()
    try:
        r = run(code, method)
    except Exception as e:
        print(f"\n===== {code.upper()} ({method} decomposition) FAILED: {e}")
        continue
    print(f"\n===== {code.upper()} surface code [{method} decomposition], d={D}, p={P:g}  "
          f"({time.time()-t0:.0f}s) =====")
    print(f"  mechanisms={r['M']}  detectors={r['ND']}  pairs={r['M']*(r['M']-1)//2:,}")
    print(f"  single-fault failures  MWPM={r['sf'][0]}  CM(1)={r['sf'][1]}  (must be 0,0)")
    print(f"  2-fault pairs where CM(1) FAILS but MWPM SUCCEEDS: {r['nfail']}")
    print(f"     -> summed weight  sum p_i p_j = {r['fw']:.3e}  (the p^2 CM-only LER channel)")
    print(f"     -> both constituents are CNOT (DEPOLARIZE2): {r['both_cnot']}/{r['nfail']}")
    print(f"     -> Pauli-weight composition (1=single-qubit,2=two-qubit): {dict(r['weights'])}")
    if r["nfail"] == 0:
        print("     => NO effective-distance loss: CM keeps the MWPM distance on this code.")
    else:
        print("     => EFFECTIVE-DISTANCE LOSS present on this code.")
