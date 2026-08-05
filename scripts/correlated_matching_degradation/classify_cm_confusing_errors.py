"""Classify the physical errors in the 2-fault pairs that confuse hard CM (d=5).

We already know CM(alpha=1) loses effective distance at low p: there exist 2-fault
configurations that MWPM decodes correctly but hard CM fails. Here we determine WHICH
physical mechanisms make up those pairs, to answer:

  Are the confusing pairs always built from HYPEREDGE errors -- errors with mixed X&Z
  character (a single-qubit Y, or a CNOT DEPOLARIZE2 pair like XZ/YY) that link the two
  matching graphs and hence create CM correlations -- or can a pair whose errors are all
  "graphlike" (pure-X or pure-Z: XX, ZZ, XI, ZI, X, Z, measurement/reset flips) also
  confuse CM?

Method: enumerate all DEM error mechanisms, tag each with (gate, Pauli signature,
hyperedge?) via stim.explain_detector_error_model_errors; decode every mechanism PAIR
with MWPM and CM(1); collect the pairs where CM(1) fails but MWPM succeeds; tally the
hyperedge content of their two constituent errors.

hyperedge := the error has both an X-type component (X or Y on some qubit) AND a Z-type
component (Z or Y on some qubit) -> it flips detectors of both stabilizer types and is a
source of cross-graph correlation for CM. (A Y is a hyperedge; XX, ZZ, XI, ZI are not.)
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

circ = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=D, rounds=D,
    after_clifford_depolarization=P, before_measure_flip_probability=P,
    after_reset_flip_probability=P, before_round_data_depolarization=P)
dem = decompose_errors_for_stim_surface_code_coords(
    circ.detector_error_model(decompose_errors=False))
ND = dem.num_detectors
mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
cmat = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)


# ---- mechanisms from the decoding DEM: net detectors, obs, prob, has-separator --------
def net_dets_obs(targets):
    dets, obs, sep = set(), 0, False
    for t in targets:
        if t.is_relative_detector_id():
            dets ^= {t.val}
        elif t.is_logical_observable_id():
            obs ^= 1
        elif t.is_separator():
            sep = True
    return frozenset(dets), obs, sep


mechs = []                      # (prob, frozenset dets, obs, dem_has_separator)
for inst in dem.flattened():
    if inst.type != "error":
        continue
    dets, obs, sep = net_dets_obs(inst.targets_copy())
    mechs.append((inst.args_copy()[0], dets, obs, sep))
M = len(mechs)
probs = np.array([m[0] for m in mechs])
det_masks = np.zeros((M, ND), dtype=np.uint8)
obs_arr = np.zeros(M, dtype=np.uint8)
for i, (pr, dets, ob, sep) in enumerate(mechs):
    det_masks[i, list(dets)] = 1
    obs_arr[i] = ob
print(f"d={D} p={P:g}: {M} mechanisms, {ND} detectors", flush=True)


# ---- physical classification of each net-syndrome via explain -------------------------
def classify(loc):
    gate = loc.instruction_targets.gate
    paulis = "".join(sorted(g.gate_target.pauli_type for g in loc.flipped_pauli_product))
    x_ish = any(c in "XY" for c in paulis)
    z_ish = any(c in "ZY" for c in paulis)
    return gate, paulis, (x_ish and z_ish), len(paulis)   # ..., weight = # non-identity qubits


detmap = {}     # frozenset(dets) -> set of (gate, paulis, hyper)
for e in circ.explain_detector_error_model_errors(reduce_to_one_representative_error=True):
    dets = frozenset(t.dem_target.val for t in e.dem_error_terms
                     if t.dem_target.is_relative_detector_id())
    if not e.circuit_error_locations:
        continue
    detmap.setdefault(dets, set()).add(classify(e.circuit_error_locations[0]))


def tag(i):
    """Return (gate, paulis, hyper) for mechanism i (representative); '?' if unmatched."""
    opts = detmap.get(mechs[i][1])
    if not opts:
        return ("?", "?", mechs[i][3], 0)       # fall back to DEM-separator flag
    # prefer a 2-qubit (CNOT) explanation if several errors share the syndrome
    return sorted(opts, key=lambda o: (o[0] != "DEPOLARIZE2", o))[0]


hyper = np.array([tag(i)[2] for i in range(M)], dtype=bool)      # Pauli-based (mixed X&Z)
hyper_sep = np.array([mechs[i][3] for i in range(M)], dtype=bool)  # CM DEM decomposes it
n_unmatched = sum(1 for i in range(M) if mechs[i][1] not in detmap)
print(f"classified {M - n_unmatched}/{M} mechanisms via explain "
      f"({hyper.sum()} hyperedge, {(~hyper).sum()} graphlike)")
print(f"hyperedge defs agree (Pauli mixed-X&Z  vs  CM-DEM separator): "
      f"{(hyper == hyper_sep).mean():.4f}  ({int((hyper != hyper_sep).sum())} of {M} differ)\n",
      flush=True)


# ---- decode all pairs; collect pairs where CM(1) fails but MWPM succeeds ---------------
def decode(shots):
    return (mwpm.decode_batch(shots)[:, 0],
            cmat.decode_batch(shots, enable_correlations=True, alpha=1.0)[:, 0])


fail_ij = []
t0 = time.time()
CH = 400
for i0 in range(0, M, CH):
    blocks, iidx, jidx = [], [], []
    for i in range(i0, min(i0 + CH, M)):
        j = np.arange(i + 1, M)
        if j.size:
            blocks.append(det_masks[i] ^ det_masks[i + 1:])
            iidx.append(np.full(j.size, i)); jidx.append(j)
    if not blocks:
        continue
    shots = np.vstack(blocks)
    ii, jj = np.concatenate(iidx), np.concatenate(jidx)
    ob = obs_arr[ii] ^ obs_arr[jj]
    pm, p1 = decode(shots)
    sel = (p1 != ob) & (pm == ob)                  # CM(1) confused, MWPM fine
    for k in np.flatnonzero(sel):
        fail_ij.append((int(ii[k]), int(jj[k])))
print(f"decoded {M*(M-1)//2:,} pairs in {time.time()-t0:.0f}s -> "
      f"{len(fail_ij)} pairs where CM(1) fails and MWPM succeeds\n", flush=True)


# ---- tally the hyperedge content of the confusing pairs -------------------------------
by_nhyper = Counter()
gate_pairs = Counter()
sig_in_fail = Counter()
graphlike_only_examples = []
for i, j in fail_ij:
    ti, tj = tag(i), tag(j)
    nh = int(ti[2]) + int(tj[2])
    by_nhyper[nh] += 1
    gate_pairs[tuple(sorted((ti[0], tj[0])))] += 1
    for t in (ti, tj):
        sig_in_fail[(t[0], t[1], "hyper" if t[2] else "graphlike")] += 1
    if nh == 0 and len(graphlike_only_examples) < 12:
        graphlike_only_examples.append((i, j, ti, tj))

# same tally but with the CM-DEM-separator definition of hyperedge (cross-check)
by_nhyper_sep = Counter()
for i, j in fail_ij:
    by_nhyper_sep[int(hyper_sep[i]) + int(hyper_sep[j])] += 1

print("confusing pairs by number of HYPEREDGE constituents (0, 1 or 2 of the 2 errors):")
for nh in (0, 1, 2):
    print(f"   {nh} hyperedge error(s): {by_nhyper.get(nh, 0):>5d} pairs")
print("\ngate-type of the two constituent errors:")
for gp, n in gate_pairs.most_common():
    print(f"   {gp[0]:<12s} + {gp[1]:<12s} : {n}")
print("\nconstituent error signatures appearing in confusing pairs "
      "(gate, Pauli, class -> count of appearances):")
for (g, s, cls), n in sig_in_fail.most_common():
    print(f"   {g:<12s} {s:<4s} {cls:<9s} : {n}")

print("\ncross-check with CM-DEM-separator definition -> pairs by #hyperedge constituents:")
for nh in (0, 1, 2):
    print(f"   {nh}: {by_nhyper_sep.get(nh, 0):>5d} pairs")

# --- Pauli WEIGHT analysis: are two single-qubit (weight-1) errors ever enough? --------
by_w = Counter()
w11_examples = []
for i, j in fail_ij:
    ti, tj = tag(i), tag(j)
    by_w[tuple(sorted((ti[3], tj[3])))] += 1
    if ti[3] == 1 and tj[3] == 1 and len(w11_examples) < 12:
        w11_examples.append((i, j, ti, tj))
print("\nconfusing pairs by Pauli WEIGHT of the two errors (1 = single-qubit, 2 = two-qubit):")
for k in sorted(by_w):
    print(f"   weights {k}: {by_w[k]:>5d} pairs")
n11 = by_w.get((1, 1), 0)
print(f"\n=> pairs where BOTH errors are single-qubit weight (1,1): {n11}")
for i, j, ti, tj in w11_examples:
    print(f"     {i:4d} (gate={ti[0]}, {ti[1]}, {'hyper' if ti[2] else 'graphlike'})  +  "
          f"{j:4d} (gate={tj[0]}, {tj[1]}, {'hyper' if tj[2] else 'graphlike'})")

print(f"\n=> confusing pairs with ZERO hyperedge errors (both graphlike): "
      f"{by_nhyper.get(0, 0)}")
if graphlike_only_examples:
    print("   examples (i,j and their (gate,Pauli,hyper) tags):")
    for i, j, ti, tj in graphlike_only_examples:
        print(f"     {i:4d} {ti}   +   {j:4d} {tj}")
else:
    print("   -> NONE: every confusing pair contains at least one hyperedge (mixed X&Z) error.")
