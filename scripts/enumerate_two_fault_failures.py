"""Exhaustive 2-fault enumeration: does CM(alpha=1) lose effective distance?

For d=5 (and optionally d=7), p=2e-4:
  - enumerate every DEM error mechanism (detectors, obs, prob)
  - check every single mechanism decodes correctly (fault distance >= 2)
  - enumerate ALL pairs of mechanisms; decode the XOR syndrome with
    MWPM / CM(alpha=1) / CM(alpha=alpha_best)
  - count failing pairs per decoder and sum their probability weight
    -> predicted 2-error LER contribution  sum_{failing pairs} p_i p_j
  - compare with the measured excess LER of CM(1) over best-alpha at that p.
"""
import os, sys, time
import numpy as np
import stim, pymatching

ROOT = r"c:\Users\cesar\Documents\Python\master-thesis\Master-thesis-RL-QEC"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords

D, P, ALPHA_BEST = 5, 2e-4, 0.2

circ = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=D, rounds=D,
    after_clifford_depolarization=P, before_measure_flip_probability=P,
    after_reset_flip_probability=P, before_round_data_depolarization=P)
dem = decompose_errors_for_stim_surface_code_coords(
    circ.detector_error_model(decompose_errors=False))
ND = dem.num_detectors

mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
cmat = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

# --- extract mechanisms: (prob, detector-set (parity-reduced), obs flip) -------------
mechs = []
for inst in dem.flattened():
    if inst.type != "error":
        continue
    pr = inst.args_copy()[0]
    dets, obs = set(), 0
    for t in inst.targets_copy():
        if t.is_relative_detector_id():
            dets ^= {t.val}
        elif t.is_logical_observable_id():
            obs ^= 1
    mechs.append((pr, tuple(sorted(dets)), obs))
M = len(mechs)
probs = np.array([m[0] for m in mechs])
print(f"d={D} p={P:g}: {M} mechanisms, {ND} detectors")

det_masks = np.zeros((M, ND), dtype=np.uint8)
obs_arr = np.zeros(M, dtype=np.uint8)
for i, (pr, dets, ob) in enumerate(mechs):
    det_masks[i, list(dets)] = 1
    obs_arr[i] = ob

def decode_all(shots):
    pm = mwpm.decode_batch(shots)[:, 0]
    p1 = cmat.decode_batch(shots, enable_correlations=True, alpha=1.0)[:, 0]
    pb = cmat.decode_batch(shots, enable_correlations=True, alpha=ALPHA_BEST)[:, 0]
    return pm, p1, pb

# --- single mechanisms: all decoders must be perfect (fault distance >= 2) -----------
pm, p1, pb = decode_all(det_masks)
print("single-fault failures  MWPM:", int((pm != obs_arr).sum()),
      " CM(1):", int((p1 != obs_arr).sum()),
      f" CM({ALPHA_BEST}):", int((pb != obs_arr).sum()))

# --- all pairs ------------------------------------------------------------------------
t0 = time.time()
fail_w = {"mwpm": 0.0, "cm1": 0.0, "cmb": 0.0}
fail_n = {"mwpm": 0, "cm1": 0, "cmb": 0}
n_cm1_only = 0
examples = []
CH = 400  # rows of i per chunk
for i0 in range(0, M, CH):
    i1 = min(i0 + CH, M)
    blocks, iidx, jidx = [], [], []
    for i in range(i0, i1):
        j = np.arange(i + 1, M)
        if j.size == 0:
            continue
        blocks.append(det_masks[i] ^ det_masks[i + 1:])
        iidx.append(np.full(j.size, i)); jidx.append(j)
    if not blocks:
        continue
    shots = np.vstack(blocks)
    ii = np.concatenate(iidx); jj = np.concatenate(jidx)
    ob = obs_arr[ii] ^ obs_arr[jj]
    pm, p1, pb = decode_all(shots)
    w = probs[ii] * probs[jj]
    for key, pred in (("mwpm", pm), ("cm1", p1), ("cmb", pb)):
        bad = pred != ob
        fail_n[key] += int(bad.sum()); fail_w[key] += float(w[bad].sum())
    sel = (p1 != ob) & (pm == ob) & (pb == ob)
    n_cm1_only += int(sel.sum())
    if len(examples) < 8:
        for k in np.flatnonzero(sel)[: 8 - len(examples)]:
            examples.append((int(ii[k]), int(jj[k]), float(w[k])))
    if (i0 // CH) % 5 == 0:
        print(f"  rows {i0:4d}/{M}  t={time.time()-t0:.0f}s", flush=True)

npairs = M * (M - 1) // 2
print(f"\nAll {npairs:,} mechanism pairs decoded in {time.time()-t0:.0f}s")
print(f"{'decoder':10s} {'failing pairs':>14s} {'sum p_i p_j (2-err LER)':>26s}")
for key, name in (("mwpm", "MWPM"), ("cm1", "CM(1)"), ("cmb", f"CM({ALPHA_BEST})")):
    print(f"{name:10s} {fail_n[key]:>14,} {fail_w[key]:>26.3e}")
print(f"\npairs where ONLY CM(1) fails (MWPM & damped correct): {n_cm1_only:,}")
print(f"measured LER at p={P:g}: MWPM 1.056e-6, CM(1) 9.62e-7, best-a 5.54e-7"
      f"  -> measured excess CM(1)-best = 4.1e-7")
print("\nexample CM(1)-only failing pairs (i, j, p_i*p_j):")
for i, j, w in examples:
    print(f"  mech {i} dets={mechs[i][1]} obs={mechs[i][2]}  +  "
          f"mech {j} dets={mechs[j][1]} obs={mechs[j][2]}   w={w:.2e}")
