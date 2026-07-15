"""2-fault channel weight W2(alpha, p): at which alpha does the channel switch on?"""
import os, sys, time
import numpy as np
import stim, pymatching

ROOT = r"c:\Users\cesar\Documents\Python\master-thesis\Master-thesis-RL-QEC"
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords

D = 5
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

for P in (1e-4, 2e-4, 1e-3, 4e-3):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=D,
        after_clifford_depolarization=P, before_measure_flip_probability=P,
        after_reset_flip_probability=P, before_round_data_depolarization=P)
    dem = decompose_errors_for_stim_surface_code_coords(
        circ.detector_error_model(decompose_errors=False))
    ND = dem.num_detectors
    cmat = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

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
    det_masks = np.zeros((M, ND), dtype=np.uint8)
    obs_arr = np.zeros(M, dtype=np.uint8)
    for i, (pr, dets, ob) in enumerate(mechs):
        det_masks[i, list(dets)] = 1
        obs_arr[i] = ob

    # build all pair syndromes once
    blocks, ii, jj = [], [], []
    for i in range(M):
        j = np.arange(i + 1, M)
        if j.size:
            blocks.append(det_masks[i] ^ det_masks[i + 1:])
            ii.append(np.full(j.size, i)); jj.append(j)
    shots = np.vstack(blocks)
    ii = np.concatenate(ii); jj = np.concatenate(jj)
    ob = obs_arr[ii] ^ obs_arr[jj]
    w = probs[ii] * probs[jj]

    out = []
    for a in ALPHAS:
        pred = cmat.decode_batch(shots, enable_correlations=True, alpha=a)[:, 0]
        bad = pred != ob
        # normalise the channel weight to (p/1e-3)^2 so different p are comparable
        Wnorm = float(w[bad].sum()) * (1e-3 / P) ** 2
        out.append((a, int(bad.sum()), Wnorm))
    print(f"p={P:.0e}: " + "  ".join(f"a={a:g}:{n} ({Wn:.1e})" for a, n, Wn in out), flush=True)
print("\n(count of failing 2-fault pairs, and channel weight normalised to p=1e-3 scale)")
