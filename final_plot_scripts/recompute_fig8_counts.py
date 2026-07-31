#!/usr/bin/env python
"""Recompute the fig-8 count caches with the NEW alpha*(p) (from reg_cm_alpha_scan_new).

Sub-threshold (weight (d-1)/2):  d5 weight-2 exhaustive; d7 weight-3 hyperedge-triple sampling
                                 scaled by C(H,3).
Threshold (weight (d+1)/2):      d5 weight-3; d7 weight-4 uniform-tuple sampling, per 1e6.

For every configuration the same shots are decoded by MWPM, CM(alpha=1) and RCM(alpha*) (CRN).
Output: data/cm_counts_fig8_newalpha.npz
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords  # noqa: E402

scan = pd.read_csv(os.path.join(ROOT, "data", "reg_cm_alpha_scan_new_combined.csv"))
PS = [2e-3, 1e-3, 4e-4, 2e-4, 1e-4]


def best_alpha(d, p):
    s = scan[(scan.distance == d) & (np.abs(scan.p - p) < 1e-12) & (scan.decoder == "cm") & (scan.alpha < 1.0)]
    return float(s.loc[s.ler.idxmin()].alpha) if len(s) else 1.0


def build(D, P):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=D,
                               after_clifford_depolarization=P, before_measure_flip_probability=P,
                               after_reset_flip_probability=P, before_round_data_depolarization=P)
    dem = decompose_errors_for_stim_surface_code_coords(c.detector_error_model(decompose_errors=False))
    return (dem, pymatching.Matching.from_detector_error_model(dem, enable_correlations=False),
            pymatching.Matching.from_detector_error_model(dem, enable_correlations=True))


def mechanisms(dem):
    ND = dem.num_detectors; ms, hy = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        d, o, sep = set(), 0, False
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): d ^= {t.val}
            elif t.is_logical_observable_id(): o ^= 1
            elif t.is_separator(): sep = True
        ms.append((frozenset(d), o)); hy.append(sep)
    M = len(ms); dm = np.zeros((M, ND), np.uint8); ob = np.zeros(M, np.uint8)
    for i, (dd, o) in enumerate(ms):
        dm[i, list(dd)] = 1; ob[i] = o
    return dm, ob, np.array(hy)


def dec(cm, s, a):
    return cm.decode_batch(s, enable_correlations=True, alpha=a)[:, 0]


def count_2fault(D, P, a):
    dem, mwpm, cm = build(D, P); dm, ob, hy = mechanisms(dem); M = len(dm)
    n1 = nr = nm = 0
    for i0 in range(0, M, 400):
        blocks, ii, jj = [], [], []
        for x in range(i0, min(i0 + 400, M)):
            b = np.arange(x + 1, M)
            if b.size:
                blocks.append(dm[x] ^ dm[x + 1:]); ii.append(np.full(b.size, x)); jj.append(b)
        if not blocks:
            continue
        s = np.vstack(blocks); ii = np.concatenate(ii); jj = np.concatenate(jj); obo = ob[ii] ^ ob[jj]
        pm = mwpm.decode_batch(s)[:, 0]
        n1 += int((dec(cm, s, 1.0) != obo).sum()); nr += int((dec(cm, s, a) != obo).sum())
        nm += int((pm != obo).sum())
    return n1, nr, nm


def count_3fault_hyper(D, P, a, budget):
    dem, mwpm, cm = build(D, P); dm, ob, hy = mechanisms(dem)
    H = np.flatnonzero(hy); nH = len(H); nTri = nH * (nH - 1) * (nH - 2) / 6.0
    rng = np.random.default_rng(0); tot = n1 = nr = nm = 0; t0 = time.time()
    while time.time() - t0 < budget:
        B = 400000
        i, j, k = rng.choice(H, B), rng.choice(H, B), rng.choice(H, B)
        ok = (i != j) & (j != k) & (i != k); i, j, k = i[ok], j[ok], k[ok]
        s = dm[i] ^ dm[j] ^ dm[k]; obo = ob[i] ^ ob[j] ^ ob[k]
        pm = mwpm.decode_batch(s)[:, 0]
        tot += len(i); n1 += int((dec(cm, s, 1.0) != obo).sum())
        nr += int((dec(cm, s, a) != obo).sum()); nm += int((pm != obo).sum())
    f = nTri / tot
    est = lambda n: (n * f, n * f / np.sqrt(max(n, 1)))
    return est(n1), est(nr), nm * f


def count_wfault_uniform(D, P, w, a, budget):
    dem, mwpm, cm = build(D, P); dm, ob, hy = mechanisms(dem); M = len(dm)
    rng = np.random.default_rng(0); tot = n1 = nr = nm = 0; t0 = time.time()
    while time.time() - t0 < budget:
        B = 100000
        idx = rng.integers(0, M, size=(w, B)); ok = np.ones(B, bool)
        for x in range(w):
            for y in range(x + 1, w):
                ok &= idx[x] != idx[y]
        idx = idx[:, ok]
        s = dm[idx[0]].copy(); obo = ob[idx[0]].copy()
        for r in range(1, w):
            s ^= dm[idx[r]]; obo ^= ob[idx[r]]
        tot += idx.shape[1]
        nm += int((mwpm.decode_batch(s)[:, 0] != obo).sum())
        n1 += int((dec(cm, s, 1.0) != obo).sum()); nr += int((dec(cm, s, a) != obo).sum())
    f = 1e6 / tot
    e = lambda n: (n * f, n * f / np.sqrt(max(n, 1)))
    return e(nm), e(n1), e(nr)


out = {"P": np.array(PS)}
# ---------------- sub-threshold ----------------
print("SUB d5 weight-2 (exhaustive)")
s = [count_2fault(5, P, best_alpha(5, P)) for P in PS]
out["sub_d5_cm1"] = np.array([x[0] for x in s]); out["sub_d5_astar"] = np.array([x[1] for x in s])
out["sub_d5_mw"] = np.array([x[2] for x in s])
print("SUB d7 weight-3 (hyperedge sampled, 90s/p)")
r7 = [count_3fault_hyper(7, P, best_alpha(7, P), 90) for P in PS]
out["sub_d7_cm1"] = np.array([x[0][0] for x in r7]); out["sub_d7_cm1e"] = np.array([x[0][1] for x in r7])
out["sub_d7_astar"] = np.array([x[1][0] for x in r7]); out["sub_d7_astare"] = np.array([x[1][1] for x in r7])
out["sub_d7_mw"] = np.array([x[2] for x in r7])
# ---------------- threshold ----------------
print("THR d5 weight-3 (uniform sampled, 25s/p)")
t5 = [count_wfault_uniform(5, P, 3, best_alpha(5, P), 25) for P in PS]
for key, idx in (("mw", 0), ("cm1", 1), ("astar", 2)):
    out[f"thr_d5_{key}"] = np.array([t5[i][idx][0] for i in range(len(PS))])
    out[f"thr_d5_{key}e"] = np.array([t5[i][idx][1] for i in range(len(PS))])
print("THR d7 weight-4 (uniform sampled, 90s/p)")
t7 = [count_wfault_uniform(7, P, 4, best_alpha(7, P), 90) for P in PS]
for key, idx in (("mw", 0), ("cm1", 1), ("astar", 2)):
    out[f"thr_d7_{key}"] = np.array([t7[i][idx][0] for i in range(len(PS))])
    out[f"thr_d7_{key}e"] = np.array([t7[i][idx][1] for i in range(len(PS))])
out["astar_d5"] = np.array([best_alpha(5, P) for P in PS])
out["astar_d7"] = np.array([best_alpha(7, P) for P in PS])

np.savez(os.path.join(ROOT, "data", "cm_counts_fig8_newalpha.npz"), **out)
print("saved data/cm_counts_fig8_newalpha.npz")
print("alpha* d5:", out["astar_d5"], " d7:", out["astar_d7"])
