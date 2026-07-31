#!/usr/bin/env python
"""Noise-scaling study on Google's calibrated Sycamore DEM (qec3v5 d=5, Z, r=5).

Take the DATA-CALIBRATED DEM (pij) as a realistic device noise MODEL, scale every error
probability p_i -> s*p_i (uniform quieter device, correlation structure preserved), Monte-Carlo
sample synthetic shots from the scaled DEM, and decode with MWPM / CM(alpha=1) / RCM(alpha grid)
[all fast, on the SAME shots per scale] and belief-matching (only at the higher scales, where its
per-shot BP cost is affordable).

Writes data/qec3v5_noise_scaling.csv incrementally (one row per scale).
"""
import csv
import os
import sys
import time

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from beliefmatching import BeliefMatching  # noqa: E402

DEM_PATH = os.path.join(ROOT, "google_qec3v5_experiment_data",
                        "surface_code_bZ_d5_r05_center_5_5", "pij_from_even_for_odd.dem")
OUT = os.path.join(ROOT, "data", "qec3v5_noise_scaling.csv")

SCALES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]        # 1.0 == hard CM
BM_SCALES = {1.0, 0.6, 0.4, 0.25}              # belief-matching only where it's affordable
TARGET_ERR = 200                               # stop a scale once CM has this many errors
MAX_SHOTS = 20_000_000
CHUNK = 500_000
BM_SHOTS = 100_000


def scale_dem(dem, s):
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            out.append("error", min(inst.args_copy()[0] * s, 0.5), inst.targets_copy())
        elif inst.type == "detector":
            out.append(inst)
    return out


def main():
    dem0 = stim.DetectorErrorModel.from_file(DEM_PATH)
    fields = (["s", "p_med", "n_shots", "mwpm_err"]
              + [f"cm_err_a{a}" for a in ALPHAS] + ["bm_shots", "bm_err"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for s in SCALES:
        t0 = time.time()
        dem = scale_dem(dem0, s)
        p_med = float(np.median([i.args_copy()[0] for i in dem.flattened() if i.type == "error"]))
        smp = dem.compile_sampler(seed=12345)
        mw = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
        cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)

        n_shots = mwpm_err = 0
        cm_err = {a: 0 for a in ALPHAS}
        while cm_err[1.0] < TARGET_ERR and n_shots < MAX_SHOTS:
            d, o, _ = smp.sample(CHUNK)
            d = np.asarray(d, bool); o = np.asarray(o, np.uint8).reshape(-1)
            mwpm_err += int((np.asarray(mw.decode_batch(d)).reshape(-1) != o).sum())
            for a in ALPHAS:
                pred = np.asarray(cm.decode_batch(d, enable_correlations=True, alpha=a)).reshape(-1)
                cm_err[a] += int((pred != o).sum())
            n_shots += CHUNK

        bm_shots = bm_err = 0
        if s in BM_SCALES:
            bm = BeliefMatching(dem, max_bp_iters=20)
            d, o, _ = smp.sample(BM_SHOTS)
            d = np.asarray(d, bool); o = np.asarray(o, np.uint8).reshape(-1)
            bm_err = int((np.asarray(bm.decode_batch(d)).reshape(-1) != o).sum())
            bm_shots = BM_SHOTS

        row = {"s": s, "p_med": p_med, "n_shots": n_shots, "mwpm_err": mwpm_err,
               "bm_shots": bm_shots, "bm_err": bm_err}
        row.update({f"cm_err_a{a}": cm_err[a] for a in ALPHAS})
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"s={s:<5} p_med={p_med:.2e} shots={n_shots:>12,} "
              f"MWPM={mwpm_err/n_shots:.3e} CM={cm_err[1.0]/n_shots:.3e} "
              f"RCM0.5={cm_err[0.5]/n_shots:.3e} "
              f"BM={'-' if not bm_shots else f'{bm_err/bm_shots:.3e}'} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("Done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
