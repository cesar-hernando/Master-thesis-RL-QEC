#!/usr/bin/env python
"""Belief-matching (5 and 20 BP iters) on the noise-scaled Spitz-calibrated pij DEM, to overlay on
the pij ratio-vs-s figure.  For each scale s we sample from the scaled pij DEM and decode MWPM +
BM(5) + BM(20) on the SAME shots (common random numbers), so LER_BM/LER_MWPM is a clean ratio.

BM's per-shot BP cost makes deep-subthreshold scales unaffordable, so we only go down to the scales
where BM can reach enough errors within the shot/time budget.  Writes data/qec3v5_pij_bm_scaling.csv.
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

BASE = os.path.join(ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")
OUT = os.path.join(ROOT, "data", "qec3v5_pij_bm_scaling.csv")

SCALES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1]     # BM too slow below this
TARGET_BM_ERR = 150
MAX_SHOTS = 3_000_000
MAX_SECONDS = 900          # per-scale wall-clock budget
CHUNK = 20_000


def scale_dem(dem, s):
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            out.append("error", min(inst.args_copy()[0] * s, 0.5), inst.targets_copy())
        elif inst.type == "detector":
            out.append(inst)
    return out


def main():
    pij = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))
    fields = ["s", "p_med", "n_shots", "mwpm_err", "bm5_err", "bm20_err"]
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for s in SCALES:
        t0 = time.time()
        dem = scale_dem(pij, s)
        p_med = float(np.median([i.args_copy()[0] for i in dem.flattened() if i.type == "error"]))
        smp = dem.compile_sampler(seed=2025)
        mw = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
        bm5 = BeliefMatching(dem, max_bp_iters=5)
        bm20 = BeliefMatching(dem, max_bp_iters=20)
        n = mwe = e5 = e20 = 0
        while (min(e5, e20) < TARGET_BM_ERR and n < MAX_SHOTS
               and time.time() - t0 < MAX_SECONDS):
            d, o, _ = smp.sample(CHUNK)
            d = np.asarray(d, bool); o = np.asarray(o, np.uint8).reshape(-1)
            mwe += int((np.asarray(mw.decode_batch(d)).reshape(-1) != o).sum())
            e5 += int((np.asarray(bm5.decode_batch(d)).reshape(-1) != o).sum())
            e20 += int((np.asarray(bm20.decode_batch(d)).reshape(-1) != o).sum())
            n += CHUNK
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(
                {"s": s, "p_med": p_med, "n_shots": n, "mwpm_err": mwe, "bm5_err": e5,
                 "bm20_err": e20})
        print(f"s={s:<5} p_med={p_med:.2e} shots={n:>9,} MWPM={mwe/n:.3e} "
              f"BM5={e5/n:.3e} BM20={e20/n:.3e} ({time.time()-t0:.0f}s)", flush=True)
    print("Done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
