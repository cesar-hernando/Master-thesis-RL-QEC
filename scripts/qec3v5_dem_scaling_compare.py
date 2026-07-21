#!/usr/bin/env python
"""Noise-scaling comparison of MWPM / CM / RCM across THREE DEM models of the same Google qec3v5
experiment (d=5, Z, r=5).  CSV-ONLY (no plots) -- meant to run on the cluster.

  pij         : Google's Spitz pairwise-CALIBRATED DEM (data-measured probabilities)
  analytical  : Google's analytical DEM = Stim decompose_errors=True (forward-model probabilities)
  proj        : decompose_errors.py (basis-strict x%2 decomposition) of the SAME analytical noise

For each (DEM, scale s) we scale every error probability p_i -> s*p_i (a uniformly quieter future
device), sample synthetic shots from the scaled DEM, and decode with MWPM and CM at
alpha = 0.1, 0.2, ..., 0.9, 1.0 (alpha=1.0 is standard correlated matching) -- all on the SAME shots.
Sampling continues until the RAREST decoder has >= --min-errors logical errors (or --max-shots /
--max-seconds is hit).  s=1 is Sycamore's own calibrated noise; the smallest scale takes the
effective physical error rate (pij p_med ~ 2.4e-3 at s=1) down to ~1e-4.

Parallelization
---------------
There are 3 DEMs x 9 scales = 27 (dem, s) tasks.  On the cluster run ONE array task per (dem, s)
via --task-index 0..26 (each writes its own one-row CSV).  WITHIN a task the Monte-Carlo is
parallelised across the task's cores with --n-workers (each worker samples+decodes an independent
shot-stream; counts are summed).  Concatenate the per-task CSVs at the end, or use --combine.

Usage
-----
    # one (dem, s) task, 16 workers, own CSV (cluster array):
    python qec3v5_dem_scaling_compare.py --task-index 0 --n-workers 16 \
        --out data/qec3v5_dem_scaling/task_0.csv

    # everything in one process (local):
    python qec3v5_dem_scaling_compare.py --n-workers 8 --out data/qec3v5_dem_scaling_compare.csv

    # merge per-task CSVs into one:
    python qec3v5_dem_scaling_compare.py --combine \
        --in-dir data/qec3v5_dem_scaling --out data/qec3v5_dem_scaling_compare.csv
"""
import argparse
import csv
import glob
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_using_detector_assignment  # noqa: E402

BASE = os.path.join(ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")

SCALES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05, 0.04]
ALPHAS = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1, 0.2, ..., 0.9, 1.0
DEMS = ["pij", "analytical", "proj"]
TASKS = [(dem, s) for dem in DEMS for s in SCALES]    # 27 tasks; index = dem_i*len(SCALES)+s_i

FIELDS = (["dem", "s", "p_med", "n_shots", "mwpm_err"] + [f"cm_err_a{a}" for a in ALPHAS]
          + ["mwpm_ler_ratio"] + [f"cm_ler_ratio_a{a}" for a in ALPHAS])


# ----------------------------------------------------------------------------- DEM construction
def scale_dem(dem, s):
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            out.append("error", min(inst.args_copy()[0] * s, 0.5), inst.targets_copy())
        elif inst.type == "detector":
            out.append(inst)
    return out


def build_dems():
    circ = stim.Circuit.from_file(os.path.join(BASE, "circuit_noisy.stim"))
    raw = circ.detector_error_model(decompose_errors=False)
    an = stim.DetectorErrorModel.from_file(os.path.join(BASE, "circuit_detector_error_model.dem"))
    pij = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))
    coords = an.get_detector_coordinates()
    proj = decompose_errors_using_detector_assignment(
        raw, lambda d: int(round(coords[d][0])) % 2, strip_undecomposable_errors=True)
    return {"pij": pij, "analytical": an, "proj": proj}


# ---------------------------------------------------------------------------- parallel sampling
# Per-worker globals built once in the initializer (avoids re-parsing the DEM every chunk).
_W = {}


def _init_worker(dem_text, chunk):
    dem = stim.DetectorErrorModel(dem_text)
    _W["dem"] = dem
    _W["chunk"] = chunk
    _W["mw"] = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    _W["cm"] = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)


def _work(seed):
    """Sample one chunk with a unique seed, decode MWPM + all alphas, return (n, mwpm_err, {a:err})."""
    smp = _W["dem"].compile_sampler(seed=seed)
    d, o, _ = smp.sample(_W["chunk"])
    d = np.asarray(d, bool); o = np.asarray(o, np.uint8).reshape(-1)
    mwpm_err = int((np.asarray(_W["mw"].decode_batch(d)).reshape(-1) != o).sum())
    cm_err = {}
    for a in ALPHAS:
        pred = np.asarray(_W["cm"].decode_batch(d, enable_correlations=True, alpha=a)).reshape(-1)
        cm_err[a] = int((pred != o).sum())
    return _W["chunk"], mwpm_err, cm_err


def _write_row(out_path, dem_name, s, p_med, n_shots, mwpm_err, cm_err):
    row = {"dem": dem_name, "s": s, "p_med": p_med, "n_shots": n_shots, "mwpm_err": mwpm_err,
           "mwpm_ler_ratio": 1.0}
    row.update({f"cm_err_a{a}": cm_err[a] for a in ALPHAS})
    denom = max(mwpm_err, 1)
    row.update({f"cm_ler_ratio_a{a}": cm_err[a] / denom for a in ALPHAS})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:      # single-row file, rewritten each checkpoint
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerow(row)


def run_task(dem_name, dem0, s, out_path, n_workers, min_errors, max_shots, max_seconds, chunk,
             base_seed):
    t0 = time.time()
    dem = scale_dem(dem0, s)
    dem_text = str(dem)
    p_med = float(np.median([i.args_copy()[0] for i in dem.flattened() if i.type == "error"]))
    n_shots = mwpm_err = 0
    cm_err = {a: 0 for a in ALPHAS}
    rnd = 0
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(dem_text, chunk)) as pool:
        while (min(mwpm_err, *cm_err.values()) < min_errors and n_shots < max_shots
               and time.time() - t0 < max_seconds):
            seeds = [base_seed + rnd * n_workers + i for i in range(n_workers)]
            rnd += 1
            for n, mwe, cme in pool.imap_unordered(_work, seeds):
                n_shots += n; mwpm_err += mwe
                for a in ALPHAS:
                    cm_err[a] += cme[a]
            _write_row(out_path, dem_name, s, p_med, n_shots, mwpm_err, cm_err)  # checkpoint
    min_err = min(mwpm_err, *cm_err.values())
    cap = "  [CAPPED<min-errors]" if min_err < min_errors else ""
    print(f"{dem_name:<11} s={s:<5} p_med={p_med:.2e} shots={n_shots:>12,} min_err={min_err:>6} "
          f"MWPM={mwpm_err/n_shots:.3e} CM={cm_err[1.0]/n_shots:.3e} "
          f"RCM0.5={cm_err[0.5]/n_shots:.3e} ({time.time()-t0:.0f}s){cap}", flush=True)


# --------------------------------------------------------------------------------------- combine
def combine(in_dir, out_path):
    files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.extend(list(csv.DictReader(f)))
    order = {(d, str(float(s))): i for i, (d, s) in enumerate(TASKS)}
    rows.sort(key=lambda r: order.get((r["dem"], str(float(r["s"]))), 1e9))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"combined {len(rows)} rows from {len(files)} files -> {out_path}")


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-index", type=int, default=None,
                    help="0..26 -> one (dem, s) task. Omit to run all 27 sequentially.")
    ap.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--min-errors", type=int, default=500,
                    help="stop a task once the RAREST decoder reaches this many logical errors")
    ap.add_argument("--max-shots", type=int, default=200_000_000)
    ap.add_argument("--max-seconds", type=float, default=float("inf"),
                    help="graceful walltime cap per task (checkpoints the CSV each round)")
    ap.add_argument("--chunk", type=int, default=500_000, help="shots per worker per round")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "qec3v5_dem_scaling_compare.csv"))
    ap.add_argument("--combine", action="store_true", help="merge per-task CSVs from --in-dir")
    ap.add_argument("--in-dir", default=os.path.join(ROOT, "data", "qec3v5_dem_scaling"))
    args = ap.parse_args()

    if args.combine:
        combine(args.in_dir, args.out)
        return

    dems = build_dems()
    if args.task_index is not None:
        dem_name, s = TASKS[args.task_index]
        base_seed = args.seed + args.task_index * 10_000_000
        run_task(dem_name, dems[dem_name], s, args.out, args.n_workers, args.min_errors,
                 args.max_shots, args.max_seconds, args.chunk, base_seed)
    else:
        # run everything, one process, into a single multi-row CSV
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        for idx, (dem_name, s) in enumerate(TASKS):
            tmp = args.out + f".task{idx}"
            base_seed = args.seed + idx * 10_000_000
            run_task(dem_name, dems[dem_name], s, tmp, args.n_workers, args.min_errors,
                     args.max_shots, args.max_seconds, args.chunk, base_seed)
            with open(tmp) as tf:
                row = list(csv.DictReader(tf))[0]
            with open(args.out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            os.remove(tmp)
        print("Done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
