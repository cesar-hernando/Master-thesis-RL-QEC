"""
LER vs physical error rate for the Tesseract decoder (run under Linux / WSL2 / cluster).

Rotated surface-code Z-memory, circuit-level depolarizing noise, swept over a p-grid processed from
HIGH p to LOW p (the feasible points first).  One or several code distances can be swept in a single
run (``--distance 5,7``).  Adaptive shot count: keep sampling a given (d, p) point until
``>= --min-errors`` Tesseract logical errors OR ``--max-shots`` OR ``--max-seconds`` is hit.  MWPM
(public pymatching, on the coordinate-aware decomposed DEM) is decoded on the SAME shots as a free,
directly-comparable baseline (common random numbers).

Tesseract is a slow *single-shot* decoder, so it is the bottleneck.  Within a single (d, p) point the
shot decoding is parallelised across ``--n-workers`` processes (each decodes an independent
shot-stream; their error/shot counts are summed).  Rows are check-pointed to the CSV every round, so a
slow/low-p point never costs you the earlier ones and a killed job keeps its partial counts.

FEASIBILITY: deep sub-threshold points need enormous shot counts (e.g. at p=2e-4, LER~1e-6 => ~3e8
shots for 300 errors) and will hit --max-shots / --max-seconds first (wider error bars; the actual
error count is always recorded).  Use --max-seconds and/or the cluster for the low-p tail.

Deps:  pip install tesseract-decoder stim numpy pymatching
Examples:
  # laptop, d=5, custom small grid, single process:
  python ler_vs_p_tesseract.py --distance 5 --p-list 8e-4,6e-4,4e-4,2e-4 --n-workers 1 --min-errors 300
  # sweep d=5 and d=7 on the default grid, 16 workers:
  python ler_vs_p_tesseract.py --distance 5,7 --n-workers 16
  # one (distance, p) unit for a SLURM array task (index 0 == first (d, p) pair):
  python ler_vs_p_tesseract.py --distance 7 --task-index $SLURM_ARRAY_TASK_ID \
         --n-workers $SLURM_CPUS_PER_TASK --out data/ler_vs_p_tesseract/tess_task0.csv
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tesseract_decoder import tesseract                                          # noqa: E402
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords  # noqa: E402

# default grid (HIGH -> LOW, feasible points first); same grid as reg_cm_alpha_scan d=7.
P_ALL = [1e-2, 7e-3, 4e-3, 2e-3, 1e-3, 7e-4, 4e-4, 2e-4, 1e-4]
FIELDS = ["distance", "p", "tess_mean", "tess_std", "mwpm_mean", "mwpm_std",
          "n_shots", "tess_errors", "mwpm_errors", "det_beam", "seconds"]

_G = {}   # per-worker globals (built once per (d, p) by the Pool initializer)


def make_circuit(D, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=D, distance=D,
        before_round_data_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, after_clifford_depolarization=p)


def _init(D, p, det_beam):
    circ = make_circuit(D, p)
    _G["circ"] = circ
    _G["n_obs"] = circ.num_observables
    _G["tess"] = tesseract.TesseractConfig(
        dem=circ.detector_error_model(decompose_errors=False), det_beam=det_beam).compile_decoder()
    # MWPM: coordinate-aware (tesseract-style) decomposition, not Stim's heuristic.
    _G["mwpm"] = pymatching.Matching.from_detector_error_model(
        decompose_errors_for_stim_surface_code_coords(
            circ.detector_error_model(decompose_errors=False)))


def _decode(args):
    """Sample + decode `n` shots with a private seed; return (n, tess_err, mwpm_err) on the SAME shots."""
    seed, n = args
    circ, n_obs, tess, mwpm = _G["circ"], _G["n_obs"], _G["tess"], _G["mwpm"]
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)
    dets = np.asarray(dets, dtype=bool); obs = np.asarray(obs, dtype=np.uint8)
    pred_m = np.asarray(mwpm.decode_batch(dets), dtype=np.uint8)
    mwpm_err = int(np.sum(np.any(pred_m != obs, axis=1)))
    tess_err = 0
    for i in range(n):
        pred_t = np.asarray(tess.decode(dets[i]), dtype=np.uint8).reshape(n_obs)
        if np.any(pred_t != obs[i]):
            tess_err += 1
    return n, tess_err, mwpm_err


def se(ler, n):
    return float(np.sqrt(ler * (1.0 - ler) / n)) if n > 0 else float("nan")


def flush(out, rows, partial=None):
    """Atomically (re)write the CSV = completed rows [+ current partial row]."""
    all_rows = rows + ([partial] if partial else [])
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(all_rows)
    os.replace(tmp, out)


def run_p(D, p, task_id, a, rows, out, t_start):
    """Sweep one (D, p) point to --min-errors (or a cap); check-point every round.  Returns the row."""
    n_shots = tess_err = mwpm_err = 0
    seed_base = a.seed + task_id * 10_000_000
    t0 = time.time()
    rnd = 0
    partial = {"distance": D, "p": p, "tess_mean": float("nan"), "tess_std": float("nan"),
               "mwpm_mean": float("nan"), "mwpm_std": float("nan"), "n_shots": 0,
               "tess_errors": 0, "mwpm_errors": 0, "det_beam": a.det_beam, "seconds": 0.0}
    with Pool(a.n_workers, initializer=_init, initargs=(D, p, a.det_beam)) as pool:
        while (tess_err < a.min_errors and n_shots < a.max_shots
               and (time.time() - t_start) < a.max_seconds):
            batch = [(seed_base + rnd * a.n_workers + w, a.chunk) for w in range(a.n_workers)]
            for n, te, me in pool.map(_decode, batch):
                n_shots += n; tess_err += te; mwpm_err += me
            rnd += 1
            tl, ml = tess_err / n_shots, mwpm_err / n_shots
            partial = {"distance": D, "p": p, "tess_mean": tl, "tess_std": se(tl, n_shots),
                       "mwpm_mean": ml, "mwpm_std": se(ml, n_shots), "n_shots": n_shots,
                       "tess_errors": tess_err, "mwpm_errors": mwpm_err, "det_beam": a.det_beam,
                       "seconds": round(time.time() - t0, 1)}
            flush(out, rows, partial)                       # checkpoint every round
            print(f"  d={D} p={p:<7g} {n_shots:>14,} shots | tess_err={tess_err:>4} "
                  f"mwpm_err={mwpm_err:>4} | {n_shots/(time.time()-t0):,.0f} sh/s", flush=True)
    if tess_err < a.min_errors:
        print(f"  [!] d={D} p={p}: stopped with {tess_err} tesseract errors "
              f"(target {a.min_errors}) — wide error bar.", flush=True)
    return partial


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", required=True,
                    help="code distance(s), comma-separated, e.g. '5' or '5,7'")
    ap.add_argument("--p-list", default=None,
                    help="comma-separated physical error rates; overrides the default grid")
    ap.add_argument("--task-index", type=int, default=None,
                    help="run only the i-th (distance, p) pair of the flattened grid (SLURM array); "
                         "omit to sweep all pairs")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-errors", type=int, default=500)
    ap.add_argument("--max-shots", type=int, default=2_000_000_000_000)
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--det-beam", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=20_000, help="shots per worker per round")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    distances = [int(x) for x in a.distance.split(",") if x.strip()]
    P = [float(x) for x in a.p_list.split(",")] if a.p_list else list(P_ALL)
    tasks = [(d, p) for d in distances for p in P]          # flattened (distance, p) grid

    if a.task_index is not None:
        selected = [(a.task_index, tasks[a.task_index])]    # one (d, p) unit
    else:
        selected = list(enumerate(tasks))                   # all pairs

    if a.out:
        out = a.out
    elif a.task_index is not None:
        out = f"data/ler_vs_p_tesseract_task{a.task_index}.csv"
    elif len(distances) == 1:
        out = f"data/ler_vs_p_tesseract_d{distances[0]}.csv"
    else:
        out = "data/ler_vs_p_tesseract.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"distances={distances}  p-grid={P}  "
          f"{'task '+str(a.task_index) if a.task_index is not None else 'all pairs'}  "
          f"min_errors={a.min_errors}  n_workers={a.n_workers}  det_beam={a.det_beam}  "
          f"max_seconds={a.max_seconds}\nwriting -> {out}\n", flush=True)
    t_start = time.time()
    rows = []
    flush(out, rows)                                        # header immediately
    for task_id, (D, p) in selected:
        row = run_p(D, p, task_id, a, rows, out, t_start)
        rows.append(row); flush(out, rows)
        print(f"  -> d={D} p={p}: tess LER={row['tess_mean']:.3e} +/- {row['tess_std']:.1e} | "
              f"mwpm LER={row['mwpm_mean']:.3e}  (saved)\n", flush=True)
        if (time.time() - t_start) >= a.max_seconds:
            print("  reached --max-seconds; stopping.", flush=True); break
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
