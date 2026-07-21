"""
LER vs physical error rate for the Tesseract decoder, distance 7 (run under Linux / cluster).

Rotated surface-code Z-memory, circuit-level depolarizing noise, swept over the SAME p-grid
as the new reg_cm_alpha_scan d=7 data, processed from HIGH p to LOW p (the feasible points
first).  Adaptive shot count: keep sampling until >= --min-errors Tesseract logical errors OR
--max-shots OR --max-seconds is hit.  MWPM (public pymatching) is decoded on the SAME shots as a
free, directly-comparable baseline (CRN).

Tesseract is a slow *single-shot* decoder, so it is the bottleneck.  Within a single (d, p) point
the shot decoding is parallelised across `--n-workers` processes (each decodes an independent
shot-stream; their error/shot counts are summed).  Rows are check-pointed to the CSV every round,
so a slow/low-p point never costs you the earlier ones and a killed job keeps its partial counts.

Deps:  pip install tesseract-decoder stim numpy pymatching
Examples:
  # all p (descending), 16 workers, one CSV:
  python ler_vs_p_tesseract_d7.py --n-workers 16
  # one p (for a SLURM array task): p-index 0 == highest p (1e-2)
  python ler_vs_p_tesseract_d7.py --p-index $SLURM_ARRAY_TASK_ID --n-workers $SLURM_CPUS_PER_TASK \
         --out data/ler_vs_p_tesseract_d7/tess_d7_p0.csv
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
import stim
import pymatching
from tesseract_decoder import tesseract

from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords

# same grid as reg_cm_alpha_scan d=7, HIGH -> LOW (feasible points first)
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


def run_p(D, p, p_index, a, rows, out, t_start):
    n_shots = tess_err = mwpm_err = 0
    seed_base = a.seed + p_index * 10_000_000
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--p-index", type=int, default=None,
                    help="run only P_ALL[p-index] (for a SLURM array task); omit to sweep all")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-errors", type=int, default=500)
    ap.add_argument("--max-shots", type=int, default=2_000_000_000_000)
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--det-beam", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=20_000, help="shots per worker per round")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    D = a.distance
    if a.p_index is not None:
        plist = [(a.p_index, P_ALL[a.p_index])]
    else:
        plist = list(enumerate(P_ALL))
    out = a.out or (f"data/ler_vs_p_tesseract_d{D}.csv" if a.p_index is None
                    else f"data/ler_vs_p_tesseract_d{D}_p{a.p_index}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"d={D}  p-grid={[P_ALL[i] for i, _ in plist]}  min_errors={a.min_errors}  "
          f"n_workers={a.n_workers}  det_beam={a.det_beam}  max_seconds={a.max_seconds}\n"
          f"writing -> {out}\n", flush=True)
    t_start = time.time()
    rows = []
    flush(out, rows)                                        # header immediately
    for p_index, p in plist:
        row = run_p(D, p, p_index, a, rows, out, t_start)
        rows.append(row); flush(out, rows)
        print(f"  -> d={D} p={p}: tess LER={row['tess_mean']:.3e} +/- {row['tess_std']:.1e} | "
              f"mwpm LER={row['mwpm_mean']:.3e}  (saved)\n", flush=True)
        if (time.time() - t_start) >= a.max_seconds:
            print("  reached --max-seconds; stopping.", flush=True); break
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
