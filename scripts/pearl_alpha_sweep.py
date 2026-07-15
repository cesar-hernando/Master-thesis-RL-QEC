#!/usr/bin/env python
"""Pearl-CM alpha sweep vs p -- script version of notebooks/pearl_alpha_sweep.ipynb.

Same methodology as the notebook:
  * for each p, decode the SAME streamed syndromes (common random numbers) with
    every alpha in the grid (plus 1.0 = ordinary CM, the reference and stopping
    criterion),
  * keep sampling chunks until CM (alpha=1.0) reaches `target_errors` logical
    errors (or `max_shots` is hit),
  * record per-(p, alpha) logical-error counts.

Differences from the notebook (what you asked for):
  * Live terminal progress every chunk: elapsed, shots, throughput, per-alpha
    running LER, and an ETA derived from CM's progress toward target_errors.
  * The CSV is rewritten every time a p-block finishes (all its alphas complete
    together under common random numbers), and -- optionally -- checkpointed
    mid-run, so an interrupted run keeps its completed/partial points.

Why it is slow (see also the second pass in two_pass_correlated_matching.py):
  the low-p point needs ~1e8 shots to collect a few hundred logical errors, and
  the correlated second pass is a per-shot Python loop. The high-p point is
  effectively free. Trade precision for speed with --target-errors / --chunk, or
  run a single p with --p-values.

Usage:
  python scripts/pearl_alpha_sweep.py                      # notebook defaults
  python scripts/pearl_alpha_sweep.py --p-values 3e-4      # just the slow point
  python scripts/pearl_alpha_sweep.py --target-errors 100  # faster, noisier
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from NeuralCM.syndrome_data_generation import SyndromeDataGenerator
from NeuralCM.two_pass_correlated_matching import TwoPassCorrelatedMatching


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--n-rounds", type=int, default=7)
    ap.add_argument("--bypass-threshold", type=int, default=2)
    ap.add_argument("--p-values", type=str, default="5e-4,7e-4,5e-3",
                    help="comma-separated physical error rates")
    ap.add_argument("--alpha-grid", type=str, default="0.4,0.7,0.9",
                    help="comma-separated alphas (1.0 is always added)")
    ap.add_argument("--target-errors", type=int, default=300)
    ap.add_argument("--max-shots", type=int, default=1_000_000_000_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-csv", type=str, default="data/pearl_alpha_sweep_vs_p_d7.csv")
    ap.add_argument("--print-every", type=int, default=1,
                    help="print a progress line every N chunks")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="rewrite CSV with partial counts every N chunks (0 = off)")
    return ap.parse_args()


def make_generator(p, args):
    return SyndromeDataGenerator(
        distance=args.distance, n_rounds=args.n_rounds, mismatch=1.0,
        noise_model={"version": "built-in",
                     "after_clifford_depolarization": p,
                     "before_measure_flip_probability": p,
                     "after_reset_flip_probability": p,
                     "before_round_data_depolarization": p, "p_gate_zz": 0.0},
        memory_type="z", n_shots=args.chunk, qec_code="surface_code")


def fmt_time(s):
    if s is None or s == float("inf") or s != s:
        return "  --:--:--"
    s = int(s)
    return f"{s // 3600:3d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def write_csv(path, results):
    """results: dict[(p, alpha)] -> (shots, errors). Rewrites the whole file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "alpha", "shots", "errors", "ler", "ler_std"])
        for (p, a), (shots, err) in sorted(results.items()):
            ler = err / shots if shots else 0.0
            std = (ler * (1 - ler) / shots) ** 0.5 if shots else 0.0
            w.writerow([p, a, shots, err, ler, std])
    os.replace(tmp, path)  # atomic: a reader never sees a half-written file


def main():
    args = parse_args()
    p_values = [float(x) for x in args.p_values.split(",")]
    eval_alphas = sorted(set(float(x) for x in args.alpha_grid.split(",")) | {1.0})

    print(f"Pearl-CM alpha sweep | d={args.distance} rounds={args.n_rounds} "
          f"bypass={args.bypass_threshold}")
    print(f"  p_values     = {p_values}")
    print(f"  eval_alphas  = {eval_alphas}  (1.0 = CM reference / stopping)")
    print(f"  target_errors={args.target_errors}  chunk={args.chunk:,}  "
          f"max_shots={args.max_shots:,}")
    print(f"  out_csv      = {args.out_csv}", flush=True)

    results = {}  # (p, alpha) -> (shots, errors); persisted to CSV
    t_all = time.time()

    for p in p_values:
        gen = make_generator(p, args)
        base_circuit, base_dem, _ = gen.generate_base_circuit()
        dec = TwoPassCorrelatedMatching(base_dem, alpha=1.0,
                                        bypass_threshold=args.bypass_threshold)
        sampler = base_circuit.compile_detector_sampler(seed=args.seed)

        err = {a: 0 for a in eval_alphas}
        shots = 0
        chunk_i = 0
        t_p = time.time()
        print(f"\n=== p={p:.3e} ===", flush=True)

        while shots < args.max_shots:
            syn, obs = sampler.sample(shots=args.chunk, separate_observables=True)
            obs = obs.flatten()
            for a in eval_alphas:
                dec.alpha = float(a)
                err[a] += int(np.sum(dec.decode_batch(syn)[:, 0] != obs))
            shots += args.chunk
            chunk_i += 1
            del syn, obs
            gc.collect()

            done = err[1.0] >= args.target_errors
            if chunk_i % args.print_every == 0 or done:
                elapsed = time.time() - t_p
                rate = shots / elapsed if elapsed else 0.0
                cm = err[1.0]
                eta = elapsed * (args.target_errors / cm - 1) if cm > 0 else float("inf")
                per_alpha = "  ".join(
                    f"a{a:g}:{err[a]:>4d}/{err[a]/shots:.2e}" for a in eval_alphas)
                print(f"[{fmt_time(elapsed)}] p={p:.1e} shots={shots:>12,} "
                      f"{rate:>6,.0f}/s | CM {cm:>3d}/{args.target_errors} "
                      f"ETA{fmt_time(eta)} | {per_alpha}", flush=True)

            if args.checkpoint_every and chunk_i % args.checkpoint_every == 0 and not done:
                snap = dict(results)
                for a in eval_alphas:
                    snap[(p, a)] = (shots, err[a])
                write_csv(args.out_csv, snap)

            if done:
                break

        # Finalize this p: commit all its (p, alpha) rows.
        for a in eval_alphas:
            results[(p, a)] = (shots, err[a])
        write_csv(args.out_csv, results)

        cm_ler = err[1.0] / shots
        best_a = min(eval_alphas, key=lambda a: err[a])
        best_ler = err[best_a] / shots
        verdict = ("CM optimal" if best_ler >= cm_ler
                   else f"best a={best_a:g} improves {100 * (1 - best_ler / cm_ler):.1f}%")
        print(f"--- DONE p={p:.3e} in {fmt_time(time.time() - t_p)} | shots={shots:,} | "
              f"CM LER={cm_ler:.3e} | {verdict} | wrote {len(eval_alphas)} rows -> "
              f"{args.out_csv}", flush=True)

        del gen, base_circuit, base_dem, dec, sampler
        gc.collect()

    print(f"\nAll p done in {fmt_time(time.time() - t_all)}. CSV: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
