#!/usr/bin/env python
"""Regularized Correlated Matching alpha-scan: LER + latency vs (distance, p, alpha).

Sibling of scripts/regularized_cm_sweep.py, but with a FIXED alpha grid
(0.1, 0.2, ..., 0.9) for every p. These runs are for analysing the *dependence of the
logical error rate on alpha* (the full LER(alpha) curve), not for locating the precise
optimum -- so the grid is the same at every p rather than a bracket around alpha*(p).

Rotated surface code, Z-memory, circuit-level depolarizing noise, Tesseract
(coordinate-aware) DEM decomposition. For every (distance, p) it decodes streamed shots
(common random numbers across all decoders) with:

  * MWPM                      (enable_correlations=False)
  * CM (hard, alpha=1.0)      (enable_correlations=True, no damping)   -- always included
  * Regularized CM at each alpha in --alphas (default 0.1..0.9)

Shared with the sweep (see NeuralCM.mc_collect):
  * Adaptive batching: keep sampling until the BEST decoder has accumulated
    >= --target-errors logical errors, so EVERY alpha on the grid has at least that many
    -> a clean, tight LER(alpha) curve. Stops at --max-shots or --max-seconds.
  * --n-workers N: independent-seed worker processes, pooled, CRN preserved -> one (d,p)
    point uses N cores (set N = cpus-per-task) for the expensive low-p / high-distance points.
  * Incremental atomic CSV every step; decode timing (us/shot per decoder); flexible CLI.

The `best_at_endpoint` column flags when the lowest-LER alpha sits on a grid endpoint
(e.g. best == 0.1 means alpha* is below the grid) -- informative for the analysis.

Examples:
  python scripts/regularized_cm_alpha_scan.py --distances 5 --p-values 1e-2,4e-3,1e-3,4e-4,1e-4
  python scripts/regularized_cm_alpha_scan.py --distances 7 --p-values 1e-4 --n-workers 16 --max-seconds 82000
"""
import argparse
import csv
import os
import sys
import time

import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.mc_collect import collect


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=str, default="5,7",
                    help="comma-separated code distances (one per job to parallelise)")
    ap.add_argument("--p-values", type=str,
                    default="1e-2,7e-3,4e-3,2e-3,1e-3,7e-4,4e-4,2e-4,1e-4",
                    help="comma-separated physical error rates")
    ap.add_argument("--rounds", type=int, default=0, help="syndrome rounds (0 = use distance)")

    ap.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="fixed comma-separated damped alphas, tested at EVERY p. "
                         "Hard CM (alpha=1.0) and MWPM are always added on top.")

    ap.add_argument("--target-errors", type=int, default=500,
                    help="sample until the BEST decoder reaches this many errors (so every "
                         "alpha on the grid has >= this many); 500 here since the fixed grid "
                         "tries more alphas than the optimum-finding sweep")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="shots per sampling+decode chunk")
    ap.add_argument("--max-shots", type=int, default=20_000_000_000, help="per-(d,p) shot cap")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-time budget for the WHOLE run (0 = unlimited); stops gracefully")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="independent-seed worker processes per (d,p); set = cpus-per-task. "
                         "CRN across alphas is preserved when pooling.")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--out-csv", type=str, default="",
                    help="output CSV (default: data/reg_cm_alpha_scan_d<dists>[_<tag>].csv). "
                         "Use a UNIQUE path per parallel job.")
    ap.add_argument("--tag", type=str, default="", help="suffix for the default out-csv name")
    return ap.parse_args()


# =============================================================================
# CSV (incremental, atomic)
# =============================================================================
FIELDS = ["distance", "rounds", "p", "decoder", "alpha", "shots", "errors",
          "ler", "ler_std", "decode_seconds", "us_per_shot", "is_best", "best_at_endpoint"]


def _ler_std(err, shots):
    l = err / shots if shots else 0.0
    return l, (l * (1 - l) / shots) ** 0.5 if shots else 0.0


def write_csv(path, results):
    """results[(d,p)] = dict(rounds, shots, err={key:n}, tsec={key:s}). Atomic rewrite."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (d, p) in sorted(results):
            r = results[(d, p)]
            shots = r["shots"]
            best_key = min(r["err"], key=lambda k: r["err"][k])
            damped = sorted(a for (dec, a) in r["err"] if dec == "cm" and a < 1.0)
            best_endpoint = 0
            if best_key[0] == "cm" and best_key[1] < 1.0 and damped:
                if best_key[1] in (damped[0], damped[-1]):
                    best_endpoint = 1
            for key in sorted(r["err"], key=lambda k: (k[0], k[1] if k[1] is not None else -1)):
                dec, a = key
                err = r["err"][key]
                tsec = r["tsec"][key]
                ler, std = _ler_std(err, shots)
                w.writerow({
                    "distance": d, "rounds": r["rounds"], "p": p,
                    "decoder": dec, "alpha": "" if a is None else a,
                    "shots": shots, "errors": err, "ler": ler, "ler_std": std,
                    "decode_seconds": round(tsec, 4),
                    "us_per_shot": round(tsec / shots * 1e6, 4) if shots else 0.0,
                    "is_best": int(key == best_key),
                    "best_at_endpoint": best_endpoint,
                })
    os.replace(tmp, path)


# =============================================================================
# Main
# =============================================================================
def fmt_time(s):
    s = int(s)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main():
    args = parse_args()
    distances = [int(x) for x in args.distances.split(",")]
    p_values = [float(x) for x in args.p_values.split(",")]

    alphas = sorted(set(round(float(x), 4) for x in args.alphas.split(",")))
    if 1.0 not in alphas:
        alphas = alphas + [1.0]

    out_csv = args.out_csv
    if not out_csv:
        tag = f"_{args.tag}" if args.tag else ""
        out_csv = f"data/reg_cm_alpha_scan_d{'-'.join(map(str, distances))}{tag}.csv"

    print(f"pymatching: {pymatching.__file__}")
    print(f"distances={distances}  p_values={p_values}  n_workers={args.n_workers}")
    print(f"alphas (fixed, all p)={alphas}")
    print(f"target_errors={args.target_errors}  chunk={args.chunk:,}  "
          f"max_shots={args.max_shots:,}  max_seconds={args.max_seconds or 'inf'}")
    print(f"out_csv={out_csv}\n", flush=True)

    results = {}
    t_all = time.time()
    stop_all = False

    for d in distances:
        if stop_all:
            break
        rounds = d if args.rounds == 0 else args.rounds
        for p in p_values:
            if stop_all:
                break
            deadline = (t_all + args.max_seconds) if args.max_seconds else None
            t0 = time.time()

            keys = [("mwpm", None)] + [("cm", a) for a in alphas]
            results[(d, p)] = dict(rounds=rounds, shots=0,
                                   err={k: 0 for k in keys}, tsec={k: 0.0 for k in keys})
            print(f"=== d={d} r={rounds} p={p:.3e} | alphas={alphas} ===", flush=True)

            def upd(err, tsec, shots, _dp=(d, p)):
                results[_dp]["err"], results[_dp]["tsec"], results[_dp]["shots"] = err, tsec, shots
                write_csv(out_csv, results)
                best = min(err, key=lambda k: err[k])
                ba = best[1] if best[0] == "cm" else "MWPM"
                print(f"  shots={shots:>12,} best={ba} err={err[best]}/{args.target_errors} | "
                      f"MWPM={err[('mwpm', None)]} CM1={err[('cm', 1.0)]} | "
                      f"{fmt_time(time.time()-t0)}", flush=True)

            err, tsec, shots, reason = collect(
                d=d, rounds=rounds, p=p, alphas=alphas,
                target_errors=args.target_errors, n_workers=args.n_workers,
                seed=args.seed, chunk=args.chunk, max_shots=args.max_shots,
                deadline=deadline, on_update=upd)

            damped = sorted(a for a in alphas if a < 1.0)
            best_key = min(err, key=lambda k: err[k])
            best_a = best_key[1] if best_key[0] == "cm" else None
            endpoint = (best_a is not None and best_a < 1.0 and damped
                        and best_a in (damped[0], damped[-1]))
            l_best, _ = _ler_std(err[best_key], shots)
            l_mwpm, _ = _ler_std(err[("mwpm", None)], shots)
            l_cm1, _ = _ler_std(err[("cm", 1.0)], shots)
            print(f"  -> shots={shots:,} | best={'a='+str(best_a) if best_a else 'MWPM'} "
                  f"LER={l_best:.3e} | MWPM={l_mwpm:.3e} | CM(1)={l_cm1:.3e} | "
                  f"MWPM/shot={tsec[('mwpm',None)]/shots*1e6:.2f}us "
                  f"CM(1)/shot={tsec[('cm',1.0)]/shots*1e6:.2f}us"
                  + ("  [best at grid endpoint -> alpha* outside "
                     f"[{damped[0]:g},{damped[-1]:g}]]" if endpoint else "")
                  + f" | wrote {out_csv}\n", flush=True)

            if reason == "deadline":
                print("  [walltime budget reached — stopping]", flush=True)
                stop_all = True

    print(f"done in {fmt_time(time.time() - t_all)}  ->  {out_csv}")


if __name__ == "__main__":
    main()
