#!/usr/bin/env python
"""Regularized Correlated Matching sweep: LER + latency vs (distance, p, alpha).

For each (distance, p) it decodes streamed shots (common random numbers across all
decoders) with MWPM, hard CM (alpha=1.0), and a small grid of ~4 damped alphas, in a
SINGLE pass, and reports the best. The 4 alphas are chosen from the coarser
`reg_cm_alpha_scan` results (the fixed 0.1..0.9 LER-vs-alpha curves): we look up the
scan's best alpha a* for this (d,p) and refine around it at 0.05 resolution ->

  * a* = 0.1  (scan's low edge; optimum at/below 0.1) -> [0.01, 0.05, 0.10, 0.15]
  * a* = 0.9  (scan's high edge; optimum toward hard CM) -> [0.80, 0.85, 0.90, 0.95]
  * interior a* -> a 0.05 grid straddling a*, leaning to the better-scoring scan
    neighbour, e.g. a*=0.4 -> [0.35, 0.40, 0.45, 0.50] (or [0.30, 0.35, 0.40, 0.45]).

Since the scan already brackets the optimum (a* beat both 0.1-spaced neighbours), the
refined best lands on an intermediate value. For (d,p) not in the scan (e.g. d=9) it falls
back to the nearest scanned distance at the same p (a*(p) is nearly distance-independent).

Rotated surface code, Z-memory, circuit-level depolarizing noise, Tesseract
(coordinate-aware) DEM decomposition.

Parallelism / HPC:
  * --n-workers N spreads sampling over N independent-seed processes, pooled. CRN across
    alphas is preserved, so one (d,p) point uses N cores (set N = cpus-per-task).
  * Incremental atomic CSV every step; decode timing (us/shot per decoder); --max-seconds
    wall-time budget with graceful stop; pass a subset of --distances/--p-values + a unique
    --out-csv to parallelise across cluster array tasks.

Examples:
  python scripts/regularized_cm_sweep.py --distances 5 --p-values 4e-3 --n-workers 8
  python scripts/regularized_cm_sweep.py --distances 9 --p-values 1e-3 --n-workers 16   # d=9 -> d=7 scan
  python scripts/regularized_cm_sweep.py --distances 5 --p-values 4e-3 --alphas 0.5,0.6,0.7  # override
"""
import argparse
import csv
import glob
import math
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

    # --- alpha grid: chosen from the reg_cm_alpha_scan results ---
    ap.add_argument("--alphas", type=str, default="",
                    help="explicit comma-separated alphas for ALL p (overrides the scan-based "
                         "selection). 1.0 (hard CM) is always added.")
    ap.add_argument("--scan-csv", type=str,
                    default="data/reg_cm_alpha_scan/*.csv,data/reg_cm_alpha_scan_*.csv",
                    help="comma-separated glob(s) of reg_cm_alpha_scan CSVs used to pick alphas")

    # --- statistics / batching ---
    ap.add_argument("--target-errors", type=int, default=1000,
                    help="sample until the BEST decoder reaches this many logical errors")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="shots per sampling+decode chunk")
    ap.add_argument("--max-shots", type=int, default=20_000_000_000, help="per-(d,p) shot cap")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-time budget for the WHOLE run (0 = unlimited); stops gracefully")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="independent-seed worker processes per (d,p); set = cpus-per-task. "
                         "CRN across alphas is preserved when pooling.")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--out-csv", type=str, default="",
                    help="output CSV (default: data/regularized_cm_sweep_d<dists>[_<tag>].csv). "
                         "Use a UNIQUE path per parallel job.")
    ap.add_argument("--tag", type=str, default="", help="suffix for the default out-csv name")
    return ap.parse_args()


# =============================================================================
# Scan-driven alpha selection
# =============================================================================
def read_scan_best(globs):
    """Parse reg_cm_alpha_scan CSVs -> {(d, p): (a_star, ler_below, ler_above)}.

    a_star = argmin-LER damped alpha on the scan grid; ler_below/above are the LERs of its
    grid neighbours (inf if a_star is on the grid edge)."""
    files = []
    for gp in globs:
        files += glob.glob(gp.strip())
    rows = {}
    for f in sorted(set(files)):
        try:
            for r in csv.DictReader(open(f)):
                if r.get("decoder") != "cm" or not r.get("alpha"):
                    continue
                a = float(r["alpha"])
                if a >= 1.0:
                    continue
                d, p = int(r["distance"]), float(r["p"])
                sh, ler = int(r["shots"]), float(r["ler"])
                key = (d, p)
                rows.setdefault(key, {})
                # keep the entry with the most shots if an (d,p,alpha) appears in >1 file
                if a not in rows[key] or sh > rows[key][a][1]:
                    rows[key][a] = (ler, sh)
        except Exception as e:
            print(f"  [warn] could not read scan file {f}: {e}", flush=True)
    best = {}
    for key, bya in rows.items():
        alphas = sorted(bya)
        if not alphas:
            continue
        a_star = min(alphas, key=lambda a: bya[a][0])
        i = alphas.index(a_star)
        ler_lo = bya[alphas[i - 1]][0] if i > 0 else math.inf
        ler_hi = bya[alphas[i + 1]][0] if i < len(alphas) - 1 else math.inf
        best[key] = (round(a_star, 4), ler_lo, ler_hi)
    return best


def lookup_scan(best, d, p):
    """Return ((a_star, ler_lo, ler_hi), provenance) with distance/p fallbacks."""
    def pmatch(pp):
        return abs(pp - p) <= 1e-3 * p + 1e-12
    exact = [(dd, pp) for (dd, pp) in best if dd == d and pmatch(pp)]
    if exact:
        return best[exact[0]], f"scan d={d} p={exact[0][1]:g}"
    same_p = [(dd, pp) for (dd, pp) in best if pmatch(pp)]
    if same_p:                                   # e.g. d=9 -> nearest scanned distance
        k = min(same_p, key=lambda kv: abs(kv[0] - d))
        return best[k], f"scan nearest d={k[0]} (p={k[1]:g})"
    same_d = [(dd, pp) for (dd, pp) in best if dd == d]
    if same_d:                                   # unscanned p -> nearest p (log space)
        k = min(same_d, key=lambda kv: abs(math.log(kv[1]) - math.log(p)))
        return best[k], f"scan d={d} nearest p={k[1]:g}"
    if best:                                      # last resort: global nearest
        k = min(best, key=lambda kv: abs(kv[0] - d) + abs(math.log(kv[1]) - math.log(p)))
        return best[k], f"scan nearest d={k[0]} p={k[1]:g}"
    return (0.5, math.inf, math.inf), "DEFAULT (no scan found)"


def scan_to_alphas(a_star, ler_lo, ler_hi):
    """4 damped alphas refining around the scan's best a* (finer below 0.1)."""
    a = round(a_star, 2)
    if a <= 0.10:                 # scan's low edge -> explore finer below 0.1
        return [0.01, 0.05, 0.10, 0.15]
    if a >= 0.90:                 # scan's high edge -> explore toward hard CM (1.0 auto-added)
        return [0.80, 0.85, 0.90, 0.95]
    if ler_lo <= ler_hi:          # optimum leans below a*
        return [round(a - 0.10, 2), round(a - 0.05, 2), a, round(a + 0.05, 2)]
    return [round(a - 0.05, 2), a, round(a + 0.05, 2), round(a + 0.10, 2)]  # leans above


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

    scan_best = {} if args.alphas.strip() else read_scan_best(args.scan_csv.split(","))

    out_csv = args.out_csv
    if not out_csv:
        tag = f"_{args.tag}" if args.tag else ""
        out_csv = f"data/regularized_cm_sweep_d{'-'.join(map(str, distances))}{tag}.csv"

    print(f"pymatching: {pymatching.__file__}")
    print(f"distances={distances}  p_values={p_values}  n_workers={args.n_workers}")
    print(f"scan points loaded: {len(scan_best)}  (from {args.scan_csv})")
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

            # -- pick the 4 damped alphas --
            if args.alphas.strip():
                damped = sorted(set(round(float(x), 4) for x in args.alphas.split(",")
                                    if float(x) < 1.0))
                prov = "explicit --alphas"
            else:
                (a_star, ler_lo, ler_hi), prov = lookup_scan(scan_best, d, p)
                damped = [a for a in scan_to_alphas(a_star, ler_lo, ler_hi) if a > 0]
            alphas = sorted(set([float(a) for a in damped] + [1.0]))
            print(f"=== d={d} r={rounds} p={p:.3e} | {prov} -> alphas={alphas} ===", flush=True)

            keys = [("mwpm", None)] + [("cm", a) for a in alphas]
            results[(d, p)] = dict(rounds=rounds, shots=0,
                                   err={k: 0 for k in keys}, tsec={k: 0.0 for k in keys})

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

            damped_a = sorted(a for a in alphas if a < 1.0)
            best_key = min(err, key=lambda k: err[k])
            best_a = best_key[1] if best_key[0] == "cm" else None
            endpoint = (best_a is not None and best_a < 1.0 and damped_a
                        and best_a in (damped_a[0], damped_a[-1]))
            l_best, _ = _ler_std(err[best_key], shots)
            l_mwpm, _ = _ler_std(err[("mwpm", None)], shots)
            l_cm1, _ = _ler_std(err[("cm", 1.0)], shots)
            print(f"  -> shots={shots:,} | best={'a='+str(best_a) if best_a else 'MWPM'} "
                  f"LER={l_best:.3e} | MWPM={l_mwpm:.3e} | CM(1)={l_cm1:.3e} | "
                  f"MWPM/shot={tsec[('mwpm',None)]/shots*1e6:.2f}us "
                  f"CM(1)/shot={tsec[('cm',1.0)]/shots*1e6:.2f}us"
                  + ("  [best at grid endpoint -> optimum outside "
                     f"[{damped_a[0]:g},{damped_a[-1]:g}]]" if endpoint else "")
                  + f" | wrote {out_csv}\n", flush=True)

            if reason == "deadline":
                print("  [walltime budget reached — stopping]", flush=True)
                stop_all = True

    print(f"done in {fmt_time(time.time() - t_all)}  ->  {out_csv}")


if __name__ == "__main__":
    main()
