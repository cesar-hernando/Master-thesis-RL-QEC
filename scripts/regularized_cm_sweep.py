#!/usr/bin/env python
"""Regularized Correlated Matching sweep: LER + latency vs (distance, p, alpha).

Rotated surface code, Z-memory, circuit-level depolarizing noise, Tesseract
(coordinate-aware) DEM decomposition. For every (distance, p) it decodes streamed
shots (common random numbers across all decoders) with:

  * MWPM                      (enable_correlations=False)
  * CM (hard, alpha=1.0)      (enable_correlations=True, no damping)   -- always included
  * Regularized CM at each alpha in a SMART, per-p grid centred on the predicted
    optimum (so we don't sweep 0..1 for every p). The centre is the Pearl-fit
    alpha*(p) = L / (1 + (p0/p)^m) shifted UP to the damped-rule optimum via
    alpha_d = alpha_p + (1-alpha_p)*r(p), r = median P(mu|~nu)/P(mu|nu) from the DEM
    (the regularized rule drops Pearl's second term, so its optimum is higher; the
    two agree as p -> 0). Disable with --no-damped-correction.

Design for HPC + latency analysis:
  * Adaptive batching: keep sampling until the BEST decoder (fewest errors) has
    accumulated >= --target-errors logical errors, so the best-alpha LER has a
    tight confidence interval. Stops early at --max-shots or --max-seconds.
  * Incremental CSV: the results file is rewritten (atomically) every chunk, so a
    job killed by a walltime limit still leaves partial, usable results.
  * Decode timing: total wall-seconds and microseconds-per-shot are recorded per
    decoder for a throughput / latency comparison (MWPM vs hard CM vs regularized).
  * Flexible CLI: pass a subset of --distances / --p-values per job and a unique
    --out-csv (auto-named by distance) to parallelise across cluster jobs.

The best alpha should land on an INTERMEDIATE grid value; if the argmin sits on a
grid endpoint the run flags it (stdout + `best_at_endpoint` CSV column) so you can
re-run that point with a wider --alpha-span.

Examples:
  # one job per distance (the default out-csv name already encodes the distance):
  python scripts/regularized_cm_sweep.py --distances 5 --p-values 1e-2,4e-3,1e-3,4e-4,1e-4
  # split a distance's p-values across jobs with distinct tags:
  python scripts/regularized_cm_sweep.py --distances 7 --p-values 1e-4 --tag lowp --max-seconds 82000
  # explicit alphas (overrides the smart grid) for all p:
  python scripts/regularized_cm_sweep.py --distances 5 --p-values 4e-3 --alphas 0.4,0.6,0.8
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords
from NeuralCM.decoding_graph import DecodingGraph


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

    # --- smart alpha grid (per p): centred on alpha*(p) = L / (1 + (p0/p)^m) ---
    ap.add_argument("--alphas", type=str, default="",
                    help="explicit comma-separated alphas for ALL p (overrides the smart grid). "
                         "1.0 (hard CM) is always added.")
    ap.add_argument("--n-alphas", type=int, default=4,
                    help="number of damped alphas per p (kept small; fewer = faster). "
                         "Hard CM (alpha=1.0) and MWPM are always added on top.")
    ap.add_argument("--Lstar", type=float, default=0.80, help="saturating-law plateau L (Pearl fit)")
    ap.add_argument("--p0star", type=float, default=7e-4, help="saturating-law crossover p0 (Pearl fit)")
    ap.add_argument("--mstar", type=float, default=0.70, help="saturating-law exponent m (Pearl fit)")
    ap.add_argument("--alpha-min", type=float, default=0.02, help="floor for tested alphas")
    ap.add_argument("--damped-correction", action=argparse.BooleanOptionalAction, default=True,
                    help="shift the Pearl-fit alpha*(p) up to the DAMPED-rule optimum using "
                         "alpha_d = alpha_p + (1-alpha_p)*r(p), r = median P(mu|~nu)/P(mu|nu) from the DEM "
                         "(this rule drops Pearl's second term). Use --no-damped-correction for raw Pearl alpha*.")

    # --- statistics / batching ---
    ap.add_argument("--target-errors", type=int, default=1000,
                    help="sample until the BEST decoder reaches this many logical errors")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="shots per sampling+decode chunk")
    ap.add_argument("--max-shots", type=int, default=20_000_000_000, help="per-(d,p) shot cap")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-time budget for the WHOLE run (0 = unlimited); "
                         "the run stops gracefully and writes results before this")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--out-csv", type=str, default="",
                    help="output CSV (default: data/regularized_cm_sweep_d<dists>[_<tag>].csv). "
                         "Use a UNIQUE path per parallel job.")
    ap.add_argument("--tag", type=str, default="", help="suffix for the default out-csv name")
    return ap.parse_args()


# =============================================================================
# Model + alpha grid
# =============================================================================
def make_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def pearl_alpha_star(p, L, p0, m):
    """Saturating law alpha*(p) = L / (1 + (p0/p)^m), fit to PEARL's rule."""
    return L / (1.0 + (p0 / p) ** m)


def damped_correction_r(graph):
    """Median r = P(mu|~nu) / P(mu|nu) over correlated edge pairs, from the DEM stats.

    Pearl's implied prob is  alpha*P(mu|nu) + (1-alpha)*P(mu|~nu); the damped rule keeps
    only the first term. Matching the effective boost gives the optimal-alpha shift
    alpha_damped = alpha_pearl + (1-alpha_pearl)*r, with r ~ O(p) (so the two rules agree
    as p -> 0 and diverge near threshold). P(mu|nu)=corr/occ_nu, P(mu|~nu)=(occ_mu-corr)/(1-occ_nu).
    """
    if graph.n_line_edges == 0:
        return 0.0
    occ, corr = graph.base_p, graph.initial_corr_tracer
    src, dst = graph.line_edge_index
    rs = []
    for k in range(graph.n_line_edges):
        a, b = int(src[k]), int(dst[k])
        for mu, nu in ((a, b), (b, a)):
            p_cond = corr[k] / max(occ[nu], 1e-12)
            if p_cond <= 1e-9:
                continue
            p_neg = max((occ[mu] - corr[k]) / max(1.0 - occ[nu], 1e-12), 0.0)
            rs.append(p_neg / p_cond)
    return float(np.median(rs)) if rs else 0.0


def smart_alphas(center, args):
    """A small grid of ROUND alpha values bracketing `center`, plus hard CM (1.0).

    Values >= 0.1 use ONE decimal (0.1, 0.2, ...); x.x5 midpoints are added only in the
    low-alpha region (< 0.3), where 0.1-wide steps are coarse relative to alpha*; values
    < 0.1 may be finer. The window is centred so `center` is bracketed (never a damped
    endpoint). Fewer points => faster, so --n-alphas is kept small.
    """
    if args.alphas.strip():
        grid = [float(x) for x in args.alphas.split(",")]
    else:
        ladder = [0.02, 0.03, 0.05, 0.07,                        # alpha < 0.1 (finer allowed)
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # one-decimal
        if center < 0.30:                                        # x5 only where 0.1 steps are coarse
            ladder += [0.15, 0.25]
        ladder = sorted(a for a in set(ladder) if args.alpha_min <= a <= 0.95)
        lad = np.array(ladder)
        n = max(3, args.n_alphas)
        i = int(np.argmin(np.abs(lad - center)))                 # nearest ladder value to alpha*
        lo = max(0, min(i - n // 2, len(lad) - n))
        sel = list(lad[lo:lo + n])
        # guarantee alpha* sits strictly inside the damped grid (not an endpoint)
        if center <= sel[0] and lo > 0:
            sel = [ladder[lo - 1]] + sel
        if center >= sel[-1] and lo + n < len(ladder):
            sel = sel + [ladder[lo + n]]
        grid = sel
    grid = sorted(set(round(float(a), 3) for a in grid if a > 0))
    if 1.0 not in grid:                       # always include hard CM (enable_correlations, no damping)
        grid.append(1.0)
    return sorted(set(grid))


# =============================================================================
# CSV (incremental, atomic)
# =============================================================================
FIELDS = ["distance", "rounds", "p", "decoder", "alpha", "shots", "errors",
          "ler", "ler_std", "decode_seconds", "us_per_shot", "is_best", "best_at_endpoint"]


def _ler_std(err, shots):
    l = err / shots if shots else 0.0
    return l, (l * (1 - l) / shots) ** 0.5 if shots else 0.0


def write_csv(path, results):
    """results[(d,p)] = dict(rounds, shots, alphas, err={key:n}, tsec={key:s}).
    key = ('mwpm', None) or ('cm', alpha).  Rewrites the whole file atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (d, p) in sorted(results):
            r = results[(d, p)]
            shots = r["shots"]
            # best decoder = fewest errors; flag if it is a damped-alpha grid endpoint
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

    out_csv = args.out_csv
    if not out_csv:
        tag = f"_{args.tag}" if args.tag else ""
        out_csv = f"data/regularized_cm_sweep_d{'-'.join(map(str, distances))}{tag}.csv"

    print(f"pymatching: {pymatching.__file__}")
    print(f"distances={distances}  p_values={p_values}")
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
            circ = make_circuit(d, rounds, p)
            dem = decompose_errors_for_stim_surface_code_coords(
                circ.detector_error_model(decompose_errors=False))
            mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
            cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            sampler = circ.compile_detector_sampler(seed=args.seed)

            # alpha grid centred on the DAMPED-rule optimum: Pearl-fit alpha*(p), then
            # shifted up by the (second-term) correction r(p) measured from this DEM.
            a_pearl = pearl_alpha_star(p, args.Lstar, args.p0star, args.mstar)
            r = 0.0
            a_center = a_pearl
            if args.damped_correction and not args.alphas.strip():
                r = damped_correction_r(DecodingGraph.from_dem(dem))
                a_center = min(a_pearl + (1.0 - a_pearl) * r, 0.98)
            alphas = smart_alphas(a_center, args)

            # keys: MWPM + one CM entry per alpha (alpha=1.0 == hard "CM, no alpha")
            keys = [("mwpm", None)] + [("cm", a) for a in alphas]
            err = {k: 0 for k in keys}
            tsec = {k: 0.0 for k in keys}
            shots = 0
            results[(d, p)] = dict(rounds=rounds, shots=0, alphas=alphas, err=err, tsec=tsec)

            t0 = time.time()
            print(f"=== d={d} r={rounds} p={p:.3e} | detectors={circ.num_detectors} | "
                  f"alpha*_pearl={a_pearl:.3f} r(p)={r:.3f} -> alpha*_damped={a_center:.3f} | "
                  f"alphas={alphas} ===", flush=True)

            while shots < args.max_shots:
                det, obs = sampler.sample(shots=args.chunk, separate_observables=True,
                                          bit_packed=True)
                ob = (obs[:, 0] & 1)

                ta = time.perf_counter()
                pm = mwpm.decode_batch(det, bit_packed_shots=True)[:, 0]
                tsec[("mwpm", None)] += time.perf_counter() - ta
                err[("mwpm", None)] += int(np.count_nonzero(pm != ob))

                for a in alphas:
                    ta = time.perf_counter()
                    pa = cm.decode_batch(det, bit_packed_shots=True,
                                         enable_correlations=True, alpha=a)[:, 0]
                    tsec[("cm", a)] += time.perf_counter() - ta
                    err[("cm", a)] += int(np.count_nonzero(pa != ob))

                shots += args.chunk
                results[(d, p)]["shots"] = shots
                write_csv(out_csv, results)             # incremental update every chunk

                best_key = min(err, key=lambda k: err[k])
                best_err = err[best_key]
                ba = best_key[1] if best_key[0] == "cm" else "MWPM"
                print(f"  shots={shots:>13,} | best={ba} err={best_err}/{args.target_errors} | "
                      f"MWPM={err[('mwpm', None)]} CM1={err[('cm', 1.0)]} | "
                      f"{fmt_time(time.time() - t0)}", flush=True)

                if best_err >= args.target_errors:
                    break
                if args.max_seconds and (time.time() - t_all) >= args.max_seconds:
                    print("  [walltime budget reached — stopping gracefully]", flush=True)
                    stop_all = True
                    break

            # per-point summary + endpoint flag
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
                  + ("  [BEST AT GRID ENDPOINT -> widen grid: raise --n-alphas or "
                     "adjust --mstar/--Lstar]" if endpoint else "")
                  + f" | wrote {out_csv}\n", flush=True)

    print(f"done in {fmt_time(time.time() - t_all)}  ->  {out_csv}")


if __name__ == "__main__":
    main()
