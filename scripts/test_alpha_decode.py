#!/usr/bin/env python
"""Sweep MWPM vs Pearl's Correlated Matching over distance, p, and alpha.

Rotated surface code Z-memory, circuit-level depolarizing noise, Tesseract
(coordinate-aware) error decomposition. For each (distance, p) it streams shots
with COMMON random numbers across all decoders until CM (alpha=1.0) reaches
--target-errors (or --max-shots), decoding every chunk with:

  * uncorrelated MWPM            (Stim native decomposition, enable_correlations=False)
  * regularized correlated matching (enable_correlations=True) at alpha=1.0 (ordinary CM)
                                    and each --alphas value (Pearl soft evidence).
                                    1.0 is always included as the CM reference / stop rule.

Results are written incrementally to CSV and plotted:
  * LER vs p (one panel per distance: MWPM, CM, each alpha)
  * LER / LER(MWPM) vs alpha (one panel per distance, one line per p)

Usage:
  python scripts/test_alpha_decode.py
  python scripts/test_alpha_decode.py --distances 3,5,7 --p-values 3e-4,1e-3,3e-3 \
      --alphas 0.2,0.3,0.4,0.7 --target-errors 500
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import stim
import pymatching
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=str, default="7", help="comma-separated code distances")
    ap.add_argument("--p-values", type=str, default="4e-4,2e-4,1e-4",
                    help="comma-separated physical error rates")
    ap.add_argument("--alphas", type=str, default="0.05,0.1,0.15,0.2,0.3,0.4",
                    help="comma-separated alphas (1.0 is always added as the CM reference)")
    ap.add_argument("--rounds", type=int, default=0, help="syndrome rounds (0 = use distance)")
    ap.add_argument("--target-errors", type=int, default=500,
                    help="per (d,p): sample until CM (alpha=1.0) reaches this many errors")
    ap.add_argument("--max-shots", type=int, default=1_000_000_000_000_000_000)
    ap.add_argument("--chunk", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-csv", type=str, default="data/alpha_sweep_extended_fixed_MWPM_d7_lowp.csv")
    ap.add_argument("--plots-dir", type=str, default="plots")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def make_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def count_err(pred_bits, obs_bits):
    """# shots whose bit-packed prediction differs from the bit-packed observables."""
    return int(np.count_nonzero(np.any(pred_bits != obs_bits, axis=1)))


def fmt_time(s):
    s = int(s)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def ler_std(err, shots):
    l = err / shots if shots else 0.0
    return l, (l * (1 - l) / shots) ** 0.5 if shots else 0.0


def write_csv(path, results, eval_alphas):
    """results[(d,p)] = dict(rounds, shots, mwpm, alphas={a: err}). Atomic rewrite."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "rounds", "p", "decoder", "alpha", "shots", "errors", "ler", "ler_std"])
        for (d, p) in sorted(results):
            r = results[(d, p)]
            shots = r["shots"]
            l, s = ler_std(r["mwpm"], shots)
            w.writerow([d, r["rounds"], p, "mwpm", "", shots, r["mwpm"], l, s])
            for a in eval_alphas:
                e = r["alphas"][a]
                l, s = ler_std(e, shots)
                w.writerow([d, r["rounds"], p, "cm", a, shots, e, l, s])
    os.replace(tmp, path)


def make_plots(results, eval_alphas, args):
    distances = sorted({d for (d, _) in results})
    os.makedirs(args.plots_dir, exist_ok=True)

    # --- Plot 1: LER vs p, one panel per distance ---
    fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 5), squeeze=False)
    acolors = plt.cm.viridis(np.linspace(0.0, 0.85, len(eval_alphas)))
    for ax, d in zip(axes[0], distances):
        ps = sorted({p for (dd, p) in results if dd == d})
        if not ps:
            continue
        shots = np.array([results[(d, p)]["shots"] for p in ps])
        mwpm = np.array([results[(d, p)]["mwpm"] for p in ps]) / shots
        ax.plot(ps, mwpm, "k--o", lw=1.6, ms=5, label="MWPM")
        for a, c in zip(eval_alphas, acolors):
            ler = np.array([results[(d, p)]["alphas"][a] for p in ps]) / shots
            lbl = "CM (α=1.0)" if a == 1.0 else f"α={a:g}"
            ax.plot(ps, ler, "-o", color=c, ms=4, lw=1.4, label=lbl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate")
        ax.set_title(f"d={d}"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("LER vs p — MWPM vs correlated matching (Pearl α)")
    fig.tight_layout()
    p1 = os.path.join(args.plots_dir, "alpha_sweep_ler_vs_p_extended_fixed_MWPM_d7_lowp.csv")
    fig.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig)

    # --- Plot 2: LER / LER(α=1.0) vs alpha, one panel per distance, one line per p ---
    fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 5), squeeze=False)
    for ax, d in zip(axes[0], distances):
        ps = sorted({p for (dd, p) in results if dd == d})
        pcolors = plt.cm.plasma(np.linspace(0.0, 0.85, max(len(ps), 1)))
        for p, c in zip(ps, pcolors):
            r = results[(d, p)]
            cm = r["alphas"][1.0]
            if cm == 0:
                continue
            ratio = [r["alphas"][a] / cm for a in eval_alphas]
            ax.plot(eval_alphas, ratio, "-o", color=c, ms=4, label=f"p={p:.0e}")
            ax.plot([1.0], [r["mwpm"] / cm], "*", color=c, ms=11)  # MWPM/CM ref (star)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xlabel("α  (1.0 = ordinary CM)"); ax.set_ylabel("LER / LER(α=1.0)")
        ax.set_title(f"d={d}  (★ = MWPM/CM)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Relative LER vs α (below 1 beats CM; lower is better)")
    fig.tight_layout()
    p2 = os.path.join(args.plots_dir, "alpha_sweep_ratio_vs_alpha_extended_fixed_MWPM_d7_lowp.csv")
    fig.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved plots: {p1}\n            {p2}")


def main():
    args = parse_args()
    distances = [int(x) for x in args.distances.split(",")]
    p_values = [float(x) for x in args.p_values.split(",")]
    eval_alphas = sorted(set(float(x) for x in args.alphas.split(",")) | {1.0})

    print(f"pymatching: {pymatching.__file__}")
    print(f"distances={distances}  p_values={p_values}  eval_alphas={eval_alphas}")
    print(f"target_errors={args.target_errors}  max_shots={args.max_shots:,}  "
          f"chunk={args.chunk:,}  out_csv={args.out_csv}\n")

    results = {}
    bw_checked = False
    t_all = time.time()

    for d in distances:
        rounds = d if args.rounds == 0 else args.rounds
        for p in p_values:
            circuit = make_circuit(d, rounds, p)
            dem_tess = decompose_errors_for_stim_surface_code_coords(
                circuit.detector_error_model(decompose_errors=False))
            matcher = pymatching.Matching.from_detector_error_model(dem_tess, enable_correlations=True)
            mwpm = pymatching.Matching.from_detector_error_model(
                dem_tess, enable_correlations=False)
            sampler = circuit.compile_detector_sampler(seed=args.seed)

            err_mwpm = 0
            err_a = {a: 0 for a in eval_alphas}
            shots = 0
            t0 = time.time()
            print(f"=== d={d} rounds={rounds} p={p:.3e} | detectors={circuit.num_detectors} ===")

            while shots < args.max_shots:
                det, obs = sampler.sample(shots=args.chunk, separate_observables=True, bit_packed=True)
                pred_m = mwpm.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True)
                err_mwpm += count_err(pred_m, obs)
                for a in eval_alphas:
                    pred = matcher.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True,
                                                enable_correlations=True, alpha=a)
                    err_a[a] += count_err(pred, obs)
                    if not bw_checked and a == 1.0:
                        pred0 = matcher.decode_batch(det, bit_packed_shots=True,
                                                     bit_packed_predictions=True, enable_correlations=True)
                        print(f"  [backward-compat] no-alpha == alpha=1.0: {np.array_equal(pred0, pred)}")
                        bw_checked = True
                shots += args.chunk
                best_a = min(eval_alphas, key=lambda a: err_a[a])
                print(f"  shots={shots:>12,} | MWPM={err_mwpm:>4d} | CM={err_a[1.0]:>4d}/{args.target_errors} | "
                      f"best α={best_a:g}:{err_a[best_a]:>4d} | {fmt_time(time.time() - t0)}", flush=True)
                if err_a[1.0] >= args.target_errors:
                    break

            results[(d, p)] = dict(rounds=rounds, shots=shots, mwpm=err_mwpm, alphas=dict(err_a))
            write_csv(args.out_csv, results, eval_alphas)

            cm_ler = err_a[1.0] / shots
            best_a = min(eval_alphas, key=lambda a: err_a[a])
            print(f"  -> shots={shots:,} | MWPM LER={err_mwpm / shots:.3e} | CM LER={cm_ler:.3e} | "
                  f"best α={best_a:g} ({err_a[best_a] / shots:.3e}, "
                  f"{err_a[best_a] / max(err_a[1.0], 1):.3f}×CM) | wrote {args.out_csv}\n", flush=True)

    print(f"all done in {fmt_time(time.time() - t_all)}")
    if not args.no_plots:
        make_plots(results, eval_alphas, args)


if __name__ == "__main__":
    main()
