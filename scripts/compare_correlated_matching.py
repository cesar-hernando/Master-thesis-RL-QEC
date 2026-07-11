"""
Compare correlated matching against plain MWPM across physical error rates p.

Decoders
--------
  * "corr" : PyMatching correlated matching (enable_correlations=True) built on a
             DEM whose hyperedges are decomposed with the geometry-aware
             "tesseract" decomposition
             (decompose_errors_for_stim_surface_code_coords), NOT Stim's native
             decompose_errors=True.
  * "mwpm" : plain minimum-weight perfect matching (enable_correlations=False) on
             Stim's natively decomposed DEM (decompose_errors=True).

Distances
---------
By default d = 3, 5, 7, 9 with target_errors = 300.

p ranges
--------
  * d = 3, 5 : evaluated over the full p grid down to p = 2e-4.
  * d = 7, 9 : p is swept downward and stopped once the correlated-matching LER
               drops to (or below) the reference LER = correlated-matching LER of
               d = 5 at p = 2e-4. This avoids simulating LERs so small that
               collecting `target_errors` becomes infeasible, while pushing each
               higher distance as low in p as is comparable to d=5 @ 2e-4.

Usage:
    python scripts/compare_correlated_matching.py
    python scripts/compare_correlated_matching.py --distances 3 5 7 9 --target-errors 300
"""

import argparse
import os

import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords

# Fine p grid (descending). d=3,5 use the whole thing; d=7,9 stop early.
P_GRID = np.array([
    0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.0015,
    0.001, 0.0008, 0.0006, 0.0004, 0.0003, 0.0002,
])

# Distances swept fully (and used to define the reference LER).
FULL_SWEEP_DISTANCES = (3, 5)
REFERENCE_DISTANCE = 5
REFERENCE_P = 2e-4

DECODERS = ("corr", "mwpm")
DEC_LABEL = {"corr": "Correlated matching (tesseract)", "mwpm": "MWPM (Stim decomp.)"}
DEC_COLOR = {"corr": "#d62728", "mwpm": "#1f77b4"}
DEC_MARKER = {"corr": "o", "mwpm": "s"}


def build_circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )


def build_decoders(circuit):
    """Return (corr_matcher, mwpm_matcher) for a circuit."""
    # Correlated matching on the tesseract-decomposed DEM.
    dem_raw = circuit.detector_error_model(decompose_errors=False)
    dem_tess = decompose_errors_for_stim_surface_code_coords(dem_raw)
    corr = pymatching.Matching.from_detector_error_model(dem_tess, enable_correlations=True)

    # Plain MWPM on Stim's native decomposition.
    dem_stim = circuit.detector_error_model(decompose_errors=True)
    mwpm = pymatching.Matching.from_detector_error_model(dem_stim, enable_correlations=False)
    return corr, mwpm


def _ler_and_err(n_err, n_shots):
    ler = n_err / n_shots
    err = np.sqrt((ler * (1.0 - ler)) / n_shots)
    return ler, err


def evaluate_point(d, p, target_errors, max_shots, batch_size):
    """Evaluate correlated matching and MWPM for one (d, p)."""
    print(f"\n--- Distance {d}, p = {p:.5e} ---")

    circuit = build_circuit(d, p)
    sampler = circuit.compile_detector_sampler()
    corr, mwpm = build_decoders(circuit)

    err = {"corr": 0, "mwpm": 0}
    total_shots = 0

    while total_shots < max_shots:
        syndromes, true_obs = sampler.sample(shots=batch_size, separate_observables=True)
        true_obs = true_obs.flatten()

        pred_c = corr.decode_batch(syndromes, enable_correlations=True)
        err["corr"] += int(np.sum(pred_c[:, 0] != true_obs))

        pred_m = mwpm.decode_batch(syndromes, enable_correlations=False)
        err["mwpm"] += int(np.sum(pred_m[:, 0] != true_obs))

        total_shots += batch_size
        if min(err.values()) >= target_errors:
            break

    out = {"shots": total_shots}
    print(f"Total Shots: {total_shots:,}")
    for dec in DECODERS:
        ler, e = _ler_and_err(err[dec], total_shots)
        out[dec] = (ler, e)
        print(f"  {DEC_LABEL[dec]:32s}: {err[dec]:6d} errors | LER {ler:.4e} ± {e:.2e}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--out-png", type=str,
                        default="plots/compare_correlated_matching_vs_mwpm.png")
    parser.add_argument("--out-csv", type=str,
                        default="data/compare_correlated_matching_vs_mwpm.csv")
    parser.add_argument("--max-shots", type=int, default=1000_000_000)
    parser.add_argument("--target-errors", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # results[d][decoder] -> {'p': [...], 'ler': [...], 'err': [...]}
    results = {d: {dec: {'p': [], 'ler': [], 'err': []} for dec in DECODERS}
               for d in args.distances}

    # Sweep the "full" distances first so we can read off the reference LER.
    reference_ler = None
    with open(args.out_csv, "w") as f:
        f.write("distance,p,shots,ler_corr,err_corr,ler_mwpm,err_mwpm\n")

        def record(d, p, m):
            ler_c, e_c = m["corr"]
            ler_m, e_m = m["mwpm"]
            for dec, (ler, e) in (("corr", (ler_c, e_c)), ("mwpm", (ler_m, e_m))):
                results[d][dec]['p'].append(p)
                results[d][dec]['ler'].append(ler)
                results[d][dec]['err'].append(e)
            f.write(f"{d},{p},{m['shots']},{ler_c},{e_c},{ler_m},{e_m}\n")
            f.flush()

        # --- d = 3, 5: full p sweep. ---
        full_distances = [d for d in args.distances if d in FULL_SWEEP_DISTANCES]
        for d in full_distances:
            for p in P_GRID:
                m = evaluate_point(d, p, args.target_errors, args.max_shots, args.batch_size)
                record(d, p, m)
                if d == REFERENCE_DISTANCE and np.isclose(p, REFERENCE_P):
                    reference_ler = m["corr"][0]

        if reference_ler is None:
            # Reference distance not requested: fall back to the lowest LER seen.
            print("\n[warn] reference LER not measured (d=5 @ p=2e-4 not in sweep); "
                  "high distances will sweep the full p grid.")
            reference_ler = 0.0
        else:
            print(f"\nReference LER (d={REFERENCE_DISTANCE} @ p={REFERENCE_P:.0e}, "
                  f"correlated matching) = {reference_ler:.4e}")

        # --- d = 7, 9 (and any other non-full distances): descend until LER <= reference. ---
        high_distances = [d for d in args.distances if d not in FULL_SWEEP_DISTANCES]
        for d in high_distances:
            for p in P_GRID:
                m = evaluate_point(d, p, args.target_errors, args.max_shots, args.batch_size)
                record(d, p, m)
                if m["corr"][0] <= reference_ler:
                    print(f"  -> d={d}: corr LER {m['corr'][0]:.4e} reached reference "
                          f"{reference_ler:.4e} at p={p:.2e}; stopping descent.")
                    break

    plot_results(args.distances, results, args.out_png, reference_ler)
    print(f"\nData saved to {args.out_csv}")


def plot_results(distances, results, out_png, reference_ler):
    fig, ax = plt.subplots(figsize=(9, 7))

    dist_color = plt.cm.viridis(np.linspace(0, 0.85, len(distances)))
    for d, base_color in zip(distances, dist_color):
        for dec in DECODERS:
            r = results[d][dec]
            if not r['p']:
                continue
            order = np.argsort(r['p'])
            p_arr = np.array(r['p'])[order]
            ler_arr = np.array(r['ler'])[order]
            err_arr = np.array(r['err'])[order]
            ax.errorbar(
                p_arr, ler_arr, yerr=err_arr,
                color=base_color,
                linestyle="-" if dec == "corr" else "--",
                marker=DEC_MARKER[dec], capsize=3, linewidth=1.8, markersize=6,
                label=f"d={d}, {DEC_LABEL[dec]}",
            )

    if reference_ler and reference_ler > 0:
        ax.axhline(reference_ler, color="gray", linestyle=":", linewidth=1.5,
                   label=f"reference LER (d={REFERENCE_DISTANCE} @ p={REFERENCE_P:.0e})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate (p)")
    ax.set_ylabel("Logical error rate (LER)")
    ax.set_title("Correlated matching (tesseract) vs. MWPM")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_png}")


if __name__ == "__main__":
    main()
