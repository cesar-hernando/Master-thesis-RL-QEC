"""Single (distance, p) point runner for ncm_ler_sweep_d3d5d7.py, meant to be launched as one
SLURM array task so the still-missing points of data/ncm_ler_sweep_d3d5d7.csv can be computed
in parallel across GPUs instead of one at a time in a serial sweep.

Reuses run_point()/append_row() from the main sweep script unchanged, so the physics/RNG
seeding is bit-for-bit identical to what a serial run of the same point would have produced --
the seed depends on the point's index in the canonical p-grid, which this script reconstructs
from --p-hi/--p-lo/--n-p exactly as the main script does.

IMPORTANT: the defaults below are --p-hi=1e-2, --p-lo=2e-4, --n-p=9. That p-lo is NOT
ncm_ler_sweep_d3d5d7.py's own default (1e-4) -- it is the grid actually used to produce the
existing rows of data/ncm_ler_sweep_d3d5d7.csv, reverse-engineered from the CSV's own p-values
and log-step (0.212 decades) rather than assumed. Passing the wrong grid here would look fine
(the point still runs) but silently give it a different RNG seed than a serial continuation of
the same sweep would have used. Leave these three alone unless you deliberately rebuilt the
whole CSV on a different grid.

Writes its OWN csv (one header + one data row) rather than appending to the shared master csv,
so N array tasks can run concurrently with no write contention. Merge the results back with
merge_ncm_sweep_results.py once the array job finishes.

Usage (usually invoked by job_scripts/ncm_ler_sweep_parallel.job, not by hand):
    python scripts/ler_comparison/ncm_ler_sweep_task.py --distance 7 --p 2e-4 \
        --out-csv data/ncm_ler_sweep_parallel/ncm_sweep_d7_p2e-4.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncm_ler_sweep_d3d5d7 import p_grid, key, run_point, append_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)

    # canonical grid the point's index (-> RNG seed) is looked up in -- see the IMPORTANT
    # note above; keep these equal to what actually produced data/ncm_ler_sweep_d3d5d7.csv
    ap.add_argument("--p-hi", type=float, default=1e-2)
    ap.add_argument("--p-lo", type=float, default=2e-4)
    ap.add_argument("--n-p", type=int, default=9)

    ap.add_argument("--model-path", type=str,
                    default="models/best_ncm_model/slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.pth")
    ap.add_argument("--n-rounds", type=int, default=None, help="default: r = distance")

    ap.add_argument("--target-errors", type=int, default=500)
    ap.add_argument("--max-shots", type=int, default=1_000_000_000)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="wall-clock cap for this point; 0 = unlimited (matches how the rest "
                         "of data/ncm_ler_sweep_d3d5d7.csv was generated -- every existing row "
                         "has target_reached=1, none were accepted capped/partial). A SLURM "
                         "--time limit still applies on top of this: if it fires first, the "
                         "process is SIGKILLed with NOTHING written for this point (run_point "
                         "never returns to append_row) and you just resubmit that array index.")
    ap.add_argument("--chunk-shots", type=int, default=200_000)
    ap.add_argument("--log-every", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-gate-zz", type=float, default=0.0)
    ap.add_argument("--mismatch", type=float, default=1.0)

    # must match the training configuration of the loaded policy -- same defaults as the
    # main sweep script, do not change unless you also regenerate the whole CSV with them
    ap.add_argument("--action-scale", type=float, default=5.0)
    ap.add_argument("--bypass-threshold", type=int, default=2)
    ap.add_argument("--local-action-hops", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--use-endpoint-firing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--start-from-oracle", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    if not os.path.exists(args.model_path):
        sys.exit(f"ERROR: model not found: {args.model_path}")

    ps = p_grid(args.p_hi, args.p_lo, args.n_p)
    p_index_map = {key(0, pp): i for i, pp in enumerate(ps)}
    idx_key = key(0, args.p)
    if idx_key not in p_index_map:
        sys.exit(f"ERROR: p={args.p:.6g} is not on the canonical grid "
                 f"{[f'{pp:.4e}' for pp in ps]} -- pass the exact grid value (check "
                 f"data/ncm_ler_sweep_d3d5d7.csv) or adjust --p-hi/--p-lo/--n-p to match how "
                 f"that grid was actually built.")
    p_index = p_index_map[idx_key]

    print(f"model      : {args.model_path}")
    print(f"point      : distance={args.distance}, p={args.p:.6e} (grid index {p_index})")
    print(f"target     : {args.target_errors} logical errors, all three decoders")
    print(f"output     : {args.out_csv}\n", flush=True)

    row = run_point(args, args.distance, args.p, p_index)
    append_row(args.out_csv, row)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
