"""LER sweep of MWPM / Correlated Matching / Neural CM for the same fixed policy at d = 3, 5, 7.

Targets a fixed number of logical errors per (d, p) point so every point carries the same
relative binomial uncertainty, 1/sqrt(target_errors) ~ 4.5% at 500 errors.

Scheduling.  Points are ordered cheapest-first: distance ascending (outer), p descending (inner),
so d=3 at p=1e-2 runs first.  Points with d >= 5 and p < --defer-below are pushed to the very end,
because their shot cost explodes as p^-(d+1)/2 and they would otherwise block the whole sweep.

Every completed point is appended to the CSV immediately (and fsync'd), so the sweep can be killed
and resumed at any time: on restart, points already present in the CSV are skipped.

Cost warning.  Reaching 500 errors needs ~500/LER shots.  At d=7, p=1e-4 the LER is ~1e-9, i.e.
~5e11 shots -- not reachable.  Such points terminate on --max-shots / --max-minutes and are written
with target_reached=0; treat their LER as an upper bound, or drop them.

Usage
    python scripts/ler_comparison/ncm_ler_sweep_d3d5d7.py
    python scripts/ler_comparison/ncm_ler_sweep_d3d5d7.py --distances 3 --n-p 5 --target-errors 100
"""

import argparse
import os
import sys
import time

import numpy as np

from NeuralCM.drifted_matching_env import DriftedMatchingEnv
from NeuralCM.gnn_sac_agent import SACAgent
from NeuralCM.syndrome_data_generation import SyndromeDataGenerator

# reuse the env/agent wiring that produced the existing ler_vs_p_* CSVs so the numbers stay
# comparable with the published ones
from neural_vs_cm_mwpm import make_generator, make_env, evaluate_one_episode  # noqa: E402

COLUMNS = [
    "distance", "n_rounds", "p",
    "static_mean", "static_std", "corr_mean", "corr_std", "neural_mean", "neural_std",
    "n_shots", "static_errors", "corr_errors", "neural_errors",
    "target_reached", "seconds",
]
LEGACY_COLUMNS = ["p", "static_mean", "static_std", "corr_mean", "corr_std",
                  "neural_mean", "neural_std"]


# --------------------------------------------------------------------------------------------
# scheduling
# --------------------------------------------------------------------------------------------
def p_grid(p_hi, p_lo, n):
    """n points from p_hi down to p_lo, uniform in log p."""
    return np.logspace(np.log10(p_hi), np.log10(p_lo), n)


def build_schedule(distances, ps, defer_below):
    """(distance, p) points, cheap first; expensive low-p points at d>=5 pushed to the end."""
    main, deferred = [], []
    for d in sorted(distances):                      # low d first
        for p in ps:                                 # high p first (ps is descending)
            (deferred if (d >= 5 and p < defer_below) else main).append((d, float(p)))
    return main + deferred


# --------------------------------------------------------------------------------------------
# incremental csv
# --------------------------------------------------------------------------------------------
def key(d, p):
    return (int(d), float(f"{p:.6g}"))


def load_done(path):
    """(distance, p) keys already in the CSV, so a restart resumes instead of recomputing."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        if header != COLUMNS:
            sys.exit(f"ERROR: {path} exists with an unexpected header.\n"
                     f"  found:    {header}\n  expected: {COLUMNS}\n"
                     f"Move it aside or pass a different --out-csv.")
        for line in f:
            if not line.strip():
                continue
            row = line.split(",")
            done.add(key(int(row[0]), float(row[2])))
    return done


def append_row(path, row):
    """Append one point and flush to disk, so a kill -9 loses at most the point in flight."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(COLUMNS) + "\n")
        f.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_legacy_split(path, out_dir, prefix):
    """Per-distance CSVs in the 7-column schema the thesis figure scripts already read."""
    if not os.path.exists(path):
        return
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                rows.append(dict(zip(COLUMNS, line.rstrip("\n").split(","))))
    os.makedirs(out_dir, exist_ok=True)
    for d in sorted({int(r["distance"]) for r in rows}):
        sub = sorted((r for r in rows if int(r["distance"]) == d), key=lambda r: float(r["p"]))
        with open(os.path.join(out_dir, f"{prefix}_d{d}.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(LEGACY_COLUMNS) + "\n")
            for r in sub:
                f.write(",".join(r[c] for c in LEGACY_COLUMNS) + "\n")


# --------------------------------------------------------------------------------------------
# one (distance, p) point
# --------------------------------------------------------------------------------------------
def run_point(args, d, p, p_index):
    t0 = time.time()
    n_rounds = d if args.n_rounds is None else args.n_rounds

    generator = make_generator(p=p, n_shots=args.chunk_shots, distance=d, n_rounds=n_rounds,
                               mismatch=args.mismatch, p_gate_zz=args.p_gate_zz)
    env = make_env(generator=generator, n_shots=args.chunk_shots, action_scale=args.action_scale,
                   local_action_hops=args.local_action_hops,
                   start_from_oracle=args.start_from_oracle,
                   use_endpoint_firing=args.use_endpoint_firing,
                   burn_in_steps=0, mismatch=args.mismatch)

    sample_obs, _ = env.reset(seed=args.seed)
    agent = SACAgent(node_dim=sample_obs["node_features"].shape[1], hidden_dim=args.hidden_dim,
                     static_edge_index=env.line_edge_index, n_layers=args.n_layers, lr=args.lr,
                     gamma=0.0, tau=0.005, alpha=args.alpha)
    agent.load_models(args.model_path)

    shots = static_err = corr_err = neural_err = 0
    chunk = 0
    deadline = t0 + 60.0 * args.max_minutes if args.max_minutes > 0 else np.inf

    while shots < args.max_shots and time.time() < deadline:
        # deterministic and collision-free across (distance, p, chunk)
        seed = args.seed + 1_000_003 * d + 7_919 * p_index + chunk
        out = evaluate_one_episode(env=env, agent=agent, seed=seed,
                                   bypass_threshold=args.bypass_threshold, burn_in_steps=0)
        static_err += out["static_errors"]
        corr_err += out["corr_errors"]
        neural_err += out["neural_errors"]
        shots += out["n_steps"]
        chunk += 1

        # require the target on ALL three decoders, so every curve gets the same precision
        if min(static_err, corr_err, neural_err) >= args.target_errors:
            break
        if chunk % args.log_every == 0:
            print(f"    [d={d} p={p:.3e}] {shots:,} shots | errors "
                  f"mwpm={static_err} cm={corr_err} ncm={neural_err} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    reached = int(min(static_err, corr_err, neural_err) >= args.target_errors)

    def ler_and_se(n_err):
        ler = n_err / shots if shots else float("nan")
        return ler, float(np.sqrt(ler * (1.0 - ler) / shots)) if shots else float("nan")

    s_m, s_s = ler_and_se(static_err)
    c_m, c_s = ler_and_se(corr_err)
    n_m, n_s = ler_and_se(neural_err)
    dt = time.time() - t0

    print(f"  d={d} p={p:.4e} | {shots:,} shots in {dt / 60:.1f} min | "
          f"MWPM {s_m:.4e}+/-{s_s:.1e} ({static_err}) | "
          f"CM {c_m:.4e}+/-{c_s:.1e} ({corr_err}) | "
          f"NCM {n_m:.4e}+/-{n_s:.1e} ({neural_err})"
          f"{'' if reached else '  [TARGET NOT REACHED]'}", flush=True)

    return {
        "distance": d, "n_rounds": n_rounds, "p": p,
        "static_mean": s_m, "static_std": s_s,
        "corr_mean": c_m, "corr_std": c_s,
        "neural_mean": n_m, "neural_std": n_s,
        "n_shots": shots, "static_errors": static_err, "corr_errors": corr_err,
        "neural_errors": neural_err, "target_reached": reached, "seconds": round(dt, 1),
    }


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=str,
                    default="models/best_ncm_model/slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.pth")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--p-hi", type=float, default=1e-2)
    ap.add_argument("--p-lo", type=float, default=1e-4)
    ap.add_argument("--n-p", type=int, default=9, help="points per decade-pair, uniform in log p")
    ap.add_argument("--defer-below", type=float, default=4e-4,
                    help="points with d>=5 and p below this run last")
    ap.add_argument("--n-rounds", type=int, default=None, help="default: r = d")

    ap.add_argument("--target-errors", type=int, default=500)
    ap.add_argument("--max-shots", type=int, default=1_000_000_000)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="wall-clock cap per (d,p) point; 0 = unlimited")
    ap.add_argument("--chunk-shots", type=int, default=200_000)
    ap.add_argument("--log-every", type=int, default=5, help="progress line every N chunks")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-gate-zz", type=float, default=0.0)
    ap.add_argument("--mismatch", type=float, default=1.0)

    # must match the training configuration of the loaded policy
    ap.add_argument("--action-scale", type=float, default=5.0)
    ap.add_argument("--bypass-threshold", type=int, default=2)
    ap.add_argument("--local-action-hops", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--use-endpoint-firing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--start-from-oracle", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--out-csv", type=str, default="data/ncm_ler_sweep_d3d5d7.csv")
    ap.add_argument("--legacy-split-dir", type=str, default="data",
                    help="also emit per-distance CSVs in the 7-column figure schema")
    ap.add_argument("--legacy-prefix", type=str, default="ler_vs_p_ncm_sweep")
    args = ap.parse_args()

    ps = p_grid(args.p_hi, args.p_lo, args.n_p)
    p_index = {key(0, p): i for i, p in enumerate(ps)}
    schedule = build_schedule(args.distances, ps, args.defer_below)
    done = load_done(args.out_csv)
    todo = [(d, p) for d, p in schedule if key(d, p) not in done]

    print(f"model      : {args.model_path}")
    step = f"{np.log10(ps[0] / ps[1]):.3f} decades" if len(ps) > 1 else "single point"
    print(f"p grid     : {', '.join(f'{p:.3e}' for p in ps)}  (log step {step})")
    print(f"target     : {args.target_errors} logical errors per (d,p), all three decoders")
    print(f"schedule   : {len(schedule)} points, {len(done)} already in CSV, {len(todo)} to run")
    print(f"deferred   : d>=5 and p<{args.defer_below:.1e} -> "
          f"{sum(1 for d, p in schedule if d >= 5 and p < args.defer_below)} points at the end")
    print(f"output     : {args.out_csv}\n")

    if not os.path.exists(args.model_path):
        sys.exit(f"ERROR: model not found: {args.model_path}")

    t0 = time.time()
    for i, (d, p) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] d={d}, p={p:.4e}  (elapsed {(time.time() - t0) / 60:.1f} min)",
              flush=True)
        row = run_point(args, d, p, p_index[key(0, p)])
        append_row(args.out_csv, row)
        if args.legacy_split_dir:
            write_legacy_split(args.out_csv, args.legacy_split_dir, args.legacy_prefix)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {args.out_csv}")


if __name__ == "__main__":
    main()
