"""
Plot logical error rate vs physical error rate for three decoders:
1) Standard MWPM (static, uncorrelated)
2) Correlated matching (algorithmic correlated matching)
3) Neural correlated matching (fixed SAC-GNN policy)

The experiment uses mismatch=1 and a very large update_period so graph updates never happen
within an episode.
"""

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


def parse_p_values(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No physical error rates provided.")
    return values


def make_generator(p: float, n_shots: int, distance: int, n_rounds: int, mismatch: float, p_gate_zz: float):
    noise_model = {
        "version": "built-in",
        "after_clifford_depolarization": p,
        "before_measure_flip_probability": p,
        "after_reset_flip_probability": p,
        "before_round_data_depolarization": p,
        "p_gate_zz": p_gate_zz,
    }

    return SyndromeDataGenerator(
        distance=distance,
        n_rounds=n_rounds,
        mismatch=mismatch,
        noise_model=noise_model,
        memory_type="z",
        n_shots=n_shots,
        qec_code="surface_code",
    )


def make_env(
    generator: SyndromeDataGenerator,
    n_shots: int,
    action_scale: float,
    local_action_hops: int,
    start_from_oracle: bool,
    use_endpoint_firing: bool,
    burn_in_steps: int = 0,
    mismatch: float = 1.0,
):
    # Under drift (mismatch>1) with a burn-in phase AND CMA tracer enabled
    # (start_from_oracle=False), let the env's update mechanism fire every 1000
    # shots so the decoder can re-align during burn-in. If start_from_oracle is
    # True the CMA path is disabled inside the env anyway, so keep the graph
    # frozen and skip the meaningless periodic ticks.
    if burn_in_steps > 0 and mismatch > 1.0 and not start_from_oracle:
        update_period = 1000
    else:
        update_period = n_shots + 1

    return DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=local_action_hops,
        action_scale=action_scale,
        update_period=update_period,
        prior_shots=1_000,
        n_test_shots=0,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        use_log_joint_prob=False,
        start_from_oracle=start_from_oracle,
        use_endpoint_firing=use_endpoint_firing,
        update_with="Spitz",
        train_mode=False,
    )


def evaluate_one_episode(env: DriftedMatchingEnv, agent: SACAgent, seed: int,
                         bypass_threshold: int, burn_in_steps: int = 0):
    obs, _ = env.reset(seed=seed)

    done = False
    neural_errors = 0
    static_errors = 0
    corr_errors = 0
    steps = 0          # counts only LER-evaluated shots (post burn-in)
    burn_in_taken = 0

    while not done:
        in_burn_in = burn_in_taken < burn_in_steps

        if in_burn_in:
            # Pure observation phase: action is zero (lets the env's CMA/graph
            # update mechanism calibrate without the agent perturbing).
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
        else:
            n_flashes = int(np.sum(env.current_syndrome != 0))
            if n_flashes <= bypass_threshold:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
            else:
                action = agent.select_action(obs, evaluate=True)

        next_obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if in_burn_in:
            burn_in_taken += 1
        else:
            true_obs = info["true_obs"]
            neural_errors += int(info["pred_obs"] != true_obs)
            static_errors += int(info["static_pred_obs"] != true_obs)
            corr_errors += int(info["oracle_pred_obs"] != true_obs)
            steps += 1

        obs = next_obs

    return {
        "n_steps": steps,
        "neural_errors": neural_errors,
        "static_errors": static_errors,
        "corr_errors": corr_errors,
    }


def run_sweep(args):
    p_values = parse_p_values(args.p_values)

    all_results = {
        "p": [],
        "static_mean": [],
        "static_std": [],
        "corr_mean": [],
        "corr_std": [],
        "neural_mean": [],
        "neural_std": [],
    }

    print("Running LER sweep with adaptive shots to ensure reliable error bounds...")
    print(f"Model: {args.model_path}")
    print(f"Physical error rates: {p_values}")
    print(f"Target Errors per p: {args.target_errors}")
    print(f"Max Shots per p: {args.max_shots:,}")

    for p in p_values:
        generator = make_generator(
            p=p,
            n_shots=args.chunk_shots,
            distance=args.distance,
            n_rounds=args.n_rounds,
            mismatch=args.mismatch,
            p_gate_zz=args.p_gate_zz,
        )
        env = make_env(
            generator=generator,
            n_shots=args.chunk_shots,
            action_scale=args.action_scale,
            local_action_hops=args.local_action_hops,
            start_from_oracle=args.start_from_oracle,
            use_endpoint_firing=args.use_endpoint_firing,
            burn_in_steps=args.burn_in_steps,
            mismatch=args.mismatch,
        )

        sample_obs, _ = env.reset(seed=args.seed)
        node_dim = sample_obs["node_features"].shape[1]

        agent = SACAgent(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            static_edge_index=env.line_edge_index,
            n_layers=args.n_layers,
            lr=args.lr,
            gamma=0.0,
            tau=0.005,
            alpha=args.alpha,
        )
        agent.load_models(args.model_path)

        total_shots = 0
        total_static_errs = 0
        total_corr_errs = 0
        total_neural_errs = 0
        chunk_idx = 0

        # Loop until the best decoder reaches the target error threshold (or we hit max shots)
        while total_shots < args.max_shots:
            # Deterministic distinct seed per chunk
            seed = args.seed + int(100_000 * p) + chunk_idx 
            
            out = evaluate_one_episode(
                env=env,
                agent=agent,
                seed=seed,
                bypass_threshold=args.bypass_threshold,
                burn_in_steps=args.burn_in_steps,
            )
            
            total_static_errs += out["static_errors"]
            total_corr_errs += out["corr_errors"]
            total_neural_errs += out["neural_errors"]
            total_shots += out["n_steps"]
            chunk_idx += 1

            # Check if all decoders have accumulated enough errors to establish a reliable LER
            min_errors_observed = min(total_static_errs, total_corr_errs, total_neural_errs)
            if min_errors_observed >= args.target_errors:
                break

        # Calculate exact Binomial LER
        static_ler = total_static_errs / total_shots
        corr_ler = total_corr_errs / total_shots
        neural_ler = total_neural_errs / total_shots

        # Calculate Binomial Standard Error: sqrt(p(1-p)/N)
        static_std = np.sqrt((static_ler * (1.0 - static_ler)) / total_shots)
        corr_std = np.sqrt((corr_ler * (1.0 - corr_ler)) / total_shots)
        neural_std = np.sqrt((neural_ler * (1.0 - neural_ler)) / total_shots)

        all_results["p"].append(p)
        all_results["static_mean"].append(float(static_ler))
        all_results["static_std"].append(float(static_std))
        all_results["corr_mean"].append(float(corr_ler))
        all_results["corr_std"].append(float(corr_std))
        all_results["neural_mean"].append(float(neural_ler))
        all_results["neural_std"].append(float(neural_std))

        print(f"\n--- Results for p={p:.5f} ---")
        print(f"Total Shots Evaluated: {total_shots:,}")
        print(f"  Standard MWPM : {total_static_errs} errors | LER: {static_ler:.6e} ± {static_std:.6e}")
        print(f"  Corr Matching : {total_corr_errs} errors | LER: {corr_ler:.6e} ± {corr_std:.6e}")
        print(f"  Neural Agent  : {total_neural_errs} errors | LER: {neural_ler:.6e} ± {neural_std:.6e}")

    return all_results


def save_csv(results: Dict[str, List[float]], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = [
        "p",
        "static_mean",
        "static_std",
        "corr_mean",
        "corr_std",
        "neural_mean",
        "neural_std",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(len(results["p"])):
            row = [
                results["p"][i],
                results["static_mean"][i],
                results["static_std"][i],
                results["corr_mean"][i],
                results["corr_std"][i],
                results["neural_mean"][i],
                results["neural_std"][i],
            ]
            f.write(",".join(str(x) for x in row) + "\n")


def make_plot(results: Dict[str, List[float]], out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    p = np.array(results["p"], dtype=np.float64)

    static_mean = np.array(results["static_mean"], dtype=np.float64)
    static_std = np.array(results["static_std"], dtype=np.float64)

    corr_mean = np.array(results["corr_mean"], dtype=np.float64)
    corr_std = np.array(results["corr_std"], dtype=np.float64)

    neural_mean = np.array(results["neural_mean"], dtype=np.float64)
    neural_std = np.array(results["neural_std"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot with error bars (yerr = binomial standard error). 
    ax.errorbar(p, static_mean, yerr=static_std, fmt='o-', capsize=4, label='Standard MWPM', linewidth=2)
    ax.errorbar(p, corr_mean, yerr=corr_std, fmt='s-', capsize=4, label='Correlated Matching', linewidth=2)
    ax.errorbar(p, neural_mean, yerr=neural_std, fmt='^-', capsize=4, label='Neural Correlated Matching', linewidth=2)

    ax.set_yscale('log')
    ax.set_xscale('linear')

    ax.set_title("Logical Error Rate vs Physical Error Rate")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot LER vs physical error rate for MWPM, correlated matching, and fixed neural policy."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hyperparam_tun_v2/slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.pth",
        help="Path to fixed learned SAC-GNN policy.",
    )
    parser.add_argument(
        "--p-values",
        type=str,
        default="0.001, 0.0008, 0.0006",
        help="Comma-separated physical error rates.",
    )
    
    # Adaptive Shot Parameters
    parser.add_argument("--target-errors", type=int, default=300, help="Stop evaluating a 'p' when the best decoder hits this many errors.")
    parser.add_argument("--max-shots", type=int, default=1_000_000_000, help="Absolute max shots per 'p' to prevent infinite loops at low p.")
    parser.add_argument("--chunk-shots", type=int, default=200_000, help="Number of shots generated per environment reset.")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--p-gate-zz", type=float, default=0.0)

    parser.add_argument("--action-scale", type=float, default=5.0)
    parser.add_argument("--bypass-threshold", type=int, default=2)
    parser.add_argument("--local-action-hops", type=int, default=1)

    # Drift + burn-in
    parser.add_argument(
        "--mismatch", type=float, default=10.0,
        help="Log-uniform drift factor M (per-edge noise prob scaled in [1/M, M]). "
             "1.0 disables drift.",
    )
    parser.add_argument(
        "--burn-in-steps", type=int, default=0,
        help="Pre-evaluation shots run with zero action; not counted in LER. "
             "If >0 AND --mismatch>1, the env's update_period switches from frozen "
             "to 1000 so CMA-driven graph alignment fires during burn-in.",
    )

    # Env feature toggles (must match how the loaded model was trained)
    parser.add_argument(
        "--use-endpoint-firing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add (d1_fired, d2_fired, is_boundary) endpoint features to each node. Use --no-use-endpoint-firing to disable.",
    )
    parser.add_argument(
        "--start-from-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed each episode at drifted-oracle weights and disable CMA tracer alignment. Use --no-start-from-oracle to disable.",
    )

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.01)

    parser.add_argument("--out-png", type=str, default="plots/ler_vs_p_slowgpu_v2_qec_graph_optuna_run_d7_trial_0040_best.png")
    parser.add_argument("--out-csv", type=str, default="data/ler_vs_p_slowgpu_v2_qec_graph_optuna_run_d7_trial_0040_best.csv")

    args = parser.parse_args()

    results = run_sweep(args)
    save_csv(results, args.out_csv)
    make_plot(results, args.out_png)

    print(f"\nSaved plot to: {args.out_png}")
    print(f"Saved table to: {args.out_csv}")


if __name__ == "__main__":
    main()