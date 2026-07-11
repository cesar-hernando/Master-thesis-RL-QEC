"""
Compare the linear correlated-matching agent (REINFORCE / PPO-trained, NO critic) against
PyMatching correlated matching, across a range of physical error rates, at mismatch=1.

All decoders are evaluated on *identical* syndromes (same seeds) so the comparison is
apples-to-apples. For each physical error rate `p` we report the logical error rate (LER) of:

  - MWPM    : plain minimum-weight perfect matching (no correlations)   [info "first_pass_obs"]
    - CM      : PyMatching correlated matching                            [decode_batch(enable_correlations=True)]
  - Neural  : the linear agent's learned 2nd-pass reweighting            [agent action -> info "pred_obs"]

The env is frozen at the oracle (= base, since mismatch=1) weights with no in-episode graph
updates and no CMA alignment reweighting, so the agent / CM perturb on top of the correct
marginal-MWPM first pass. (No oracle decoder is evaluated: at mismatch=1 it is redundant.)

Example
-------
    python scripts/compare_linear_vs_cm.py \
        --model-path models/linear/linear_model_0_best.pth \
        --distance 5 --n-rounds 5 \
        --p-min 1e-3 --p-max 1e-2 --n-p 8 \
        --target-errors 200 --max-shots 400000
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pymatching

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


def make_generator(p, n_shots, distance, n_rounds, mismatch, p_gate_zz):
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


def make_env(generator, n_shots, local_action_hops):
    # Linear-CM env: -log joint-prob edge feature, action_scale=1 so the agent's
    # coefficients map directly to weight deltas.
    #   * start_from_oracle=True  -> seed at oracle (= base, since mismatch=1) weights AND
    #                                disable all CMA alignment reweighting.
    #   * update_period > n_shots -> no in-episode graph updates either.
    #   * train_mode=True         -> SKIP the expensive per-shot oracle/static batch decode
    #                                in reset() (we don't need the oracle here). The plain-MWPM
    #                                baseline is read from the first pass (info["first_pass_obs"]).
    return DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=local_action_hops,
        action_scale=1.0,
        update_period=n_shots + 1,        # never fire an in-episode graph update
        prior_shots=1_000,
        n_test_shots=0,
        use_pearson_correlation=False,
        use_syndrome_features=False,
        use_log_joint_prob=True,          # required by the linear CM actor
        start_from_oracle=True,           # freeze at oracle (= base at mismatch=1) weights, no alignment
        use_endpoint_firing=False,
        update_with="DGR",
        train_mode=True,                  # skip oracle/static precompute (not needed at mismatch=1)
    )


def evaluate_chunk(env, agent, cm_matching, seed, bypass_threshold):
    """
    Two deterministic passes over the SAME syndromes (same seed):
      Pass A applies the neural agent (and reads plain MWPM from the first pass for free),
            Pass B applies PyMatching correlated matching in batch mode.
    Returns error counts and the number of evaluated shots.
    """
    # --- Pass A: neural agent (MWPM = the first pass it builds on) ---
    obs, _ = env.reset(seed=seed)
    done = False
    neural = mwpm = steps = 0
    while not done:
        n_flashes = int(np.sum(env.current_syndrome != 0))
        if n_flashes <= bypass_threshold:
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
        else:
            action = agent.select_action(obs, evaluate=True)   # greedy, no exploration noise
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t = info["true_obs"]
        neural += int(info["pred_obs"] != t)
        mwpm += int(info["first_pass_obs"] != t)               # plain MWPM (oracle=base weights)
        steps += 1

    # --- Pass B: correlated matching in batch mode (identical syndromes) ---
    obs, _ = env.reset(seed=seed)
    cm_pred = np.asarray(
        cm_matching.decode_batch(env.syndrome_batch, enable_correlations=True),
        dtype=np.uint8,
    ).reshape(-1)
    cm = int(np.sum(cm_pred != env.true_obs_batch))

    return {"steps": steps, "neural": neural, "mwpm": mwpm, "cm": cm}


def run_sweep(args):
    p_values = np.logspace(np.log10(args.p_min), np.log10(args.p_max), args.n_p)

    results = {k: [] for k in ["p", "mwpm", "cm", "neural",
                               "mwpm_std", "cm_std", "neural_std"]}

    print(f"Linear-CM agent vs analytical CM | distance={args.distance}, mismatch={args.mismatch}")
    print(f"Model: {args.model_path}")
    print(f"p values: {[f'{p:.2e}' for p in p_values]}")
    print(f"Target errors per p: {args.target_errors} | Max shots per p: {args.max_shots:,}\n")

    for p in p_values:
        generator = make_generator(p, args.chunk_shots, args.distance, args.n_rounds,
                                   args.mismatch, args.p_gate_zz)
        env = make_env(generator, args.chunk_shots, args.local_action_hops)
        cm_matching = pymatching.Matching.from_detector_error_model(
            env.base_dem, enable_correlations=True
        )

        sample_obs, _ = env.reset(seed=args.seed)
        node_dim = sample_obs["node_features"].shape[1]

        agent = SACAgent(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            static_edge_index=env.line_edge_index,
            n_layers=args.n_layers,
            gamma=0.0,
            actor_type="linear_cm",
            action_scale=1.0,
            # std is irrelevant for evaluate=True; fixed_std just matches the saved
            # actor's 'log_std' so the state_dict loads cleanly.
            linear_cm_fixed_std=args.fixed_std,
        )
        agent.load_models(args.model_path)

        tot = {"steps": 0, "neural": 0, "mwpm": 0, "cm": 0}
        chunk_idx = 0
        while tot["steps"] < args.max_shots:
            seed = args.seed + int(1e5 * p) + chunk_idx
            out = evaluate_chunk(env, agent, cm_matching, seed, args.bypass_threshold)
            for k in tot:
                tot[k] += out[k]
            chunk_idx += 1
            # Stop once the decoders all have enough errors for a stable LER.
            if min(tot["neural"], tot["cm"], tot["mwpm"]) >= args.target_errors:
                break

        n = tot["steps"]

        def ler_std(errs):
            ler = errs / n if n else float("nan")
            std = np.sqrt(ler * (1.0 - ler) / n) if n else float("nan")
            return ler, std

        mwpm_ler, mwpm_std = ler_std(tot["mwpm"])
        cm_ler, cm_std = ler_std(tot["cm"])
        neural_ler, neural_std = ler_std(tot["neural"])

        results["p"].append(float(p))
        results["mwpm"].append(mwpm_ler);   results["mwpm_std"].append(mwpm_std)
        results["cm"].append(cm_ler);       results["cm_std"].append(cm_std)
        results["neural"].append(neural_ler); results["neural_std"].append(neural_std)

        print(f"--- p={p:.3e} | shots={n:,} ---")
        print(f"  MWPM   : {tot['mwpm']:6d} err | LER {mwpm_ler:.3e} +/- {mwpm_std:.1e}")
        print(f"  CM     : {tot['cm']:6d} err | LER {cm_ler:.3e} +/- {cm_std:.1e}")
        print(f"  Neural : {tot['neural']:6d} err | LER {neural_ler:.3e} +/- {neural_std:.1e}")
        if cm_ler > 0:
            print(f"  -> Neural/CM LER ratio: {neural_ler / cm_ler:.3f} "
                  f"({'Neural better' if neural_ler < cm_ler else 'CM better'})\n")

    return results, p_values


def plot_results(results, args):
    os.makedirs("plots", exist_ok=True)
    p = np.array(results["p"])

    fig, ax = plt.subplots(figsize=(8, 6))
    series = [
        ("mwpm",   "MWPM (no correlations)", "#9467bd", "o"),
        ("cm",     "Correlated Matching",    "#ff7f0e", "s"),
        ("neural", "Neural CM (linear)",     "#2ca02c", "D"),
    ]
    for key, label, color, marker in series:
        y = np.array(results[key])
        yerr = np.array(results[f"{key}_std"])
        ax.errorbar(p, y, yerr=yerr, label=label, color=color, marker=marker,
                    capsize=3, linewidth=1.8, markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate  p")
    ax.set_ylabel("Logical error rate  (LER)")
    ax.set_title(f"Linear neural CM vs analytical CM  (d={args.distance}, mismatch={args.mismatch})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out = os.path.join("plots", args.output)
    fig.savefig(out, dpi=300)
    print(f"\nSaved plot to {out}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="LER vs p: linear CM agent vs PyMatching correlated matching (mismatch=1).")
    ap.add_argument("--model-path", required=True, help="Path to the trained linear_cm model (.pth)")
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=5)
    ap.add_argument("--mismatch", type=float, default=1.0)
    ap.add_argument("--p-min", type=float, default=2e-4)
    ap.add_argument("--p-max", type=float, default=1e-2)
    ap.add_argument("--n-p", type=int, default=10, help="Number of log-spaced p values")
    ap.add_argument("--p-gate-zz", type=float, default=0.0)
    ap.add_argument("--bypass-threshold", type=int, default=2)
    ap.add_argument("--local-action-hops", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=256, help="Must match the trained model")
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--fixed-std", type=float, default=1.5, help="Only for state_dict load; unused at eval")
    ap.add_argument("--chunk-shots", type=int, default=1_000_000, help="Shots per episode/seed")
    ap.add_argument("--target-errors", type=int, default=500, help="Stop a p once each decoder hits this many errors")
    ap.add_argument("--max-shots", type=int, default=10_000_000_000_000, help="Cap on shots per p")
    ap.add_argument("--seed", type=int, default=20_001)
    ap.add_argument("--output", type=str, default="ler_vs_p_linear_ppo_vs_cm.png")
    args = ap.parse_args()

    results, _ = run_sweep(args)
    plot_results(results, args)

    # Persist the raw numbers next to the plot for later re-plotting.
    npz_path = os.path.join("plots", os.path.splitext(args.output)[0] + ".npz")
    np.savez(npz_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved raw results to {npz_path}")


if __name__ == "__main__":
    main()
