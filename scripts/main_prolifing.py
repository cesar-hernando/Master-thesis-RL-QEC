"""
Profile the full main workflow in test mode with method-level timing.

This keeps the same general profiling style as scripts/test_env_profiling.py
(no deep phase profiling) but runs the engine test pipeline.

Usage examples:
    python scripts/main_prolifing.py --model-path models/sac_gnn_30.pth
    python scripts/main_prolifing.py --n-shots 5000 --test-episodes 3 --top 25
    python scripts/main_prolifing.py --profile-cpu --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from functools import wraps

import numpy as np
import pymatching

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.engine import test
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


@dataclass
class TimerStat:
    total_s: float = 0.0
    calls: int = 0

    def add(self, delta_s: float) -> None:
        self.total_s += delta_s
        self.calls += 1


class MethodProfiler:
    """Lightweight profiler for selected bound methods."""

    def __init__(self) -> None:
        self.stats: dict[str, TimerStat] = {}

    def add_timing(self, key: str, delta_s: float) -> None:
        self.stats.setdefault(key, TimerStat())
        self.stats[key].add(delta_s)

    def wrap_method(self, obj: object, method_name: str, key: str | None = None) -> None:
        if not hasattr(obj, method_name):
            return

        original = getattr(obj, method_name)
        if not callable(original):
            return

        stat_key = key or method_name
        self.stats.setdefault(stat_key, TimerStat())

        @wraps(original)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.stats[stat_key].add(time.perf_counter() - t0)

        setattr(obj, method_name, wrapped)


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile scripts/main.py behavior in test mode.")
    parser.add_argument("--model-path", type=str, default="models/sac_gnn_30.pth")

    # Environment settings (same semantics as scripts/main.py)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-gate-zz", type=float, default=0.0)
    parser.add_argument("--mismatch", type=float, default=30.0)
    parser.add_argument("--n-shots", type=int, default=6500)
    parser.add_argument("--burn-in-steps", type=int, default=1500)
    parser.add_argument("--bypass-threshold", type=int, default=2)
    parser.add_argument("--action-scale", type=float, default=3.0)
    parser.add_argument("--update-period", type=int, default=100)
    parser.add_argument("--prior-shots", type=int, default=1000)
    parser.add_argument("--oracle-reward-coef", type=float, default=0.0)
    parser.add_argument("--local-action-only", action="store_true", default=True)
    parser.add_argument("--global-action", action="store_true", help="Disable local action mask.")
    parser.add_argument("--local-action-hops", type=int, default=1)

    # Agent settings
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.01)

    # Test settings
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Profiling output
    parser.add_argument("--profile-cpu", action="store_true", help="Enable cProfile around full run.")
    parser.add_argument(
    "--deep-profile-step",
    action="store_true",
    help="Enable phase-by-phase profiling inside env.step.",)
    parser.add_argument("--top", type=int, default=25, help="Top entries to print for profiles.")
    return parser.parse_args()


def print_ranked_stats(title: str, data: dict[str, TimerStat], top: int, total_ref_s: float | None = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    sorted_items = sorted(data.items(), key=lambda kv: kv[1].total_s, reverse=True)
    if not sorted_items:
        print("No data collected.")
        return

    shown = sorted_items[:top]
    print(f"{'name':40s} {'total(s)':>10s} {'calls':>8s} {'avg(ms)':>10s} {'share%':>8s}")
    for name, stat in shown:
        avg_ms = (1000.0 * stat.total_s / stat.calls) if stat.calls else 0.0
        share = 100.0 * stat.total_s / total_ref_s if total_ref_s and total_ref_s > 0 else 0.0
        print(f"{name:40s} {stat.total_s:10.4f} {stat.calls:8d} {avg_ms:10.3f} {share:8.2f}")


def print_step_phase_breakdown(profiler_stats: dict[str, TimerStat]) -> None:
    """Always-visible breakdown of env.step internal phases as % of env.step total."""
    step_stat = profiler_stats.get("env.step")
    phase_items = sorted(
        [(k, v) for k, v in profiler_stats.items() if k.startswith("env.step.phase.")],
        key=lambda kv: kv[1].total_s,
        reverse=True,
    )
    if not phase_items:
        print("\nenv.step phase breakdown: no phase data collected (run with --deep-profile-step).")
        return

    step_total = step_stat.total_s if step_stat else sum(v.total_s for _, v in phase_items)

    title = "env.step internal phase breakdown (% of env.step total)"
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'phase':40s} {'total(s)':>10s} {'calls':>8s} {'avg(ms)':>10s} {'of step%':>9s}")
    for name, stat in phase_items:
        label = name.replace("env.step.phase.", "")
        avg_ms = (1000.0 * stat.total_s / stat.calls) if stat.calls else 0.0
        share = 100.0 * stat.total_s / step_total if step_total > 0 else 0.0
        print(f"{label:40s} {stat.total_s:10.4f} {stat.calls:8d} {avg_ms:10.3f} {share:9.2f}")

    accounted = sum(v.total_s for _, v in phase_items)
    overhead = step_total - accounted
    print(f"{'[unaccounted overhead]':40s} {overhead:10.4f} {'':>8s} {'':>10s} {100.0 * overhead / step_total if step_total > 0 else 0.0:9.2f}")


def build_config(args: argparse.Namespace) -> dict:
    local_action_only = args.local_action_only and not args.global_action
    return {
        "MODE": "test",
        "model_path": args.model_path,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "p": args.p,
        "p_gate_zz": args.p_gate_zz,
        "mismatch": args.mismatch,
        "n_shots": args.n_shots,
        "burn_in_steps": args.burn_in_steps,
        "bypass_threshold": args.bypass_threshold,
        "action_scale": args.action_scale,
        "update_period": args.update_period,
        "prior_shots": args.prior_shots,
        "oracle_reward_coef": args.oracle_reward_coef,
        "local_action_only": local_action_only,
        "local_action_hops": args.local_action_hops,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": args.alpha,
        "test_episodes": args.test_episodes,
    }


def install_detailed_step_profiler(env: DriftedMatchingEnv, profiler: MethodProfiler) -> None:
    """Replace env.step with a phase-instrumented equivalent for detailed profiling."""

    def _record(key: str, t_start: float) -> None:
        profiler.add_timing(key, time.perf_counter() - t_start)

    def profiled_step(action):
        t_step = time.perf_counter()

        assert env.current_syndrome is not None, "Call reset() before step()."

        t = time.perf_counter()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != env.n_dec_edges:
            raise ValueError(f"Action has shape {action.shape}; expected ({env.n_dec_edges},)")
        _record("env.step.phase.validate_action", t)

        t = time.perf_counter()
        if env.local_action_only:
            mask = env.current_action_mask.astype(np.float32)
            applied_delta = action * mask * env.action_scale
        else:
            applied_delta = action * env.action_scale

        second_pass_edge_reweights = None

        if not np.any(applied_delta):
            second_pass_matching = env.current_matching
        else:
            second_pass_weights = np.clip(env.current_weights + applied_delta, env.min_weight, env.max_weight)
            if env.supports_edge_reweights:
                second_pass_matching = env.current_matching
                second_pass_edge_reweights = env._build_edge_reweights(second_pass_weights)
            else:
                second_pass_matching = pymatching.Matching.from_check_matrix(env.H, weights=second_pass_weights)
        _record("env.step.phase.apply_action", t)

        t = time.perf_counter()
        selected_edges_2, pred_obs = env.syndrome_data_generator.get_solution_edges(
            matching=second_pass_matching,
            syndrome_volume=env.current_syndrome,
            enable_correlations=True,
            edge_reweights=second_pass_edge_reweights,
            return_predicted_obs=True,
            pair_to_idx_matrix=env.pair_to_idx_matrix,
            fault_array=env.fault_array,
        )
        _record("env.step.phase.second_pass_decode", t)

        t = time.perf_counter()
        selected_idx_2 = env._selected_edge_indices_from_pairs(selected_edges_2)
        _record("env.step.phase.selected_idx_lookup", t)

        t = time.perf_counter()
        env._accumulate_occurrence(selected_idx_2)
        env._accumulate_correlation(selected_idx_2)
        env.shots_since_update += 1
        _record("env.step.phase.accumulate_tracers", t)

        t = time.perf_counter()
        if env.shots_since_update >= env.update_period:
            env._apply_cma_and_update_graph()
            env.shots_since_update = 0
            env.corr_mse_error = np.mean((env.pearson_correlations - env.oracle_correlations) ** 2)
            env.corr_mse_error_static = np.mean((env.pearson_correlations - env.initial_pearson_corr) ** 2)
            env.weights_mse_error = np.mean((env.current_weights - env.oracle_weights) ** 2)
            env.weights_mse_error_static = np.mean((env.current_weights - env.initial_base_weights) ** 2)
        _record("env.step.phase.periodic_cma_update", t)

        t = time.perf_counter()
        agent_correct = pred_obs == env.current_true_obs
        first_pass_correct = env.current_first_pass_pred_obs == env.current_true_obs

        if agent_correct and not first_pass_correct:
            logical_reward = +1.0
        elif not agent_correct and first_pass_correct:
            logical_reward = -1.0
        else:
            logical_reward = 0.0

        reward = logical_reward
        oracle_similarity = None

        if env.oracle_reward_coef > 0.0:
            oracle_edges = env.oracle_solution_edges_batch[env.step_count]
            oracle_idx = env._selected_edge_indices_from_pairs(oracle_edges)
            oracle_similarity = env._edge_set_jaccard(selected_idx_2, oracle_idx)
            oracle_reward = (2.0 * oracle_similarity) - 1.0
            reward += env.oracle_reward_coef * oracle_reward
        _record("env.step.phase.reward", t)

        t = time.perf_counter()
        env.step_count += 1
        terminated = False
        truncated = env.step_count >= env.max_steps

        info = {
            "logical_error": not (agent_correct),
            "true_obs": env.current_true_obs,
            "pred_obs": pred_obs,
            "first_pass_obs": env.current_first_pass_pred_obs,
            "oracle_pred_obs": env.oracle_predicted_obs_batch[env.step_count - 1],
            "static_pred_obs": env.static_predicted_obs_batch[env.step_count - 1],
            "reward_logical": logical_reward,
            "reward_total": float(reward),
            "oracle_similarity_jaccard": float(oracle_similarity) if oracle_similarity is not None else None,
            "selected_edges_first_pass_idx": env.current_first_pass_selected_idx.copy()
            if env.current_first_pass_selected_idx is not None
            else None,
            "selected_edges_second_pass_idx": selected_idx_2.copy(),
            "action_mask": env.current_action_mask.copy() if env.current_action_mask is not None else None,
            "weights_mse_error": env.weights_mse_error,
            "corr_mse_error": env.corr_mse_error,
            "weights_mse_error_static": env.weights_mse_error_static,
            "corr_mse_error_static": env.corr_mse_error_static,
        }
        _record("env.step.phase.info_pack", t)

        t = time.perf_counter()
        if not truncated:
            next_obs = env._prepare_next_observation()
        else:
            next_obs = None
        _record("env.step.phase.next_observation", t)

        profiler.add_timing("env.step", time.perf_counter() - t_step)
        return next_obs, float(reward), terminated, truncated, info

    env.step = profiled_step


def run_profiled_test(args: argparse.Namespace) -> None:
    config = build_config(args)
    profiler = MethodProfiler()
    top_level_stats = {
        "init.total": TimerStat(),
        "init.env_reset_for_node_dim": TimerStat(),
        "engine.test_total": TimerStat(),
    }

    cprof: cProfile.Profile | None = cProfile.Profile() if args.profile_cpu else None
    if cprof is not None:
        cprof.enable()

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    generator = SyndromeDataGenerator(
        distance=config["distance"],
        n_rounds=config["n_rounds"],
        mismatch=config["mismatch"],
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": config["p"],
            "before_measure_flip_probability": config["p"],
            "after_reset_flip_probability": config["p"],
            "before_round_data_depolarization": config["p"],
            "p_gate_zz": config["p_gate_zz"],
        },
        memory_type="z",
        n_shots=config["n_shots"],
        qec_code="surface_code",
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=config["local_action_only"],
        local_action_hops=config["local_action_hops"],
        action_scale=config["action_scale"],
        update_period=config["update_period"],
        prior_shots=config["prior_shots"],
        oracle_reward_coef=config["oracle_reward_coef"],
        use_pearson_correlation=True,
        use_syndrome_features=False,
        update_with="DGR",
    )

    # General method profiling; optionally replace env.step with deep phase profiling.
    env_methods = [
        "reset",
        "_prepare_next_observation",
        "_compute_action_mask",
        "_selected_edge_indices_from_pairs",
        "_accumulate_occurrence",
        "_accumulate_correlation",
        "_apply_cma_and_update_graph",
        "compute_pearson_correlations",
    ]

    if not args.deep_profile_step:
        env_methods.insert(1, "step")

    for method_name in env_methods:
        profiler.wrap_method(env, method_name, key=f"env.{method_name}")

    if args.deep_profile_step:
        install_detailed_step_profiler(env, profiler)

    gen_methods = [
        "generate_drifted_circuit",
        "simulate_syndrome_data",
        "get_solution_edges",
        "get_solution_edges_batch",
    ]
    for method_name in gen_methods:
        profiler.wrap_method(env.syndrome_data_generator, method_name, key=f"gen.{method_name}")

    t_reset = time.perf_counter()
    sample_obs, _ = env.reset(seed=args.seed)
    top_level_stats["init.env_reset_for_node_dim"].add(time.perf_counter() - t_reset)
    node_dim = sample_obs["node_features"].shape[1]

    agent = SACAgent(
        node_dim=node_dim,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
    )

    profiler.wrap_method(agent, "select_action", key="agent.select_action")
    profiler.wrap_method(agent, "load_models", key="agent.load_models")

    top_level_stats["init.total"].add(time.perf_counter() - t0)

    t_test = time.perf_counter()
    test(env, agent, config)
    top_level_stats["engine.test_total"].add(time.perf_counter() - t_test)

    total_runtime = time.perf_counter() - t_total

    if cprof is not None:
        cprof.disable()

    print("\nRun summary")
    print("-----------")
    print(f"mode                 : test")
    print(f"model_path           : {config['model_path']}")
    print(f"distance, rounds     : {config['distance']}, {config['n_rounds']}")
    print(f"n_shots, episodes    : {config['n_shots']}, {config['test_episodes']}")
    print(f"burn_in_steps        : {config['burn_in_steps']}")
    print(f"total runtime (s)    : {total_runtime:.4f}")

    print_ranked_stats(
        title="Instrumented method breakdown",
        data=profiler.stats,
        top=args.top,
        total_ref_s=total_runtime,
    )

    if args.deep_profile_step:
        print_step_phase_breakdown(profiler.stats)

    print_ranked_stats(
        title="Top-level workflow totals",
        data=top_level_stats,
        top=args.top,
        total_ref_s=total_runtime,
    )

    if cprof is not None:
        buffer = io.StringIO()
        stats = pstats.Stats(cprof, stream=buffer).strip_dirs().sort_stats("cumtime")
        stats.print_stats(args.top)
        print("\nTop cProfile entries (cumtime)")
        print("------------------------------")
        print(buffer.getvalue())


def main() -> None:
    args = build_argparser()
    run_profiled_test(args)


if __name__ == "__main__":
    main()
