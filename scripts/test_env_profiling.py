"""
Profile `DriftedMatchingEnv` to identify where reset and step time is spent.

Usage examples:
    python scripts/test_env_profiling.py --n-shots 5000 --profile-cpu
    python scripts/test_env_profiling.py --n-shots 8000 --top 20
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable

import numpy as np

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
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
    parser = argparse.ArgumentParser(description="Profile DriftedMatchingEnv reset/step internals.")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--mismatch", type=float, default=30.0)
    parser.add_argument("--n-shots", type=int, default=50_000)
    parser.add_argument("--burn-in-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-period", type=int, default=1000)
    parser.add_argument("--prior-shots", type=int, default=1000)
    parser.add_argument("--local-action-only", action="store_true", default=True)
    parser.add_argument("--local-action-hops", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=3.0)
    parser.add_argument("--profile-cpu", action="store_true", help="Enable cProfile around reset+loop.")
    parser.add_argument(
        "--deep-profile-prepare",
        action="store_true",
        help="Enable phase-by-phase profiling inside env._prepare_next_observation.",
    )
    parser.add_argument("--top", type=int, default=25, help="Top entries to print for profiles.")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> DriftedMatchingEnv:
    generator = SyndromeDataGenerator(
        distance=args.distance,
        n_rounds=args.n_rounds,
        mismatch=args.mismatch,
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": args.p,
            "before_measure_flip_probability": args.p,
            "after_reset_flip_probability": args.p,
            "before_round_data_depolarization": args.p,
            "p_gate_zz": 0.0,
        },
        memory_type="z",
        n_shots=args.n_shots,
        qec_code="surface_code",
    )

    return DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=args.local_action_only,
        local_action_hops=args.local_action_hops,
        action_scale=args.action_scale,
        update_period=args.update_period,
        prior_shots=args.prior_shots,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        update_with="DGR",
        render_mode=None,
    )


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
        if total_ref_s and total_ref_s > 0:
            share = 100.0 * stat.total_s / total_ref_s
        else:
            share = 0.0
        print(f"{name:40s} {stat.total_s:10.4f} {stat.calls:8d} {avg_ms:10.3f} {share:8.2f}")


def install_detailed_prepare_profiler(env: DriftedMatchingEnv, profiler: MethodProfiler) -> None:
    """Replace _prepare_next_observation with a phase-instrumented equivalent for profiling."""

    def _record(key: str, t_start: float) -> None:
        profiler.add_timing(key, time.perf_counter() - t_start)

    def profiled_prepare_next_observation():
        t_total = time.perf_counter()

        t = time.perf_counter()
        syndrome = env.syndrome_batch[env.step_count]
        true_obs = env.true_obs_batch[env.step_count]
        _record("env._prepare.phase.read_current_shot", t)

        t = time.perf_counter()
        selected_edges_1, first_pass_pred_obs = env.syndrome_data_generator.get_solution_edges(
            matching=env.current_matching,
            syndrome_volume=syndrome,
            enable_correlations=False,
            return_predicted_obs=True,
            pair_to_idx_matrix=env.pair_to_idx_matrix,
            fault_array=env.fault_array,
        )
        _record("env._prepare.phase.first_pass_decode", t)

        t = time.perf_counter()
        selected_idx_1 = env._selected_edge_indices_from_pairs(selected_edges_1)
        _record("env._prepare.phase.selected_idx_lookup", t)

        t = time.perf_counter()
        selected_flag = np.zeros(env.n_dec_edges, dtype=np.float32)
        if selected_idx_1 is not None and len(selected_idx_1) > 0:
            selected_flag[selected_idx_1] = 1.0
        _record("env._prepare.phase.selected_flag_build", t)

        t = time.perf_counter()
        action_mask = env._compute_action_mask(selected_idx_1)
        _record("env._prepare.phase.action_mask", t)

        t = time.perf_counter()
        env.current_syndrome = syndrome
        env.current_true_obs = true_obs
        env.current_first_pass_pred_obs = first_pass_pred_obs
        env.current_first_pass_selected_idx = selected_idx_1
        env.current_action_mask = action_mask
        _record("env._prepare.phase.cache_state", t)

        t = time.perf_counter()
        if env.use_pearson_correlation:
            if env.n_line_edges > 0:
                dgr_edge_feat = env.pearson_correlations
            else:
                dgr_edge_feat = np.zeros(0, dtype=np.float32)
        else:
            dgr_edge_feat = env.corr_tracer
        _record("env._prepare.phase.edge_feature_compute", t)

        t = time.perf_counter()
        if env.use_syndrome_features:
            node_feats = np.stack([env.current_weights, selected_flag, env.spitz_tracer], axis=1)
            if env.n_line_edges > 0:
                edge_feats = np.stack([dgr_edge_feat, env.remm_tracer], axis=1)
            else:
                edge_feats = np.zeros((0, 2), dtype=np.float32)
        else:
            # Fallback to the original DGR-only sizes
            env.node_feats[:, 0] = env.current_weights
            env.node_feats[:, 1] = selected_flag
            if env.n_line_edges > 0:
                env.edge_feats[:, 0] = dgr_edge_feat

        _record("env._prepare.phase.feature_assembly", t)

        t = time.perf_counter()
        obs = {
            "node_features": env.node_feats,
            "edge_index": env.line_edge_index,
            "edge_attr": env.edge_feats,
            "action_mask": action_mask,
        }
        _record("env._prepare.phase.obs_pack", t)

        profiler.add_timing("env._prepare_next_observation", time.perf_counter() - t_total)
        return obs

    env._prepare_next_observation = profiled_prepare_next_observation


def run_loop(args: argparse.Namespace) -> None:
    env = build_env(args)
    profiler = MethodProfiler()

    # Time internal env methods that are likely hotspots in update/reset behavior.
    env_methods = [
        "_compute_action_mask",
        "_selected_edge_indices_from_pairs",
        "_accumulate_occurrence",
        "_accumulate_correlation",
        "_apply_cma_and_update_graph",
        "compute_pearson_correlations",
    ]
    if not args.deep_profile_prepare:
        env_methods.insert(0, "_prepare_next_observation")

    for method_name in env_methods:
        profiler.wrap_method(env, method_name, key=f"env.{method_name}")

    if args.deep_profile_prepare:
        install_detailed_prepare_profiler(env, profiler)

    # Time data-generator methods frequently called from reset/step.
    gen_methods = [
        "generate_drifted_circuit",
        "simulate_syndrome_data",
        "get_solution_edges",
        "get_solution_edges_batch",
    ]
    for method_name in gen_methods:
        profiler.wrap_method(env.syndrome_data_generator, method_name, key=f"gen.{method_name}")

    reset_stats = {"env.reset": TimerStat()}
    step_stats = {"env.step_total": TimerStat()}

    # Optional full CPU callgraph profiler.
    cprof: cProfile.Profile | None = cProfile.Profile() if args.profile_cpu else None

    if cprof is not None:
        cprof.enable()

    t0 = time.perf_counter()
    r0 = time.perf_counter()
    obs, info = env.reset(seed=args.seed)
    reset_stats["env.reset"].add(time.perf_counter() - r0)

    terminated = False
    truncated = False
    step_idx = 0
    n_shots = args.n_shots
    action = np.zeros(env.n_dec_edges, dtype=np.float32)

    while not (terminated or truncated):
        s0 = time.perf_counter()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        step_stats["env.step_total"].add(time.perf_counter() - s0)

        obs = next_obs
        step_idx += 1

        if args.progress_every > 0 and step_idx % args.progress_every == 0:
            print(f"progress: {step_idx}/{n_shots} steps")

    episode_s = time.perf_counter() - t0

    if cprof is not None:
        cprof.disable()

    print("\nRun summary")
    print("-----------")
    print(f"shots configured      : {n_shots}")
    print(f"steps executed        : {step_idx}")
    print(f"total runtime (s)     : {episode_s:.4f}")
    print(f"reset runtime (s)     : {reset_stats['env.reset'].total_s:.4f}")
    print(f"loop runtime (s)      : {step_stats['env.step_total'].total_s:.4f}")
    if step_idx > 0:
        print(f"avg step runtime (ms) : {1000.0 * step_stats['env.step_total'].total_s / step_idx:.3f}")

    print_ranked_stats(
        title="Instrumented method breakdown",
        data=profiler.stats,
        top=args.top,
        total_ref_s=episode_s,
    )

    print_ranked_stats(
        title="Top-level reset/step totals",
        data={**reset_stats, **step_stats},
        top=args.top,
        total_ref_s=episode_s,
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
    run_loop(args)


if __name__ == "__main__":
    main()
