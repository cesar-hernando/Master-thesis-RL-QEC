import time
import numpy as np

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv

CONFIG = {
    "distance": 3,
    "n_rounds": 3,
    "mismatch": 30.0,
    "p": 0.004,
    "n_shots": 220,
    "local_action_only": True,
    "local_action_hops": 1,
    "action_scale": 3.0,
    "update_period": 100,
    "prior_shots": 1000,
}


def make_env() -> DriftedMatchingEnv:
    gen = SyndromeDataGenerator(
        distance=CONFIG["distance"],
        n_rounds=CONFIG["n_rounds"],
        mismatch=CONFIG["mismatch"],
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": CONFIG["p"],
            "before_measure_flip_probability": CONFIG["p"],
            "after_reset_flip_probability": CONFIG["p"],
            "before_round_data_depolarization": CONFIG["p"],
            "p_gate_zz": 0.0,
        },
        memory_type="z",
        n_shots=CONFIG["n_shots"],
        qec_code="surface_code",
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=gen,
        local_action_only=CONFIG["local_action_only"],
        local_action_hops=CONFIG["local_action_hops"],
        action_scale=CONFIG["action_scale"],
        update_period=CONFIG["update_period"],
        prior_shots=CONFIG["prior_shots"],
        oracle_reward_coef=0.0,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        update_with="DGR",
    )
    return env


def run_case(use_reweights: bool, repeats: int = 3, steps: int = 200) -> dict:
    per_step_ms = []

    for rep in range(repeats):
        env = make_env()
        obs, info = env.reset(seed=123 + rep)
        env.supports_edge_reweights = use_reweights

        rng = np.random.default_rng(2026)
        actions = rng.uniform(-1.0, 1.0, size=(steps, env.n_dec_edges)).astype(np.float32)

        t0 = time.perf_counter()
        executed = 0
        for i in range(steps):
            action = actions[i]
            obs, reward, terminated, truncated, info = env.step(action)
            executed += 1
            if terminated or truncated:
                break

        dt = time.perf_counter() - t0
        per_step_ms.append(1000.0 * dt / executed)

    arr = np.array(per_step_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "runs": per_step_ms,
    }


def main() -> None:
    before = run_case(use_reweights=False)
    after = run_case(use_reweights=True)

    speedup = before["mean_ms"] / after["mean_ms"] if after["mean_ms"] > 0 else float("nan")
    improvement = (1.0 - after["mean_ms"] / before["mean_ms"]) * 100.0 if before["mean_ms"] > 0 else float("nan")

    print("BENCHMARK (distance=3, rounds=3, steps<=200, repeats=3)")
    print(
        f"BEFORE (force rebuild): mean={before['mean_ms']:.3f} ms/step, "
        f"std={before['std_ms']:.3f}, runs={[round(x, 3) for x in before['runs']]}"
    )
    print(
        f"AFTER  (edge reweights): mean={after['mean_ms']:.3f} ms/step, "
        f"std={after['std_ms']:.3f}, runs={[round(x, 3) for x in after['runs']]}"
    )
    print(f"SPEEDUP: {speedup:.2f}x  ({improvement:.1f}% faster)")


if __name__ == "__main__":
    main()
