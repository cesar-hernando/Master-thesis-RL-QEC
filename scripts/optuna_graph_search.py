from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.engine import train, validate
from adaptiveQRL.gnn_sac_agent import GraphReplayBuffer, SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from sandbox_optuna_graph_space import optuna_categorical_choices, preset_by_id, trial_presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute-aware Optuna runner for this RL-QEC repo: one long training run per trial."
        )
    )

    parser.add_argument("--study-name", type=str, default="qec_graph_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("models/optuna_results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/optuna"))
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--trial-preset-id",
        type=int,
        default=None,
        help="Preset index from sandbox_optuna_graph_space.trial_presets().",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials to run in this process (keep at 1 for cluster jobs).",
    )
    parser.add_argument(
        "--sampler",
        choices=["random", "tpe"],
        default="random",
        help="Ignored when --trial-preset-id is set (queued fixed trial).",
    )
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--skip-if-completed",
        action="store_true",
        help="Exit quickly if this preset_id is already completed in the study.",
    )

    # Environment settings
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-gate-zz", type=float, default=0.0)
    parser.add_argument("--mismatch", type=float, default=30.0)
    parser.add_argument("--n-shots", type=int, default=65_000)
    parser.add_argument("--n-test-shots", type=int, default=0)
    parser.add_argument("--burn-in-steps", type=int, default=15_000)
    parser.add_argument("--bypass-threshold", type=int, default=2)
    parser.add_argument("--prior-shots", type=int, default=1_000)
    parser.add_argument("--update-period", type=int, default=1_000)
    parser.add_argument("--local-action-only", action="store_true", default=True)
    parser.add_argument("--global-action", action="store_true", help="Disable local action mask.")

    # Agent constants (kept fixed for contextual-bandit semantics unless explicitly changed)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Training schedule settings
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)

    return parser.parse_args()


def build_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    sampler_seed = args.seed if args.sampler_seed is None else args.sampler_seed
    if args.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=args.n_startup_trials)
    return optuna.samplers.RandomSampler(seed=sampler_seed)


def find_completed_trial_for_preset(study: optuna.Study, preset_id: int) -> optuna.trial.FrozenTrial | None:
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.user_attrs.get("preset_id") == preset_id:
            return t
    return None


def suggest_run_config(trial: optuna.Trial) -> dict:
    choices = optuna_categorical_choices()
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", choices["hidden_dim"]),
        "n_layers": trial.suggest_categorical("n_layers", choices["n_layers"]),
        "lr": trial.suggest_categorical("lr", choices["lr"]),
        "alpha": trial.suggest_categorical("alpha", choices["alpha"]),
        "batch_size": trial.suggest_categorical("batch_size", choices["batch_size"]),
        "update_frequency": trial.suggest_categorical(
            "update_frequency", choices["update_frequency"]
        ),
        "local_action_hops": trial.suggest_categorical(
            "local_action_hops", choices["local_action_hops"]
        ),
        "action_scale": trial.suggest_categorical("action_scale", choices["action_scale"]),
    }


def objective_factory(args: argparse.Namespace, trial_root: Path):
    local_action_only = args.local_action_only and not args.global_action

    def objective(trial: optuna.Trial) -> float:
        run_cfg = suggest_run_config(trial)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model_path = trial_root / f"{args.study_name}_trial_{trial.number:04d}.pth"

        config = {
            "MODE": "train",
            "model_path": str(model_path),
            "distance": args.distance,
            "n_rounds": args.n_rounds,
            "p": args.p,
            "p_gate_zz": args.p_gate_zz,
            "mismatch": args.mismatch,
            "n_shots": args.n_shots,
            "n_test_shots": args.n_test_shots,
            "burn_in_steps": args.burn_in_steps,
            "bypass_threshold": args.bypass_threshold,
            "action_scale": run_cfg["action_scale"],
            "update_period": args.update_period,
            "prior_shots": args.prior_shots,
            "local_action_only": local_action_only,
            "local_action_hops": run_cfg["local_action_hops"],
            "n_layers": run_cfg["n_layers"],
            "hidden_dim": run_cfg["hidden_dim"],
            "lr": run_cfg["lr"],
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": run_cfg["alpha"],
            "batch_size": run_cfg["batch_size"],
            "buffer_capacity": args.buffer_capacity,
            "update_frequency": run_cfg["update_frequency"],
            "train_episodes": args.train_episodes,
            "test_episodes": 3,
        }

        start_s = time.time()

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
            n_test_shots=config["n_test_shots"],
            use_pearson_correlation=True,
            use_syndrome_features=False,
            update_with="DGR",
            train_mode=True,
        )

        sample_obs, _ = env.reset(seed=args.seed)
        node_dim = sample_obs["node_features"].shape[1]

        agent = SACAgent(
            node_dim=node_dim,
            hidden_dim=config["hidden_dim"],
            static_edge_index=env.line_edge_index,
            lr=config["lr"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            n_layers=config["n_layers"],
        )
        buffer = GraphReplayBuffer(capacity=config["buffer_capacity"])

        train(env, agent, buffer, config)

        best_model_path = str(model_path).replace(".pth", "_best.pth")
        if Path(best_model_path).exists():
            agent.load_models(best_model_path)
            scored_model_path = best_model_path
        else:
            scored_model_path = str(model_path)

        score = float(validate(env, agent, config))

        trial.set_user_attr("runtime_seconds", time.time() - start_s)
        trial.set_user_attr("model_path", scored_model_path)
        trial.set_user_attr("preset_id", args.trial_preset_id)

        return score

    return objective


def main() -> None:
    args = parse_args()

    if args.distance < 3:
        raise ValueError("--distance must be >= 3")
    if args.n_rounds < 1:
        raise ValueError("--n-rounds must be >= 1")
    if args.n_shots <= args.burn_in_steps:
        raise ValueError("--n-shots must be greater than --burn-in-steps")
    if args.gamma != 0.0:
        print("[WARN] gamma is not 0.0. This departs from contextual bandit semantics.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    trial_root = (args.model_dir / args.study_name).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    if storage is None:
        storage_path = (args.results_dir / f"{args.study_name}.sqlite3").resolve()
        storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=build_sampler(args),
        storage=storage,
        load_if_exists=True,
    )

    if args.trial_preset_id is not None:
        preset = preset_by_id(args.trial_preset_id)

        if args.skip_if_completed:
            existing = find_completed_trial_for_preset(study, args.trial_preset_id)
            if existing is not None:
                print(
                    f"Preset {args.trial_preset_id} already complete as trial {existing.number}; skipping."
                )
                return

        queued_params = {
            "hidden_dim": preset["hidden_dim"],
            "n_layers": preset["n_layers"],
            "lr": preset["lr"],
            "alpha": preset["alpha"],
            "batch_size": preset["batch_size"],
            "update_frequency": preset["update_frequency"],
            "local_action_hops": preset["local_action_hops"],
            "action_scale": preset["action_scale"],
        }
        study.enqueue_trial(queued_params)
        print(f"Queued fixed preset_id={args.trial_preset_id}: {json.dumps(queued_params)}")

    print(f"Study: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Total preset count: {len(trial_presets())}")
    print(f"Running n_trials={args.n_trials}, timeout_sec={args.timeout_sec}")

    study.optimize(
        objective_factory(args, trial_root),
        n_trials=args.n_trials,
        timeout=(None if args.timeout_sec <= 0 else args.timeout_sec),
    )

    best = {
        "study_name": args.study_name,
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials_total": len(study.trials),
    }
    out_json = args.results_dir / f"{args.study_name}_best.json"
    out_json.write_text(json.dumps(best, indent=2) + "\n")

    print("Best trial summary:")
    print(json.dumps(best, indent=2))
    print(f"Saved best summary: {out_json}")


if __name__ == "__main__":
    main()
