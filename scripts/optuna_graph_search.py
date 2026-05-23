from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import optuna
import torch

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.engine import evaluate_low_p_oracle, train, validate
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
        "--startup-jitter-max-sec",
        type=float,
        default=5.0,
        help=(
            "Maximum random startup delay in seconds before connecting to Optuna storage. "
            "Useful for reducing SQLite init contention in array jobs."
        ),
    )
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

    # Optuna-target evaluation (post-training): LER at low p, no drift, oracle init.
    # Defaults: 4M shots, all in a single episode.
    parser.add_argument("--target-eval-p", type=float, default=0.001)
    parser.add_argument("--target-eval-mismatch", type=float, default=1.0)
    parser.add_argument("--target-eval-shots", type=int, default=4_000_000)
    parser.add_argument("--target-eval-shots-per-episode", type=int, default=4_000_000)
    parser.add_argument("--target-eval-seed", type=int, default=20001)

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


def create_study_with_sqlite_race_retry(
    *,
    study_name: str,
    sampler: optuna.samplers.BaseSampler,
    storage: str,
    max_retries: int = 8,
) -> optuna.Study:
    """Create or load an Optuna study, tolerating SQLite schema init races.

    With array jobs, multiple processes can try to create Optuna's tables at once.
    SQLite may then raise "table studies already exists" during startup even with
    load_if_exists=True. A short retry resolves this once the other process commits.
    """

    for attempt in range(max_retries):
        try:
            return optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=sampler,
                storage=storage,
                load_if_exists=True,
            )
        except Exception as exc:
            msg = str(exc)
            # Broaden the net: Catch ANY "already exists" or "locked" error during init
            is_sqlite_race = (
                "sqlite3.OperationalError" in msg and 
                ("already exists" in msg or "database is locked" in msg)
            )
            if (not is_sqlite_race) or attempt == max_retries - 1:
                raise

            # Brief jittered backoff to let the competing process finish schema setup.
            sleep_s = 0.2 * (attempt + 1) + random.uniform(0.0, 0.2)
            print(
                "[WARN] SQLite schema init race while creating study; "
                f"retrying in {sleep_s:.2f}s ({attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unreachable: failed to create/load study after retries")


def suggest_run_config(trial: optuna.Trial) -> dict:
    choices = optuna_categorical_choices()
    return {
        "hidden_dim":          trial.suggest_categorical("hidden_dim",          choices["hidden_dim"]),
        "n_layers":            trial.suggest_categorical("n_layers",            choices["n_layers"]),
        "lr":                  trial.suggest_categorical("lr",                  choices["lr"]),
        "alpha_lr":            trial.suggest_categorical("alpha_lr",            choices["alpha_lr"]),
        "alpha":               trial.suggest_categorical("alpha",               choices["alpha"]),
        "batch_size":          trial.suggest_categorical("batch_size",          choices["batch_size"]),
        "update_frequency":    trial.suggest_categorical("update_frequency",    choices["update_frequency"]),
        "local_action_hops":   trial.suggest_categorical("local_action_hops",   choices["local_action_hops"]),
        "action_scale":        trial.suggest_categorical("action_scale",        choices["action_scale"]),
        "target_entropy":      trial.suggest_categorical("target_entropy",      choices["target_entropy"]),
        "tau":                 trial.suggest_categorical("tau",                 choices["tau"]),
        "mlp_head":            trial.suggest_categorical("mlp_head",            choices["mlp_head"]),
        "mismatch":            trial.suggest_categorical("mismatch",            choices["mismatch"]),
        "buffer_capacity":     trial.suggest_categorical("buffer_capacity",     choices["buffer_capacity"]),
        "n_shots":             trial.suggest_categorical("n_shots",             choices["n_shots"]),
        "burn_in_steps":       trial.suggest_categorical("burn_in_steps",       choices["burn_in_steps"]),
        "train_episodes":      trial.suggest_categorical("train_episodes",      choices["train_episodes"]),
        "start_from_oracle":   trial.suggest_categorical("start_from_oracle",   choices["start_from_oracle"]),
        "use_endpoint_firing": trial.suggest_categorical("use_endpoint_firing", choices["use_endpoint_firing"]),
        "p":                   trial.suggest_categorical("p",                   choices["p"]),
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
            "training_metrics_filename": f"{args.study_name}_trial_{trial.number:04d}_training_metrics.png",
            "distance": args.distance,
            "n_rounds": args.n_rounds,
            "p": run_cfg["p"],
            "p_gate_zz": args.p_gate_zz,
            "mismatch": run_cfg["mismatch"],
            "n_shots": run_cfg["n_shots"],
            "n_test_shots": args.n_test_shots,
            "burn_in_steps": run_cfg["burn_in_steps"],
            "bypass_threshold": args.bypass_threshold,
            "action_scale": run_cfg["action_scale"],
            "update_period": args.update_period,
            "prior_shots": args.prior_shots,
            "local_action_only": local_action_only,
            "local_action_hops": run_cfg["local_action_hops"],
            "n_layers": run_cfg["n_layers"],
            "hidden_dim": run_cfg["hidden_dim"],
            "lr": run_cfg["lr"],
            "alpha_lr": run_cfg["alpha_lr"],
            "gamma": args.gamma,
            "tau": run_cfg["tau"],
            "alpha": run_cfg["alpha"],
            "batch_size": run_cfg["batch_size"],
            "buffer_capacity": run_cfg["buffer_capacity"],
            "update_frequency": run_cfg["update_frequency"],
            "target_entropy": run_cfg["target_entropy"],
            "mlp_head": run_cfg["mlp_head"],
            "train_episodes": run_cfg["train_episodes"],
            "start_from_oracle": run_cfg["start_from_oracle"],
            "use_endpoint_firing": run_cfg["use_endpoint_firing"],
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
            use_endpoint_firing=config["use_endpoint_firing"],
            start_from_oracle=config["start_from_oracle"],
            update_with="DGR",
            train_mode=True,
        )

        sample_obs, _ = env.reset(seed=args.seed)
        node_dim = sample_obs["node_features"].shape[1]

        # Sentinel: alpha_lr == 0.0 means "reuse actor lr" (SACAgent treats None this way).
        alpha_lr_value = None if config["alpha_lr"] == 0.0 else config["alpha_lr"]
        agent = SACAgent(
            node_dim=node_dim,
            hidden_dim=config["hidden_dim"],
            static_edge_index=env.line_edge_index,
            lr=config["lr"],
            alpha_lr=alpha_lr_value,
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            n_layers=config["n_layers"],
            target_entropy=config["target_entropy"],
            mlp_head=config["mlp_head"],
        )
        buffer = GraphReplayBuffer(capacity=config["buffer_capacity"])

        train(env, agent, buffer, config)

        best_model_path = str(model_path).replace(".pth", "_best.pth")
        if Path(best_model_path).exists():
            agent.load_models(best_model_path)
            scored_model_path = best_model_path
        else:
            scored_model_path = str(model_path)

        # Optuna objective: LER on a fresh env with no drift (mismatch=1.0),
        # oracle-initialised weights (alignment reweighting disabled), and
        # low physical error rate (p=0.001 by default). Higher = better, so
        # the returned value is the NEGATIVE neural LER.
        eval_result = evaluate_low_p_oracle(
            agent=agent,
            config=config,
            p_eval=args.target_eval_p,
            mismatch_eval=args.target_eval_mismatch,
            n_eval_shots=args.target_eval_shots,
            shots_per_episode=args.target_eval_shots_per_episode,
            seed=args.target_eval_seed,
        )

        # Also run the original isolated-validation metric on the training env
        # for backwards comparability with prior runs; saved as user_attr only.
        legacy_isolated_improvement = float(validate(env, agent, config))

        score = -float(eval_result["neural_ler"])

        trial.set_user_attr("runtime_seconds", time.time() - start_s)
        trial.set_user_attr("model_path", scored_model_path)
        trial.set_user_attr("preset_id", args.trial_preset_id)
        trial.set_user_attr("neural_ler", float(eval_result["neural_ler"]))
        trial.set_user_attr("corr_ler", float(eval_result["corr_ler"]))
        trial.set_user_attr("static_ler", float(eval_result["static_ler"]))
        trial.set_user_attr("neural_errors", int(eval_result["neural_errors"]))
        trial.set_user_attr("corr_errors", int(eval_result["corr_errors"]))
        trial.set_user_attr("static_errors", int(eval_result["static_errors"]))
        trial.set_user_attr("eval_total_shots", int(eval_result["total_shots"]))
        trial.set_user_attr("eval_p", float(eval_result["p_eval"]))
        trial.set_user_attr("eval_mismatch", float(eval_result["mismatch_eval"]))
        trial.set_user_attr("legacy_isolated_improvement", legacy_isolated_improvement)

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

    if args.startup_jitter_max_sec > 0:
        # Get the Slurm array ID (default to 0 if running locally)
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        
        # Base delay of 2s per job ID, plus a tiny random jitter
        delay_s = (array_id * 2.0) + random.uniform(0.0, args.startup_jitter_max_sec)
        
        print(
            f"Applying staggered startup delay: sleeping {delay_s:.2f}s "
            f"(array_id={array_id})"
        )
        time.sleep(delay_s)

    study = create_study_with_sqlite_race_retry(
        study_name=args.study_name,
        sampler=build_sampler(args),
        storage=storage,
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
            "hidden_dim":          preset["hidden_dim"],
            "n_layers":            preset["n_layers"],
            "lr":                  preset["lr"],
            "alpha_lr":            preset["alpha_lr"],
            "alpha":               preset["alpha"],
            "batch_size":          preset["batch_size"],
            "update_frequency":    preset["update_frequency"],
            "local_action_hops":   preset["local_action_hops"],
            "action_scale":        preset["action_scale"],
            "target_entropy":      preset["target_entropy"],
            "tau":                 preset["tau"],
            "mlp_head":            preset["mlp_head"],
            "mismatch":            preset["mismatch"],
            "buffer_capacity":     preset["buffer_capacity"],
            "n_shots":             preset["n_shots"],
            "burn_in_steps":       preset["burn_in_steps"],
            "train_episodes":      preset["train_episodes"],
            "start_from_oracle":   preset["start_from_oracle"],
            "use_endpoint_firing": preset["use_endpoint_firing"],
            "p":                   preset["p"],
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
