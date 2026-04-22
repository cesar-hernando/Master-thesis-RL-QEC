from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptunaSearchSpace:
    """Small, compute-aware categorical search space for 14h training runs."""

    hidden_dim: tuple[int, ...] = (128, 256)
    n_layers: tuple[int, ...] = (1, 2)
    lr: tuple[float, ...] = (5e-5, 1e-4, 2e-4)
    alpha: tuple[float, ...] = (0.005, 0.01, 0.02)
    batch_size: tuple[int, ...] = (64, 128)
    update_frequency: tuple[int, ...] = (50, 100, 200)
    local_action_hops: tuple[int, ...] = (1, 2)
    action_scale: tuple[float, ...] = (2.5, 3.0)


def base_trial_config() -> dict:
    """A stable default used by all presets unless explicitly overridden."""
    return {
        "hidden_dim": 256,
        "n_layers": 1,
        "lr": 1e-4,
        "alpha": 0.01,
        "batch_size": 64,
        "update_frequency": 100,
        "local_action_hops": 1,
        "action_scale": 3.0,
    }


def trial_presets() -> list[dict]:
    """
    Hand-picked presets (small but meaningful).
    Each list element is one full training run (one Slurm array task).
    """
    base = base_trial_config()
    variants = [
        {},
        {"hidden_dim": 128},
        {"n_layers": 2},
        {"hidden_dim": 128, "n_layers": 2},
        {"batch_size": 128},
        {"lr": 5e-5},
        {"lr": 2e-4},
        {"alpha": 0.005},
        {"alpha": 0.02},
        {"update_frequency": 50},
        {"update_frequency": 200},
        {"local_action_hops": 2, "n_layers": 2},
        {"action_scale": 2.5},
        {"action_scale": 2.5, "lr": 5e-5},
    ]

    presets: list[dict] = []
    for idx, override in enumerate(variants):
        cfg = dict(base)
        cfg.update(override)
        cfg["preset_id"] = idx
        presets.append(cfg)
    return presets


def preset_by_id(preset_id: int) -> dict:
    presets = trial_presets()
    if preset_id < 0 or preset_id >= len(presets):
        raise ValueError(
            f"Invalid preset_id={preset_id}. Must be in [0, {len(presets) - 1}]."
        )
    return presets[preset_id]


def optuna_categorical_choices() -> dict[str, list]:
    """Valid categorical domains for Optuna suggest_categorical calls."""
    space = OptunaSearchSpace()
    return {
        "hidden_dim": list(space.hidden_dim),
        "n_layers": list(space.n_layers),
        "lr": list(space.lr),
        "alpha": list(space.alpha),
        "batch_size": list(space.batch_size),
        "update_frequency": list(space.update_frequency),
        "local_action_hops": list(space.local_action_hops),
        "action_scale": list(space.action_scale),
    }
