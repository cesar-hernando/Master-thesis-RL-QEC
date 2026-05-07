from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE BASELINE
#
# Matches model 64 — best fully-trained run as of May 2026 (6.913% best val).
#
# FIXED (not varied — settled by models 55-65):
#   gamma=0.0, n_shots=65_000, burn_in=15_000, mismatch=30.0, d=5,
#   p=0.004, use_pearson_correlation=True, train_episodes=500,
#   target_entropy=-1.0, tau=0.005
#
# KEY LESSONS FROM MODELS 55-65
# ──────────────────────────────
#  • lr=1e-5  → complete failure (model 62)
#  • lr=5e-5  → creates a long dead zone (~350/500 eps near-zero reward)
#              before a fragile critic-spike breakthrough; not reliable
#  • lr=1e-4  → consistent, no dead zone; best safe choice (model 64: 6.913%)
#  • lr=2e-4  → UNTESTED; expected to escape dead-zone faster; gradient
#              clipping (max_norm=1) already limits instability risk
#  • bs=64    → dropped (lower than bs=128 in all comparisons)
#  • bs=256   → model 65 (lr=5e-5) hit 7.545% but only after the dead zone;
#              lr=1e-4 + bs=256 + 500ep is still the key missing test
#  • target_entropy ≠ -1.0 → -2.0 catastrophic (model 58); -0.5 noisy; fix at -1.0
#  • tau variations → not worth it when gamma=0.0 (single critic, no target net)
#  • update_frequency=100 → was standard in models 40-43 (best era pre-55);
#    raised to 1000 without re-evaluation → must be re-tested with hd=256
# ─────────────────────────────────────────────────────────────────────────────

def base_trial_config() -> dict:
    """
    Reference configuration (preset 0 — model 64).

    All other presets override exactly one or two keys so causal attribution
    is unambiguous.  Three-key overrides are reserved for the final
    "max-aggressive combo" preset and are clearly labelled.
    """
    return {
        # Architecture
        "hidden_dim":        256,
        "n_layers":          1,
        "mlp_head":          "standard",   # head variant: standard|narrow|deep|wide
        # Policy search
        "lr":                1e-4,
        "batch_size":        128,
        "update_frequency":  1000,         # SAC gradient step every N env steps
        "target_entropy":    -1.0,         # fixed — auto-tunes alpha
        # Agent dynamics (fixed)
        "alpha":             0.01,         # initial entropy coefficient
        "tau":               0.005,        # fixed — tau variations not useful at gamma=0
        # Action
        "action_scale":      5.0,
        "local_action_hops": 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRESET TABLE  (15 configs, IDs 0–14)
#
# One parameter changes at a time unless explicitly labelled as an
# interaction test (groups I1 / I2).
#
# Design rationale per group
# ──────────────────────────
#  A   lr=2e-4   Avoid the dead-zone plateau caused by lr=5e-5; 2e-4 has
#                not been tested and is the natural "try higher" candidate.
#
#  B   bs=256    lr=1e-4 + bs=256 + 500ep is the single most important
#                missing data point: model 57b had it at 300ep (6.166%),
#                model 65 had bs=256 but with lr=5e-5 (7.545% after dead zone).
#
#  C   upd_freq  update_frequency=100 was the standard in models 40-43
#                (best era before model 55); the change to 1000 was never
#                ablated. 500 tests the midpoint.
#
#  D   arch      n_layers=2 + hops=2 has never been tested at hd=256.
#                hd=128 checks whether the 256-dim capacity is actually needed.
#
#  E   scale     action_scale=3.0 isolates the 3→5 change made in model 45.
#
#  F   MLP head  Three head shape alternatives (narrow / deep / wide)
#                to check if the current hourglass [H→2H→H→1] is a bottleneck.
#
#  I1  lr×uf     Higher lr + more frequent updates: faster gradient signal
#                with less staleness.
#
#  I2  lr×uf×bs  Max-aggressive combo: all three "faster training" axes
#                combined. Expensive but informative if the individual
#                improvements hold.
# ─────────────────────────────────────────────────────────────────────────────

_VARIANTS: list[tuple[str, str, dict]] = [
    # id  group  description                                          override
    # ── Baseline ─────────────────────────────────────────────────────────────
    ("A",  "Baseline — model 64 ref  (lr=1e-4, bs=128)",             {}),
    # ── Group A: Learning Rate ────────────────────────────────────────────────
    ("A",  "lr=2e-4  (avoid dead-zone plateau; untested)",           {"lr": 2e-4}),
    # ── Group B: Batch Size ───────────────────────────────────────────────────
    ("B",  "bs=256  (key missing: lr=1e-4 × bs=256 × 500ep)",        {"batch_size": 256}),
    ("B",  "lr=2e-4 × bs=256  (LR–BS interaction)",                  {"lr": 2e-4, "batch_size": 256}),
    # ── Group C: SAC Update Frequency ────────────────────────────────────────
    ("C",  "update_freq=100  (10× grad steps; was best in ep40-43)", {"update_frequency": 100}),
    ("C",  "update_freq=100 × bs=256  (uf×bs interaction)",          {"update_frequency": 100, "batch_size": 256}),
    ("C",  "update_freq=500  (intermediate)",                         {"update_frequency": 500}),
    # ── Group D: Architecture ─────────────────────────────────────────────────
    ("D",  "n_layers=2 + hops=2  (deep GCN, hd=256, untested)",
           {"n_layers": 2, "local_action_hops": 2}),
    ("D",  "hidden_dim=128  (capacity-reduction check)",             {"hidden_dim": 128}),
    # ── Group E: Action Scale ─────────────────────────────────────────────────
    ("E",  "action_scale=3.0  (isolate the 3→5 change in ep45)",     {"action_scale": 3.0}),
    # ── Group F: MLP Head Alternatives ───────────────────────────────────────
    ("F",  "head: narrow  [H→H→1]  (shallower, fewer params)",       {"mlp_head": "narrow"}),
    ("F",  "head: deep    [H→H→H/2→H/4→1]  (pyramidal)",            {"mlp_head": "deep"}),
    ("F",  "head: wide    [H→4H→1]  (single fat expansion layer)",   {"mlp_head": "wide"}),
    # ── Group I1: LR × Update-Frequency Interaction ──────────────────────────
    ("I1", "lr=2e-4 × update_freq=100  (fast lr + frequent updates)",
           {"lr": 2e-4, "update_frequency": 100}),
    # ── Group I2: Max-Aggressive Combo ────────────────────────────────────────
    ("I2", "lr=2e-4 × uf=100 × bs=256  (all-aggressive combo)",
           {"lr": 2e-4, "update_frequency": 100, "batch_size": 256}),
]


def trial_presets() -> list[dict]:
    """Return all 15 preset configurations as a list of dicts."""
    base = base_trial_config()
    presets: list[dict] = []
    for preset_id, (group, description, override) in enumerate(_VARIANTS):
        cfg = dict(base)
        cfg.update(override)
        cfg["preset_id"]   = preset_id
        cfg["group"]       = group
        cfg["description"] = description
        presets.append(cfg)
    return presets


def preset_by_id(preset_id: int) -> dict:
    presets = trial_presets()
    if preset_id < 0 or preset_id >= len(presets):
        raise ValueError(
            f"Invalid preset_id={preset_id}. Must be in [0, {len(presets) - 1}]."
        )
    return presets[preset_id]


def preset_summary() -> None:
    """Print a human-readable table showing each preset and its diff from base."""
    presets = trial_presets()
    base = base_trial_config()
    print(f"{'ID':>3}  {'Grp':>3}  {'Description':<55}  Changed vs baseline")
    print("-" * 105)
    for p in presets:
        changed = {k: p[k] for k in base if p[k] != base[k]}
        print(
            f"{p['preset_id']:>3}  {p['group']:>3}  {p['description']:<55}  "
            f"{changed if changed else '—'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptunaSearchSpace:
    """Categorical search space derived from the manual grid (for Optuna)."""
    hidden_dim:        tuple[int,   ...] = (128, 256)
    n_layers:          tuple[int,   ...] = (1, 2)
    lr:                tuple[float, ...] = (1e-4, 2e-4)
    alpha:             tuple[float, ...] = (0.01,)
    batch_size:        tuple[int,   ...] = (128, 256)
    update_frequency:  tuple[int,   ...] = (100, 500, 1000)
    local_action_hops: tuple[int,   ...] = (1, 2)
    action_scale:      tuple[float, ...] = (3.0, 5.0)
    target_entropy:    tuple[float, ...] = (-1.0,)
    tau:               tuple[float, ...] = (0.005,)
    mlp_head:          tuple[str,   ...] = ("standard", "narrow", "deep", "wide")


def optuna_categorical_choices() -> dict[str, list]:
    """Valid categorical domains for Optuna suggest_categorical calls."""
    space = OptunaSearchSpace()
    return {
        "hidden_dim":        list(space.hidden_dim),
        "n_layers":          list(space.n_layers),
        "lr":                list(space.lr),
        "alpha":             list(space.alpha),
        "batch_size":        list(space.batch_size),
        "update_frequency":  list(space.update_frequency),
        "local_action_hops": list(space.local_action_hops),
        "action_scale":      list(space.action_scale),
        "target_entropy":    list(space.target_entropy),
        "tau":               list(space.tau),
        "mlp_head":          list(space.mlp_head),
    }
