from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE BASELINE
#
# Matches the argparse defaults of scripts/main.py — the configuration that
# every long training run uses unless an axis is explicitly overridden in
# the .job script. Anchor model: model 64 (best fully-trained run as of May
# 2026, 6.913% best val) which used these defaults plus bs=128 / hd=256 /
# action_scale=5.0 carried over from the model 55-65 era.
#
# FIXED (not varied here):
#   gamma=0.0, n_shots=65_000, burn_in=15_000, d=5,
#   p=0.004, use_pearson_correlation=True, train_episodes=500,
#   target_entropy=-1.0, tau=0.005, alpha=0.01,
#   update_frequency=100   ← the actual value in every long run
#
# VARIED ENV AXIS:
#   mismatch — baseline 30.0 (drift factor range, 30x). Group H sweeps it
#              downward to test whether the policy is bottlenecked by env
#              hardness rather than training hyperparameters.
#
# KEY LESSONS FROM PRIOR SWEEPS
# ──────────────────────────────
#  • lr=1e-5  → complete failure (model 62)
#  • lr=5e-5  → long dead zone (~350/500 eps near-zero reward) before a
#               fragile critic-spike breakthrough; not reliable
#  • lr=1e-4  → consistent, no dead zone; safe choice (model 64: 6.913%)
#  • lr=2e-4  → UNTESTED; expected to escape dead-zone faster; gradient
#               clipping (max_norm=1) already limits instability risk
#  • bs=64    → dropped (consistently worse than bs=128)
#  • bs=256   → model 65 (lr=5e-5) hit 7.545% but only after the dead zone;
#               lr=1e-4 × bs=256 × 500ep is still the key missing test
#  • target_entropy ≠ -1.0 → -2.0 catastrophic (model 58); -0.5 noisy
#  • tau variations → not useful at gamma=0.0 (single critic, no target net)
#  • update_frequency: 100 is the long-standing default and is the value
#    used by every model whose results we trust. Raising it to 500 or 1000
#    was tried and gave clearly worse results — those values are DELIBERATELY
#    excluded from this search space and from the Optuna domain below.
# ─────────────────────────────────────────────────────────────────────────────

def base_trial_config() -> dict:
    """
    Reference configuration (preset 0 — argparse defaults equivalent to the
    model 64 run).

    All other presets override exactly one or two keys so causal attribution
    is unambiguous. A small number of three-key overrides are reserved for
    the "max-aggressive combo" presets and are clearly labelled.
    """
    return {
        # Architecture
        "hidden_dim":        256,
        "n_layers":          1,
        "mlp_head":          "standard",   # head variant: standard|narrow|deep|wide
        # Policy search
        "lr":                1e-4,
        "batch_size":        128,
        "update_frequency":  100,          # SAC gradient step every N env steps — FIXED
        "target_entropy":    -1.0,         # fixed — auto-tunes alpha
        # Agent dynamics (fixed)
        "alpha":             0.01,         # initial entropy coefficient
        "tau":               0.005,        # fixed — tau variations not useful at gamma=0
        # Action
        "action_scale":      5.0,
        "local_action_hops": 1,
        # Environment (only entries that the preset table actually sweeps)
        "mismatch":          30.0,         # drift factor range (1.0 = no drift)
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRESET TABLE  (15 configs, IDs 0–14)
#
# update_frequency is held at the baseline value 100 in every preset:
# 500 and 1000 were already tried and gave bad results, so we don't burn
# compute on them again. The 15 slots therefore probe lr, batch size,
# architecture, action scale, MLP head shape, and their interactions.
#
# One parameter changes at a time unless explicitly labelled as an
# interaction test (group I).
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
#  D   arch      Architectural axis is "depth" — n_layers and
#                local_action_hops are coupled and must always match (the
#                action mask is meaningful only over the GCN's receptive
#                field, so depth = n_layers = local_action_hops by
#                construction). depth=2 is the only depth variant probed
#                here; depth=3 was considered and dropped (cost vs. expected
#                gain). hd=128 also checks whether the 256-dim capacity is
#                actually needed.
#
#  E   scale     action_scale=3.0 isolates the 3→5 change made in model 45.
#
#  F   MLP head  Three head shape alternatives (narrow / deep / wide)
#                to check if the current hourglass [H→2H→H→1] is a bottleneck.
#
#  H   env       mismatch=10.0 reduces the drift factor range from 30x to
#                10x — same noise model, less time-varying severity. Tests
#                whether the headroom in LER vs. baseline MWPM is bounded
#                by training hyperparameters or by the underlying drift
#                difficulty itself; if mismatch=10 lifts performance
#                disproportionately, the bottleneck is the env, not the agent.
#
#  I   combos    Interactions among the "best candidate" axes (lr, bs, depth)
#                plus one env × training cross (mismatch × lr) that asks
#                whether training tweaks compound with an easier env or
#                saturate. update_frequency is intentionally absent from
#                every combo, and any combo that touches depth always sets
#                both n_layers and local_action_hops to keep them coupled.
# ─────────────────────────────────────────────────────────────────────────────

_VARIANTS: list[tuple[str, str, dict]] = [
    # id  group  description                                            override
    # ── Baseline ─────────────────────────────────────────────────────────────
    ("0",  "Baseline — argparse defaults (lr=1e-4, bs=128, uf=100)",   {}),
    # ── Group A: Learning Rate ────────────────────────────────────────────────
    ("A",  "lr=2e-4  (avoid dead-zone plateau; untested)",             {"lr": 2e-4}),
    # ── Group H: Env hardness (drift factor range) ───────────────────────────
    ("H",  "mismatch=10.0  (3x less drift; easier env, isolate hardness)",
                                                                       {"mismatch": 10.0}),
    # ── Group B: Batch Size ───────────────────────────────────────────────────
    ("B",  "bs=256  (key missing: lr=1e-4 × bs=256 × 500ep)",          {"batch_size": 256}),
    ("B",  "lr=2e-4 × bs=256  (LR–BS interaction)",                    {"lr": 2e-4, "batch_size": 256}),
    # ── Group D: Architecture (depth = n_layers = local_action_hops) ─────────
    ("D",  "depth=2  (n_layers=2 × hops=2; matched receptive field)",
           {"n_layers": 2, "local_action_hops": 2}),
    ("D",  "hidden_dim=128  (capacity-reduction check)",               {"hidden_dim": 128}),
    # ── Group I: Env × Training interaction ──────────────────────────────────
    ("I",  "mismatch=10 × lr=2e-4  (env × LR; do tweaks compound with easier env?)",
           {"mismatch": 10.0, "lr": 2e-4}),
    # ── Group E: Action Scale ─────────────────────────────────────────────────
    ("E",  "action_scale=3.0  (isolate the 3→5 change in ep45)",       {"action_scale": 3.0}),
    # ── Group F: MLP Head Alternatives ───────────────────────────────────────
    ("F",  "head: narrow  [H→H→1]  (shallower, fewer params)",         {"mlp_head": "narrow"}),
    ("F",  "head: deep    [H→H→H/2→H/4→1]  (pyramidal)",               {"mlp_head": "deep"}),
    ("F",  "head: wide    [H→4H→1]  (single fat expansion layer)",     {"mlp_head": "wide"}),
    # ── Group I: Interactions (depth axis is always coupled) ─────────────────
    ("I",  "lr=2e-4 × depth=2  (fast lr + deeper GCN)",
           {"lr": 2e-4, "n_layers": 2, "local_action_hops": 2}),
    ("I",  "bs=256 × depth=2  (large batch + deeper GCN)",
           {"batch_size": 256, "n_layers": 2, "local_action_hops": 2}),
    ("I",  "lr=2e-4 × bs=256 × depth=2  (max-aggressive combo)",
           {"lr": 2e-4, "batch_size": 256, "n_layers": 2, "local_action_hops": 2}),
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
    """
    Categorical search space derived from the manual grid (for Optuna).

    Constraint: n_layers and local_action_hops must always be set to the
    same value (the depth axis). Their tuples carry the same options here
    so the Optuna sampler can be wired to suggest a single 'depth' index
    and apply it to both keys.
    """
    hidden_dim:        tuple[int,   ...] = (128, 256)
    n_layers:          tuple[int,   ...] = (1, 2)         # coupled with local_action_hops
    lr:                tuple[float, ...] = (1e-4, 2e-4)
    alpha:             tuple[float, ...] = (0.01,)
    batch_size:        tuple[int,   ...] = (128, 256)
    update_frequency:  tuple[int,   ...] = (100,)         # fixed — 500/1000 confirmed worse
    local_action_hops: tuple[int,   ...] = (1, 2)         # coupled with n_layers
    action_scale:      tuple[float, ...] = (3.0, 5.0)
    target_entropy:    tuple[float, ...] = (-1.0,)
    tau:               tuple[float, ...] = (0.005,)
    mlp_head:          tuple[str,   ...] = ("standard", "narrow", "deep", "wide")
    mismatch:          tuple[float, ...] = (10.0, 30.0)   # env drift factor range


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
        "mismatch":          list(space.mismatch),
    }
