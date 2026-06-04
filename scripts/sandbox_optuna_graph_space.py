from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# ROUND 3 SEARCH SPACE  (study: qec_graph_optuna_run_d5_r5_v3)
#
# Goal of this round: pin down the best (lr × edge-feature) recipe for the
# ORACLE-INITIALISED setup, run long enough (700 eps) to let the slow lr=5e-5
# learner finish its climb, and — for the first time — compare the two
# correlation-graph edge-feature encodings (Pearson vs log-joint-prob).
#
# ── EVIDENCE THIS ROUND IS BUILT ON ─────────────────────────────────────────
# (validation = best "Relative Improvement vs pure CMA"; M=10 is the easy env,
#  M=30 the hard benchmark)
#
#  Round-1 Optuna (qec_graph_optuna_run_d5, 500 eps, bs=128, buffer=100k):
#    • M=30: lr=1e-4 best (T00 6.74%); lr=2e-4 worse (5.82%); bs=256 worse at
#      1e-4 (T03 5.92%); action_scale=3≈5; mlp_head standard ≥ narrow/deep/wide;
#      hidden_dim=128 worse; n_layers=2 / hops=2 COLLAPSE (~0).
#    • M=10: ~12% (T02 12.14% at lr=1e-4) — easy env gives ~2× the signal.
#
#  60-70 standalone long runs (M=30, 500 eps, Pearson features):
#    • run65  lr=5e-5 bs=256 te=-1 buffer=100k → 7.55% (BEST M=30, still
#      climbing at ep 500 → motivates 700 eps).
#    • run64  lr=1e-4 bs=128 → 6.91%.   run63 lr=5e-5 bs=128 → 5.50%.
#    • run62  lr=1e-5 → total failure.   te=-0.5 (run60/61) worse than -1.
#    • ORACLE: run69 (oracle, buffer=1M, no s-feat) → 6.78% ≈ run64, BUT
#      run67 (oracle, buffer=100k) → 5.35%. ⇒ with oracle, buffer=1M >> 100k.
#    • endpoint_firing=True ALWAYS hurt (run66/68/70 < their no-endpoint twins).
#
#  Round-2 Optuna (qec_graph_optuna_run_d5_vv2, 500 eps, bs=256, buffer=1M):
#    • Best M=30: preset 4 (lr=5e-5, alpha_lr=0) → 6.83%.  Best M=10: preset
#      18 (lr=1e-4) → 11.93%, preset 9 → 11.66%, preset 11 (oracle) → 11.42%.
#    • alpha_lr ≠ 0 (decoupled slow-entropy) → BROKEN: every M=10 alpha_lr>0
#      preset (5,6,7,12,15,17,19) failed to even cross the reward gate, and at
#      M=30 it was slower + lower (preset 8 6.21% < preset 4 6.83%). ⇒ DROPPED.
#    • alpha=0.1 warm-init → worse (preset 10 5.47% < 4; preset 2 failed).
#    • use_endpoint_firing=True → worst M=30 (preset 14 3.78%). ⇒ DROPPED.
#
# ── DECISIONS FOR ROUND 3 (all fixed unless listed as a varied axis) ────────
#  start_from_oracle = True   (user requirement; also makes reward cross the
#                              ep-gate early → avoids the round-2 "never
#                              validated" failures, since oracle init starts
#                              the policy near a high-reward regime)
#  mismatch          = 30.0        (hard benchmark; fixed for the whole round)
#  p                 = 4e-3        (0.004 — the standard rate used in rounds 1-2)
#  n_shots           = 50_000      (fixed; the oracle-precedent episode length)
#  batch_size        = 256         (unlocks lr=5e-5, the run65 champion recipe)
#  hidden_dim=256, mlp_head="standard", n_layers=1, local_action_hops=1
#  alpha=0.01 (usual init), alpha_lr=0.0 (reuse actor lr — decoupling is broken)
#  use_endpoint_firing = False     (always hurt)
#  burn_in_steps = 0               (oracle init ⇒ no random burn-in needed)
#  target_entropy=-1.0, tau=0.005, action_scale=5.0, update_frequency=100
#  train_episodes = 700            (run65 still climbing at 500)
#
#  VARIED AXES (2 × 2 × 2 = 8 presets):
#    lr              ∈ {5e-5, 1e-4}
#    edge_feature    ∈ {"pearson", "joint_prob"}   ← NEW: correlation-graph edge
#                      encoding. "pearson"   = use_pearson_correlation=True,
#                      use_log_joint_prob=False (everything so far); "joint_prob"
#                      = use_pearson_correlation=False, use_log_joint_prob=True
#                      (UNTESTED — first look at the log-joint-prob encoding).
#                      NB: a fair pearson-vs-joint_prob comparison under
#                      start_from_oracle requires the env to seed the co-occurrence
#                      tracer at the drifted-oracle joint probs (fixed in
#                      drifted_matching_env.reset()); otherwise the joint_prob arm
#                      silently sees stale, non-drifted base-DEM values.
#    buffer_capacity ∈ {500_000, 1_000_000}   ← oracle needed >100k (run69 6.78%
#                      vs run67 5.35%); 1M was the prior default. Probe whether
#                      500k already captures the gain (cheaper / less stale).
# ─────────────────────────────────────────────────────────────────────────────

def base_trial_config() -> dict:
    """
    Reference configuration for round 3 (preset 0): the oracle-initialised,
    buffer=1M, bs=256, lr=5e-5 recipe at the hard env (M=30), p=0.004,
    Pearson edge features, n_shots=50k, trained for 700 episodes.
    """
    return {
        # Architecture
        "hidden_dim":          256,
        "n_layers":            1,
        "mlp_head":            "standard",
        # Policy / critic optimization
        "lr":                  5e-5,
        "alpha_lr":            0.0,         # 0.0 ⇒ reuse actor lr (decoupling broken)
        "batch_size":          256,
        "update_frequency":    100,
        "target_entropy":      -1.0,
        # SAC dynamics
        "alpha":               0.01,        # usual initial entropy temperature
        "tau":                 0.005,
        # Action
        "action_scale":        5.0,
        "local_action_hops":   1,
        # Environment  (mismatch / p / n_shots fixed for the whole round)
        "mismatch":            30.0,
        "p":                   0.004,
        "edge_feature":        "pearson",   # "pearson" | "joint_prob"
        # Replay
        "buffer_capacity":     1_000_000,
        # Schedule
        "n_shots":             50_000,
        "burn_in_steps":       0,           # oracle init ⇒ no burn-in
        "train_episodes":      700,
        # Env feature toggles
        "start_from_oracle":   True,
        "use_endpoint_firing": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRESET TABLE  (8 configs, IDs 0–7)
#
# Each preset lists ONLY the keys that differ from base. Full
# lr × edge_feature × buffer_capacity factorial at the fixed hard-env / low-p /
# 50k-shot operating point. Base = lr=5e-5, Pearson, buffer=1M.
# ─────────────────────────────────────────────────────────────────────────────

_VARIANTS: list[tuple[str, str, dict]] = [
    # id  group  description                                                  override
    ("A", "lr=5e-5 x Pearson    x buf=1M   (base; M=30, p=0.004, ns=50k, oracle, bs256, 700 eps)",
        {}),
    ("A", "lr=5e-5 x Pearson    x buf=500k",
        {"buffer_capacity": 500_000}),
    ("A", "lr=5e-5 x joint_prob x buf=1M",
        {"edge_feature": "joint_prob"}),
    ("A", "lr=5e-5 x joint_prob x buf=500k",
        {"edge_feature": "joint_prob", "buffer_capacity": 500_000}),
    ("A", "lr=1e-4 x Pearson    x buf=1M",
        {"lr": 1e-4}),
    ("A", "lr=1e-4 x Pearson    x buf=500k",
        {"lr": 1e-4, "buffer_capacity": 500_000}),
    ("A", "lr=1e-4 x joint_prob x buf=1M",
        {"lr": 1e-4, "edge_feature": "joint_prob"}),
    ("A", "lr=1e-4 x joint_prob x buf=500k",
        {"lr": 1e-4, "edge_feature": "joint_prob", "buffer_capacity": 500_000}),
]


def trial_presets() -> list[dict]:
    """Return all preset configurations as a list of dicts."""
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
    print(f"{'ID':>3}  {'Grp':>3}  {'Description':<70}  Changed vs baseline")
    print("-" * 130)
    for p in presets:
        changed = {k: p[k] for k in base if p[k] != base[k]}
        print(
            f"{p['preset_id']:>3}  {p['group']:>3}  {p['description']:<70}  "
            f"{changed if changed else '—'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA INTERFACE
#
# Every key the objective reads from suggest_run_config must appear here.
#
# Constraint: n_layers and local_action_hops must always equal each other
# (the depth axis), kept at 1 (depth=2 collapses).
#
# edge_feature is a single categorical that the objective translates into the
# two mutually-exclusive env flags (use_pearson_correlation / use_log_joint_prob)
# — encoding it as one choice prevents the random sampler from ever picking the
# illegal "both True" combination.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptunaSearchSpace:
    # Architecture
    hidden_dim:          tuple[int,   ...] = (256,)
    n_layers:            tuple[int,   ...] = (1,)
    local_action_hops:   tuple[int,   ...] = (1,)
    mlp_head:            tuple[str,   ...] = ("standard",)
    # Optimization
    lr:                  tuple[float, ...] = (5e-5, 1e-4)
    alpha_lr:            tuple[float, ...] = (0.0,)              # 0.0 ⇒ reuse actor lr
    alpha:               tuple[float, ...] = (0.01,)
    batch_size:          tuple[int,   ...] = (256,)
    update_frequency:    tuple[int,   ...] = (100,)
    target_entropy:      tuple[float, ...] = (-1.0,)
    tau:                 tuple[float, ...] = (0.005,)
    # Action / env
    action_scale:        tuple[float, ...] = (5.0,)
    mismatch:            tuple[float, ...] = (30.0,)
    p:                   tuple[float, ...] = (0.004,)
    edge_feature:        tuple[str,   ...] = ("pearson", "joint_prob")
    # Replay + schedule
    buffer_capacity:     tuple[int,   ...] = (500_000, 1_000_000)
    n_shots:             tuple[int,   ...] = (50_000,)
    burn_in_steps:       tuple[int,   ...] = (0,)
    train_episodes:      tuple[int,   ...] = (700,)
    # Env feature toggles
    start_from_oracle:   tuple[bool,  ...] = (True,)
    use_endpoint_firing: tuple[bool,  ...] = (False,)


def optuna_categorical_choices() -> dict[str, list]:
    """Valid categorical domains for Optuna suggest_categorical calls."""
    space = OptunaSearchSpace()
    return {
        "hidden_dim":          list(space.hidden_dim),
        "n_layers":            list(space.n_layers),
        "local_action_hops":   list(space.local_action_hops),
        "mlp_head":            list(space.mlp_head),
        "lr":                  list(space.lr),
        "alpha_lr":            list(space.alpha_lr),
        "alpha":               list(space.alpha),
        "batch_size":          list(space.batch_size),
        "update_frequency":    list(space.update_frequency),
        "target_entropy":      list(space.target_entropy),
        "tau":                 list(space.tau),
        "action_scale":        list(space.action_scale),
        "mismatch":            list(space.mismatch),
        "p":                   list(space.p),
        "edge_feature":        list(space.edge_feature),
        "buffer_capacity":     list(space.buffer_capacity),
        "n_shots":             list(space.n_shots),
        "burn_in_steps":       list(space.burn_in_steps),
        "train_episodes":      list(space.train_episodes),
        "start_from_oracle":   list(space.start_from_oracle),
        "use_endpoint_firing": list(space.use_endpoint_firing),
    }
