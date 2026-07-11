from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE BASELINE (v2 study: qec_graph_optuna_run_d5_r5_v2)
#
# Anchor model: run 65 (best M=30 long run to date, 7.545% best validation,
# 6.817% final ep-500 validation, 22.5 h wall) — config: lr=5e-5, bs=256,
# hd=256, n_layers=1, action_scale=5, burn_in=15k, n_shots=65k, train_eps=500,
# buffer=100k, alpha_init=0.01, target_entropy=-1.0, mismatch=30, p=0.004.
#
# The base config below mirrors the run 65 recipe so that every preset's diff
# is small and causally interpretable.
#
# KEY LESSONS FROM PRIOR SWEEPS (kept; do NOT re-test here)
# ─────────────────────────────────────────────────────────
#  • lr=1e-5  → complete failure (model 62)
#  • lr=5e-5  → run 65: 7.55% best, still climbing at ep 500 → needs more eps
#  • lr=1e-4  → run 64: 6.91% best (safe but lower ceiling than 5e-5+bs256)
#  • bs=64    → consistently worse → dropped
#  • bs=256   → unlocks lr=5e-5 → kept as default
#  • n_layers=2 / depth=2  → collapsed to ~0 in prior Optuna → dropped
#  • mlp_head deep / wide / narrow ≈ standard in prior Optuna → standard is default
#  • target_entropy ≠ -1.0 → -2.0 catastrophic, -0.5 noisy → kept at -1.0
#  • tau ≠ 0.005 (single critic at gamma=0) → not useful → kept
#  • update_frequency ∈ {500, 1000} → worse than 100 → kept at 100
#  • action_scale=3.0 (model 45 era) → worse than 5.0 → kept
#  • use_endpoint_firing=True (run 66/68/70) → lower critic MSE but no LER win,
#    sometimes catastrophic late collapse → probed sparingly here (2 slots)
#  • start_from_oracle=True (run 67/68/69/70) → wash at M=30, untested at M=10
#    → probed at M=10 (2 slots) + 1 at M=30 with new combo
#  • buffer_capacity=1M helped the no-s-feat oracle setup (run 69 ≈ run 64)
#    → adopted as the v2 DEFAULT (base config). 100k retained only as a
#      control in group C to confirm the 1M default is actually load-bearing.
#
# NEW LEVERS FIRST TIME IN THE SEARCH SPACE
# ─────────────────────────────────────────
#  • alpha_lr (separate LR for the entropy temperature optimizer): default
#    None means "reuse actor lr" (backward compatible). Values 1e-5 / 3e-5
#    keep entropy bonus alive into late training — directly motivated by
#    the alpha-collapse seen by ep ~100 in every prior training plot.
#  • Higher initial alpha (0.05, 0.1) combined with small alpha_lr → "warm
#    + slow-decay" entropy schedule.
#  • Longer training (train_episodes=800, 1000) at lr=5e-5 → run 65 was
#    still climbing at ep 500; the highest-confidence simple move.
#  • Mid p=0.006 → headroom over MWPM is larger at higher physical rate
#    (cf. ler_vs_p_mwpm_corr_neural_static figure: 4.08e-3 → 2.12e-3 = 48%
#    gap at p=0.006 vs 49% at p=0.004 → similar gap, easier signal).
#  • mismatch=10 is the strongest single lever from the previous Optuna
#    (0.121 vs 0.05 at M=30) — most new presets sit at M=10.
# ─────────────────────────────────────────────────────────────────────────────

def base_trial_config() -> dict:
    """
    Reference configuration (preset 0 — replicates run 65 with one change:
    mismatch=10 instead of 30, since M=10 was the strongest lever found in
    the previous Optuna study).
    """
    return {
        # Architecture
        "hidden_dim":          256,
        "n_layers":            1,
        "mlp_head":            "standard",
        # Policy / critic optimization
        "lr":                  5e-5,
        "alpha_lr":            0.0,         # 0.0 ⇒ reuse actor lr (Optuna-friendly sentinel)
        "batch_size":          256,
        "update_frequency":    100,
        "target_entropy":      -1.0,
        # SAC dynamics
        "alpha":               0.01,
        "tau":                 0.005,
        # Action
        "action_scale":        5.0,
        "local_action_hops":   1,
        # Environment
        "mismatch":            10.0,        # easy env (best lever from prior study)
        "p":                   0.004,
        # Replay
        "buffer_capacity":     1_000_000,   # new v2 default (run 69 showed 1M >= 100k, more stable)
        # Schedule
        "n_shots":             65_000,
        "burn_in_steps":       15_000,
        "train_episodes":      500,
        # Env feature toggles
        "start_from_oracle":   False,
        "use_endpoint_firing": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRESET TABLE  (20 configs, IDs 0–19)
#
# Each preset lists ONLY the keys that differ from base. One axis per slot
# unless tagged as a combo. Combos are kept to the few high-expected-value
# crosses of axes that look promising independently.
#
# Design rationale per group
# ──────────────────────────
#  A   anchor / minimal-Δ slots that move only mismatch, p, or training
#      length — give us a clean control bar for everything else.
#  B   alpha_lr slots (warm + slow-decay) at both M=10 and M=30.
#  C   buffer=100k controls (since 1M is now the default, these isolate
#      whether the 1M default actually helps vs the historical 100k value).
#  D   start_from_oracle revisits at M=10 (untested) + a combo at M=30.
#  E   use_endpoint_firing sparingly (2 slots) — only paired with other
#      changes since previous runs (66/68/70) showed it doesn't win alone.
#  F   "All-in" combinations: the cumulative best of axes B + schedule.
#  G   one architectural variant: a "wide" MLP head paired with the higher
#      lr — head shape was a near-wash last time but combos may differ.
#  H   very-slow probe: lr=3e-5 + alpha_lr=1e-5 + eps=1000 + n_shots=80k.
# ─────────────────────────────────────────────────────────────────────────────

_VARIANTS: list[tuple[str, str, dict]] = [
    # id  group  description                                                  override
    # NOTE: every preset trains for the same 500 episodes (base default).
    # The only remaining "training compute" axis is n_shots per episode.
    # ── A: minimal-Delta anchors (buffer=1M from base) ──────────────────────
    ("A", "Anchor: run 65 recipe at M=10 (lr=5e-5 x bs=256 x buffer=1M)",
        {}),
    ("A", "M=30 anchor (literal run 65 reproduction: buffer=100k)",
        {"mismatch": 30.0, "buffer_capacity": 100_000}),
    ("A", "M=10 x alpha=0.1 (warm init only, default alpha decay)",
        {"alpha": 0.1}),
    ("A", "M=10 x n_shots=80k (more shots/episode at default lr)",
        {"n_shots": 80_000}),
    ("A", "M=30 (buffer=1M, 500 eps) — hard-env comparison to preset 0",
        {"mismatch": 30.0}),
    # ── B: decoupled alpha_lr (slower entropy decay) ─────────────────────────
    ("B", "M=10 x alpha_lr=3e-5 (slow entropy decay)",
        {"alpha_lr": 3e-5}),
    ("B", "M=10 x alpha_lr=1e-5 (very slow entropy decay)",
        {"alpha_lr": 1e-5}),
    ("B", "M=10 x alpha=0.1 x alpha_lr=1e-5 (warm + slow-decay)",
        {"alpha": 0.1, "alpha_lr": 1e-5}),
    ("B", "M=30 x alpha_lr=3e-5 (does slow-alpha help hard env?)",
        {"mismatch": 30.0, "alpha_lr": 3e-5}),
    # ── C: small-buffer control (M=30 small-buffer is preset 1) ─────────────
    ("C", "M=10 x buffer=100k (small-buffer control for easy env)",
        {"buffer_capacity": 100_000}),
    ("A", "M=30 x alpha=0.1 (warm init only at hard env; pairs with preset 2)",
        {"mismatch": 30.0, "alpha": 0.1}),
    # ── D: start_from_oracle revisited (buffer=1M from base) ────────────────
    ("D", "M=10 x start_from_oracle=True x burn_in=0 x n_shots=50k",
        {"start_from_oracle": True, "burn_in_steps": 0, "n_shots": 50_000}),
    ("D", "M=10 x start_from_oracle x alpha_lr=1e-5 x burn_in=0 x n_shots=50k",
        {"start_from_oracle": True, "burn_in_steps": 0, "n_shots": 50_000,
         "alpha_lr": 1e-5}),
    # ── E: endpoint_firing paired with other changes ─────────────────────────
    ("E", "M=10 x endpoint_firing=True x alpha_lr=1e-5 (sharper critic + warm alpha)",
        {"use_endpoint_firing": True, "alpha_lr": 1e-5}),
    ("E", "M=30 x endpoint_firing=True (buffer=1M from base)",
        {"mismatch": 30.0, "use_endpoint_firing": True}),
    # ── F: all-in combos (highest expected value) ────────────────────────────
    ("F", "M=10 ALL-IN: alpha_lr=1e-5 x alpha=0.05 x n_shots=80k",
        {"alpha": 0.05, "alpha_lr": 1e-5, "n_shots": 80_000}),
    ("F", "M=30 ALL-IN: alpha_lr=3e-5 x n_shots=80k",
        {"mismatch": 30.0, "alpha_lr": 3e-5, "n_shots": 80_000}),
    # ── G: physical p shift + MLP head tweak ─────────────────────────────────
    ("G", "M=10 x p=0.006 x alpha_lr=1e-5 (more LER headroom at higher p)",
        {"p": 0.006, "alpha_lr": 1e-5}),
    ("G", "M=10 x wide head x lr=1e-4 (wide hourglass + safer lr)",
        {"mlp_head": "wide", "lr": 1e-4}),
    # ── H: high-volatility safety probe (slow everything) ────────────────────
    ("H", "M=10 x lr=3e-5 x alpha_lr=1e-5 x n_shots=80k (very slow learner)",
        {"lr": 3e-5, "alpha_lr": 1e-5, "n_shots": 80_000}),
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
# Every key the objective reads from suggest_run_config must appear here. New
# categorical axes added in v2: alpha_lr, buffer_capacity, n_shots,
# burn_in_steps, train_episodes, start_from_oracle, use_endpoint_firing, p.
#
# Constraint: n_layers and local_action_hops must always equal each other
# (the depth axis). Kept at 1 in v2 since depth=2 was decisively worse last
# time; both tuples are length-1 here.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptunaSearchSpace:
    # Architecture
    hidden_dim:          tuple[int,   ...] = (256,)
    n_layers:            tuple[int,   ...] = (1,)
    local_action_hops:   tuple[int,   ...] = (1,)
    mlp_head:            tuple[str,   ...] = ("standard", "wide")
    # Optimization
    lr:                  tuple[float, ...] = (3e-5, 5e-5, 1e-4)
    alpha_lr:            tuple[float, ...] = (0.0, 1e-5, 3e-5)     # 0.0 ⇒ reuse actor lr
    alpha:               tuple[float, ...] = (0.01, 0.05, 0.1)
    batch_size:          tuple[int,   ...] = (256,)
    update_frequency:    tuple[int,   ...] = (100,)
    target_entropy:      tuple[float, ...] = (-1.0,)
    tau:                 tuple[float, ...] = (0.005,)
    # Action / env
    action_scale:        tuple[float, ...] = (5.0,)
    mismatch:            tuple[float, ...] = (10.0, 30.0)
    p:                   tuple[float, ...] = (0.004, 0.006)
    # Replay + schedule
    buffer_capacity:     tuple[int,   ...] = (100_000, 1_000_000)
    n_shots:             tuple[int,   ...] = (50_000, 65_000, 80_000)
    burn_in_steps:       tuple[int,   ...] = (0, 15_000)
    train_episodes:      tuple[int,   ...] = (500,)
    # Env feature toggles
    start_from_oracle:   tuple[bool,  ...] = (False, True)
    use_endpoint_firing: tuple[bool,  ...] = (False, True)


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
        "buffer_capacity":     list(space.buffer_capacity),
        "n_shots":             list(space.n_shots),
        "burn_in_steps":       list(space.burn_in_steps),
        "train_episodes":      list(space.train_episodes),
        "start_from_oracle":   list(space.start_from_oracle),
        "use_endpoint_firing": list(space.use_endpoint_firing),
    }
