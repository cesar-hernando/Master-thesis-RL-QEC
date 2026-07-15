"""
Script to compare the step-by-step convergence of DGR and Spitz tracers.
Uses independent test shots (held-out batch) to evaluate LER at each update
period, avoiding the bias of running averages computed on the same shots used
to update the tracers.

Metrics tracked per update period:
  - Test LER  (independent batch, unbiased)
  - Weight MSE vs Oracle
  - Correlation MSE vs Oracle
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from NeuralCM.syndrome_data_generation import SyndromeDataGenerator
from NeuralCM.drifted_matching_env import DriftedMatchingEnv


# ==========================================
# CONFIGURATION
# ==========================================
P_VALUES  = [0.003, 0.005]
MISMATCHES = [10.0, 30.0]

N_SHOTS      = 150_000    # Shots per episode (tracer update data)
N_TEST_SHOTS = 1_0000_000   # Independent held-out batch for LER evaluation
UPDATE_PERIOD = 1_000      # Tracer CMA update frequency (shots between graph updates)
SEED = 2024

DISTANCE    = 5
N_ROUNDS    = 5
MEMORY_TYPE = 'z'


# ==========================================
# EPISODE EXECUTION
# ==========================================
def run_convergence_episode(generator, tracer_method, seed):
    """
    Runs a full episode with zero action and tracks:
      - Test LER at each graph update (from independent held-out batch)
      - Weight MSE vs Oracle at each graph update
      - Correlation MSE vs Oracle at each graph update

    The LER is evaluated on a separate N_TEST_SHOTS batch that was never
    used to update the tracers, avoiding the optimistic bias of running
    averages computed on training shots.

    Parameters
    ----------
    generator     : SyndromeDataGenerator configured for the desired noise regime
    tracer_method : 'DGR' or 'Spitz'
    seed          : integer seed for reproducibility

    Returns
    -------
    test_ler_history  : (n_updates + 1,) array — LER at step 0 and after each update
    oracle_ler        : scalar — oracle decoder LER (constant reference)
    static_ler        : scalar — static decoder LER (constant reference)
    weights_mse       : (n_updates + 1,) array — weight MSE vs oracle
    corr_mse          : (n_updates + 1,) array — correlation MSE vs oracle
    weights_mse_static: (n_updates + 1,) array — weight MSE vs static base graph
    corr_mse_static   : (n_updates + 1,) array — corr MSE vs static base graph
    run_time          : wall-clock seconds
    """
    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=1,
        action_scale=1.0,
        update_period=UPDATE_PERIOD,
        #prior_shots=1 if tracer_method == 'Spitz' else 1000,
        prior_shots=1000,
        n_test_shots=N_TEST_SHOTS,   # ← independent held-out LER batch
        use_pearson_correlation=True,
        use_syndrome_features=(tracer_method == 'Spitz'),
        update_with=tracer_method,
        train_mode=False,
    )

    obs, info = env.reset(seed=seed)

    # Step 0: record initial state before any tracer updates
    static_ler = info["initial_test_ler"]
    oracle_ler = info["oracle_ler"]

    n_updates = N_SHOTS // UPDATE_PERIOD  # number of graph updates in the episode
    n_points  = n_updates + 1            # include step 0

    test_ler_history   = np.zeros(n_points, dtype=np.float32)
    weights_mse        = np.zeros(n_points, dtype=np.float32)
    corr_mse           = np.zeros(n_points, dtype=np.float32)
    weights_mse_static = np.zeros(n_points, dtype=np.float32)
    corr_mse_static    = np.zeros(n_points, dtype=np.float32)

    # Point 0: values from env.reset() info
    test_ler_history[0]   = static_ler
    weights_mse[0]        = info["weights_mse_error"]
    corr_mse[0]           = info["corr_mse_error"]
    # Static MSE is 0 at reset (adaptive decoder starts identical to static),
    # so both tracers start at the same point.
    weights_mse_static[0] = info["weights_mse_error"]
    corr_mse_static[0]    = info["corr_mse_error"]

    action     = np.zeros(env.n_dec_edges, dtype=np.float32)
    step_idx   = 0
    terminated = False
    truncated  = False
    start_time = time.time()

    while not (terminated or truncated):
        _, _, terminated, truncated, step_info = env.step(action)

        # The env only updates test_ler and MSE metrics after a graph update.
        # Graph updates happen every UPDATE_PERIOD shots.
        if (step_idx + 1) % UPDATE_PERIOD == 0:
            update_idx = (step_idx + 1) // UPDATE_PERIOD

            # test_ler comes from the independent N_TEST_SHOTS batch
            # (evaluated inside env._apply_cma_and_update_graph via
            #  current_matching.decode_batch on self.test_syndrome_batch)
            test_ler_history[update_idx]   = step_info["test_ler"]
            weights_mse[update_idx]        = step_info["weights_mse_error"]
            corr_mse[update_idx]           = step_info["corr_mse_error"]
            weights_mse_static[update_idx] = step_info["weights_mse_error_static"]
            corr_mse_static[update_idx]    = step_info["corr_mse_error_static"]

        step_idx += 1

    run_time = time.time() - start_time

    snapshot = {
        'final_weights':  env.current_weights.copy(),
        'final_pearson':  env.pearson_correlations.copy(),
        'static_weights': env.initial_base_weights.copy(),
        'oracle_weights': env.oracle_weights.copy(),
        'static_pearson': env.initial_pearson_corr.copy(),
        'oracle_pearson': env.oracle_correlations.copy(),
    }

    return (
        test_ler_history,
        oracle_ler,
        static_ler,
        weights_mse,
        corr_mse,
        weights_mse_static,
        corr_mse_static,
        run_time,
        snapshot,
    )


# ==========================================
# MAIN SWEEP & PLOTTING
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DUAL-METRIC CONVERGENCE ANALYSIS (Independent Test LER)")
    print(f"  Sweeping: {len(P_VALUES)} p-values × {len(MISMATCHES)} mismatches")
    print(f"  N_SHOTS={N_SHOTS:,}  |  N_TEST_SHOTS={N_TEST_SHOTS:,}  |  UPDATE_PERIOD={UPDATE_PERIOD}")
    print("=" * 60)

    n_rows = len(P_VALUES)
    n_cols = len(MISMATCHES)

    # Shared x-axis: shot count at each evaluation point (0, 100, 200, …)
    x_shots = np.arange(0, N_SHOTS + 1, UPDATE_PERIOD)  # length = n_updates + 1

    fig_ler,  axes_ler  = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    fig_wmse, axes_wmse = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    fig_cmse, axes_cmse = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)

    total_start = time.time()

    for r, p in enumerate(P_VALUES):
        for c, mismatch in enumerate(MISMATCHES):
            print(f"\n[*] p={p}  |  mismatch={mismatch}x")
            print("    Generating physics ... ", end="", flush=True)

            generator = SyndromeDataGenerator(
                distance=DISTANCE,
                n_rounds=N_ROUNDS,
                mismatch=mismatch,
                noise_model={
                    "version": "built-in",
                    "after_clifford_depolarization":    p,
                    "before_measure_flip_probability":  p,
                    "after_reset_flip_probability":     p,
                    "before_round_data_depolarization": p,
                    "p_gate_zz": 0.0,
                },
                memory_type=MEMORY_TYPE,
                n_shots=N_SHOTS,
                qec_code='surface_code',
            )
            print("done.")

            print("    Running DGR   ... ", end="", flush=True)
            (dgr_ler, oracle_ler, static_ler,
             dgr_w_mse, dgr_c_mse, dgr_w_static, dgr_c_static,
             t_dgr, dgr_snap) = run_convergence_episode(generator, 'DGR', SEED)
            print(f"done ({t_dgr:.1f}s)  | final LER={dgr_ler[-1]:.2e}")

            print("    Running Spitz ... ", end="", flush=True)
            (spitz_ler, _, _,
             spitz_w_mse, spitz_c_mse, _, _,
             t_spitz, spitz_snap) = run_convergence_episode(generator, 'Spitz', SEED)
            print(f"done ({t_spitz:.1f}s)  | final LER={spitz_ler[-1]:.2e}")

            # ── Plot 1: Test LER convergence ──────────────────────────────────
            ax = axes_ler[r, c]
            ax.semilogy(x_shots, dgr_ler,   'b-',  lw=2.5, label="DGR")
            ax.semilogy(x_shots, spitz_ler, 'r-',  lw=2.5, label="Spitz")
            ax.axhline(static_ler, color='k', ls='--', lw=1.5, label="Static")
            ax.axhline(oracle_ler, color='g', ls=':',  lw=2.0, label="Oracle")
            ax.set_title(f"Test LER  (p={p}, drift={mismatch}x)\n"
                         f"[{N_TEST_SHOTS:,} independent shots per evaluation]")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if r == n_rows - 1: ax.set_xlabel("Training shots processed")
            if c == 0:          ax.set_ylabel("Test LER")
            if r == 0 and c == 0: ax.legend(fontsize=10, loc="upper right")

            # ── Plot 2: Weight MSE vs Oracle ──────────────────────────────────
            ax = axes_wmse[r, c]
            ax.semilogy(x_shots, dgr_w_mse,   'b-',  lw=2.5, label="DGR")
            ax.semilogy(x_shots, spitz_w_mse, 'r-',  lw=2.5, label="Spitz")
            ax.semilogy(x_shots, dgr_w_static, 'k--', lw=1.5, label="Static base")
            ax.set_title(f"Weight MSE vs Oracle  (p={p}, drift={mismatch}x)")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if r == n_rows - 1: ax.set_xlabel("Training shots processed")
            if c == 0:          ax.set_ylabel("Weight MSE")
            if r == 0 and c == 0: ax.legend(fontsize=10, loc="upper right")

            # ── Plot 3: Correlation MSE vs Oracle ─────────────────────────────
            ax = axes_cmse[r, c]
            ax.semilogy(x_shots, dgr_c_mse,   'b-',  lw=2.5, label="DGR")
            ax.semilogy(x_shots, spitz_c_mse, 'r-',  lw=2.5, label="Spitz")
            ax.semilogy(x_shots, dgr_c_static, 'k--', lw=1.5, label="Static base")
            ax.set_title(f"Correlation MSE vs Oracle  (p={p}, drift={mismatch}x)")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if r == n_rows - 1: ax.set_xlabel("Training shots processed")
            if c == 0:          ax.set_ylabel("Correlation MSE")
            if r == 0 and c == 0: ax.legend(fontsize=10, loc="upper right")

            # ── Plot 4: Final weight distribution histograms ──────────────────
            weight_sources = [
                ("Static",  dgr_snap['static_weights'], 'k'),
                ("Oracle",  dgr_snap['oracle_weights'], 'g'),
                ("DGR",     dgr_snap['final_weights'],  'b'),
                ("Spitz",   spitz_snap['final_weights'],'r'),
            ]
            w_min = min(arr.min() for _, arr, _ in weight_sources)
            w_max = max(arr.max() for _, arr, _ in weight_sources)
            w_bins = np.linspace(w_min, w_max, 41)

            fig_w_hist, axes_w_hist = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
            for ax, (label, arr, color) in zip(axes_w_hist.flat, weight_sources):
                ax.hist(arr, bins=w_bins, color=color, alpha=0.75, edgecolor='black')
                ax.set_title(f"{label}  (n={arr.size}, mean={arr.mean():.2f})")
                ax.grid(True, ls="--", alpha=0.4)
            for ax in axes_w_hist[-1, :]:
                ax.set_xlabel("Edge weight")
            for ax in axes_w_hist[:, 0]:
                ax.set_ylabel("Edge count")
            fig_w_hist.suptitle(
                f"Final weight distribution  (p={p}, drift={mismatch}x)",
                fontsize=14
            )
            fig_w_hist.tight_layout()
            fig_w_hist.savefig(f"weight_hist_p{p}_m{mismatch}.png", dpi=300)
            plt.close(fig_w_hist)

            # ── Plot 5: Final Pearson correlation distribution histograms ─────
            corr_sources = [
                ("Static",  dgr_snap['static_pearson'], 'k'),
                ("Oracle",  dgr_snap['oracle_pearson'], 'g'),
                ("DGR",     dgr_snap['final_pearson'],  'b'),
                ("Spitz",   spitz_snap['final_pearson'],'r'),
            ]
            c_min = min(arr.min() for _, arr, _ in corr_sources)
            c_max = max(arr.max() for _, arr, _ in corr_sources)
            c_bins = np.linspace(c_min, c_max, 41)

            fig_c_hist, axes_c_hist = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
            for ax, (label, arr, color) in zip(axes_c_hist.flat, corr_sources):
                ax.hist(arr, bins=c_bins, color=color, alpha=0.75, edgecolor='black')
                ax.set_title(f"{label}  (n={arr.size}, mean={arr.mean():.3f})")
                ax.grid(True, ls="--", alpha=0.4)
            for ax in axes_c_hist[-1, :]:
                ax.set_xlabel("Pearson correlation")
            for ax in axes_c_hist[:, 0]:
                ax.set_ylabel("Line-edge count")
            fig_c_hist.suptitle(
                f"Final Pearson correlation distribution  (p={p}, drift={mismatch}x)",
                fontsize=14
            )
            fig_c_hist.tight_layout()
            fig_c_hist.savefig(f"pearson_hist_p{p}_m{mismatch}.png", dpi=300)
            plt.close(fig_c_hist)

    print(f"\nAll sweeps finished in {(time.time()-total_start)/60:.2f} min.")

    for fig, fname in [
        (fig_ler,  "tracer_test_ler_convergence.png"),
        (fig_wmse, "tracer_weight_mse_evolution.png"),
        (fig_cmse, "tracer_correlation_mse_evolution.png"),
    ]:
        fig.tight_layout()
        fig.savefig(fname, dpi=300)
        print(f"[*] Saved '{fname}'")

    plt.show()