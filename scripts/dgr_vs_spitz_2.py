"""
Script to compare the step-by-step convergence of DGR and Spitz tracers
across a grid of different physical error rates (p) and drift severities.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv

# ==========================================
# CONFIGURATION
# ==========================================
# Parameter grids to sweep (Keep these small to avoid hours of execution time!)
P_VALUES = [0.001, 0.003, 0.005]
MISMATCHES = [10.0, 20.0, 30.0]

N_SHOTS = 100_000          # Shots per episode (lowered slightly to accommodate grid sweep)
N_TEST_SHOTS = 1000_000     # Batch size for evaluating the Test LER
UPDATE_PERIOD = 1_000     # How often the environment updates the graph and calculates LER
SEED = 2024               # Fixed seed for a fair apples-to-apples comparison

# Fixed Code Parameters
DISTANCE = 5
N_ROUNDS = 5
MEMORY_TYPE = 'z'

def run_convergence_episode(generator, tracer_method, seed):
    """
    Runs a full episode passively (Zero Action) and tracks the Test LER.
    """
    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=1,
        action_scale=1.0, 
        update_period=UPDATE_PERIOD,
        prior_shots=1 if tracer_method == 'Spitz' else 1000,  # Spitz can start with fewer shots
        n_test_shots=N_TEST_SHOTS,
        use_pearson_correlation=True,
        use_syndrome_features=True if tracer_method == 'Spitz' else False,
        update_with=tracer_method,
        train_mode=False
    )
    
    obs, info = env.reset(seed=seed)
    
    static_ler = info["initial_test_ler"]
    oracle_ler = info["oracle_ler"]
    
    n_updates = (N_SHOTS // UPDATE_PERIOD) + 1
    test_ler_history = np.zeros(n_updates, dtype=np.float32)
    test_ler_history[0] = static_ler
    
    terminated = False
    truncated = False
    step_idx = 0
    action = np.zeros(env.n_dec_edges, dtype=np.float32)
    
    start_time = time.time()
    
    while not (terminated or truncated):
        _, _, terminated, truncated, step_info = env.step(action)
        
        if (step_idx + 1) % UPDATE_PERIOD == 0:
            update_idx = (step_idx + 1) // UPDATE_PERIOD
            test_ler_history[update_idx] = step_info["test_ler"]
                
        step_idx += 1
        
    run_time = time.time() - start_time
    return test_ler_history, static_ler, oracle_ler, run_time


if __name__ == "__main__":
    print("=" * 60)
    print(f"STARTING GRID CONVERGENCE ANALYSIS")
    print(f"Sweeping: {len(P_VALUES)} p-values x {len(MISMATCHES)} mismatches")
    print("=" * 60)
    
    # Setup the plot grid
    n_rows = len(P_VALUES)
    n_cols = len(MISMATCHES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    x_steps = np.arange(0, N_SHOTS + 1, UPDATE_PERIOD)
    
    total_start_time = time.time()
    
    for r, p in enumerate(P_VALUES):
        for c, mismatch in enumerate(MISMATCHES):
            print(f"\n[*] Evaluating [p = {p} | Mismatch = {mismatch}x]")
            print("  -> Generating Physics...", end="", flush=True)
            
            generator = SyndromeDataGenerator(
                distance=DISTANCE, 
                n_rounds=N_ROUNDS, 
                mismatch=mismatch, 
                noise_model={
                    "version": "built-in",
                    "after_clifford_depolarization": p,
                    "before_measure_flip_probability": p,
                    "after_reset_flip_probability": p,
                    "before_round_data_depolarization": p,
                    "p_gate_zz": 0.0, 
                }, 
                memory_type=MEMORY_TYPE, 
                n_shots=N_SHOTS, 
                qec_code='surface_code'
            )
            print(" Done.")
            
            print("  -> Running DGR...", end="", flush=True)
            dgr_history, static_ler, oracle_ler, t_dgr = run_convergence_episode(generator, 'DGR', SEED)
            print(f" ({t_dgr:.1f}s)")
            
            print("  -> Running Spitz...", end="", flush=True)
            spitz_history, _, _, t_spitz = run_convergence_episode(generator, 'Spitz', SEED)
            print(f" ({t_spitz:.1f}s)")
            
            # --- Plotting to the specific subplot ---
            ax = axes[r, c]
            
            ax.semilogy(x_steps, dgr_history, 'b-', linewidth=2, label="Adaptive (DGR Tracer)")
            ax.semilogy(x_steps, spitz_history, 'r-', linewidth=2, label="Adaptive (Spitz Tracer)")
            
            ax.semilogy(x_steps, np.ones_like(x_steps) * static_ler, 'k--', linewidth=1.5, label="Static Decoder")
            ax.semilogy(x_steps, np.ones_like(x_steps) * oracle_ler, 'g:', linewidth=2, label="Oracle Decoder")
            
            ax.set_title(f"p = {p} | Drift = {mismatch}x")
            ax.grid(True, which="both", ls="--", alpha=0.6)
            
            # Only add labels to outer edges to keep it clean
            if r == n_rows - 1:
                ax.set_xlabel("Shots Processed")
            if c == 0:
                ax.set_ylabel("Test LER")
                
            # Only put the legend in the first subplot
            if r == 0 and c == 0:
                ax.legend(fontsize=10, loc="upper right")
                
    print("\n" + "=" * 60)
    print(f"All sweeps finished in {(time.time() - total_start_time)/60:.2f} minutes.")
    
    plt.tight_layout()
    plt.savefig("tracer_convergence_grid.png", dpi=300)
    plt.show()