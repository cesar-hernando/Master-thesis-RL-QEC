"""
Comparative analysis script to evaluate the performance of DGR vs. Spitz 
adaptive tracer methods across different physical error rates (p) and 
drift severities (mismatch).
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv

# ==========================================
# CONFIGURATION
# ==========================================
N_SHOTS = 50_000         # Shots per episode (reduced from 100k for faster sweeping)
N_TEST_SHOTS = 100_000    # Shots used to evaluate the true LER of the graph
UPDATE_PERIOD = 1_000
SEED = 42

# Parameter grids to sweep
P_VALUES = [0.001, 0.003, 0.005]
MISMATCHES = [5.0, 10.0, 20.0, 30.0]
TRACERS = ['DGR', 'Spitz']

# Fixed Code Parameters
DISTANCE = 5
N_ROUNDS = 5
MEMORY_TYPE = 'z'

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate_tracer(generator, tracer_method, seed):
    """Initializes the env with the given tracer and runs a zero-action episode."""
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
    
    _, info = env.reset(seed=seed)
    
    static_ler = info["initial_test_ler"]
    oracle_ler = info["oracle_ler"]
    
    # Run the environment passively (zero action) to let the tracer adapt
    terminated = False
    truncated = False
    action = np.zeros(env.n_dec_edges, dtype=np.float32)
    
    while not (terminated or truncated):
        _, _, terminated, truncated, step_info = env.step(action)
        
    final_test_ler = step_info["test_ler"]
    
    return static_ler, oracle_ler, final_test_ler


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Starting Tracer Comparison: {len(P_VALUES)} p-values x {len(MISMATCHES)} mismatches")
    print("=" * 60)
    
    # Dictionary to store results for plotting
    results = {p: {'Static': [], 'Oracle': [], 'DGR': [], 'Spitz': []} for p in P_VALUES}
    
    total_start_time = time.time()
    
    for p in P_VALUES:
        print(f"\nEvaluating Base Error Rate: p = {p}")
        print("-" * 40)
        
        for mismatch in MISMATCHES:
            loop_start = time.time()
            print(f"  -> Mismatch = {mismatch} | Generating Physics...", end="", flush=True)
            
            # 1. Initialize Generator ONCE per (p, mismatch) pair
            noise_model = {
                "version": "built-in",
                "after_clifford_depolarization": p,
                "before_measure_flip_probability": p,
                "after_reset_flip_probability": p,
                "before_round_data_depolarization": p,
                "p_gate_zz": 0.0, 
            }
            
            generator = SyndromeDataGenerator(
                distance=DISTANCE, 
                n_rounds=N_ROUNDS, 
                mismatch=mismatch, 
                noise_model=noise_model, 
                memory_type=MEMORY_TYPE, 
                n_shots=N_SHOTS, 
                qec_code='surface_code'
            )
            print(" Done.", flush=True)
            
            # 2. Evaluate Tracers
            final_lers = {}
            for tracer in TRACERS:
                static_ler, oracle_ler, final_ler = evaluate_tracer(generator, tracer, seed=SEED)
                final_lers[tracer] = final_ler
            
            # Store results
            results[p]['Static'].append(static_ler)
            results[p]['Oracle'].append(oracle_ler)
            results[p]['DGR'].append(final_lers['DGR'])
            results[p]['Spitz'].append(final_lers['Spitz'])
            
            loop_time = time.time() - loop_start
            print(f"     Static: {static_ler:.5f} | Oracle: {oracle_ler:.5f} | DGR: {final_lers['DGR']:.5f} | Spitz: {final_lers['Spitz']:.5f} ({loop_time:.1f}s)")
            
    print("\n" + "=" * 60)
    print(f"All evaluations finished in {(time.time() - total_start_time)/60:.2f} minutes.")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(1, len(P_VALUES), figsize=(5 * len(P_VALUES), 5), sharey=False)
    
    if len(P_VALUES) == 1:
        axes = [axes]
        
    for idx, p in enumerate(P_VALUES):
        ax = axes[idx]
        
        ax.semilogy(MISMATCHES, results[p]['Static'], 'k--', label="Static Decoder", marker='x')
        ax.semilogy(MISMATCHES, results[p]['DGR'], 'b-', label="Adaptive (DGR)", marker='o')
        ax.semilogy(MISMATCHES, results[p]['Spitz'], 'r-', label="Adaptive (Spitz)", marker='s')
        ax.semilogy(MISMATCHES, results[p]['Oracle'], 'g:', label="Oracle Decoder", marker='*')
        
        ax.set_title(f"Base Error Rate: p = {p}")
        ax.set_xlabel("Drift Mismatch Multiplier")
        ax.set_ylabel("Test LER")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig("tracer_comparison_results.png", dpi=300)
    plt.show()