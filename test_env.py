"""
In this file, we perform a basic sanity test of the DriftedMatchingEnv 
and its integration with the SyndromeDataGenerator.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv


n_shots = 1_000_000
verbose = False

# Setup the physical simulation
generator = SyndromeDataGenerator(
    distance=3, 
    n_rounds=3, 
    mismatch=20.0,  # Drift multiplier
    noise_model={
        "version": "built-in",
        "after_clifford_depolarization": 0.001,
        "before_measure_flip_probability": 0.001,
        "after_reset_flip_probability": 0.001,
        "before_round_data_depolarization": 0.001,
    }, 
    memory_type='z', 
    n_shots=n_shots, 
    qec_code='surface_code'
)

env = DriftedMatchingEnv(
    syndrome_data_generator=generator,
    local_action_only=True,
    local_action_hops=1,
    action_scale = 3.0,
    update_period=1_000,
    prior_shots=1_000,
    oracle_reward_coef=1.0,
    use_pearson_correlation=True,
    use_syndrome_features=False,
    update_with='DGR',
    render_mode="human"
)

print(f"\nGraph Topology Built Successfully:")
print(f" - Total Decoding Edges (GNN Nodes): {env.n_dec_edges}")
print(f" - Total Line Graph Connections (GNN Edges): {env.n_line_edges}")

# Pre-allocate metric arrays for speed
rewards = np.zeros(n_shots, dtype=np.float32)
logical_errors = np.zeros(n_shots, dtype=np.float32)
oracle_errors = np.zeros(n_shots, dtype=np.float32)
static_errors = np.zeros(n_shots, dtype=np.float32)
weights_mse_error = np.zeros(n_shots + 1, dtype=np.float32)
corr_mse_error = np.zeros(n_shots + 1, dtype=np.float32)

n_logical_flips = 0

start_time = time.time()
obs, info = env.reset(seed=42)

# Record the starting weight error and correlation error
weights_mse_error[0] = info["weights_mse_error"]
corr_mse_error[0] = info["corr_mse_error"]

terminated = False
truncated = False
step_idx = 0

#print("\nOracle Pearson correlations:\n", env.oracle_correlations)
#print("\nInitial Pearson correlations:\n", env.pearson_correlations)

print("\n--- Starting Episode Loop ---")
while not (terminated or truncated):
    #if (step_idx + 1) % (n_shots // 10) == 0:
    #    print(f"Completed {100 * (step_idx + 1) / n_shots:.0f}% of the episode")
    
    # Step the environment
    action = env.action_space.sample() 
    #action = np.zeros(env.n_dec_edges, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    # Store step information
    rewards[step_idx] = reward
    weights_mse_error[step_idx + 1] = step_info["weights_mse_error"]
    corr_mse_error[step_idx + 1] = step_info["corr_mse_error"]
    logical_errors[step_idx] = float(step_info["logical_error"])
    oracle_errors[step_idx] = float(step_info["oracle_pred_obs"] != step_info["true_obs"])
    static_errors[step_idx] = float(step_info["static_pred_obs"] != step_info["true_obs"])
    
    if step_info["true_obs"]:
        n_logical_flips += 1

    step_idx += 1


#print("Final Pearson Correlations:\n", env.pearson_correlations)
end_time = time.time()
print(f"\nEpisode finished! Time taken: {end_time - start_time:.2f} seconds")

# Calculate running averages instantly using np.cumsum
steps_array = np.arange(1, n_shots + 1)
avg_reward = np.cumsum(rewards) / steps_array
logical_error_rate = np.cumsum(logical_errors) / steps_array
logical_error_rate_oracle = np.cumsum(oracle_errors) / steps_array
logical_error_rate_static = np.cumsum(static_errors) / steps_array

n_logical_errors = int(np.sum(logical_errors))
n_logical_errors_oracle = int(np.sum(oracle_errors))
n_logical_errors_static = int(np.sum(static_errors))

print("Number of logical errors of our decoder: ", n_logical_errors)
print("Number of logical errors of oracle decoder: ", n_logical_errors_oracle)
print("Number of logical errors of static decoder: ", n_logical_errors_static)
print("Number of logical flips: ", n_logical_flips)
print("LER (Our) = ", n_logical_errors/n_shots)
print("LER (Oracle) = ", n_logical_errors_oracle/n_shots)
print("LER (Static) = ", n_logical_errors_static/n_shots)
print("Relative LER (Our) = ", n_logical_errors/n_logical_errors_oracle)
print("Relative LER (Mismatched) = ", n_logical_errors_static/n_logical_errors_oracle)

# Generate the correct x-axis steps for the detached plot
start_plot_idx = 1000
x_steps = steps_array[start_plot_idx:]

############
# PLOTTING #
############

# 1. Reward evolution
plt.figure()
plt.plot(x_steps, rewards[start_plot_idx:])
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.show()

# 2. Average reward
plt.figure()
plt.plot(x_steps, avg_reward[start_plot_idx:])
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.show()

# 3. Weights error
plt.figure()
plt.plot(x_steps, weights_mse_error[start_plot_idx+1:])
plt.xlabel("Step")
plt.ylabel("Weight Error (MSE)")
plt.grid()
plt.show()

# Semi-Log y scale
plt.figure()
plt.semilogy(x_steps, weights_mse_error[start_plot_idx+1:])
plt.xlabel("Step")
plt.ylabel("Weight Error (MSE)")
plt.grid()
plt.show()

# 4. Correlations error
plt.figure()
plt.plot(x_steps, corr_mse_error[start_plot_idx+1:])
plt.xlabel("Step")
plt.ylabel("Correlation Error (MSE)")
plt.grid()
plt.show()

# Semi-log y scale
plt.figure()
plt.semilogy(x_steps, corr_mse_error[start_plot_idx+1:])
plt.xlabel("Step")
plt.ylabel("Correlation Error (MSE)")
plt.grid()
plt.show()

# 5. Logical error rate comparison
plt.figure()
plt.semilogy(x_steps, logical_error_rate[start_plot_idx:], label="Our")
plt.semilogy(x_steps, logical_error_rate_oracle[start_plot_idx:], label="Oracle")
plt.semilogy(x_steps, logical_error_rate_static[start_plot_idx:], '--', label="Static")
plt.xlabel("Step")
plt.ylabel("Logical error rate")
plt.grid()
plt.legend()
plt.show()

env.render()