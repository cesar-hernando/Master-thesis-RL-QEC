"""
In this file, we perform a basic sanity test of the DriftedMatchingEnv 
and its integration with the SyndromeDataGenerator.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv

# We will set n_shots to 100,000
n_shots = 100_000
verbose = False

# Setup the physical simulation
generator = SyndromeDataGenerator(
    distance=5, 
    n_rounds=5, 
    mismatch=10.0,  # Drift multiplier
    noise_model={
        "version": "built-in",
        "after_clifford_depolarization": 0.0,
        "before_measure_flip_probability": 0.01,
        "after_reset_flip_probability": 0.0,
        "before_round_data_depolarization": 0.01,
    }, 
    memory_type='z', 
    n_shots=n_shots, 
    qec_code='surface_code'
)

# Setting local_action_hops=1 captures Y errors and nearest-neighbor crosstalk
env = DriftedMatchingEnv(
    syndrome_data_generator=generator,
    local_action_only=True,
    local_action_hops=1,
    update_period=100,
    prior_shots=100,
    oracle_reward_coef=0.5,
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

n_logical_flips = 0

start_time = time.time()
obs, info = env.reset(seed=42)

# Record the starting weight error
weights_mse_error[0] = info["weights_mse_error"]

terminated = False
truncated = False
step_idx = 0

print("\n--- Starting Episode Loop ---")
while not (terminated or truncated) and step_idx < n_shots:
    if (step_idx + 1) % (n_shots // 10) == 0:
        print(f"Completed {100 * (step_idx + 1) / n_shots:.0f}% of the episode")
    
    # Step the environment
    action = np.zeros(env.n_dec_edges, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)


    # Fast indexed storage
    rewards[step_idx] = reward
    weights_mse_error[step_idx + 1] = step_info["weights_mse_error"]
    logical_errors[step_idx] = float(step_info["logical_error"])
    oracle_errors[step_idx] = float(step_info["oracle_pred_obs"] != step_info["true_obs"])
    static_errors[step_idx] = float(step_info["static_pred_obs"] != step_info["true_obs"])
    
    if step_info["true_obs"]:
        n_logical_flips += 1
        
    step_idx += 1

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

# Plotting
plt.figure()
plt.plot(steps_array, avg_reward)
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.show()

plt.figure()
plt.loglog(np.arange(n_shots + 1), weights_mse_error)
plt.xlabel("Step")
plt.ylabel("Weight Error (MSE)")
plt.grid()
plt.show()

# Generate the correct x-axis steps for the detached plot
start_plot_idx = 1000
x_steps = steps_array[start_plot_idx:]

plt.figure()
plt.loglog(x_steps, logical_error_rate[start_plot_idx:], label="Our")
plt.loglog(x_steps, logical_error_rate_oracle[start_plot_idx:], label="Oracle")
plt.loglog(x_steps, logical_error_rate_static[start_plot_idx:], label="Static")
plt.xlabel("Step")
plt.ylabel("Logical error rate")
plt.grid()
plt.legend()
plt.show()

env.render()