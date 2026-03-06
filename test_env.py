"""
In this file, we perform a basic sanity test of the DriftedMatchingEnv 
and its integration with the SyndromeDataGenerator.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv


# We will set n_shots to 5 so our episode length is exactly 5 steps
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

# Setting xz_crosstalk_radius=2.1 captures Y errors and nearest-neighbor crosstalk
env = DriftedMatchingEnv(
    syndrome_data_generator=generator,
    local_action_only=True,
    local_action_hops=1,
    update_period=100,
    prior_shots=100,
    oracle_reward_coef=0.5,
    render_mode = "human"
)

print(f"\nGraph Topology Built Successfully:")
print(f" - Total Decoding Edges (GNN Nodes): {env.n_dec_edges}")
print(f" - Total Line Graph Connections (GNN Edges): {env.n_line_edges}")


step_num = 0
cum_reward = 0.0
avg_reward = []
logical_error_rate = []
n_logical_errors = 0
n_logical_errors_oracle = 0
logical_error_rate_oracle = []
n_logical_flips = 0
n_logical_errors_static = 0
logical_error_rate_static = []
start_time = time.time()

terminated = False
truncated = False
obs, info = env.reset(seed=42)
weights_mse_error = [info["weights_mse_error"]]

while not (terminated or truncated):
    step_num += 1

    if step_num % (n_shots/10)== 0:
        print(f"Completed {100*step_num/n_shots:.0f}% of the episode")
    
    # Sample a random continuous action between [-1.0, 1.0]
    #action = env.action_space.sample()
    action = np.zeros(env.n_dec_edges, dtype=np.float32)
    
    # Step the environment
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    # Episode analysis
    cum_reward += reward
    avg_reward.append(cum_reward/step_num)
    weights_mse_error.append(step_info["weights_mse_error"])
    logical_error = step_info["logical_error"]
    if logical_error:
        n_logical_errors += 1
    logical_error_rate.append(n_logical_errors/step_num)

    if step_info["oracle_pred_obs"] != step_info["true_obs"]:
        n_logical_errors_oracle += 1
    logical_error_rate_oracle.append(n_logical_errors_oracle/step_num)

    if step_info["static_pred_obs"] != step_info["true_obs"]:
        n_logical_errors_static += 1
    logical_error_rate_static.append(n_logical_errors_static/step_num)

    if step_info["true_obs"]:
        n_logical_flips += 1
    
    if verbose:
        print(f"\nStep {step_num}:")
        print(f"  - Logical Error Occurred: {bool(step_info['logical_error'])}")
        print(f"  - Total Reward: {reward:.4f}")
        
        if step_info['oracle_similarity_jaccard'] is not None:
            print(f"  - Jaccard Similarity to Oracle: {step_info['oracle_similarity_jaccard']:.4f}")
            
        unlocked_actions = int(next_obs["action_mask"].sum())
        print(f"  - Unlocked Actions for NEXT step: {unlocked_actions}")

end_time = time.time()
episode_time = end_time - start_time
print(f"\nEpisode finished! Time taken: {episode_time} (s)")


# Plot the average reward
plt.figure()
plt.plot(avg_reward)
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.show()

# Plot the weight error
plt.figure()
plt.loglog(weights_mse_error)
plt.xlabel("Step")
plt.ylabel("Weight Error (MSE)")
plt.grid()
plt.show()

print("Number of logical errors of our decoder: ", n_logical_errors)
print("Number of logical errors of oracle decoder: ", n_logical_errors_oracle)
print("Number of logical errors of static decoder: ", n_logical_errors_static)
print("Number of logical flips: ", n_logical_flips)

# Plot the logical error rate evolution# Generate the correct x-axis steps
x_steps = range(1000, len(logical_error_rate))
plt.figure()
plt.loglog(x_steps, logical_error_rate[1000:], label="Our")
plt.loglog(x_steps, logical_error_rate_oracle[1000:], label="Oracle")
plt.loglog(x_steps, logical_error_rate_static[1000:], label="Static")
plt.xlabel("Step")
plt.ylabel("Logical error rate")
plt.grid()
plt.legend()
plt.show()

env.render()

