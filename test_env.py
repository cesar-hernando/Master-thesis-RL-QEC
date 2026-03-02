"""
In this file, we perform a basic sanity test of the DriftedMatchingEnv 
and its integration with the SyndromeDataGenerator.
"""

import numpy as np
import matplotlib.pyplot as plt

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv


# We will set n_shots to 5 so our episode length is exactly 5 steps
n_shots = 100_000
verbose = False

# Setup the physical simulation
generator = SyndromeDataGenerator(
    distance=5, 
    n_rounds=5, 
    mismatch=50.0,  # Drift multiplier
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

# Setting xz_crosstalk_radius=2.1 captures Y errors and nearest-neighbor crosstalk
env = DriftedMatchingEnv(
    syndrome_data_generator=generator,
    local_action_only=True,
    local_action_hops=1,
    xz_crosstalk_radius=1.5, 
    alpha=0.01,
    oracle_reward_coef=0.0,
    render_mode = "human"
)

print(f"\nGraph Topology Built Successfully:")
print(f" - Total Decoding Edges (GNN Nodes): {env.n_dec_edges}")
print(f" - Total Line Graph Connections (GNN Edges): {env.n_line_edges}")

# Let's count how many X-Z links the new geometric search actually created!
xz_link_count = 0
# Re-infer edge types to check the connections
# We quickly repeat the mod-4 check on the dec_edge_list
edge_types = []
for u, v in env.dec_edge_list:
    c = env.detector_coords.get(u, env.detector_coords.get(v, None))
    if c:
        j, i_y = int(round(c[0])), int(round(c[1]))
        if (i_y % 4 == 0 and j % 4 == 0) or (i_y % 4 == 2 and j % 4 == 2):
            edge_types.append("X")
        else:
            edge_types.append("Z")
    else:
        edge_types.append("Unknown")

# Check the line graph connections
src = env.line_edge_index[0]
dst = env.line_edge_index[1]
for i in range(env.n_line_edges):
    if edge_types[src[i]] != edge_types[dst[i]] and edge_types[src[i]] != "Unknown" and edge_types[dst[i]] != "Unknown":
        xz_link_count += 1
        
# Since line edges are directed in the PyG format, we divide by 2 for unique undirected pairs
print(f" - Unique X-Z Crosstalk Connections Found: {xz_link_count}")

print("\n--- Testing env.reset() ---")
obs, info = env.reset(seed=42)
print("Observation Keys:", list(obs.keys()))
print("Node Features Shape:", obs["node_features"].shape)

# Check the action mask
unlocked_actions = int(obs["action_mask"].sum())
print(f"Initial Shot - Unlocked Actions (due to masking): {unlocked_actions} / {env.n_dec_edges}")


print("\n--- Testing Episode Loop (env.step) ---")
terminated = False
truncated = False
step_num = 0
cum_reward = 0.0
avg_reward = []
weights_mse_error = [info["weights_mse_error"]]
logical_error_rate = []
n_logical_errors = 0
n_logical_errors_oracle = 0
logical_error_rate_oracle = []

while not (terminated or truncated):
    step_num += 1

    if step_num % (n_shots/10)== 0:
        print(f"Completed {100*step_num/n_shots:.0f}% of the episode")

    #if step_num % 10 == 0:
    #    env.render()
    
    # Sample a random continuous action between [-1.0, 1.0]
    #action = env.action_space.sample()
    action = np.zeros(env.n_dec_edges, dtype=np.float32)
    
    # Step the environment
    next_obs, reward, terminated, truncated, step_info = env.step(action)
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
    
    if verbose:
        print(f"\nStep {step_num}:")
        print(f"  - Logical Error Occurred: {bool(step_info['logical_error'])}")
        print(f"  - Total Reward: {reward:.4f}")
        
        if step_info['oracle_similarity_jaccard'] is not None:
            print(f"  - Jaccard Similarity to Oracle: {step_info['oracle_similarity_jaccard']:.4f}")
            
        unlocked_actions = int(next_obs["action_mask"].sum())
        print(f"  - Unlocked Actions for NEXT step: {unlocked_actions}")

print(f"\nEpisode finished! Truncated: {truncated}, Terminated: {terminated}")
print(f"Total steps taken: {step_num} (Should match n_shots: {n_shots})")

env.render()

'''
# Plot the average reward
plt.figure()
plt.plot(avg_reward[1000:])
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.show()

# Plot the weight error
plt.figure()
plt.plot(weights_mse_error[1000:])
plt.xlabel("Step")
plt.ylabel("Weight Error (MSE)")
plt.grid()
plt.show()
'''

# Plot the logical error rate evolution
plt.figure()
plt.plot(logical_error_rate[100:], label="Our")
plt.plot(logical_error_rate_oracle[100:], label="Oracle")
plt.xlabel("Step")
plt.ylabel("Logical error rate")
plt.grid()
plt.legend()
plt.show()

env.render()

