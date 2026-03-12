"""
Main execution script for training and testing the SAC-GNN QEC Decoder.
It handles hyperparameters, execution modes, and performance visualization.
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv
from gnn_sac_agent import SACAgent, GraphReplayBuffer

###################################
# 1. VISUALIZATION HELPERS        #
###################################

def plot_training_metrics(metrics, config):
    """Generates a 3-panel plot showing training health over episodes."""
    os.makedirs('plots', exist_ok=True)
    
    episodes = range(1, len(metrics['rewards']) + 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"SAC-GNN Training Metrics ({config['train_episodes']} Episodes)", fontsize=16)
    
    # Plot 1: Total Reward
    axs[0].plot(episodes, metrics['rewards'], color='blue', linewidth=2)
    axs[0].set_title('Episode Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Losses
    axs[1].plot(episodes, metrics['c_losses'], label='Critic Loss', color='red', alpha=0.8)
    axs[1].plot(episodes, metrics['a_losses'], label='Actor Loss', color='green', alpha=0.8)
    axs[1].set_title('Network Losses')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Weight MSE to Oracle
    axs[2].plot(episodes, metrics['mses'], color='purple', linewidth=2)
    axs[2].set_title('Final Weight MSE vs Oracle')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Mean Squared Error')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/training_metrics.png', dpi=300)
    print("Training plots saved to 'plots/training_metrics.png'")
    plt.close()


def plot_testing_metrics(test_results):
    """Generates a bar chart of LER and a cumulative error timeline."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Decoder Performance Evaluation", fontsize=16)
    
    # Plot 1: Bar Chart of Logical Error Rates
    labels = ['GNN (Our)', 'Oracle', 'Static']
    lers = [test_results['ler_gnn'], test_results['ler_oracle'], test_results['ler_static']]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    
    bars = axs[0].bar(labels, lers, color=colors, alpha=0.8)
    axs[0].set_title('Logical Error Rate (LER)')
    axs[0].set_ylabel('LER')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with exact values
    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.5f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Cumulative Errors Over Time
    shots = np.arange(1, len(test_results['cum_gnn']) + 1)
    axs[1].plot(shots, test_results['cum_gnn'], label='GNN (Our)', color='#2ca02c', linewidth=2)
    axs[1].plot(shots, test_results['cum_oracle'], label='Oracle', color='#1f77b4', linewidth=2, linestyle='--')
    axs[1].plot(shots, test_results['cum_static'], label='Static', color='#d62728', linewidth=2, linestyle='-.')
    
    axs[1].set_title('Cumulative Logical Errors')
    axs[1].set_xlabel('Total Shots Evaluated')
    axs[1].set_ylabel('Cumulative Errors')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/testing_metrics.png', dpi=300)
    print("Testing plots saved to 'plots/testing_metrics.png'")
    plt.close()


###################################
# 2. TRAINING & TESTING FUNCTIONS #
###################################

def train(env, agent, buffer, config):
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING (Episodes: {config['train_episodes']})")
    print(f"{'='*40}")
    
    # Tracking dictionaries for plotting
    metrics = {'rewards': [], 'c_losses': [], 'a_losses': [], 'mses': []}
    start_time = time.time()
    
    for episode in range(config['train_episodes']):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        ep_c_loss, ep_a_loss = [], []
        
        while not done:
            # evaluate=False enables Gaussian exploration
            action = agent.select_action(obs, evaluate=False) 
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.push(obs, action, reward, next_obs if not done else None, done)
            
            # Staggered updates
            if len(buffer) > config['batch_size'] and step_count % config['update_frequency'] == 0:
                c_loss, a_loss = agent.update(buffer, config['batch_size'])
                ep_c_loss.append(c_loss)
                ep_a_loss.append(a_loss)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
        # Record end-of-episode metrics
        avg_c_loss = np.mean(ep_c_loss) if ep_c_loss else 0.0
        avg_a_loss = np.mean(ep_a_loss) if ep_a_loss else 0.0
        
        metrics['rewards'].append(episode_reward)
        metrics['c_losses'].append(avg_c_loss)
        metrics['a_losses'].append(avg_a_loss)
        metrics['mses'].append(info['weights_mse_error'])
            
        print(f"Train Ep: {episode+1:03d}/{config['train_episodes']} | "
              f"Reward: {episode_reward:6.1f} | "
              f"MSE: {info['weights_mse_error']:.4f} | "
              f"C_Loss: {avg_c_loss:.3f}")

    run_time = time.time() - start_time
    print(f"\nTraining complete in {run_time:.2f} seconds.")
    
    # Save the trained model and plot metrics
    agent.save_models(config['model_path'])
    plot_training_metrics(metrics, config)


def test(env, agent, config):
    print(f"\n{'='*40}")
    print(f"STARTING TESTING (Episodes: {config['test_episodes']})")
    print(f"{'='*40}")
    
    # Load the best weights
    agent.load_models(config['model_path'])
    
    total_shots = config['test_episodes'] * env.max_steps
    
    # Arrays to track cumulative errors shot-by-shot for plotting
    cum_gnn = np.zeros(total_shots, dtype=np.int32)
    cum_oracle = np.zeros(total_shots, dtype=np.int32)
    cum_static = np.zeros(total_shots, dtype=np.int32)
    
    total_logical_errors, total_oracle_errors, total_static_errors = 0, 0, 0
    total_flips = 0
    global_shot_idx = 0
    
    for episode in range(config['test_episodes']):
        obs, info = env.reset()
        done = False
        
        ep_logical_errs, ep_oracle_errs, ep_static_errs = 0, 0, 0
        
        while not done:
            # evaluate=True outputs strictly optimal deterministic actions
            action = agent.select_action(obs, evaluate=True) 
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Step errors
            err_gnn = int(info["logical_error"])
            err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
            err_static = int(info["static_pred_obs"] != info["true_obs"])
            
            # Accumulate for episode printing
            ep_logical_errs += err_gnn
            ep_oracle_errs += err_oracle
            ep_static_errs += err_static
            
            if info["true_obs"]:
                total_flips += 1
                
            # Update global trackers
            total_logical_errors += err_gnn
            total_oracle_errors += err_oracle
            total_static_errors += err_static
            
            cum_gnn[global_shot_idx] = total_logical_errors
            cum_oracle[global_shot_idx] = total_oracle_errors
            cum_static[global_shot_idx] = total_static_errors
            
            obs = next_obs
            global_shot_idx += 1
            
        print(f"Test Ep {episode+1:02d}: GNN Errs: {ep_logical_errs} | Oracle Errs: {ep_oracle_errs} | Static Errs: {ep_static_errs}")

    # Compile final results
    test_results = {
        'ler_gnn': total_logical_errors / total_shots,
        'ler_oracle': total_oracle_errors / total_shots,
        'ler_static': total_static_errors / total_shots,
        'cum_gnn': cum_gnn,
        'cum_oracle': cum_oracle,
        'cum_static': cum_static
    }

    print("\n--- FINAL TEST REPORT ---")
    print(f"Total Shots: {total_shots}")
    print(f"Number of logical flips: {total_flips}")
    print(f"LER (Our GNN)  = {test_results['ler_gnn']:.5f}")
    print(f"LER (Oracle)   = {test_results['ler_oracle']:.5f}")
    print(f"LER (Static)   = {test_results['ler_static']:.5f}")
    print(f"Relative LER (Our vs Oracle)   = {total_logical_errors / max(1, total_oracle_errors):.3f}")
    print(f"Relative LER (Static vs Oracle)= {total_static_errors / max(1, total_oracle_errors):.3f}")

    # Plot the final evaluation
    plot_testing_metrics(test_results)


##################
# MAIN EXECUTION #
##################

if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'train',  
        'model_path': 'models/sac_gnn_best.pth',
        
        # Environment Settings
        'distance': 3,
        'n_rounds': 3,
        'p': 0.001,
        'mismatch': 20.0,
        'n_shots': 60_000,       # Shots per episode
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 1.0, # Phase 1: High imitation reward
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 3e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.2,          # Entropy tuning
        'batch_size': 64,
        'update_frequency': 10,
        
        # Episode Settings
        'train_episodes': 50,
        'test_episodes': 10
    }

    #######################################
    # 2. INITIALIZE ENVIRONMENT AND AGENT #
    #######################################
    print("Initializing environment...")
    generator = SyndromeDataGenerator(
        distance=CONFIG['distance'], 
        n_rounds=CONFIG['n_rounds'], 
        mismatch=CONFIG['mismatch'],  
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": CONFIG["p"],
            "before_measure_flip_probability": CONFIG["p"],
            "after_reset_flip_probability": CONFIG["p"],
            "before_round_data_depolarization": CONFIG["p"],
        }, 
        memory_type='z', 
        n_shots=CONFIG['n_shots'], 
        qec_code='surface_code'
    )
    
    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=1,
        action_scale=CONFIG['action_scale'],
        update_period=CONFIG['update_period'],
        prior_shots=CONFIG['prior_shots'],            
        oracle_reward_coef=CONFIG['oracle_reward_coef'], 
        use_pearson_correlation=True,
        use_syndrome_features=False, 
        update_with='DGR',
    )

    # Determine dynamic dimensions from environment
    sample_obs, _ = env.reset()
    NODE_DIM = sample_obs["node_features"].shape[1]
    
    # Init Agent
    agent = SACAgent(
        node_dim=NODE_DIM, 
        hidden_dim=CONFIG['hidden_dim'],
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        tau=CONFIG['tau'],
        alpha=CONFIG['alpha']
    )

    ############################
    # 3. EXECUTE SELECTED MODE #
    ############################
    if CONFIG['MODE'] == 'train':
        start_train = time.time()
        buffer = GraphReplayBuffer(capacity=50_000)
        train(env, agent, buffer, CONFIG)
        end_train = time.time()
        train_runtime = end_train - start_train
        print(f"Training run time = {train_runtime:.2f} s")
        
    elif CONFIG['MODE'] == 'test':
        start_test = time.time()
        test(env, agent, CONFIG)
        end_test = time.time()
        test_runtime = end_test - start_test
        print(f"Test run time = {test_runtime:.2f} s")
    else:
        print(f"Unknown MODE: {CONFIG['MODE']}. Please select 'train' or 'test'.")