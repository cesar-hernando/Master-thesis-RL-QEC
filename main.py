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
    """Generates a bar chart and timeline for all 5 evaluation metrics."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comprehensive Decoder Evaluation (Ablation Study)", fontsize=16)
    
    # Define the 5 categories
    labels = ['GNN (Ours)', 'Oracle', 'Zero (CMA Only)', 'Static', 'Random']
    keys = ['gnn', 'oracle', 'zero', 'static', 'random']
    
    # Colors: Green (Win), Blue (Target), Purple (Ablation), Red (Baseline), Orange (Worst)
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#d62728', '#ff7f0e']
    
    lers = [test_results[f'ler_{k}'] for k in keys]
    
    # Plot 1: Bar Chart of Logical Error Rates
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
    
    for label, key, color in zip(labels, keys, colors):
        # Use dashed/dotted lines for the static baselines to make the GNN stand out
        linestyle = '-' if key in ['gnn', 'zero', 'random'] else '--'
        axs[1].plot(shots, test_results[f'cum_{key}'], label=label, color=color, linewidth=2, linestyle=linestyle)
    
    axs[1].set_title('Cumulative Logical Errors')
    axs[1].set_xlabel('Total Shots Evaluated')
    axs[1].set_ylabel('Cumulative Errors')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/testing_metrics_comprehensive.png', dpi=300)
    print("Testing plots saved to 'plots/testing_metrics_comprehensive.png'")
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
    print(f"\n{'='*50}")
    print(f"STARTING COMPREHENSIVE ABLATION TEST (Episodes: {config['test_episodes']})")
    print(f"{'='*50}")
    
    # Load the best weights
    agent.load_models(config['model_path'])
    
    total_shots = config['test_episodes'] * env.max_steps
    policies = ['GNN', 'Zero', 'Random']
    
    # Initialize data trackers
    raw_results = {
        'Oracle': {'errors': 0, 'cum': np.zeros(total_shots, dtype=np.int32)},
        'Static': {'errors': 0, 'cum': np.zeros(total_shots, dtype=np.int32)}
    }
    for p in policies:
        raw_results[p] = {'errors': 0, 'cum': np.zeros(total_shots, dtype=np.int32)}

    # Evaluate each policy sequentially
    for policy in policies:
        print(f"\n[*] Evaluating Policy: {policy}")
        global_shot_idx = 0
        policy_errors = 0
        oracle_errors = 0
        static_errors = 0
        
        for episode in range(config['test_episodes']):
            obs, info = env.reset()
            done = False
            ep_policy_errs = 0
            
            while not done:
                # Choose action based on the current policy pass
                if policy == 'GNN':
                    action = agent.select_action(obs, evaluate=True)
                elif policy == 'Zero':
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                elif policy == 'Random':
                    action = np.random.uniform(low=-1.0, high=1.0, size=env.n_dec_edges).astype(np.float32)
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Track Errors
                err_policy = int(info["logical_error"])
                policy_errors += err_policy
                ep_policy_errs += err_policy
                
                # Only track Oracle and Static during the GNN run so we don't calculate them 3 times
                if policy == 'GNN':
                    err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
                    err_static = int(info["static_pred_obs"] != info["true_obs"])
                    oracle_errors += err_oracle
                    static_errors += err_static
                    
                    raw_results['Oracle']['cum'][global_shot_idx] = oracle_errors
                    raw_results['Static']['cum'][global_shot_idx] = static_errors

                raw_results[policy]['cum'][global_shot_idx] = policy_errors
                
                obs = next_obs
                global_shot_idx += 1
                
            print(f"  Test Ep {episode+1:02d} [{policy}]: {ep_policy_errs} errors")
            
        raw_results[policy]['errors'] = policy_errors
        
        if policy == 'GNN':
            raw_results['Oracle']['errors'] = oracle_errors
            raw_results['Static']['errors'] = static_errors

    # Format the final dictionary for the plotter
    final_metrics = {}
    for k in ['GNN', 'Zero', 'Random', 'Oracle', 'Static']:
        final_metrics[f'ler_{k.lower()}'] = raw_results[k]['errors'] / total_shots
        final_metrics[f'cum_{k.lower()}'] = raw_results[k]['cum']

    # Print Final Report
    print("\n--- FINAL ABLATION TEST REPORT ---")
    print(f"Total Shots Evaluated per Policy: {total_shots}")
    print(f"LER (GNN)     = {final_metrics['ler_gnn']:.5f}  <-- Our Agent")
    print(f"LER (Oracle)  = {final_metrics['ler_oracle']:.5f}  <-- Perfect Marginal Knowledge")
    print(f"LER (Zero)    = {final_metrics['ler_zero']:.5f}  <-- CMA Tracer Only")
    print(f"LER (Static)  = {final_metrics['ler_static']:.5f}  <-- Undrifted Baseline")
    print(f"LER (Random)  = {final_metrics['ler_random']:.5f}  <-- Chaos Baseline")

    plot_testing_metrics(final_metrics)




if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'test',  
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