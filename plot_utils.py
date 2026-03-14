'''
This module includes helper functions to plot different training
and testing metrics, such as the evolution of the reward, logical
error rates, weights and correlations.
'''

import matplotlib.pyplot as plt
import os
import numpy as np


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_weight_correlations(wm):
    """Generates a 2x2 plot showing step-by-step evolution of weights over an average episode."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Intra-Episode Evolution: GNN vs. Zero (CMA) vs. Random\n(Averaged across test episodes with ±1 Std Dev)", fontsize=16)
    
    # Colors matching the bar chart
    c_gnn = '#2ca02c'   # Green
    c_zero = '#9467bd'  # Purple
    c_rand = '#ff7f0e'  # Orange

    def plot_evolution_with_std(ax, data_list, label, color, linestyle='-'):
        """Helper to compute step-wise mean/std and plot with a shaded confidence interval."""
        # Convert list of lists (episodes x steps) to a 2D numpy array
        data = np.array(data_list)
        
        # Calculate mean and std across the episodes (axis 0)
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        steps = np.arange(1, len(mean_vals) + 1)
        
        # Plot the mean line
        ax.plot(steps, mean_vals, label=label, color=color, linewidth=2, linestyle=linestyle)
        # Fill the standard deviation area
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

    # --- ROW 1: MSE ---
    # Plot 1: MSE vs Oracle
    plot_evolution_with_std(axs[0, 0], wm['mse_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 0], wm['mse_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 0], wm['mse_random_oracle'], 'Random', c_rand, ':')
    axs[0, 0].set_title('Weight Difference (vs Oracle Weights)')
    axs[0, 0].set_ylabel('Mean Squared Error')
    axs[0, 0].set_xlabel('Step (Shot) within Episode')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: MSE vs Static
    plot_evolution_with_std(axs[0, 1], wm['mse_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 1], wm['mse_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 1], wm['mse_random_static'], 'Random', c_rand, ':')
    axs[0, 1].set_title('Weight Difference (vs Static Undrifted Graph)')
    axs[0, 1].set_xlabel('Step (Shot) within Episode')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # --- ROW 2: CORRELATIONS ---
    # Plot 3: Correlation vs Oracle
    plot_evolution_with_std(axs[1, 0], wm['p_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 0], wm['p_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 0], wm['p_random_oracle'], 'Random', c_rand, ':')
    axs[1, 0].set_title('Correlation (vs Oracle)')
    axs[1, 0].set_ylabel('Correlation Coefficient / Error')
    axs[1, 0].set_xlabel('Step (Shot) within Episode')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Correlation vs Static
    plot_evolution_with_std(axs[1, 1], wm['p_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 1], wm['p_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 1], wm['p_random_static'], 'Random', c_rand, ':')
    axs[1, 1].set_title('Correlation (vs Static Undrifted Graph)')
    axs[1, 1].set_xlabel('Step (Shot) within Episode')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/weight_evolution_comprehensive.png', dpi=300)
    print("Weight evolution plots saved to 'plots/weight_evolution_comprehensive.png'")
    plt.close()
    

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