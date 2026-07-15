'''
This module includes helper functions to plot different training
and testing metrics, such as the evolution of the reward, logical
error rates, weights and correlations.
'''

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_action_topography(direct, neighbors, filename='action_topography_hist.png'):
    """Adds a visual comparison of how the GNN treats selected edges vs context edges."""
    if not direct and not neighbors:
        print("No active actions collected for topography plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    ax1.hist(direct, bins=50, color='royalblue', alpha=0.7, log=True)
    ax1.set_title("Direct Actions\n(Edges MWPM Selected in Pass 1)")
    ax1.set_xlabel("Action Value")
    ax1.set_ylabel("Frequency (log scale)")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    ax2.hist(neighbors, bins=50, color='darkorange', alpha=0.7, log=True)
    ax2.set_title("Neighbor Actions\n(Contextual Masked Edges)")
    ax2.set_xlabel("Action Value")
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.suptitle("Action Topography Breakdown", fontsize=16)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def plot_syndrome_count_histogram(syndrome_counts, bypass_threshold=4, filename='syndrome_counts_hist.png'):
    """Plots a histogram of the number of active syndrome flashes per shot."""
    if not len(syndrome_counts):
        print("No syndrome data to plot!")
        return
        
    print("\nGenerating syndrome count histogram...")
    plt.figure(figsize=(10, 6))
    
    min_val = int(np.min(syndrome_counts))
    max_val = int(np.max(syndrome_counts))
    bins = np.arange(min_val, max_val + 2) - 0.5 
    
    plt.hist(syndrome_counts, bins=bins, color='darkorange', edgecolor='black', alpha=0.75)
    plt.axvline(x=bypass_threshold + 0.5, color='red', linestyle='dashed', linewidth=2, 
                label=f'Bypass Threshold (N={bypass_threshold})')
    
    plt.title('Distribution of Syndrome Flashes per Shot', fontsize=14)
    plt.xlabel('Number of Active Syndromes (Defects)', fontsize=12)
    plt.ylabel('Frequency (Number of Shots)', fontsize=12)
    
    step = max(1, (max_val - min_val) // 15)
    plt.xticks(np.arange(min_val, max_val + 1, step=step))
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def plot_action_histogram(raw_actions, active_actions, filename='action_histogram.png'):
    """Plots a dual histogram of the raw GNN actions and the actively applied actions."""
    if not raw_actions and not active_actions:
        print("No actions to plot!")
        return
        
    print("\nGenerating action histograms...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(raw_actions, bins=100, color='royalblue', alpha=0.8, log=True)
    plt.title('All Raw Actions (Output by GNN)', fontsize=14)
    plt.xlabel('Action Value (tanh bounded [-1, 1])', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    if active_actions:
        plt.hist(active_actions, bins=100, color='forestgreen', alpha=0.8, log=True)
    plt.title('Active Actions (Where Mask == 1)', fontsize=14)
    plt.xlabel('Action Value (Applied to Graph)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def plot_weight_correlations(wm, p_val=None):
    """Generates a 2x2 plot showing step-by-step evolution of weights over an average episode."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    title_suffix = f" (p={p_val})" if p_val else ""
    fig.suptitle(f"Intra-Episode Evolution{title_suffix}: GNN vs. Zero (CMA) vs. CM\n(Averaged across test episodes with ±1 Std Dev)", fontsize=16)
    
    c_gnn = '#2ca02c'   # Green
    c_zero = '#9467bd'  # Purple
    c_cm = '#ff7f0e'    # Orange 

    def plot_evolution_with_std(ax, data_list, label, color, linestyle='-'):
        if not data_list or len(data_list) == 0:
            return
        data = np.array(data_list)
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        steps = np.arange(1, len(mean_vals) + 1)
        ax.plot(steps, mean_vals, label=label, color=color, linewidth=2, linestyle=linestyle)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

    plot_evolution_with_std(axs[0, 0], wm['mse_sac_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 0], wm['mse_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 0], wm['mse_cm_oracle'], 'Correlated Matching', c_cm, linestyle='--')
    axs[0, 0].set_title('Weight Difference (vs Oracle Weights)')
    axs[0, 0].set_ylabel('Mean Squared Error')
    axs[0, 0].set_xlabel('Step (Shot) within Episode')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    plot_evolution_with_std(axs[0, 1], wm['mse_sac_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 1], wm['mse_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 1], wm['mse_cm_static'], 'Correlated Matching', c_cm, linestyle='--')
    axs[0, 1].set_title('Weight Difference (vs Static Undrifted Graph)')
    axs[0, 1].set_xlabel('Step (Shot) within Episode')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    plot_evolution_with_std(axs[1, 0], wm['p_sac_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 0], wm['p_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 0], wm['p_cm_oracle'], 'Correlated Matching', c_cm, linestyle='--')
    axs[1, 0].set_title('Correlation (vs Oracle)')
    axs[1, 0].set_ylabel('Correlation Coefficient / Error')
    axs[1, 0].set_xlabel('Step (Shot) within Episode')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    plot_evolution_with_std(axs[1, 1], wm['p_sac_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 1], wm['p_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 1], wm['p_cm_static'], 'Correlated Matching', c_cm, linestyle='--')
    axs[1, 1].set_title('Correlation (vs Static Undrifted Graph)')
    axs[1, 1].set_xlabel('Step (Shot) within Episode')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filename = f'weight_evolution_p_{p_val}.png' if p_val else 'weight_evolution_comprehensive.png'
    plt.savefig(f'plots/{filename}', dpi=300)
    print(f"Weight evolution plots saved to 'plots/{filename}'")
    plt.close()


def plot_training_metrics(metrics, config):
    """Generates a 4-panel plot showing training health over episodes."""
    os.makedirs('plots', exist_ok=True)
    episodes = range(1, len(metrics['rewards']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten() 
    
    # Calculate total episodes properly
    total_eps = len(config['curriculum_p']) * config['train_episodes_per_p']
    fig.suptitle(f"SAC-GNN Curriculum Training Metrics ({total_eps} Total Episodes)", fontsize=16, fontweight='bold', y=0.98)
    
    axs[0].plot(episodes, metrics['rewards'], color='blue', linewidth=2)
    axs[0].set_title('Episode Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    if 'alphas' in metrics:
        axs[1].plot(episodes, metrics['alphas'], color='purple', linewidth=2)
        axs[1].set_title('Alpha (Entropy Temperature)')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Alpha Value')
        axs[1].grid(True, linestyle='--', alpha=0.7)
    
    axs[2].plot(episodes, metrics['c_losses'], label='Critic Loss', color='red', linewidth=2)
    axs[2].set_title('Critic Loss (MSE)')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    axs[3].plot(episodes, metrics['a_losses'], label='Actor Loss', color='green', linewidth=2)
    axs[3].set_title('Actor Loss')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Loss')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines to indicate curriculum phase shifts
    phase_length = config['train_episodes_per_p']
    for i in range(1, len(config['curriculum_p'])):
        for ax in axs:
            ax.axvline(x=i * phase_length, color='gray', linestyle=':', alpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    filename = config.get("training_metrics_filename", "training_metrics_curriculum.png")
    plt.savefig(f'plots/{filename}', dpi=300)
    print(f"Training plots saved to 'plots/{filename}'")
    plt.close()


def plot_testing_metrics(test_results, p_val=None):
    """Generates a bar chart and timeline for evaluation metrics for a specific p value."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" at p={p_val}" if p_val else ""
    fig.suptitle(f"Comprehensive Decoder Evaluation{title_suffix}", fontsize=16)
    
    labels = ['SAC_GNN', 'Zero (CMA Only)', 'Static', 'Oracle', 'CM']
    keys = ['sac_gnn', 'zero', 'static', 'oracle', 'cm']
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#d62728', '#ff7f0e']
    
    lers = [test_results[f'ler_{k}'] for k in keys]
    stds = [test_results[f'ler_std_{k}'] for k in keys]
    
    bars = axs[0].bar(labels, lers, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    axs[0].set_title('Logical Error Rate (LER)')
    axs[0].set_ylabel('LER')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        err = stds[i]
        axs[0].text(bar.get_x() + bar.get_width()/2.0, yval + err + (max(lers)*0.02), 
                    f'{yval:.7f}\n±{err:.7f}', ha='center', va='bottom', fontweight='bold')

    # Cumulative plot uses the actual length of the array to plot correctly
    max_len = max([len(test_results[f'cum_{k}']) for k in keys])
    shots = np.arange(1, max_len + 1)
    
    for label, key, color in zip(labels, keys, colors):
        linestyle = '-' if key in ['sac_gnn', 'zero', 'oracle', 'cm'] else '--'
        # Ensure arrays match the x-axis dimension
        cum_arr = test_results[f'cum_{key}']
        if len(cum_arr) > 0:
            axs[1].plot(shots[:len(cum_arr)], cum_arr, label=label, color=color, linewidth=2, linestyle=linestyle)
    
    axs[1].set_title('Cumulative Logical Errors')
    axs[1].set_xlabel('Total Shots Evaluated')
    axs[1].set_ylabel('Cumulative Errors')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filename = f'testing_metrics_p_{p_val}.png' if p_val else 'testing_metrics_comprehensive.png'
    plt.savefig(f'plots/{filename}', dpi=300)
    print(f"Testing plots saved to 'plots/{filename}'")
    plt.close()