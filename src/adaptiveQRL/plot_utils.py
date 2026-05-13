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
    import matplotlib.pyplot as plt
    
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
    plt.savefig(filename, dpi=300)
    plt.show()
    


def plot_syndrome_count_histogram(syndrome_counts, bypass_threshold=4, filename='syndrome_counts_hist.png'):
    """
    Plots a histogram of the number of active syndrome flashes per shot.
    Centers the bins on integers and marks the GNN bypass threshold.
    """
    if not len(syndrome_counts):
        print("No syndrome data to plot!")
        return
        
    print("\nGenerating syndrome count histogram...")
    plt.figure(figsize=(10, 6))
    
    # Calculate integer bins so the bars align perfectly with discrete counts
    min_val = int(np.min(syndrome_counts))
    max_val = int(np.max(syndrome_counts))
    bins = np.arange(min_val, max_val + 2) - 0.5 # Shift by 0.5 to center bars on integers
    
    # Plot the histogram
    plt.hist(syndrome_counts, bins=bins, color='darkorange', edgecolor='black', alpha=0.75)
    
    # Add a vertical line for the proposed GNN bypass threshold
    plt.axvline(x=bypass_threshold + 0.5, color='red', linestyle='dashed', linewidth=2, 
                label=f'Bypass Threshold (N={bypass_threshold})')
    
    plt.title('Distribution of Syndrome Flashes per Shot', fontsize=14)
    plt.xlabel('Number of Active Syndromes (Defects)', fontsize=12)
    plt.ylabel('Frequency (Number of Shots)', fontsize=12)
    
    # Ensure x-axis ticks are integers so it looks clean
    step = max(1, (max_val - min_val) // 15)
    plt.xticks(np.arange(min_val, max_val + 1, step=step))
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved syndrome count histogram to '{filename}'")


def plot_tracer_evolution_histograms(initial_joint, final_joint, initial_pearson, final_pearson, filename='./plots/tracer_histograms.png'):
    """
    Plots overlaid histograms comparing the initial DEM physics to the learned CMA tracers.
    """
    if len(initial_joint) == 0:
        print("No line graph edges found. Cannot plot correlations.")
        return

    print("\nGenerating tracer evolution histograms...")
    plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Joint Probabilities ---
    plt.subplot(1, 2, 1)
    # Use alpha=0.6 for transparent overlays
    plt.hist(initial_joint, bins=100, alpha=0.6, label='Initial (Stim DEM)', color='slategray')
    plt.hist(final_joint, bins=100, alpha=0.6, label='Final (CMA Learned)', color='darkorange')
    plt.title('Line Graph: Joint Probabilities', fontsize=14)
    plt.xlabel('Probability (P_AB)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.yscale('log') # Crucial for seeing the rare, high-correlation hyperedges
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: Pearson Correlations ---
    plt.subplot(1, 2, 2)
    plt.hist(initial_pearson, bins=100, alpha=0.6, label='Initial (Stim DEM)', color='slategray')
    plt.hist(final_pearson, bins=100, alpha=0.6, label='Final (CMA Learned)', color='crimson')
    plt.title('Line Graph: Pearson Correlations', fontsize=14)
    plt.xlabel('Correlation Coefficient [0, 1]', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved tracer histograms to '{filename}'")

def plot_action_histogram(raw_actions, active_actions, filename='action_histogram.png'):
    """
    Plots a dual histogram of the raw GNN actions and the actively applied actions.
    """
    if not raw_actions and not active_actions:
        print("No actions to plot!")
        return
        
    print("\nGenerating action histograms...")
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Every single action output by the Actor (The full 3000+ vector)
    plt.subplot(1, 2, 1)
    plt.hist(raw_actions, bins=100, color='royalblue', alpha=0.8, log=True)
    plt.title('All Raw Actions (Output by GNN)', fontsize=14)
    plt.xlabel('Action Value (tanh bounded [-1, 1])', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Only the actions that were applied (Where action_mask == 1)
    plt.subplot(1, 2, 2)
    if active_actions:
        plt.hist(active_actions, bins=100, color='forestgreen', alpha=0.8, log=True)
    plt.title('Active Actions (Where Mask == 1)', fontsize=14)
    plt.xlabel('Action Value (Applied to Graph)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved action histogram to '{filename}'")


def plot_weight_correlations(wm):
    """Generates a 2x2 plot showing step-by-step evolution of weights over an average episode."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Intra-Episode Evolution: GNN vs. Zero (CMA) vs. Random\n(Averaged across test episodes with ±1 Std Dev)", fontsize=16)
    
    # Colors matching the bar chart
    c_gnn = '#2ca02c'   # Green
    c_zero = '#9467bd'  # Purple
    c_cm = '#ff7f0e'    # Orange 

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
    plot_evolution_with_std(axs[0, 0], wm['mse_sac_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 0], wm['mse_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 0], wm['mse_cm_oracle'], 'Correlated Matching', c_cm, linestyle='--')
    axs[0, 0].set_title('Weight Difference (vs Oracle Weights)')
    axs[0, 0].set_ylabel('Mean Squared Error')
    axs[0, 0].set_xlabel('Step (Shot) within Episode')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: MSE vs Static
    plot_evolution_with_std(axs[0, 1], wm['mse_sac_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[0, 1], wm['mse_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[0, 1], wm['mse_cm_static'], 'Correlated Matching', c_cm, linestyle='--')
    axs[0, 1].set_title('Weight Difference (vs Static Undrifted Graph)')
    axs[0, 1].set_xlabel('Step (Shot) within Episode')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # --- ROW 2: CORRELATIONS ---
    # Plot 3: Correlation vs Oracle
    plot_evolution_with_std(axs[1, 0], wm['p_sac_gnn_oracle'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 0], wm['p_zero_oracle'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 0], wm['p_cm_oracle'], 'Correlated Matching', c_cm, linestyle='--')
    axs[1, 0].set_title('Correlation (vs Oracle)')
    axs[1, 0].set_ylabel('Correlation Coefficient / Error')
    axs[1, 0].set_xlabel('Step (Shot) within Episode')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Correlation vs Static
    plot_evolution_with_std(axs[1, 1], wm['p_sac_gnn_static'], 'GNN', c_gnn)
    plot_evolution_with_std(axs[1, 1], wm['p_zero_static'], 'Zero (CMA)', c_zero)
    plot_evolution_with_std(axs[1, 1], wm['p_cm_static'], 'Correlated Matching', c_cm, linestyle='--')
    axs[1, 1].set_title('Correlation (vs Static Undrifted Graph)')
    axs[1, 1].set_xlabel('Step (Shot) within Episode')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/weight_evolution_comprehensive.png', dpi=300)
    print("Weight evolution plots saved to 'plots/weight_evolution_comprehensive.png'")
    plt.close()
    

import os
import matplotlib.pyplot as plt

def plot_training_metrics(metrics, config):
    """Generates a 4-panel plot showing training health over episodes."""
    os.makedirs('plots', exist_ok=True)
    
    episodes = range(1, len(metrics['rewards']) + 1)
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()  # Flattens the 2x2 array into 1D for easier indexing
    
    fig.suptitle(f"SAC-GNN Training Metrics ({config['train_episodes']} Episodes)", fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1 (Top-Left): Total Reward
    axs[0].plot(episodes, metrics['rewards'], color='blue', linewidth=2)
    axs[0].set_title('Episode Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2 (Top-Right): Alpha Evolution
    # (Assuming you are storing alpha values in metrics['alphas'])
    if 'alphas' in metrics:
        axs[1].plot(episodes, metrics['alphas'], color='purple', linewidth=2)
        axs[1].set_title('Alpha (Entropy Temperature)')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Alpha Value')
        axs[1].grid(True, linestyle='--', alpha=0.7)
    else:
        axs[1].text(0.5, 0.5, "Alpha metrics not found in dictionary", ha='center', va='center')
        axs[1].set_title('Alpha (Entropy Temperature)')
    
    # Plot 3 (Bottom-Left): Critic Loss
    axs[2].plot(episodes, metrics['c_losses'], label='Critic Loss', color='red', linewidth=2)
    axs[2].set_title('Critic Loss (MSE)')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # Plot 4 (Bottom-Right): Actor Loss
    axs[3].plot(episodes, metrics['a_losses'], label='Actor Loss', color='green', linewidth=2)
    axs[3].set_title('Actor Loss')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Loss')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Give the suptitle a bit of breathing room
    
    plt.savefig(f'plots/{config["training_metrics_filename"]}', dpi=300)
    print(f"Training plots saved to 'plots/{config['training_metrics_filename']}'")
    plt.close()


def plot_testing_metrics(test_results):
    """Generates a bar chart and timeline for all evaluation metrics with standard deviations."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comprehensive Decoder Evaluation (Ablation Study)", fontsize=16)
    
    # Define the 5 categories
    labels = ['SAC_GNN', 'Zero (CMA Only)', 'Static', 'Oracle', 'CM']
    keys = ['sac_gnn', 'zero', 'static', 'oracle', 'cm']
    
    # Colors: Green (Win), Blue (Target), Purple (Ablation), Red (Baseline)
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#d62728', '#ff7f0e']
    
    lers = [test_results[f'ler_{k}'] for k in keys]
    stds = [test_results[f'ler_std_{k}'] for k in keys] # ADDED
    
    # Plot 1: Bar Chart of Logical Error Rates (ADDED yerr and capsize)
    bars = axs[0].bar(labels, lers, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    axs[0].set_title('Logical Error Rate (LER)')
    axs[0].set_ylabel('LER')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Give the annotations vertical breathing room so they don't get clipped
    max_top = max(y + s for y, s in zip(lers, stds)) if lers else 1.0
    axs[0].set_ylim(top=max_top * 1.30)

    # Annotate bars with mean ± binomial std using a shared scientific exponent for readability
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        err = stds[i]
        if yval > 0:
            exp = int(np.floor(np.log10(yval)))
            mant_y = yval / (10 ** exp)
            mant_e = err / (10 ** exp)
            text = f'({mant_y:.2f} ± {mant_e:.2f})\n×10$^{{{exp}}}$'
        else:
            text = f'0\n±{err:.1e}'
        axs[0].text(bar.get_x() + bar.get_width() / 2.0,
                    yval + err + max_top * 0.02,
                    text, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Cumulative Errors Over Time
    shots = np.arange(1, len(test_results['cum_sac_gnn']) + 1)
    
    for label, key, color in zip(labels, keys, colors):
        # Use dashed/dotted lines for the static baselines to make the GNN stand out
        linestyle = '-' if key in ['sac_gnn', 'zero', 'oracle', 'cm'] else '--'
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