import re
import matplotlib.pyplot as plt

def generate_single_dashboard(log_file, target_run="Starting Run 3: Update Period 10", output_filename="sac_gnn_model22_dashboard.png"):
    # Data storage
    episodes = []
    rewards = []
    c_losses = []
    a_losses = []
    alphas = []
    
    # Regex pattern to match the training output lines
    log_pattern = re.compile(
        r"Train Ep:\s+(\d+)/\d+\s+\|\s+Reward:\s+([-\d.]+)\s+\|\s+MSE:\s+[\d.]+\s+\|\s+C_Loss:\s+([\d.]+)\s+\|\s+A_Loss:\s+([\d.]+)\s+\|\s+Alpha:\s+([\d.]+)"
    )

    is_target_run = False

    # 1. Parse the log file
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Detect the start of the specific run we want
                if target_run in line:
                    is_target_run = True
                # Stop parsing if we hit the next run
                elif "Starting Run" in line and is_target_run:
                    break
                
                # Extract data only if we are inside the correct run block
                if is_target_run:
                    match = log_pattern.search(line)
                    if match:
                        episodes.append(int(match.group(1)))
                        rewards.append(float(match.group(2)))
                        c_losses.append(float(match.group(3)))
                        a_losses.append(float(match.group(4)))
                        alphas.append(float(match.group(5)))
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return

    if not episodes:
        print("Error: No data found for the specified run.")
        return

    # 2. Setup the Matplotlib Figure (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'SAC-GNN Training Metrics ({len(episodes)} Episodes)', fontsize=16, fontweight='bold', y=1.02)

    # --- Plot 1: Episode Reward (Blue) ---
    axs[0, 0].plot(episodes, rewards, color='blue', linewidth=2)
    axs[0, 0].set_title('Episode Reward', fontsize=10)
    axs[0, 0].set_xlabel('Episode', fontsize=8)
    axs[0, 0].set_ylabel('Total Reward', fontsize=8)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=8)

    # --- Plot 2: Alpha (Purple) ---
    axs[0, 1].plot(episodes, alphas, color='purple', linewidth=2)
    axs[0, 1].set_title('Alpha (Entropy Temperature)', fontsize=10)
    axs[0, 1].set_xlabel('Episode', fontsize=8)
    axs[0, 1].set_ylabel('Alpha Value', fontsize=8)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=8)

    # --- Plot 3: Critic Loss (Red) ---
    axs[1, 0].plot(episodes, c_losses, color='red', linewidth=2)
    axs[1, 0].set_title('Critic Loss (MSE)', fontsize=10)
    axs[1, 0].set_xlabel('Episode', fontsize=8)
    axs[1, 0].set_ylabel('Loss', fontsize=8)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=8)

    # --- Plot 4: Actor Loss (Green) ---
    axs[1, 1].plot(episodes, a_losses, color='green', linewidth=2)
    axs[1, 1].set_title('Actor Loss', fontsize=10)
    axs[1, 1].set_xlabel('Episode', fontsize=8)
    axs[1, 1].set_ylabel('Loss', fontsize=8)
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=8)

    # Final formatting and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved successfully as '{output_filename}'")
    
    # Optionally display the plot if running in an interactive environment
    # plt.show()

# Run the function
if __name__ == "__main__":
    # Make sure 'sweep_1362718' is in the same directory as this script
    generate_single_dashboard(
        log_file="./scripts/sweep_1368359.out", 
        target_run="Starting Run 1: Update Period 1000", 
        output_filename="training_metrics_23.png"
    )

    generate_single_dashboard(
        log_file="./scripts/sweep_1368359.out", 
        target_run="Starting Run 2: Update Period 100", 
        output_filename="training_metrics_24.png"
    )