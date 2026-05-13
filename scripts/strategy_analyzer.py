import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent

def build_bidirectional_adjacency(line_edge_index):
    """
    Creates a fast lookup dictionary for neighbors and edge attributes.
    Because line_edge_index is [2, M], we make it bidirectional for easy graph traversal.
    """
    adj = {}
    src = line_edge_index[0]
    dst = line_edge_index[1]
    
    for e_idx in range(len(src)):
        u, v = src[e_idx], dst[e_idx]
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, e_idx))
        adj[v].append((u, e_idx))
        
    return adj

def compute_correlated_action(weight_A, weight_B, joint_prob, action_scale, min_weight, max_weight):
    """
    Calculates the exact analytical reweighting that standard Correlated MWPM 
    would apply. The lower bound is unclipped so it can drop below -1.0.
    """
    # W = ln((1-p)/p) -> p = 1 / (1 + exp(W))
    p_A = 1.0 / (1.0 + np.exp(weight_A))
    p_B = 1.0 / (1.0 + np.exp(weight_B))
    
    # Conditional probability P(A|B) = P(A and B) / P(B)
    if p_B > 1e-12:
        p_A_given_B = joint_prob / p_B
    else:
        p_A_given_B = p_A
        
    # Standard correlated MWPM only discounts hyperedges (increases probability)
    p_A_given_B = max(p_A, p_A_given_B) 
    p_A_given_B = np.clip(p_A_given_B, 1e-6, 0.499999)
    
    # Map back to weight
    new_w = np.log((1.0 - p_A_given_B) / p_A_given_B)
    new_w = np.clip(new_w, min_weight, max_weight)
    
    # Calculate delta and normalize to action scale
    delta_w = new_w - weight_A
    
    # Allow the action to go below -1.0, but cap at 1.0 just in case
    return min(delta_w / action_scale, 1.0)


def run_collection(env, agent, config, save_path):
    print(f"[*] Starting Data Collection for {config['n_shots']} shots...")
    
    data_records = []
    obs, info = env.reset(seed=42) # Fixed seed for reproducible analysis
    
    # Extract static topology
    adj = build_bidirectional_adjacency(env.line_edge_index)
    
    burn_in_steps = config.get('burn_in_steps', 0)
    bypass_threshold = config.get('bypass_threshold', 2)
    step_count = 0
    done = False
    
    pbar = tqdm(total=config['n_shots'])
    
    while not done:
        n_flashes = np.sum(env.current_syndrome != 0)
        step_records = [] # Temporary storage for the current step
        
        # Replicate the bypass logic from your engine.py
        if step_count < burn_in_steps or n_flashes <= bypass_threshold:
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
        else:
            action = agent.select_action(obs, evaluate=True)
            
            mask = obs["action_mask"]
            node_feats = obs["node_features"]
            edge_attrs = obs["edge_attr"]
            
            # Only look at nodes where the agent is allowed to act
            active_nodes = np.where(mask == 1.0)[0]
            full_correlated_action = np.zeros(env.n_dec_edges, dtype=np.float32)
            
            for node in active_nodes:
                weight = node_feats[node, 0]
                selected = node_feats[node, 1]  # The Pass 1 Flag
                act = action[node]
                
                correlations_with_selected = []
                weights_of_selected = []
                
                max_corr_overall = -1.0
                weight_of_highest_corr_neighbor = None
                
                # --- ANALYTICAL CORRELATED MWPM CALCULATION ---
                correlated_act = 0.0
                
                # Fetch 1-hop neighbor data
                for nbr, e_idx in adj.get(node, []):
                    nbr_weight = node_feats[nbr, 0]
                    nbr_selected = node_feats[nbr, 1]
                    corr = edge_attrs[e_idx, 0]
                    joint_prob = env.corr_tracer[e_idx] # Exact physics tracer
                    
                    if nbr_selected == 1.0:
                        correlations_with_selected.append(corr)
                        weights_of_selected.append(nbr_weight)
                        
                        # Calculate the analytical math action (unbounded negative)
                        c_act = compute_correlated_action(
                            weight, nbr_weight, joint_prob, 
                            config['action_scale'], env.min_weight, env.max_weight
                        )
                        # We take the minimum (largest discount) across all selected neighbors
                        correlated_act = min(correlated_act, c_act)
                        
                    if corr > max_corr_overall:
                        max_corr_overall = corr
                        weight_of_highest_corr_neighbor = nbr_weight
                        
                full_correlated_action[node] = correlated_act
                
                # Summarize selected neighbors
                num_selected_neighbors = len(correlations_with_selected)
                max_corr_to_selected = np.max(correlations_with_selected) if num_selected_neighbors > 0 else 0.0
                
                # Store the row (Waiting for env.step to determine the logical outcome)
                step_records.append({
                    "step": step_count,
                    "node_id": node,
                    "base_weight": weight,
                    "is_selected": selected,
                    "action_taken": act,
                    "correlated_action_taken": correlated_act,
                    "num_selected_neighbors": num_selected_neighbors,
                    "weights_of_selected_neighbors": str(weights_of_selected), 
                    "corrs_with_selected_neighbors": str(correlations_with_selected),
                    "max_corr_to_selected_neighbor": max_corr_to_selected,
                    "weight_of_highest_corr_neighbor": weight_of_highest_corr_neighbor
                })
                
            # -------------------------------------------------------------
            # PARALLEL DECODING PASS: Get the True Correlated MWPM Outcome
            # -------------------------------------------------------------
            applied_corr_delta = full_correlated_action * config['action_scale']
            if config['local_action_only']:
                applied_corr_delta *= mask
                
            if not np.any(applied_corr_delta):
                corr_pred_obs = env.current_first_pass_pred_obs
            else:
                corr_second_pass_weights = np.clip(env.current_weights + applied_corr_delta, env.min_weight, env.max_weight)
                corr_edge_reweights = env._build_edge_reweights(corr_second_pass_weights)
                
                _, corr_pred_obs = env.syndrome_data_generator.get_solution_edges(
                    matching=env.current_matching,
                    syndrome_volume=env.current_syndrome,
                    enable_correlations=False,
                    edge_reweights=corr_edge_reweights,
                    return_predicted_obs=True,
                    fault_array=env.fault_array,
                )
                
        # Step the environment (happens for both active and bypassed shots)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # If we recorded GNN actions this step, evaluate if they helped or hurt
        if len(step_records) > 0:
            true_obs = step_info["true_obs"]
            pass1_correct = (step_info["first_pass_obs"] == true_obs)
            pass2_correct = (step_info["pred_obs"] == true_obs)
            corr_correct = (corr_pred_obs == true_obs)
            
            # GNN Outcome
            if pass2_correct and not pass1_correct:
                logical_outcome = "fixed"
            elif pass1_correct and not pass2_correct:
                logical_outcome = "broken"
            else:
                logical_outcome = "unchanged"
                
            # Correlated MWPM Outcome
            if corr_correct and not pass1_correct:
                corr_logical_outcome = "fixed"
            elif pass1_correct and not corr_correct:
                corr_logical_outcome = "broken"
            else:
                corr_logical_outcome = "unchanged"
                
            # STRICT FILTER: Only save if the GNN action changed the logical outcome
            if logical_outcome in ["fixed", "broken"]:
                for record in step_records:
                    record["logical_outcome"] = logical_outcome
                    record["corr_logical_outcome"] = corr_logical_outcome
                    data_records.append(record)
                
        done = terminated or truncated
        obs = next_obs
        step_count += 1
        pbar.update(1)
        
    pbar.close()
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(data_records)
    df.to_csv(save_path, index=False)
    print(f"[*] Successfully saved {len(df)} action records to {save_path} (filtered for fixes/breaks)")


def run_analysis(data_path, output_dir):
    print(f"[*] Loading dataset from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find {data_path}")
        return

    if len(df) == 0:
        print("[!] Dataset is empty. The agent did not fix or break any shots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # ---------------------------------------------------------
    # SHOT-LEVEL OUTCOME SUMMARY
    # ---------------------------------------------------------
    if "corr_logical_outcome" in df.columns:
        print("\n" + "="*65)
        print("   SHOT-LEVEL OUTCOMES (Compared to Standard MWPM)")
        print("="*65)
        
        # Drop duplicates by step to count actual shots, not individual edges
        shot_outcomes = df.drop_duplicates(subset=["step"])
        
        # Count the outcomes for both decoders
        gnn_counts = shot_outcomes["logical_outcome"].value_counts()
        corr_counts = shot_outcomes["corr_logical_outcome"].value_counts()
        
        # Build a direct comparison DataFrame
        summary_df = pd.DataFrame({
            "Neural Correlated (SAC-GNN)": gnn_counts,
            "Analytical Correlated (Cond. prob.)": corr_counts
        }).fillna(0).astype(int)
        
        # Ensure rows are ordered logically and exist even if 0
        for idx in ["fixed", "broken", "unchanged"]:
            if idx not in summary_df.index:
                summary_df.loc[idx] = [0, 0]
        
        # Order the rows
        summary_df = summary_df.loc[["fixed", "broken", "unchanged"]]
        
        # Drop the 'unchanged' row if it is completely empty to keep the table clean
        if summary_df.loc["unchanged"].sum() == 0:
            summary_df = summary_df.drop(index="unchanged")
            
        # Add a Total row at the bottom
        summary_df.loc["Total"] = summary_df.sum()
        
        # Name the index for the final printout
        summary_df.index.name = "Outcome"
        
        try:
            print(summary_df.to_markdown())
        except ImportError:
            print(summary_df.to_string()) # Fallback if tabulate isn't installed
            
        print("="*65 + "\n")

    # Separate the data into Pass-1 edges and Neighborhood edges
    df_neighbors = df[df["is_selected"] == 0.0].copy()
    df_direct = df[df["is_selected"] == 1.0].copy()
    
    if len(df_neighbors) == 0:
        print("[!] No unselected neighbor actions found in the dataset.")
        return

    # ---------------------------------------------------------
    # Plot 1: Action Distribution (Agent vs. Correlated MWPM)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # 1. GNN Agent Actions (Solid Fill)
    sns.kdeplot(
        data=df, x="action_taken", hue="logical_outcome", 
        fill=True, common_norm=False, 
        palette={"fixed": "seagreen", "broken": "indianred"}
    )

    # 2. Correlated MWPM Actions (Dashed Outlines)
    # Assumes you have a column tracking the correlated baseline's outcome.
    sns.kdeplot(
        data=df, x="correlated_action_taken", hue="corr_logical_outcome", 
        fill=False, common_norm=False, linestyle="--", linewidth=2.5, 
        palette={"fixed": "mediumseagreen", "broken": "lightcoral", "unchanged": "dimgray"}
    )

    plt.title("Action Distribution: GNN Agent vs. Correlated MWPM Baseline")
    plt.xlabel("Action Value (Negative = Discount/Highway)")
    plt.ylabel("Density")

    # Fix the legend so it clearly separates the two models
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Outcomes", loc="upper right")

    plt.savefig(os.path.join(output_dir, "xai_1_action_distribution_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Rescue Trigger (Correlation vs. Action Overlay)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_correlated = df_neighbors[df_neighbors["max_corr_to_selected_neighbor"] > 0]
    
    sns.scatterplot(
        data=df_correlated, x="max_corr_to_selected_neighbor", y="action_taken",
        hue="logical_outcome", alpha=0.3, palette={"fixed": "seagreen", "broken": "indianred"}
    )
    sns.scatterplot(
        data=df_correlated, x="max_corr_to_selected_neighbor", y="correlated_action_taken",
        color="black", marker="x", s=50, alpha=0.8, label="Correlated MWPM (Math)"
    )
    
    plt.title("Rescue Trigger: GNN Policy vs. Correlated Math Curve")
    plt.xlabel("Max Pearson Correlation to a Pass-1 Selected Neighbor")
    plt.ylabel("Action Value")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "xai_2_rescue_correlation_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Relative Weight Comparison (1x2 Subplots)
    # ---------------------------------------------------------
    df_fixed = df_neighbors[df_neighbors["logical_outcome"] == "fixed"].copy()
    if len(df_fixed) > 0:
        df_fixed_valid = df_fixed[df_fixed["weight_of_highest_corr_neighbor"].notnull()]
        
        # Calculate global bounds for color scaling in Plot 3 encompassing both metrics
        global_min_p3 = min(df_fixed_valid["action_taken"].min(), df_fixed_valid["correlated_action_taken"].min())
        global_max_p3 = max(df_fixed_valid["action_taken"].max(), df_fixed_valid["correlated_action_taken"].max())

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
        
        # Left: Agent
        sc1 = axes[0].scatter(
            df_fixed_valid["weight_of_highest_corr_neighbor"], df_fixed_valid["base_weight"],
            c=df_fixed_valid["action_taken"], cmap="viridis", alpha=0.7, s=40, edgecolor="w",
            vmin=global_min_p3, vmax=global_max_p3
        )
        axes[0].set_title("GNN Agent Rescues")
        axes[0].set_xlabel("Base Weight of Selected Neighbor")
        axes[0].set_ylabel("Base Weight of Unselected Node")
        
        # Right: Correlated MWPM
        sc2 = axes[1].scatter(
            df_fixed_valid["weight_of_highest_corr_neighbor"], df_fixed_valid["base_weight"],
            c=df_fixed_valid["correlated_action_taken"], cmap="viridis", alpha=0.7, s=40, edgecolor="w",
            vmin=global_min_p3, vmax=global_max_p3
        )
        axes[1].set_title("Correlated MWPM Equivalents")
        axes[1].set_xlabel("Base Weight of Selected Neighbor")

        # Formatting
        for ax in axes:
            max_val = max(df_fixed_valid["weight_of_highest_corr_neighbor"].max(), df_fixed_valid["base_weight"].max())
            min_val = min(df_fixed_valid["weight_of_highest_corr_neighbor"].min(), df_fixed_valid["base_weight"].min())
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', alpha=0.5, label="Equal Weight Line")
            ax.legend()
        
        cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label(f"Action Value ({global_min_p3:.2f} to {global_max_p3:.2f})")
        
        plt.suptitle("Successful Rescues: Base Weight Matrix Comparison", y=1.02, fontsize=16)
        plt.savefig(os.path.join(output_dir, "xai_3_weight_comparison_landscape.png"), bbox_inches="tight", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 4: Action Heatmap (2x2 Matrix: Agent vs Correlated)
    # ---------------------------------------------------------
    df_d_valid = df_direct[df_direct["weight_of_highest_corr_neighbor"].notnull()]
    df_n_valid = df_neighbors[df_neighbors["weight_of_highest_corr_neighbor"].notnull()]

    # Global min and max now factor in BOTH the agent's actions and the unbounded Correlated Math
    global_min_p4 = min(
        df_d_valid["action_taken"].min(), df_n_valid["action_taken"].min(),
        df_d_valid["correlated_action_taken"].min(), df_n_valid["correlated_action_taken"].min()
    )
    global_max_p4 = max(
        df_d_valid["action_taken"].max(), df_n_valid["action_taken"].max(),
        df_d_valid["correlated_action_taken"].max(), df_n_valid["correlated_action_taken"].max()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, sharex=True)
    cmap_choice = "plasma" 
    
    # --- ROW 1: GNN AGENT ---
    axes[0, 0].hexbin(
        df_d_valid["max_corr_to_selected_neighbor"], df_d_valid["weight_of_highest_corr_neighbor"],
        C=df_d_valid["action_taken"], reduce_C_function=np.mean, gridsize=25,
        cmap=cmap_choice, vmin=global_min_p4, vmax=global_max_p4, mincnt=1, edgecolors='none'
    )
    axes[0, 0].set_title("GNN Agent: Selected Edges (Direct)")
    axes[0, 0].set_ylabel("Base Weight of Adjacent Edge")

    axes[0, 1].hexbin(
        df_n_valid["max_corr_to_selected_neighbor"], df_n_valid["weight_of_highest_corr_neighbor"],
        C=df_n_valid["action_taken"], reduce_C_function=np.mean, gridsize=25,
        cmap=cmap_choice, vmin=global_min_p4, vmax=global_max_p4, mincnt=1, edgecolors='none'
    )
    axes[0, 1].set_title("GNN Agent: Unselected Edges (Neighborhood)")

    # --- ROW 2: CORRELATED MWPM ---
    axes[1, 0].hexbin(
        df_d_valid["max_corr_to_selected_neighbor"], df_d_valid["weight_of_highest_corr_neighbor"],
        C=df_d_valid["correlated_action_taken"], reduce_C_function=np.mean, gridsize=25,
        cmap=cmap_choice, vmin=global_min_p4, vmax=global_max_p4, mincnt=1, edgecolors='none'
    )
    axes[1, 0].set_title("Correlated MWPM: Selected Edges")
    axes[1, 0].set_xlabel("Max Correlation with Selected Edge")
    axes[1, 0].set_ylabel("Base Weight of Adjacent Edge")

    hb_corr = axes[1, 1].hexbin(
        df_n_valid["max_corr_to_selected_neighbor"], df_n_valid["weight_of_highest_corr_neighbor"],
        C=df_n_valid["correlated_action_taken"], reduce_C_function=np.mean, gridsize=25,
        cmap=cmap_choice, vmin=global_min_p4, vmax=global_max_p4, mincnt=1, edgecolors='none'
    )
    axes[1, 1].set_title("Correlated MWPM: Unselected Edges")
    axes[1, 1].set_xlabel("Max Correlation with Selected Edge")

    cbar = fig.colorbar(hb_corr, ax=axes.ravel().tolist(), aspect=40, pad=0.02)
    cbar.set_label(f"Action Intensity ({global_min_p4:.2f} to {global_max_p4:.2f})")
    
    plt.suptitle("Heatmap Matrix: GNN Strategy vs. Correlated MWPM Analytical Logic", fontsize=18, y=0.95)
    plt.savefig(os.path.join(output_dir, "xai_4_action_heatmap_comparison.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 5: Action Correlation (GNN vs Analytical Math)
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 9))
    
    # Filter out baseline 0s to zoom in on where the agent actually acts
    df_action_valid = df_neighbors[(df_neighbors["max_corr_to_selected_neighbor"] > 0)].copy()

    sns.scatterplot(
        data=df_action_valid, 
        x="correlated_action_taken", 
        y="action_taken", 
        hue="logical_outcome",
        alpha=0.6,
        s=35,
        palette={"fixed": "seagreen", "broken": "indianred"}
    )
    
    # Calculate limits to draw a clean diagonal
    min_val_p5 = min(df_action_valid["correlated_action_taken"].min(), df_action_valid["action_taken"].min())
    max_val_p5 = max(df_action_valid["correlated_action_taken"].max(), df_action_valid["action_taken"].max())
    
    # Perfect Agreement Line
    plt.plot([min_val_p5, max_val_p5], [min_val_p5, max_val_p5], color='black', linestyle='--', label='Perfect Agreement (y=x)')
    
    # Trendline for the GNN's actual behavior
    sns.regplot(
        data=df_action_valid, 
        x="correlated_action_taken", 
        y="action_taken", 
        scatter=False, 
        color='blue', 
        line_kws={'linestyle': 'dotted', 'linewidth': 2},
        label='GNN Trend'
    )
    
    plt.title("Action Space Alignment: GNN Output vs. Analytical Correlated Math")
    plt.xlabel("Analytical Action (Correlated MWPM)")
    plt.ylabel("GNN Agent Action")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(os.path.join(output_dir, "xai_5_action_correlation_scatter.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 6: Logical Outcome Agreement Heatmap (Confusion Matrix)
    # ---------------------------------------------------------
    if "corr_logical_outcome" in df.columns:
        plt.figure(figsize=(7, 6))
        
        # Drop duplicates by step to count actual shots, not individual edges
        shot_outcomes = df.drop_duplicates(subset=["step"])[["logical_outcome", "corr_logical_outcome"]].copy()
        
        # Define categories to ensure a stable matrix shape even if some outcomes (like 'unchanged') are 0
        valid_outcomes = ["broken", "fixed"]
        
        # Only add 'unchanged' to the matrix if it actually appears in the dataset
        if "unchanged" in shot_outcomes["logical_outcome"].unique() or "unchanged" in shot_outcomes["corr_logical_outcome"].unique():
            valid_outcomes.append("unchanged")
            
        shot_outcomes["logical_outcome"] = pd.Categorical(shot_outcomes["logical_outcome"], categories=valid_outcomes)
        shot_outcomes["corr_logical_outcome"] = pd.Categorical(shot_outcomes["corr_logical_outcome"], categories=valid_outcomes)
        
        # Create the cross-tabulation matrix
        cm = pd.crosstab(
            shot_outcomes["logical_outcome"], 
            shot_outcomes["corr_logical_outcome"], 
            dropna=False
        )
        
        # Plot the heatmap
        sns.heatmap(
            cm, 
            annot=True,       # Write the number of shots inside the boxes
            fmt="d",          # Format as integer
            cmap="Blues",     # Darker blue = more shots
            cbar=False,       # Hide the colorbar since the numbers are written inside
            linewidths=1, 
            linecolor='white',
            square=True,
            annot_kws={"size": 14, "weight": "bold"}
        )
        
        plt.title("Decoder Agreement Matrix:\nSAC-GNN vs. Analytical Correlated MWPM", fontsize=14, pad=15)
        plt.xlabel("Analytical Correlated MWPM (Cond. prob.)", fontsize=12, labelpad=10)
        plt.ylabel("Neural Correlated (SAC-GNN)", fontsize=12, labelpad=10)
        
        # Move the X-axis labels to the top for a cleaner look
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top') 
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "xai_6_outcome_agreement_matrix.png"), dpi=300)
        plt.close()

    print(f"[*] Analysis complete. Interpretive plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Analyze GNN Agent Policy")
    parser.add_argument("--mode", type=str, choices=["collect", "analyze", "both"], default="both",
                        help="Choose whether to collect data, analyze a CSV, or both.")
    parser.add_argument("--shots", type=int, default=10_000_000, help="Number of shots to evaluate")
    parser.add_argument("--data_path", type=str, default="data/gnn_actions_p1e-3_1Ms_mode64b_m1_d7.csv", help="Path to save/load the CSV data")
    parser.add_argument("--plot_dir", type=str, default="plots/strategy_analysis/model_64b_p1e-3_m1_d7/", help="Directory to save plots")
    
    args = parser.parse_args()
    
    CONFIG = {
        'distance': 7,
        'n_rounds': 7,
        'p': 0.005,
        'p_gate_zz': 0.0,
        'mismatch': 1.0,
        'n_shots': args.shots,
        'n_test_shots': 0,
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 5.0,
        'update_period': 20_000_000,
        'prior_shots': 1_000,
        'local_action_only': True,
        'local_action_hops': 1,
        'use_pearson_correlation': True,
        'use_log_joint_prob': False,
        'n_layers': 1,
        'hidden_dim': 256,
        'lr': 1e-4,
        'gamma': 0.0,
        'tau': 0.005,
        'alpha': 0.01,
        'target_entropy': -1.0,
        'model_path': 'models/sac_gnn_64_best.pth'
    }
    
    if args.mode in ["collect", "both"]:
        print("Initializing environment...")
        generator = SyndromeDataGenerator(
            distance=CONFIG['distance'], n_rounds=CONFIG['n_rounds'], mismatch=CONFIG['mismatch'],  
            noise_model={"version": "built-in", "after_clifford_depolarization": CONFIG["p"],
                         "before_measure_flip_probability": CONFIG["p"], "after_reset_flip_probability": CONFIG["p"],
                         "before_round_data_depolarization": CONFIG["p"], "p_gate_zz": CONFIG["p_gate_zz"]}, 
            memory_type='z', n_shots=CONFIG['n_shots'], qec_code='surface_code'
        )

        env = DriftedMatchingEnv(
            syndrome_data_generator=generator, local_action_only=CONFIG['local_action_only'],
            local_action_hops=CONFIG['local_action_hops'], action_scale=CONFIG['action_scale'],
            update_period=CONFIG['update_period'], prior_shots=CONFIG['prior_shots'],
            n_test_shots=CONFIG['n_test_shots'], use_pearson_correlation=CONFIG['use_pearson_correlation'],
            use_log_joint_prob=CONFIG['use_log_joint_prob'], use_syndrome_features=False, update_with='DGR', train_mode=False
        )

        sample_obs, _ = env.reset()
        NODE_DIM = sample_obs["node_features"].shape[1]
        
        agent = SACAgent(
            node_dim=NODE_DIM, hidden_dim=CONFIG['hidden_dim'], static_edge_index=env.line_edge_index,
            lr=CONFIG['lr'], gamma=CONFIG['gamma'], tau=CONFIG['tau'], alpha=CONFIG['alpha'], n_layers=CONFIG['n_layers'], target_entropy=CONFIG['target_entropy']
        )
        
        try:
            agent.load_models(CONFIG['model_path'])
            print(f"Successfully loaded model from {CONFIG['model_path']}")
        except Exception as e:
            print(f"[!] Warning: Could not load model: {e}")
            
        run_collection(env, agent, CONFIG, args.data_path)

    if args.mode in ["analyze", "both"]:
        if os.path.exists(args.data_path):
            run_analysis(args.data_path, args.plot_dir)
        else:
            print(f"[!] Error: Data file {args.data_path} not found. Run --mode collect first.")