"""
Main execution script for training and testing the SAC-GNN QEC Decoder.
It handles hyperparameters, execution modes, and performance visualization.
"""

import time
import numpy as np

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.plot_utils import *
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


def validate(env, agent, config):
    """Rigorous isolated validation loop (Pure CMA vs. GNN) with fixed seeds."""
    print(f"  [!] Running Isolated Validation (Pure CMA vs GNN)...")

    val_episodes = config.get('val_episodes', 10)
    # Fixed deterministic seeds so both policies face identical circuits, and so
    # the metric is comparable across training episodes and across Optuna trials.
    val_seeds = [10001 + i for i in range(val_episodes)]
    
    val_burn_in = config.get('burn_in_steps', 15000)
    val_eval_shots = config.get('n_shots', 65000) - val_burn_in 
    bypass_threshold = config.get('bypass_threshold', 2)
    
    # Temporarily override the env's max steps
    original_max_steps = env.max_steps
    env.max_steps = val_burn_in + val_eval_shots
    
    total_zero_errors = 0
    total_gnn_errors = 0
    
    # ========================================================
    # PHASE 1: Pure Baseline (Zero Action) Evaluation
    # ========================================================
    for ep_idx in range(val_episodes):
        obs, _ = env.reset(seed=val_seeds[ep_idx]) 
        done = False
        step_count = 0
        
        while not done:
            # Force the Zero Action for the entire episode to build healthy CMA trackers
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if step_count >= val_burn_in:
                # If action is zero, pred_obs is effectively the baseline
                if info["pred_obs"] != info["true_obs"]:
                    total_zero_errors += 1
                    
            obs = next_obs
            step_count += 1

    # ========================================================
    # PHASE 2: GNN Evaluation
    # ========================================================
    for ep_idx in range(val_episodes):
        obs, _ = env.reset(seed=val_seeds[ep_idx]) 
        done = False
        step_count = 0
        
        while not done:
            n_flashes = np.sum(env.current_syndrome != 0)
            
            if step_count < val_burn_in or n_flashes <= bypass_threshold:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
            else:
                action = agent.select_action(obs, evaluate=True) 
                
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if step_count >= val_burn_in:
                if info["pred_obs"] != info["true_obs"]:
                    total_gnn_errors += 1
                    
            obs = next_obs
            step_count += 1
            
    # Restore original environment settings
    env.max_steps = original_max_steps
    
    # Calculate True Relative Improvement
    net_errors_saved = total_zero_errors - total_gnn_errors
    relative_improvement = (net_errors_saved / total_zero_errors) if total_zero_errors > 0 else 0.0
    
    print(f"  -> Val Results: Pure Baseline Errors: {total_zero_errors} | GNN Errors: {total_gnn_errors}")
    print(f"  -> Relative Improvement: {relative_improvement * 100:.3f}%")
    
    return relative_improvement


def evaluate_low_p_oracle(
    agent,
    config,
    p_eval: float = 0.001,
    mismatch_eval: float = 1.0,
    n_eval_shots: int = 4_000_000,
    shots_per_episode: int = 4_000_000,
    seed: int = 20001,
):
    """Optuna-objective evaluation on a fresh env: low-p, no drift, oracle init.

    Builds a brand-new DriftedMatchingEnv with `mismatch=mismatch_eval`,
    `start_from_oracle=True` (which also disables the CMA tracer alignment),
    `update_period` larger than the episode length (no in-episode updates), and
    `train_mode=False`. Runs the trained `agent` together with the env's
    built-in MWPM and correlated-matching baselines and reports per-policy LERs.

    Returns a dict with `neural_ler`, `corr_ler`, `static_ler`, plus error
    counts and the total number of shots evaluated.
    """
    print(
        f"  [!] Running Optuna-target LER eval (p={p_eval}, "
        f"mismatch={mismatch_eval}, oracle init, no alignment reweighting)..."
    )

    bypass_threshold = config.get("bypass_threshold", 2)
    distance = config.get("distance", 5)
    n_rounds = config.get("n_rounds", 5)
    p_gate_zz = config.get("p_gate_zz", 0.0)
    local_action_hops = config.get("local_action_hops", 1)
    action_scale = config.get("action_scale", 5.0)
    prior_shots = config.get("prior_shots", 1_000)
    use_endpoint_firing = config.get("use_endpoint_firing", False)

    n_episodes = max(1, (n_eval_shots + shots_per_episode - 1) // shots_per_episode)
    frozen_update_period = shots_per_episode + 1  # ensure no in-episode tracker update

    generator = SyndromeDataGenerator(
        distance=distance,
        n_rounds=n_rounds,
        mismatch=mismatch_eval,
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": p_eval,
            "before_measure_flip_probability": p_eval,
            "after_reset_flip_probability": p_eval,
            "before_round_data_depolarization": p_eval,
            "p_gate_zz": p_gate_zz,
        },
        memory_type="z",
        n_shots=shots_per_episode,
        qec_code="surface_code",
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=local_action_hops,
        action_scale=action_scale,
        update_period=frozen_update_period,
        prior_shots=prior_shots,
        n_test_shots=0,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        use_endpoint_firing=use_endpoint_firing,
        start_from_oracle=True,
        update_with="DGR",
        train_mode=False,
    )

    total_shots = 0
    total_neural_errs = 0
    total_corr_errs = 0
    total_static_errs = 0

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        done = False
        while not done:
            n_flashes = int(np.sum(env.current_syndrome != 0))
            if n_flashes <= bypass_threshold:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
            else:
                action = agent.select_action(obs, evaluate=True)

            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            true_obs = info["true_obs"]
            total_neural_errs += int(info["pred_obs"] != true_obs)
            total_static_errs += int(info["static_pred_obs"] != true_obs)
            total_corr_errs += int(info["oracle_pred_obs"] != true_obs)

            obs = next_obs
            total_shots += 1

    neural_ler = total_neural_errs / total_shots if total_shots else float("nan")
    corr_ler = total_corr_errs / total_shots if total_shots else float("nan")
    static_ler = total_static_errs / total_shots if total_shots else float("nan")

    print(
        f"  -> Eval shots: {total_shots:,} | "
        f"Static {total_static_errs} ({static_ler:.3e}) | "
        f"Corr {total_corr_errs} ({corr_ler:.3e}) | "
        f"Neural {total_neural_errs} ({neural_ler:.3e})"
    )

    return {
        "neural_ler": neural_ler,
        "corr_ler": corr_ler,
        "static_ler": static_ler,
        "total_shots": total_shots,
        "neural_errors": total_neural_errs,
        "corr_errors": total_corr_errs,
        "static_errors": total_static_errs,
        "p_eval": p_eval,
        "mismatch_eval": mismatch_eval,
    }


def train(env, agent, buffer, config):
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING (Episodes: {config['train_episodes']})")
    print(f"{'='*40}")
    
    metrics = {'rewards': [], 'c_losses': [], 'a_losses': [], 'mses': [], 'alphas': []}
    start_time = time.time()
    bypass_threshold = config.get('bypass_threshold', 2)
    
    best_val_score = -float('inf')
    best_model_path = config['model_path'].replace('.pth', '_best.pth')
    
    # The Validation Latch
    validation_triggered = False
    
    for episode in range(config['train_episodes']):
        # Reset without a seed so training data stays random and diverse!
        obs, info = env.reset(seed=None) 
        done = False
        step_count = 0
        episode_reward = 0
        ep_c_loss, ep_a_loss = [], []
        
        while not done:
            # Add burn-in period where the tracers are updated but there is no reweighting yet
            if step_count < config['burn_in_steps']:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                step_count += 1
                done = terminated or truncated
                continue

            # Check syndrome flashes directly from the environment cache
            n_flashes = np.sum(env.current_syndrome != 0)

            if n_flashes <= bypass_threshold:
                # TRIVIAL SHOT: Bypass GNN, feed zero action, and DO NOT push to replay buffer
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                # COMPLEX SHOT: Wake up SAC, evaluate graph, and SAVE to replay buffer
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
        metrics['alphas'].append(agent.log_alpha.exp().item())
            
        print(f"Train Ep: {episode+1:03d}/{config['train_episodes']} | "
              f"Reward: {episode_reward:6.1f} | "
              f"MSE: {info['weights_mse_error']:.4f} | "
              f"C_Loss: {avg_c_loss:.3f} | "
              f"A_Loss: {avg_a_loss:.3f} | "
              f"Alpha: {agent.log_alpha.exp().item():.4f}")

        # --- SMART VALIDATION LOGIC ---
        
        # 1. Check if we should unlock validation
        if not validation_triggered and episode_reward > 100:
            print(f"  [!] Reward crossed 100! Unlocking Validation phase for the rest of training.")
            validation_triggered = True

        # 2. Run validation only if unlocked AND it is the 5th episode
        if validation_triggered and (episode + 1) % 5 == 0:
            val_score = validate(env, agent, config)
            
            # Save the best model
            if val_score > best_val_score:
                print(f"  *** New Best Model! Net Score improved from {best_val_score} to {val_score} ***")
                best_val_score = val_score
                agent.save_models(best_model_path)

    run_time = time.time() - start_time
    print(f"\nTraining complete in {run_time:.2f} seconds.")
    
    # Save the final trained model (just as a backup) and plot metrics
    agent.save_models(config['model_path'])
    plot_training_metrics(metrics, config)
    
    print(f"[*] The best performing model during validation was saved to: {best_model_path}")


def test(env, agent, config):
    print(f"\n{'='*50}")
    print(f"STARTING COMPREHENSIVE ABLATION TEST")
    print(f"{'='*50}")
    
    agent.load_models(config['model_path'])
    bypass_threshold = config.get('bypass_threshold', 2)

    test_seeds = [int(np.random.randint(0, 1_000_000)) for _ in range(config['test_episodes'])]
    
    n_shots = config['n_shots']
    burn_in_steps = config.get('burn_in_steps', 0)
    eval_shots_per_ep = n_shots - burn_in_steps
    
    total_shots = config['test_episodes'] * n_shots
    total_eval_shots = config['test_episodes'] * eval_shots_per_ep 
    
    policies = ['SAC_GNN', 'Zero', 'CM']
    
    # ADDED: 'ep_lers' to store the Logical Error Rate of each individual episode
    raw_results = {
        p: {'errors': 0, 'eval_errors': 0, 'fixed_count': 0, 'broken_count': 0, 
            'cum': np.zeros(total_shots, dtype=np.int32), 'ep_lers': []} 
        for p in policies + ['Oracle', 'Static']
    }
    
    weight_metrics = {
        'mse_sac_gnn_oracle': [], 'mse_zero_oracle': [],
        'mse_sac_gnn_static': [], 'mse_zero_static': [],
        'p_sac_gnn_static': [], 'p_zero_static': [],
        'p_sac_gnn_oracle': [], 'p_zero_oracle': [],
        'mse_cm_oracle': [], 'mse_cm_static': [],
        'p_cm_static': [], 'p_cm_oracle': []

    }
    
    for policy in policies:
        print(f"\n[*] Evaluating Policy: {policy}")
        global_shot_idx = 0
        policy_errors, oracle_errors, static_errors = 0, 0, 0
        policy_eval_errs, oracle_eval_errs, static_eval_errs = 0, 0, 0
        
        policy_fixed_count = 0 
        policy_broken_count = 0 
        
        for episode in range(config['test_episodes']):
            obs, info = env.reset(seed=test_seeds[episode])
            done = False
            step_count = 0
            
            ep_eval_policy_errs = 0 
            ep_eval_oracle_errs = 0  
            ep_eval_static_errs = 0  
            ep_fixed_count = 0
            ep_broken_count = 0
            
            ep_weights_mse_oracle, ep_weights_mse_static = [], []
            ep_corr_mse_oracle, ep_corr_mse_static = [], []
            
            while not done:
                n_flashes = np.sum(env.current_syndrome != 0)

                if step_count < burn_in_steps:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    if policy == 'SAC_GNN': 
                        if n_flashes <= bypass_threshold:
                            action = np.zeros(env.n_dec_edges, dtype=np.float32)
                        else:
                            action = agent.select_action(obs, evaluate=True)

                    elif policy == 'Zero': action = np.zeros(env.n_dec_edges, dtype=np.float32)

                    elif policy == 'CM': 
                        # Use the Correlated Matching solution as the action for the CM baseline
                        if n_flashes <= bypass_threshold:
                            action = np.zeros(env.n_dec_edges, dtype=np.float32)
                        else:  
                            action = env.compute_analytical_correlated_matching_action()

                    else:
                        raise ValueError(f"Unknown policy: {policy}")
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                err_policy = int(info["logical_error"])
                
                policy_errors += err_policy
                raw_results[policy]['cum'][global_shot_idx] = policy_errors
                
                if step_count >= burn_in_steps:
                    policy_eval_errs += err_policy
                    ep_eval_policy_errs += err_policy
                    
                    if policy == 'SAC_GNN':
                        true_obs = info["true_obs"]
                        pass1_correct = (info["first_pass_obs"] == true_obs)
                        pass2_correct = (info["pred_obs"] == true_obs)
                        
                        if pass2_correct and not pass1_correct:
                            policy_fixed_count += 1
                            ep_fixed_count += 1
                        elif pass1_correct and not pass2_correct:
                            policy_broken_count += 1
                            ep_broken_count += 1

                if policy == 'SAC_GNN':
                    err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
                    err_static = int(info["static_pred_obs"] != info["true_obs"])
                    
                    oracle_errors += err_oracle
                    static_errors += err_static
                    raw_results['Oracle']['cum'][global_shot_idx] = oracle_errors
                    raw_results['Static']['cum'][global_shot_idx] = static_errors

                    if step_count >= burn_in_steps:
                        oracle_eval_errs += err_oracle
                        static_eval_errs += err_static
                        ep_eval_oracle_errs += err_oracle # ADDED
                        ep_eval_static_errs += err_static # ADDED

                if step_count >= burn_in_steps:
                    ep_weights_mse_oracle.append(info["weights_mse_error"])
                    ep_weights_mse_static.append(info["weights_mse_error_static"])
                    ep_corr_mse_oracle.append(info["corr_mse_error"])
                    ep_corr_mse_static.append(info["corr_mse_error_static"])

                obs = next_obs
                step_count += 1
                global_shot_idx += 1
                
            # ADDED: Store the LER for this specific episode
            raw_results[policy]['ep_lers'].append(ep_eval_policy_errs / eval_shots_per_ep)
            if policy == 'SAC_GNN':
                raw_results['Oracle']['ep_lers'].append(ep_eval_oracle_errs / eval_shots_per_ep)
                raw_results['Static']['ep_lers'].append(ep_eval_static_errs / eval_shots_per_ep)
            # Clean printing logic
            if policy == 'SAC_GNN':
                print(f"  Test Ep {episode+1:02d} [{policy}]: {ep_eval_policy_errs} errors | Fixed: {ep_fixed_count} | Broken: {ep_broken_count}")
            else:
                print(f"  Test Ep {episode+1:02d} [{policy}]: {ep_eval_policy_errs} errors")
            
            pol_key = policy.lower()
            weight_metrics[f'mse_{pol_key}_oracle'].append(ep_weights_mse_oracle)
            weight_metrics[f'mse_{pol_key}_static'].append(ep_weights_mse_static)
            weight_metrics[f'p_{pol_key}_oracle'].append(ep_corr_mse_oracle)
            weight_metrics[f'p_{pol_key}_static'].append(ep_corr_mse_static)
            
        raw_results[policy]['eval_errors'] = policy_eval_errs

        # Binomial standard error: sigma = sqrt(p_L * (1 - p_L) / N).
        # When mismatch == 1 every episode shares the same circuit, so shots can be pooled.
        # Otherwise each episode samples a different drifted circuit, so use per-episode shots.
        mismatch = config.get('mismatch', 1.0)
        n_for_std = eval_shots_per_ep * config['test_episodes'] if mismatch == 1.0 else eval_shots_per_ep
        policy_ler_mean = np.mean(raw_results[policy]['ep_lers'])
        policy_ler_std = np.sqrt(policy_ler_mean * (1 - policy_ler_mean) / n_for_std) if n_for_std > 0 else 0.0

        # Clean summary logic
        print(f"\n  -> {policy} Summary:")
        print(f"     * LER: {policy_ler_mean:.3e} ± {policy_ler_std:.3e} ({policy_eval_errs}/{total_eval_shots})")
        if policy == 'SAC_GNN':
            fixed_pct = (policy_fixed_count / total_eval_shots) * 100
            broken_pct = (policy_broken_count / total_eval_shots) * 100
            raw_results[policy]['fixed_count'] = policy_fixed_count
            raw_results[policy]['broken_count'] = policy_broken_count
            print(f"     * Fixed CMA Errors:  {policy_fixed_count} times ({fixed_pct:.3f}% of eval shots)")
            print(f"     * Broken CMA Succes: {policy_broken_count} times ({broken_pct:.3f}% of eval shots)")

        if policy == 'SAC_GNN':
            raw_results['Oracle']['eval_errors'] = oracle_eval_errs
            raw_results['Static']['eval_errors'] = static_eval_errs

    # Construct final metrics dictionary
    final_metrics = {}
    for k in ['SAC_GNN', 'Zero', 'Static', 'Oracle', 'CM']:
        ler_mean = np.mean(raw_results[k]['ep_lers'])
        final_metrics[f'ler_{k.lower()}'] = ler_mean
        final_metrics[f'ler_std_{k.lower()}'] = (
            np.sqrt(ler_mean * (1 - ler_mean) / n_for_std) if n_for_std > 0 else 0.0
        )
        final_metrics[f'cum_{k.lower()}'] = raw_results[k]['cum']

    plot_testing_metrics(final_metrics)
    plot_weight_correlations(weight_metrics)


def analyze_policy(env, agent, config):
    print(f"\n{'='*50}")
    print(f"STARTING POLICY ANALYSIS (Action Histogram & Syndrome Counts)")
    print(f"CATEGORIZING: Direct (Pass 1 Selected) vs. Neighbor (Masked Only)")
    print(f"{'='*50}")
    
    try:
        agent.load_models(config['model_path'])
        print(f"Successfully loaded model from {config['model_path']}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    burn_in_steps = config.get('burn_in_steps', 0)
    bypass_threshold = config.get('bypass_threshold', 4)
    
    # Original Trackers
    all_raw_actions = []
    all_active_actions = []
    all_syndrome_counts = []
    
    # New Topography Trackers
    direct_actions = []    # Actions on edges MWPM actually picked (Flag=1)
    neighbor_actions = []  # Actions on edges that are neighbors (Mask=1, Flag=0)

    episodes_to_run = min(3, config['test_episodes'])
    
    for episode in range(episodes_to_run):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        print(f"Collecting data from Episode {episode+1}/{episodes_to_run}...")
        
        while not done:
            n_flashes = np.sum(env.current_syndrome != 0) 
            all_syndrome_counts.append(n_flashes)
            
            if step_count >= burn_in_steps:
                if n_flashes <= bypass_threshold:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    action = agent.select_action(obs, evaluate=True)
                
                # --- EXISTING LOGIC: General Histograms ---
                mask = obs['action_mask']
                active_acts = action[mask > 0]
                
                if len(active_acts) > 0:
                    all_active_actions.extend(active_acts.tolist())
                all_raw_actions.extend(action.tolist())

                # --- NEW LOGIC: Topography Breakdown ---
                # flags: [:, 1] is the Pass 1 Flag (0 or 1)
                flags = obs['node_features'][:, 1]
                
                # 1. Direct: Nodes MWPM selected (Flag=1, implies Mask=1)
                direct_mask = (flags > 0)
                if np.any(direct_mask):
                    direct_actions.extend(action[direct_mask].tolist())
                
                # 2. Neighbors: Masked nodes that MWPM did NOT select
                neighbor_mask = (mask > 0) & (flags == 0)
                if np.any(neighbor_mask):
                    neighbor_actions.extend(action[neighbor_mask].tolist())
                
            else:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            step_count += 1
            
    # --- NEW LOGIC: Near-Zero Action Analysis ---
    print(f"\n{'='*50}")
    print(f"NEAR-ZERO ACTION ANALYSIS (Active Edges Only)")
    print(f"{'='*50}")
    
    active_arr = np.array(all_active_actions)
    total_active = len(active_arr)
    
    if total_active > 0:
        # Calculate thresholds (using absolute values to catch both positive and negative noise)
        abs_actions = np.abs(active_arr)
        
        exactly_zero = np.sum(abs_actions == 0.0)
        less_than_1e6 = np.sum(abs_actions < 1e-6)
        less_than_1e3 = np.sum(abs_actions < 1e-3)
        less_than_1e2 = np.sum(abs_actions < 1e-2)
        
        print(f"Total Active Actions Evaluated: {total_active:,}")
        print(f" - Exactly 0.0:     {exactly_zero:,} ({exactly_zero/total_active*100:.2f}%)")
        print(f" - < 1e-6:          {less_than_1e6:,} ({less_than_1e6/total_active*100:.2f}%)")
        print(f" - < 1e-3:          {less_than_1e3:,} ({less_than_1e3/total_active*100:.2f}%)")
        print(f" - < 1e-2:          {less_than_1e2:,} ({less_than_1e2/total_active*100:.2f}%)")
        
        print("\nInsight:")
        if exactly_zero / total_active > 0.5:
            print("The agent is actively predicting exactly zero for the majority of active edges.")
        elif less_than_1e3 / total_active > 0.5:
            print("The agent outputs a lot of numerical noise (< 1e-3). You may want to implement an action masking threshold in env.step() to snap these to exactly 0.0 and skip the second pass.")
        else:
            print("The agent is actively reweighting most edges with significant values (> 1e-3). Skipping the second pass frequently is unlikely.")
    else:
        print("No active actions were recorded during these episodes.")

    # Standard Plots
    plot_action_histogram(all_raw_actions, all_active_actions)
    plot_syndrome_count_histogram(all_syndrome_counts, bypass_threshold=bypass_threshold)
    
    # New Comparative Plot
    plot_action_topography(direct_actions, neighbor_actions)








    