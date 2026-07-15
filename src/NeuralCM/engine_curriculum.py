"""
Main execution script for training and testing the SAC-GNN QEC Decoder.
It handles hyperparameters, execution modes, and performance visualization.
"""

import time
import numpy as np

from NeuralCM.plot_utils import *


def validate(create_env, agent, config):
    """Rigorous isolated validation loop (Pure CMA vs. GNN) locked to an anchor p-value."""
    print(f"  [!] Running Isolated Validation (Pure CMA vs GNN)...")
    
    # 1. Lock validation to a stable anchor to prevent the "Moving Goalpost" problem
    val_p = 0.002
    val_shots = 20_000 
    
    val_episodes = 3
    val_seeds = [10001, 10002, 10003] 
    val_burn_in = config.get('burn_in_steps', 0)
    bypass_threshold = config.get('bypass_threshold', 2)
    
    # Create the specialized validation environment
    env = create_env(val_p, val_shots)
    
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
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if step_count >= val_burn_in:
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
            
    # Calculate True Relative Improvement
    net_errors_saved = total_zero_errors - total_gnn_errors
    relative_improvement = (net_errors_saved / total_zero_errors) if total_zero_errors > 0 else 0.0
    
    print(f"  -> Val Anchor (p={val_p}): Baseline Errors: {total_zero_errors} | GNN Errors: {total_gnn_errors}")
    print(f"  -> Relative Improvement: {relative_improvement * 100:.3f}%")
    
    return relative_improvement


def train(create_env, agent, buffer, config):
    print(f"\n{'='*40}")
    print(f"STARTING CURRICULUM TRAINING")
    print(f"{'='*40}")
    
    metrics = {'rewards': [], 'c_losses': [], 'a_losses': [], 'mses': [], 'alphas': []}
    start_time = time.time()
    bypass_threshold = config.get('bypass_threshold', 2)
    
    best_val_score = -float('inf')
    best_model_path = config['model_path'].replace('.pth', '_best.pth')
    validation_triggered = False
    
    # Outer Loop: The Curriculum
    for p_val in config['curriculum_p']:
        current_shots = config['shots_per_p'][p_val]
        print(f"\n>>> ADVANCING CURRICULUM: p = {p_val} (Shots/Ep: {current_shots:,}) <<<")
        
        # Build environment for this specific phase
        env = create_env(p_val, current_shots)
        
        for episode in range(config['train_episodes_per_p']):
            obs, info = env.reset(seed=None) 
            done = False
            step_count = 0
            episode_reward = 0
            ep_c_loss, ep_a_loss = [], []
            
            while not done:
                if step_count < config['burn_in_steps']:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    obs = next_obs
                    step_count += 1
                    done = terminated or truncated
                    continue

                n_flashes = np.sum(env.current_syndrome != 0)

                if n_flashes <= bypass_threshold:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                else:
                    action = agent.select_action(obs, evaluate=False) 
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    buffer.push(obs, action, reward, next_obs if not done else None, done)
                
                if len(buffer) > config['batch_size'] and step_count % config['update_frequency'] == 0:
                    c_loss, a_loss = agent.update(buffer, config['batch_size'])
                    ep_c_loss.append(c_loss)
                    ep_a_loss.append(a_loss)
                
                obs = next_obs
                episode_reward += reward
                step_count += 1
                
            avg_c_loss = np.mean(ep_c_loss) if ep_c_loss else 0.0
            avg_a_loss = np.mean(ep_a_loss) if ep_a_loss else 0.0
            
            metrics['rewards'].append(episode_reward)
            metrics['c_losses'].append(avg_c_loss)
            metrics['a_losses'].append(avg_a_loss)
            metrics['mses'].append(info['weights_mse_error'])
            metrics['alphas'].append(agent.log_alpha.exp().item())
                
            print(f"Train Ep: {episode+1:03d}/{config['train_episodes_per_p']} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"C_Loss: {avg_c_loss:.3f} | "
                  f"A_Loss: {avg_a_loss:.3f}")

            # Smart Validation Logic
            if not validation_triggered and episode_reward > 100:
                print(f"  [!] Reward crossed 100! Unlocking Validation phase.")
                validation_triggered = True

            if validation_triggered and (episode + 1) % 5 == 0:
                val_score = validate(create_env, agent, config)
                
                if val_score > best_val_score:
                    print(f"  *** New Best Model! Net Score improved from {best_val_score:.3f} to {val_score:.3f} ***")
                    best_val_score = val_score
                    agent.save_models(best_model_path)

    run_time = time.time() - start_time
    print(f"\nTraining complete in {run_time:.2f} seconds.")
    agent.save_models(config['model_path'])
    plot_training_metrics(metrics, config)
    print(f"[*] The best performing model during validation was saved to: {best_model_path}")


def test(create_env, agent, config):
    print(f"\n{'='*50}")
    print(f"STARTING ABLATION TEST ACROSS CURRICULUM")
    print(f"{'='*50}")
    
    agent.load_models(config['model_path'])
    bypass_threshold = config.get('bypass_threshold', 2)
    burn_in_steps = config.get('burn_in_steps', 0)
    policies = ['SAC_GNN', 'Zero', 'CM']
    
    for p_val in config['curriculum_p']:
        current_shots = config['shots_per_p'][p_val]
        eval_shots_per_ep = current_shots - burn_in_steps
        total_eval_shots = config['test_episodes_per_p'] * eval_shots_per_ep
        
        print(f"\n{'*'*40}")
        print(f" EVALUATING p = {p_val} (Total Shots: {total_eval_shots:,})")
        print(f"{'*'*40}")
        
        env = create_env(p_val, current_shots)
        
        # 1. INITIALIZE TRACKING DICTIONARIES FOR THIS SPECIFIC p_val
        raw_results = {
            p: {'errors': 0, 'eval_errors': 0, 'fixed_count': 0, 'broken_count': 0, 
                'cum': np.zeros(total_eval_shots, dtype=np.int32), 'ep_lers': []} 
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
            print(f"\n[*] Policy: {policy}")
            global_shot_idx = 0
            policy_errors, oracle_errors, static_errors = 0, 0, 0
            policy_eval_errs, oracle_eval_errs, static_eval_errs = 0, 0, 0
            policy_fixed_count, policy_broken_count = 0, 0 
            
            for episode in range(config['test_episodes_per_p']):
                obs, info = env.reset(seed=100000 + episode) 
                done = False
                step_count = 0
                ep_eval_policy_errs, ep_eval_oracle_errs, ep_eval_static_errs = 0, 0, 0 
                
                ep_weights_mse_oracle, ep_weights_mse_static = [], []
                ep_corr_mse_oracle, ep_corr_mse_static = [], []
                
                while not done:
                    n_flashes = np.sum(env.current_syndrome != 0)

                    if step_count < burn_in_steps:
                        action = np.zeros(env.n_dec_edges, dtype=np.float32)
                    else:
                        if policy == 'SAC_GNN': 
                            action = np.zeros(env.n_dec_edges, dtype=np.float32) if n_flashes <= bypass_threshold else agent.select_action(obs, evaluate=True)
                        elif policy == 'Zero': 
                            action = np.zeros(env.n_dec_edges, dtype=np.float32)
                        elif policy == 'CM': 
                            action = np.zeros(env.n_dec_edges, dtype=np.float32) if n_flashes <= bypass_threshold else env.compute_analytical_correlated_matching_action()
                        
                    next_obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    err_policy = int(info["logical_error"])
                    if step_count >= burn_in_steps:
                        policy_eval_errs += err_policy
                        ep_eval_policy_errs += err_policy
                        raw_results[policy]['cum'][global_shot_idx] = policy_eval_errs
                        
                        if policy == 'SAC_GNN':
                            true_obs = info["true_obs"]
                            pass1_correct = (info["first_pass_obs"] == true_obs)
                            pass2_correct = (info["pred_obs"] == true_obs)
                            if pass2_correct and not pass1_correct: policy_fixed_count += 1
                            elif pass1_correct and not pass2_correct: policy_broken_count += 1
                            
                            err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
                            err_static = int(info["static_pred_obs"] != info["true_obs"])
                            oracle_eval_errs += err_oracle
                            static_eval_errs += err_static
                            ep_eval_oracle_errs += err_oracle
                            ep_eval_static_errs += err_static
                            raw_results['Oracle']['cum'][global_shot_idx] = oracle_eval_errs
                            raw_results['Static']['cum'][global_shot_idx] = static_eval_errs

                        ep_weights_mse_oracle.append(info["weights_mse_error"])
                        ep_weights_mse_static.append(info["weights_mse_error_static"])
                        ep_corr_mse_oracle.append(info["corr_mse_error"])
                        ep_corr_mse_static.append(info["corr_mse_error_static"])

                        global_shot_idx += 1

                    obs = next_obs
                    step_count += 1
                    
                raw_results[policy]['ep_lers'].append(ep_eval_policy_errs / eval_shots_per_ep)
                if policy == 'SAC_GNN':
                    raw_results['Oracle']['ep_lers'].append(ep_eval_oracle_errs / eval_shots_per_ep)
                    raw_results['Static']['ep_lers'].append(ep_eval_static_errs / eval_shots_per_ep)
                    
                print(f"  Test Ep {episode+1:02d}: {ep_eval_policy_errs} errors")
                
                pol_key = policy.lower()
                weight_metrics[f'mse_{pol_key}_oracle'].append(ep_weights_mse_oracle)
                weight_metrics[f'mse_{pol_key}_static'].append(ep_weights_mse_static)
                weight_metrics[f'p_{pol_key}_oracle'].append(ep_corr_mse_oracle)
                weight_metrics[f'p_{pol_key}_static'].append(ep_corr_mse_static)
                
            # Print Summary
            policy_ler_mean = np.mean(raw_results[policy]['ep_lers'])
            policy_ler_std = np.std(raw_results[policy]['ep_lers'])
            print(f"  -> {policy} Summary (p={p_val}):")
            print(f"     * LER: {policy_ler_mean:.6e} ± {policy_ler_std:.6e} ({policy_eval_errs}/{total_eval_shots})")
            
            if policy == 'SAC_GNN':
                print(f"     * MWPM Baseline (Static) Errors: {static_eval_errs}")
                print(f"     * Perfect Oracle Errors:         {oracle_eval_errs}")
                print(f"     * Fixed CMA Errors:  {policy_fixed_count}")
                print(f"     * Broken CMA Succes: {policy_broken_count}")

        # 2. FINALIZE METRICS FOR THIS p_val
        final_metrics = {}
        for k in ['SAC_GNN', 'Zero', 'Static', 'Oracle', 'CM']:
            final_metrics[f'ler_{k.lower()}'] = np.mean(raw_results[k]['ep_lers'])
            final_metrics[f'ler_std_{k.lower()}'] = np.std(raw_results[k]['ep_lers'])
            final_metrics[f'cum_{k.lower()}'] = raw_results[k]['cum']

        # ---------------------------------------------------------
        # --> CALL TESTING PLOT FUNCTIONS HERE (inside the p loop) <--
        # ---------------------------------------------------------
        plot_testing_metrics(final_metrics, p_val)
        plot_weight_correlations(weight_metrics, p_val)


def analyze_policy(create_env, agent, config):
    print(f"\n{'='*50}")
    print(f"STARTING POLICY ANALYSIS")
    print(f"{'='*50}")
    
    try:
        agent.load_models(config['model_path'])
        print(f"Successfully loaded model from {config['model_path']}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Use a stable mid-range p to analyze action distributions
    analyze_p = 0.002
    analyze_shots = 20_000
    env = create_env(analyze_p, analyze_shots)

    burn_in_steps = config.get('burn_in_steps', 0)
    bypass_threshold = config.get('bypass_threshold', 2)
    
    all_raw_actions, all_active_actions, all_syndrome_counts = [], [], []
    direct_actions, neighbor_actions = [], []
    
    episodes_to_run = min(3, config['test_episodes_per_p'])
    
    for episode in range(episodes_to_run):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        print(f"Collecting data from Episode {episode+1}/{episodes_to_run} (p={analyze_p})...")
        
        while not done:
            n_flashes = np.sum(env.current_syndrome != 0) 
            all_syndrome_counts.append(n_flashes)
            
            if step_count >= burn_in_steps:
                if n_flashes <= bypass_threshold:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    action = agent.select_action(obs, evaluate=True)
                
                mask = obs['action_mask']
                active_acts = action[mask > 0]
                if len(active_acts) > 0: all_active_actions.extend(active_acts.tolist())
                all_raw_actions.extend(action.tolist())

                flags = obs['node_features'][:, 1]
                direct_mask = (flags > 0)
                if np.any(direct_mask): direct_actions.extend(action[direct_mask].tolist())
                
                neighbor_mask = (mask > 0) & (flags == 0)
                if np.any(neighbor_mask): neighbor_actions.extend(action[neighbor_mask].tolist())
            else:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            
    print(f"\nNEAR-ZERO ACTION ANALYSIS (Active Edges Only)")
    active_arr = np.array(all_active_actions)
    total_active = len(active_arr)
    
    if total_active > 0:
        abs_actions = np.abs(active_arr)
        exactly_zero = np.sum(abs_actions == 0.0)
        less_than_1e3 = np.sum(abs_actions < 1e-3)
        print(f"Total Active Actions Evaluated: {total_active:,}")
        print(f" - Exactly 0.0:     {exactly_zero:,} ({exactly_zero/total_active*100:.2f}%)")
        print(f" - < 1e-3:          {less_than_1e3:,} ({less_than_1e3/total_active*100:.2f}%)")
    else:
        print("No active actions were recorded.")

    plot_action_histogram(all_raw_actions, all_active_actions)
    plot_syndrome_count_histogram(all_syndrome_counts, bypass_threshold=bypass_threshold)
    plot_action_topography(direct_actions, neighbor_actions)