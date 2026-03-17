"""
Main execution script for training and testing the SAC-GNN QEC Decoder.
It handles hyperparameters, execution modes, and performance visualization.
"""

import time
import numpy as np

from syndrome_data_generation import SyndromeDataGenerator
from drifted_matching_env import DriftedMatchingEnv
from gnn_sac_agent import SACAgent, GraphReplayBuffer
from plot_utils import plot_weight_correlations, plot_training_metrics, plot_testing_metrics


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
            # Add burn-in period where the tracers are updated but there is no reweighting yet
            if step_count < config['burn_in_steps']:
                action = np.zeros(env.n_dec_edges, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                step_count += 1
                done = terminated or truncated
                continue

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
    print(f"STARTING COMPREHENSIVE ABLATION TEST")
    print(f"{'='*50}")
    
    agent.load_models(config['model_path'])

    test_seeds = [int(np.random.randint(0, 1_000_000)) for _ in range(config['test_episodes'])]
    
    n_shots = config['n_shots']
    burn_in_steps = config.get('burn_in_steps', 0)
    eval_shots_per_ep = n_shots - burn_in_steps
    
    total_shots = config['test_episodes'] * n_shots
    total_eval_shots = config['test_episodes'] * eval_shots_per_ep # For LER math
    
    policies = ['GNN', 'Zero', 'Random']
    
    raw_results = {p: {'errors': 0, 'eval_errors': 0, 'cum': np.zeros(total_shots, dtype=np.int32)} for p in policies + ['Oracle', 'Static']}
    
    weight_metrics = {
        'mse_gnn_oracle': [], 'mse_zero_oracle': [], 'mse_random_oracle': [],
        'mse_gnn_static': [], 'mse_zero_static': [], 'mse_random_static': [],
        'p_gnn_oracle': [], 'p_zero_oracle': [], 'p_random_oracle': [],
        'p_gnn_static': [], 'p_zero_static': [], 'p_random_static': []
    }

    for policy in policies:
        print(f"\n[*] Evaluating Policy: {policy}")
        global_shot_idx = 0
        policy_errors, oracle_errors, static_errors = 0, 0, 0
        policy_eval_errs, oracle_eval_errs, static_eval_errs = 0, 0, 0
        
        for episode in range(config['test_episodes']):
            obs, info = env.reset(seed=test_seeds[episode])
            done = False
            step_count = 0
            ep_eval_policy_errs = 0 
            
            ep_weights_mse_oracle, ep_weights_mse_static = [], []
            ep_corr_mse_oracle, ep_corr_mse_static = [], []
            
            while not done:
                # --- 1. BURN-IN VS ACTIVE LOGIC ---
                if step_count < burn_in_steps:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    if policy == 'GNN': action = agent.select_action(obs, evaluate=True)
                    elif policy == 'Zero': action = np.zeros(env.n_dec_edges, dtype=np.float32)
                    elif policy == 'Random': action = np.random.uniform(low=-1.0, high=1.0, size=env.n_dec_edges).astype(np.float32)
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # --- 2. ERROR TRACKING ---
                err_policy = int(info["logical_error"])
                
                # Global Tracking (For the Cumulative Plot)
                policy_errors += err_policy
                raw_results[policy]['cum'][global_shot_idx] = policy_errors
                
                # Eval Tracking (For the LER Bar Chart - Only count post-burn-in)
                if step_count >= burn_in_steps:
                    policy_eval_errs += err_policy
                    ep_eval_policy_errs += err_policy

                if policy == 'GNN':
                    err_oracle = int(info["oracle_pred_obs"] != info["true_obs"])
                    err_static = int(info["static_pred_obs"] != info["true_obs"])
                    
                    oracle_errors += err_oracle
                    static_errors += err_static
                    raw_results['Oracle']['cum'][global_shot_idx] = oracle_errors
                    raw_results['Static']['cum'][global_shot_idx] = static_errors
                    
                    if step_count >= burn_in_steps:
                        oracle_eval_errs += err_oracle
                        static_eval_errs += err_static

                # Track weight and correlations MSE errors
                if step_count >= burn_in_steps:
                    ep_weights_mse_oracle.append(info["weights_mse_error"])
                    ep_weights_mse_static.append(info["weights_mse_error_static"])
                    ep_corr_mse_oracle.append(info["corr_mse_error"])
                    ep_corr_mse_static.append(info["corr_mse_error_static"])

                obs = next_obs
                step_count += 1
                global_shot_idx += 1
                
            print(f"  Test Ep {episode+1:02d} [{policy}]: {ep_eval_policy_errs} active errors")
            
            pol_key = policy.lower()
            weight_metrics[f'mse_{pol_key}_oracle'].append(ep_weights_mse_oracle)
            weight_metrics[f'mse_{pol_key}_static'].append(ep_weights_mse_static)
            weight_metrics[f'p_{pol_key}_oracle'].append(ep_corr_mse_oracle)
            weight_metrics[f'p_{pol_key}_static'].append(ep_corr_mse_static)
            
        raw_results[policy]['eval_errors'] = policy_eval_errs
        if policy == 'GNN':
            raw_results['Oracle']['eval_errors'] = oracle_eval_errs
            raw_results['Static']['eval_errors'] = static_eval_errs

    # Format the final LER metrics using ONLY the active evaluation shots
    final_metrics = {}
    for k in ['GNN', 'Zero', 'Random', 'Oracle', 'Static']:
        final_metrics[f'ler_{k.lower()}'] = raw_results[k]['eval_errors'] / total_eval_shots
        final_metrics[f'cum_{k.lower()}'] = raw_results[k]['cum']

    plot_testing_metrics(final_metrics)
    plot_weight_correlations(weight_metrics)




if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'train',  
        'model_path': 'models/sac_gnn_6.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.002,
        'mismatch': 30.0,
        'n_shots': 30_000,       # Shots per episode
        'burn_in_steps': 15_000,
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 0.2, # Phase 1: High imitation reward
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 3e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.02,          # Entropy tuning
        'batch_size': 64,
        'update_frequency': 10,
        
        # Episode Settings
        'train_episodes': 100,
        'test_episodes': 50
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
        local_action_only=CONFIG['local_action_only'],
        local_action_hops=CONFIG['local_action_hops'],
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

    total_params = sum(p.numel() for p in agent.actor.parameters())
    print(f"Total Parameters: {total_params:,}")
    #print(len(agent.critic1.parameters()))

    '''
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

    '''