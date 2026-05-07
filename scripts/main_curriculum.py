"""
Main execution script for training, testing, and analyzing the 
SAC-GNN agent on the Drifted Matching Environment using Curriculum Learning.
"""

import time
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent, GraphReplayBuffer
from adaptiveQRL.engine_curriculum import train, test, analyze_policy

if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        'MODE': 'train',  # 'train', 'test', or 'analyze_policy'
        'model_path': 'models/sac_gnn_54_curriculum.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'mismatch': 1.0,           # Perfect baseline math
        'p_gate_zz': 0.0,          # Crosstalk ZZ error probability
        
        # --- THE CURRICULUM ---
        'curriculum_p': [0.02, 0.01, 0.005, 0.003],
        
        # Mathematically derived via N_shots ∝ 1 / p^3 to balance logical errors
        'shots_per_p': {
            0.02: 1_250,
            0.01: 10_000,     
            0.005: 20_000,
            0.003: 185_000,
        },
        'train_episodes_per_p': 50, 
        'test_episodes_per_p': 5,  # Keep low to avoid massive wait times during engine.test()
        
        # --- FROZEN GRAPH SETTINGS ---
        'update_period': 10_000_000, # Massive number so it never updates mid-episode
        'prior_shots': 1_000,
        
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 5.0,
        'local_action_only': True,
        'local_action_hops': 1, 
        'use_pearson_correlation': True,
        'use_log_joint_prob': False,  
        
        # Agent / NN Settings
        'n_layers': 1, 
        'hidden_dim': 256,
        'lr': 1e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit
        'tau': 0.005,
        'alpha': 0.01,         # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
    }

    #######################################
    # 2. ENVIRONMENT FACTORY FUNCTION     #
    #######################################
    def create_env(p_val, n_shots):
        """Builds a fresh environment for a specific curriculum step."""
        generator = SyndromeDataGenerator(
            distance=CONFIG['distance'], 
            n_rounds=CONFIG['n_rounds'], 
            mismatch=CONFIG['mismatch'],  
            noise_model={
                "version": "built-in",
                "after_clifford_depolarization": p_val,
                "before_measure_flip_probability": p_val,
                "after_reset_flip_probability": p_val,
                "before_round_data_depolarization": p_val,
                "p_gate_zz": CONFIG["p_gate_zz"]
            }, 
            memory_type='z', 
            n_shots=n_shots, 
            qec_code='surface_code'
        )

        return DriftedMatchingEnv(
            syndrome_data_generator=generator,
            local_action_only=CONFIG['local_action_only'],
            local_action_hops=CONFIG['local_action_hops'],
            action_scale=CONFIG['action_scale'],
            update_period=CONFIG['update_period'],
            prior_shots=CONFIG['prior_shots'],
            n_test_shots=0,             
            use_pearson_correlation=CONFIG['use_pearson_correlation'],
            use_syndrome_features=False, 
            use_log_joint_prob=CONFIG['use_log_joint_prob'],
            update_with='DGR',
            train_mode=(CONFIG['MODE'] == 'train')
        )

    #######################################
    # 3. INITIALIZE AGENT                 #
    #######################################
    print("Initializing Agent Dimensions...")
    # Create a dummy env just to extract the node dimensions and edge index
    dummy_env = create_env(CONFIG['curriculum_p'][0], 100)
    sample_obs, _ = dummy_env.reset()
    NODE_DIM = sample_obs["node_features"].shape[1]
    
    agent = SACAgent(
        node_dim=NODE_DIM, 
        hidden_dim=CONFIG['hidden_dim'],
        static_edge_index=dummy_env.line_edge_index,
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        tau=CONFIG['tau'],
        alpha=CONFIG['alpha'],
        n_layers=CONFIG['n_layers']
    )

    ############################
    # 4. EXECUTE SELECTED MODE #
    ############################
    if CONFIG['MODE'] == 'train':
        start_train = time.time()
        buffer = GraphReplayBuffer(capacity=CONFIG['buffer_capacity'])
        train(create_env, agent, buffer, CONFIG)
        print(f"Training run time = {time.time() - start_train:.2f} s")
        
    elif CONFIG['MODE'] == 'test':
        start_test = time.time()
        test(create_env, agent, CONFIG)
        print(f"Test run time = {time.time() - start_test:.2f} s")

    elif CONFIG['MODE'] == 'analyze_policy':
        start_analysis = time.time()
        analyze_policy(create_env, agent, CONFIG)
        print(f"Analysis run time = {time.time() - start_analysis:.2f} s")

    else:
        print(f"Unknown MODE: {CONFIG['MODE']}. Please select 'train', 'test', or 'analyze_policy'.")