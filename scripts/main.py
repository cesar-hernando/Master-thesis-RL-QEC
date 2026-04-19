"""
Main execution script for training, testing, and analyzing the 
SAC-GNN agent on the Drifted Matching Environment. 
"""

import time
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent, GraphReplayBuffer
from adaptiveQRL.engine import train, test, analyze_policy


if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        # Execution Mode: 'train','test' or 'analyze_policy'
        'MODE': 'test',  
        'model_path': 'models/sac_gnn_35_best.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.001,
        'p_gate_zz': 0.0,  # Crosstalk ZZ error probability
        'mismatch': 30.0,
        'n_shots': 150_000,       # Shots per episode
        'n_test_shots': 10_000,   # Shots for LER evaluation
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        'n_layers': 1, # Number of GNN layers (affects receptive field size)
        
        # Agent / NN Settings
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.99,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.01,          # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        
        # Episode Settings
        'train_episodes': 270,
        'test_episodes': 20
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
            "p_gate_zz": CONFIG["p_gate_zz"]
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
        n_test_shots=CONFIG['n_test_shots'],             
        use_pearson_correlation=True,
        use_syndrome_features=False, 
        update_with='DGR',
        train_mode=(CONFIG['MODE'] == 'train')
    )

    # Determine dynamic dimensions from environment
    sample_obs, _ = env.reset()
    NODE_DIM = sample_obs["node_features"].shape[1]
    
    # Init Agent
    agent = SACAgent(
        node_dim=NODE_DIM, 
        hidden_dim=CONFIG['hidden_dim'],
        static_edge_index=env.line_edge_index,  # Pass the static edge index to the agent
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        tau=CONFIG['tau'],
        alpha=CONFIG['alpha'],
        n_layers=CONFIG['n_layers']
    )

    #total_params = sum(p.numel() for p in agent.actor.parameters())
    #print(f"Total Parameters: {total_params:,}")

    ############################
    # 3. EXECUTE SELECTED MODE #
    ############################
    if CONFIG['MODE'] == 'train':
        start_train = time.time()
        buffer = GraphReplayBuffer(capacity=CONFIG['buffer_capacity'])
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

    elif CONFIG['MODE'] == 'analyze_policy':
        start_analysis = time.time()
        analyze_policy(env, agent, CONFIG)
        end_analysis = time.time()
        analysis_runtime = end_analysis - start_analysis
        print(f"Analyisis run time = {analysis_runtime:.2f} s")

    else:
        print(f"Unknown MODE: {CONFIG['MODE']}. Please select 'train' or 'test'.")