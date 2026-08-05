CONFIG = {
        # Execution Mode: 'train','test' or 'analyze_policy'
        'MODE': 'test',  
        'model_path': 'models/sac_gnn_10.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.004,
        'mismatch': 30.0,
        'n_shots': 65_000,       # Shots per episode
        'burn_in_steps': 15_000,
        'bypass_threshold': 2,
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 1.0, # Phase 1: High imitation reward
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 1e-4,
        'gamma': 0.99,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.2,          # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        
        # Episode Settings
        'train_episodes': 50,
        'test_episodes': 20
    }