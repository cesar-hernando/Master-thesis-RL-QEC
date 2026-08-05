oracle enable_correlations = False

CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'analyze_policy',  
        'model_path': 'models/sac_gnn_9.pth',
        
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
        'oracle_reward_coef': 2.0, # Phase 1: High imitation reward
        'local_action_only': False,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 1e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.2,          # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 50_000,
        'update_frequency': 10,
        
        # Episode Settings
        'train_episodes': 25,
        'test_episodes': 10
    }