enable_correlations = True for oracle
built-in Stim noise model
logical reward with respect to zero action (1st pass)

CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'analyze_policy',  
        'model_path': 'models/sac_gnn_7.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.002,
        'mismatch': 20.0,
        'n_shots': 40_000,       # Shots per episode
        'burn_in_steps': 15_000,
        'bypass_threshold': 4,
        'action_scale': 6.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 0.5, # Phase 1: High imitation reward
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 3e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.02,          # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 50_000,
        'update_frequency': 10,
        
        # Episode Settings
        'train_episodes': 20,
        'test_episodes': 5
    }