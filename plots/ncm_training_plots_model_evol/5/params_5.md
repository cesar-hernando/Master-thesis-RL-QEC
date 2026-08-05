enable_correlations = True for oracle
built-in Stim noise model

CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'test',  
        'model_path': 'models/sac_gnn_5.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.002,
        'mismatch': 30.0,
        'n_shots': 20_000,       # Shots per episode
        'burn_in_steps': 18_000,
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 0.0, # Phase 1: High imitation reward
        'local_action_only': False,
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