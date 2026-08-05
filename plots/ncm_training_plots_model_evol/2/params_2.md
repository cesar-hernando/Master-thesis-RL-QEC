enable_correlations = False for the oracle
built-in Stim noise model
local_action_only=True
local_action_hops=1


CONFIG = {
        # Execution Mode: 'train' or 'test'
        'MODE': 'test',  
        'model_path': 'models/sac_gnn_best_uncorr_or.pth',
        
        # Environment Settings
        'distance': 3,
        'n_rounds': 3,
        'p': 0.001,
        'mismatch': 20.0,
        'n_shots': 60_000,       # Shots per episode
        'action_scale': 3.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,
        'oracle_reward_coef': 0.0, # Phase 1: High imitation reward
        
        # Agent / NN Settings
        'hidden_dim': 64,
        'lr': 3e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.2,          # Entropy tuning
        'batch_size': 64,
        'update_frequency': 10,
        
        # Episode Settings
        'train_episodes': 50,
        'test_episodes': 50
    }