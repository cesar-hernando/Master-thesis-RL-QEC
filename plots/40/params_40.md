HIDDEN_DIM=128
TRAIN_EPISODES=270

echo "Starting Run 40: Gamma=0.0 | Layers=1 | Mismatch=30.0"
python -u run_job.py --mode 0 \
    --gamma 0.0 \
    --n_layers 1 \
    --update_period 100 \
    --mismatch 30.0 \
    --hidden_dim $HIDDEN_DIM \
    --train_episodes $TRAIN_EPISODES \
    --n_shots 65000 \
    --burn_in_steps 15000 \
    --model_path models/sac_gnn_40.pth \
    --training_metrics_filename training_metrics_40.png


CONFIG = {
        # Execution Mode: 'train','test' or 'analyze_policy'
        'MODE': 'test',  
        'model_path': 'models/sac_gnn_38_best.pth',
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.004,
        'p_gate_zz': 0.0,  # Crosstalk ZZ error probability
        'mismatch': 30.0,
        'n_shots': 65_000,       # Shots per episode
        'burn_in_steps': 15_000,
        'bypass_threshold': 2,
        'action_scale': 3.0,
        'update_period': 100,  # CMA update frequency
        'prior_shots': 1_000,
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        'n_layers': 1, # Number of GNN layers (affects receptive field size)
        
        # Agent / NN Settings
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.01,          # Entropy tuning
        'batch_size': 64,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        
        # Episode Settings
        'train_episodes': 270,
        'test_episodes': 20
    }