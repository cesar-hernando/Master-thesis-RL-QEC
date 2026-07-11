"""
Main execution script for training, testing, and analyzing the 
SAC-GNN agent on the Drifted Matching Environment. 
"""

import time
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent, GraphReplayBuffer
from adaptiveQRL.engine import train, train_policy_gradient, test, analyze_policy


if __name__ == "__main__":

    ######################################
    # 1. HYPERPARAMETERS & CONFIGURATION #
    ######################################
    CONFIG = {
        # Execution Mode: 'train','test' or 'analyze_policy'
        'MODE': 'train',  
        'model_path': 'models/linear/linear_model_1_start_from_CM.pth',  # Path to save/load model
        
        # Environment Settings
        'distance': 5,
        'n_rounds': 5,
        'p': 0.004,
        'p_gate_zz': 0.0,  # Crosstalk ZZ error probability
        'mismatch': 30.0,
        'n_shots': 50_000,       # Shots per episode
        'n_test_shots': 0,   # Shots for LER evaluation
        'burn_in_steps': 0,
        'bypass_threshold': 2,
        'action_scale': 1.0,
        'update_period': 1_000,  # CMA update frequency
        'prior_shots': 1_000,  # Shots for initial CMA prior
        'local_action_only': True,
        'local_action_hops': 1, # if local_action_only = False, this parameter is ignored
        'use_pearson_correlation': False,
        'use_endpoint_firing': False,  # Add [d1_fired, d2_fired, is_boundary] to node features
        'start_from_oracle': True,   # If True: seed each episode at oracle weights and disable CMA reweighting
        'use_log_joint_prob': True,  # Whether to use joint probabilities for CMA updates
        'n_layers': 1, # Number of GNN layers (affects receptive field size)

        # Agent / NN Settings
        # actor_type: 'gnn' = expressive GNN+MLP policy (default);
        #             'linear_cm' = interpretable linear policy hard-wired to the
        #             correlated-matching form (learns coefficients toward 1, -1, -1).
        #             'linear_cm' REQUIRES use_log_joint_prob=True (edge feature = -log p(e_mu,e_nu)).
        'actor_type': 'linear_cm',
        'linear_cm_squash': False,         # If False, action is unsquashed (clean linear coefficients)
        'linear_cm_init_identity': True,  # If True, init coefficients at the CM solution (1,-1,-1)

        # Training algorithm: 'sac' (actor-critic), 'reinforce' (actor-only vanilla policy
        # gradient), or 'ppo' (actor-only clipped surrogate with importance-weighted reuse).
        # 'reinforce'/'ppo' suit the contextual-bandit setting and the tiny linear_cm actor;
        # both use a zero baseline (the env reward is differential) and never touch the critic.
        'algo': 'ppo',
        'reinforce_batch': 512,            # on-policy transitions per gradient step (reinforce & ppo)
        'reinforce_std': 1.5,              # FIXED exploration std at the START of training (weight units)
        'reinforce_std_final': 0.01,        # anneal the fixed std linearly to this by the last episode (None = constant)
        'reinforce_lr': 1e-3,              # actor lr for the policy-gradient step (reinforce & ppo; None reuses 'lr')
        'ppo_clip_eps': 0.2,               # PPO trust-region clip eps (paper uses 0.4 for surface codes)
        'ppo_epochs': 4,                   # PPO gradient epochs per batch (gentle reuse; 10 diverged here)
        # Trainable exploration std (the payoff of PPO's clip). If True, std is learned,
        # initialised at reinforce_std and clamped to [reinforce_std_min, reinforce_std_max]
        # (the floor guards against collapse). NOTE: with the weak differential reward it
        # collapsed to the floor and the coefficients ran away -> keep False (fixed std=1.5).
        'reinforce_trainable_std': False,
        'reinforce_std_min': 0.3,
        'reinforce_std_max': 3.0,
        'hidden_dim': 256,
        'lr': 1e-4,
        'alpha_lr': None,       # If None, alpha optimizer reuses lr; set smaller (e.g. 1e-5) to slow entropy decay
        'gamma': 0.0,          # 0.0 for Contextual Bandit (Crucial for QEC!)
        'tau': 0.005,
        'alpha': 0.01,          # Entropy tuning
        'target_entropy': -1.0,  # Target entropy for automatic alpha tuning
        'batch_size': 256,
        'buffer_capacity': 100_000,
        'update_frequency': 100,
        
        # Episode Settings
        'train_episodes': 1000,
        'test_episodes': 20,

        # Validation (REINFORCE): every `val_every` episodes, run the greedy policy on
        # `val_episodes` fixed-seed circuits and log/plot the mean reward (low-variance).
        'val_every': 10,
        'val_episodes': 3,
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

    # The linear CM actor only matches the analytical rule when the line-graph edge
    # feature is w_{mu,nu} = -log p(e_mu, e_nu), i.e. use_log_joint_prob must be True.
    if CONFIG['actor_type'] == 'linear_cm' and not CONFIG['use_log_joint_prob']:
        raise ValueError(
            "actor_type='linear_cm' requires use_log_joint_prob=True so that the edge "
            "feature is the -log joint probability w_{mu,nu} expected by correlated matching."
        )

    # The linear CM actor is cleanest with action_scale = 1: the coefficients then map
    # DIRECTLY to weight deltas (no /action_scale rescaling) and the exploration std is
    # already in weight units. Force it regardless of the configured value.
    if CONFIG['actor_type'] == 'linear_cm' and CONFIG['action_scale'] != 1.0:
        print(f"[i] actor_type='linear_cm': overriding action_scale "
              f"{CONFIG['action_scale']} -> 1.0 (coefficients map directly to weight deltas).")
        CONFIG['action_scale'] = 1.0

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=CONFIG['local_action_only'],
        local_action_hops=CONFIG['local_action_hops'],
        action_scale=CONFIG['action_scale'],
        update_period=CONFIG['update_period'],
        prior_shots=CONFIG['prior_shots'],
        n_test_shots=CONFIG['n_test_shots'],             
        use_pearson_correlation=CONFIG['use_pearson_correlation'],
        use_syndrome_features=False,
        use_endpoint_firing=CONFIG.get('use_endpoint_firing', False),
        use_log_joint_prob=CONFIG['use_log_joint_prob'],
        start_from_oracle=CONFIG.get('start_from_oracle', False),
        update_with='DGR',
        train_mode=(CONFIG['MODE'] == 'train')
    )

    # Determine dynamic dimensions from environment
    sample_obs, _ = env.reset()
    NODE_DIM = sample_obs["node_features"].shape[1]

    # Exploration-std setup for the actor-only policy-gradient algos.
    #   * trainable std (PPO payoff): std learned, init reinforce_std, clamped [min, max].
    #   * fixed std (default REINFORCE): std held constant at reinforce_std.
    #   * SAC: learnable std with wide default bounds (managed by the alpha tuner).
    _is_pg = CONFIG['algo'] in ('reinforce', 'ppo')
    _trainable_std = _is_pg and CONFIG.get('reinforce_trainable_std', False)
    _fixed_std = CONFIG['reinforce_std'] if (_is_pg and not _trainable_std) else None
    _init_std = CONFIG['reinforce_std'] if _trainable_std else None
    _std_min = CONFIG['reinforce_std_min'] if _trainable_std else None
    _std_max = CONFIG['reinforce_std_max'] if _trainable_std else None

    # Init Agent
    agent = SACAgent(
        node_dim=NODE_DIM,
        hidden_dim=CONFIG['hidden_dim'],
        static_edge_index=env.line_edge_index,  # Pass the static edge index to the agent
        lr=CONFIG['lr'],
        alpha_lr=CONFIG.get('alpha_lr', None),
        gamma=CONFIG['gamma'],
        tau=CONFIG['tau'],
        alpha=CONFIG['alpha'],
        target_entropy=CONFIG['target_entropy'],
        n_layers=CONFIG['n_layers'],
        actor_type=CONFIG['actor_type'],
        action_scale=CONFIG['action_scale'],
        linear_cm_squash=CONFIG['linear_cm_squash'],
        linear_cm_init_identity=CONFIG['linear_cm_init_identity'],
        # Exploration std: fixed (default REINFORCE) or trainable with floor/ceiling (PPO).
        linear_cm_fixed_std=_fixed_std,
        linear_cm_init_std=_init_std,
        linear_cm_std_min=_std_min,
        linear_cm_std_max=_std_max,
    )

    #total_params = sum(p.numel() for p in agent.actor.parameters())
    #print(f"Total Parameters: {total_params:,}")

    ############################
    # 3. EXECUTE SELECTED MODE #
    ############################
    if CONFIG['MODE'] == 'train':
        start_train = time.time()
        if CONFIG.get('algo', 'sac') in ('reinforce', 'ppo'):
            train_policy_gradient(env, agent, CONFIG)
        else:
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