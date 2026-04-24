"""
Realistic PyTorch Profiler for SAC-GNN QEC Decoder.
Uses the actual Environment and Agent logic to pinpoint exact bottlenecks.
"""

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent, GraphReplayBuffer

######################################
# 1. PROFILER CONFIGURATION          #
######################################
PROFILER_CONFIG = {
    'PROFILE_MODE': 'train', # Change to 'test' to profile inference & PyMatching
    'NUM_PROFILE_STEPS': 50, # How many batches/steps to record
    
    # Realistic Environment Settings (Mirroring main.py)
    'distance': 5,
    'n_rounds': 5,
    'p': 0.001,
    'p_gate_zz': 0.0,
    'mismatch': 1.0,
    'n_shots': 5_000, # REDUCED from 500k so env.reset() doesn't take forever during profiling
    'n_test_shots': 0,
    'burn_in_steps': 100,
    'bypass_threshold': 2,
    'action_scale': 3.0,
    'update_period': 1_000, 
    'prior_shots': 1_000,
    'local_action_only': True,
    'local_action_hops': 1,
    'use_pearson_correlation': True,
    'use_log_joint_prob': False,
    'n_layers': 1,
    
    # Agent Settings
    'hidden_dim': 128,
    'lr': 1e-4,
    'gamma': 0.0, 
    'tau': 0.005,
    'alpha': 0.01,
    'batch_size': 64,
    'buffer_capacity': 10_000,
}

def setup_env_and_agent(config):
    """Initializes the real environment and agent based on the config."""
    print("[*] Initializing SyndromeDataGenerator and DriftedMatchingEnv...")
    generator = SyndromeDataGenerator(
        distance=config['distance'], 
        n_rounds=config['n_rounds'], 
        mismatch=config['mismatch'],  
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": config["p"],
            "before_measure_flip_probability": config["p"],
            "after_reset_flip_probability": config["p"],
            "before_round_data_depolarization": config["p"],
            "p_gate_zz": config["p_gate_zz"]
        }, 
        memory_type='z', 
        n_shots=config['n_shots'], 
        qec_code='surface_code'
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=config['local_action_only'],
        local_action_hops=config['local_action_hops'],
        action_scale=config['action_scale'],
        update_period=config['update_period'],
        prior_shots=config['prior_shots'],
        n_test_shots=config['n_test_shots'],             
        use_pearson_correlation=config['use_pearson_correlation'],
        use_syndrome_features=False, 
        use_log_joint_prob=config['use_log_joint_prob'],
        update_with='DGR',
        train_mode=(config['PROFILE_MODE'] == 'train')
    )

    sample_obs, _ = env.reset(seed=42)
    node_dim = sample_obs["node_features"].shape[1]
    
    print("[*] Initializing SACAgent...")
    agent = SACAgent(
        node_dim=node_dim, 
        hidden_dim=config['hidden_dim'],
        static_edge_index=env.line_edge_index,
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        n_layers=config['n_layers']
    )
    
    return env, agent, sample_obs


def profile_training(config):
    """Profiles the Replay Buffer sampling and PyTorch Backward passes."""
    print(f"\n{'='*50}")
    print(f" PROFILING TRAINING MODE (Agent Updates)")
    print(f"{'='*50}")
    
    env, agent, obs = setup_env_and_agent(config)
    buffer = GraphReplayBuffer(capacity=config['buffer_capacity'])
    
    print("[*] Pre-filling Replay Buffer (running env steps)...")
    done = False
    step_count = 0
    # Fill enough batches so we don't sample the exact same data repeatedly
    target_buffer_size = config['batch_size'] * 10 
    
    while len(buffer) < target_buffer_size and not done:
        n_flashes = np.sum(env.current_syndrome != 0)
        if step_count < config['burn_in_steps'] or n_flashes <= config['bypass_threshold']:
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
        else:
            action = agent.select_action(obs, evaluate=False)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Only push complex shots, just like engine.py
        if step_count >= config['burn_in_steps'] and n_flashes > config['bypass_threshold']:
            buffer.push(obs, action, reward, next_obs if not done else None, done)
            
        obs = next_obs
        step_count += 1
        
    print(f"[*] Buffer filled with {len(buffer)} transitions.")
    print("[*] Running 2 warmup updates...")
    agent.update(buffer, config['batch_size'])
    agent.update(buffer, config['batch_size'])

    print(f"[*] Starting Profiler for {config['NUM_PROFILE_STEPS']} gradient updates...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(config['NUM_PROFILE_STEPS']):
            with record_function("01_agent_full_update"):
                # You can nest record_functions to break it down further if needed!
                agent.update(buffer, config['batch_size'])

    print_and_save_trace(prof, "trace_realistic_training.json")


def profile_testing(config):
    """Profiles the PyMatching Environment, CMA updates, and Inference."""
    print(f"\n{'='*50}")
    print(f" PROFILING TESTING MODE (Environment & Inference)")
    print(f"{'='*50}")
    
    env, agent, obs = setup_env_and_agent(config)
    
    print("[*] Running warmup steps...")
    for _ in range(5):
        action = np.zeros(env.n_dec_edges, dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
    
    done = False
    print(f"[*] Starting Profiler for {config['NUM_PROFILE_STEPS']} environment steps...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for i in range(config['NUM_PROFILE_STEPS']):
            if done: break
            
            n_flashes = np.sum(env.current_syndrome != 0)
            
            with record_function("01_agent_select_action"):
                if n_flashes <= config['bypass_threshold']:
                    action = np.zeros(env.n_dec_edges, dtype=np.float32)
                else:
                    action = agent.select_action(obs, evaluate=True)
            
            with record_function("02_env_step_total"):
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            obs = next_obs

    print_and_save_trace(prof, "trace_realistic_testing.json")


def print_and_save_trace(prof, filename):
    """Formats and exports the profiler data."""
    print("\n" + "="*70)
    print("PROFILER RESULTS (Sorted by CPU Time)")
    print("="*70)
    # Filter out basic Python overhead so we only see PyTorch/PyMatching operations
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("PROFILER RESULTS (Sorted by CUDA Time)")
        print("="*70)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    prof.export_chrome_trace(filename)
    print(f"\n[*] Chrome Trace saved to: {filename}")
    print("[*] Open Google Chrome, navigate to chrome://tracing, and drag-and-drop the JSON file.")

if __name__ == "__main__":
    mode = PROFILER_CONFIG['PROFILE_MODE']
    if mode == 'train':
        profile_training(PROFILER_CONFIG)
    elif mode == 'test':
        profile_testing(PROFILER_CONFIG)
    else:
        print("Invalid PROFILE_MODE. Choose 'train' or 'test'.")