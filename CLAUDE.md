# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Project**: Adaptive Quantum Error Decoding under Drift Noise via Graph Reinforcement Learning  
**Primary Language**: Python 3.11 (Windows) / 3.9+ (Linux/macOS)  
**Main Framework**: PyTorch + PyTorch Geometric + Gymnasium + Stim Quantum Simulator  
**Entry Point**: `scripts/main.py`

## Installation & Environment

```bash
# Windows users (required for pre-compiled backend)
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/macOS users
source .venv/bin/activate

# Install the custom PyMatching fork (adds the regularized `alpha` on top of
# PyMatching's own correlated matching, enable_correlations=True).
# NOT a PyPI wheel: it is a C++ extension built from source via CMake, so a C++
# toolchain is required (Windows: Visual Studio Build Tools + CMake). See README
# section 3 for the Windows short-path build that avoids path-length failures.
pip install "pymatching @ git+https://github.com/cesar-hernando/PyMatching.git@69d7c049"

# Install package in development mode
pip install -e .

# Verify installation
python -c "import pymatching; import numpy as np; m = pymatching.Matching(); m.add_edge(0, 1); m.decode_to_edges_array(np.array([1, 1]), edge_reweights=np.array([[0, 1, 0.5]])); print('SUCCESS: Backend is working!')"
```

## Core Commands

### Training a New Model
```bash
# Edit scripts/main.py:
# - Set CONFIG['MODE'] = 'train'
# - Adjust CONFIG['train_episodes'] and hyperparameters as needed
python scripts/main.py
```
**Output**: Trained model saved to `models/sac_gnn_<N>.pth`

### Testing a Pretrained Model
```bash
# Edit scripts/main.py:
# - Set CONFIG['MODE'] = 'test'
# - Set CONFIG['model_path'] = 'models/sac_gnn_57_best.pth' (or another model)
# - Adjust CONFIG['test_episodes'] as needed
python scripts/main.py
```
**Output**: Logical error rates (LER) and comparison with baseline MWPM

### Analyzing a Trained Policy
```bash
# Edit scripts/main.py:
# - Set CONFIG['MODE'] = 'analyze_policy'
# - Set CONFIG['model_path'] = 'models/sac_gnn_29.pth'
python scripts/main.py
```
**Output**: Visualizations of learned weight distributions and edge importance

### Interactive Exploration
```bash
# Jupyter notebook for exploring decoding graphs and results
jupyter notebook notebooks/decoding_graph.ipynb

# Analyze learned reweighting patterns
python scripts/analyze_rl_reweighting.py

# Benchmark reweighting performance
python scripts/scaling_decoding_graph.py

# Profile environment performance
python scripts/test_env_profiling.py
```

## Project Architecture

### Data Flow & Pipeline

1. **Circuit & Graph Generation** (`surface_code_stim.py`)
   - Builds rotated surface code using Stim simulator
   - Extracts Detector Error Model (DEM) and decoding graph structure
   - Output: Decoding graph with error probabilities for each edge

2. **Syndrome Data Generation** (`syndrome_data_generation.py`)
   - Simulates quantum noise with configurable error rates
   - Multiplies error probabilities by log-uniformly sampled "drift" factors (simulating slow temporal drift)
   - Samples syndrome vectors and their corresponding logical observable labels
   - Output: Syndrome data batch for an RL episode

3. **RL Environment** (`drifted_matching_env.py`)
   - Gymnasium-compatible environment wrapping the two-pass MWPM decoder
   - **First Pass**: Standard MWPM identifies which edges are activated
   - **Agent Intervention**: GNN encodes the decoding graph as a "correlation graph" where:
     - Nodes = decoding graph edges
     - Node features = [edge_weight, is_activated_in_first_pass]
     - Edges = DEM-determined statistical correlations between decoding edges
     - Edge features = co-occurrence statistics
   - **Action**: Agent predicts local reweighting multipliers for edges near activated edges
   - **Second Pass**: MWPM re-runs with adjusted weights
   - **Reward**: Improvement in logical error prediction (contextual bandit setting: γ=0.0)

4. **GNN-SAC Agent** (`gnn_sac_agent.py`)
   - **Encoder**: Graph Convolutional Network (GCN) with configurable layers and hidden dim
   - **Actor**: Maps graph embedding → deterministic policy for edge reweighting
   - **Critic**: Two Q-networks (target + live) for off-policy learning
   - **Replay Buffer**: GraphReplayBuffer stores PyTorch Geometric Data objects
   - **Algorithm**: Soft Actor-Critic with entropy regularization for exploration

5. **Training Pipeline** (`engine.py`)
   - Train loop: Episode collection → Replay buffer accumulation → SAC updates
   - Validation: Rigorous isolated comparison (zero-action baseline vs. learned policy)
   - Testing: Statistical evaluation on held-out episodes
   - Analysis: Policy interpretation (which edges/patterns does the agent reweight?)

### Key Data Structures

**State Graph** (PyTorch Geometric Data object):
```
node_features (N_edges, 2): [edge_weight, is_activated]
edge_index (2, M_correlations): Connectivity in correlation graph
edge_attr (M_correlations, 1): Correlation strength
action_mask (N_edges): Binary mask for local action application
```

**Action Vector**: Shape (N_edges,), values ∈ [-action_scale, action_scale]  
**Observation**: Dictionary with `node_features`, `edge_attr`, `edge_index`, `action_mask`

### Key Configuration Parameters (scripts/main.py)

| Parameter | Meaning | Notes |
|-----------|---------|-------|
| `MODE` | 'train', 'test', or 'analyze_policy' | Execution mode |
| `distance` | Surface code distance | Determines problem size |
| `n_rounds` | Syndrome extraction rounds | Time dimension |
| `p` | Physical error rate | Baseline noise level |
| `mismatch` | Drift factor range | 1.0 = no drift, 30.0 = 30x noise range |
| `n_shots` | Shots per episode | Length of episode (more = longer training) |
| `local_action_only` | Boolean | If True, agent can only modify nearby edges |
| `local_action_hops` | Integer | Radius for "nearby" edges |
| `hidden_dim` | GNN hidden dimension | Model size / capacity |
| `gamma` | RL discount factor | **Must be 0.0 for QEC (contextual bandit)** |
| `tau` | Soft target update rate | Lower = slower updates |
| `batch_size` | SAC training batch | Higher = faster learning, less stable |
| `buffer_capacity` | Replay buffer size | Larger = more memory, less correlated samples |
| `train_episodes` | Episodes to train | Number of full episodes |

## Module Functions & Key Classes

### `surface_code_stim.py`
- **SurfaceCodeCircuit**: Builds Stim circuit and extracts DEM + decoding graph
  - `get_dem()`: Returns DetectorErrorModel
  - `get_decoding_graph()`: Returns PyMatching graph with edge weights

### `syndrome_data_generation.py`
- **SyndromeDataGenerator**: Generates syndrome samples with optional drift noise
  - `generate_syndrome_batch()`: Returns syndromes and observable labels
  - `get_detector_error_model()`: Access to DEM

### `drifted_matching_env.py`
- **DriftedMatchingEnv**: Gymnasium environment
  - `reset(seed)`: Initialize new episode
  - `step(action)`: Applies action, runs second MWPM pass, computes reward
  - `render()`: Visualize correlation graph (optional)

### `gnn_sac_agent.py`
- **GraphReplayBuffer**: Stores graph-structured transitions
- **GNNActor**: Policy network (GCN → action sampling)
- **GNNQNetwork**: Critic network (GCN → Q-value)
- **SACAgent**: Orchestrates training updates

### `engine.py`
- `train(env, agent, config)`: Main training loop
- `test(env, agent, config)`: Evaluation on test episodes
- `analyze_policy(env, agent, config)`: Generate policy visualizations

## Common Development Tasks

### Running a Single Training Episode in Debug Mode
```python
# In scripts/main.py, modify CONFIG:
CONFIG['MODE'] = 'test'  # Single episode
CONFIG['train_episodes'] = 1
CONFIG['n_shots'] = 100  # Small episode

# Then run with Python debugger
python -m pdb scripts/main.py
```

### Profiling the Environment
```bash
python scripts/test_env_profiling.py
```
This benchmarks syndrome generation, first-pass MWPM, and environment overhead.

### Checking MWPM Integration
```bash
# scripts/test_env.py runs integration tests
python scripts/test_env.py
```

### Comparing Multiple Models
Edit `scripts/main.py` to loop over multiple `model_path` values in `test` mode, or use:
```bash
python scripts/analyze_rl_reweighting.py  # Compares learned vs. baseline patterns
```

## Important Quirks & Gotchas

1. **Gamma Must Be 0.0**: QEC is a contextual bandit problem (no temporal credit assignment). Setting `gamma > 0` breaks the learning signal.

2. **Windows Python 3.11 Only**: Pre-compiled PyMatching wheel is Python 3.11-specific. Linux/macOS can use 3.9+.

3. **Custom PyMatching Fork**: Main dependency is a custom fork with `decode_to_edges_array()` method for edge-level reweighting. Standard PyMatching won't work.

4. **GCN Receptive Field**: `n_layers` parameter controls how far each edge "sees" correlations. Larger = more context, more computation.

5. **Episode Length**: Increasing `n_shots` increases episode length and training time quadratically (more steps per episode = more gradient updates).

6. **Local Action Masking**: If `local_action_only=True`, the action mask constrains which edges can be modified. Edges far from activated edges get forced to zero action.

7. **Seed Management**: Use fixed seeds in validation loops (hardcoded in `validate()` function) to ensure reproducible comparisons between policies.

8. **Model Checkpointing**: Best model is auto-saved during training; intermediate models are also saved. Check `models/` directory for available checkpoints.

## Code Style & Conventions

- **Environment Variables**: None required (all config in `CONFIG` dict)
- **Logging**: Uses `print()` statements (no structured logging)
- **Error Handling**: Minimal; assumes correct input formats
- **Type Hints**: Sparse (mostly in SAC agent)
- **Testing**: Ad-hoc scripts in `scripts/test_*.py`; no formal test suite

## Debugging Tips

1. **Reward Not Improving**: Check if `gamma=0.0`, validate that episodes have correct mix of improving/worsening syndromes, check if agent is updating at all (print actor loss).

2. **MWPM Errors**: Verify custom PyMatching is installed with `python -c "import pymatching; print(pymatching.__file__)"` — should point to the custom fork.

3. **Memory Issues**: Reduce `n_shots`, `hidden_dim`, or `batch_size`. Environment builds full syndrome batch in memory.

4. **Graph Construction Issues**: If correlation graph is sparse or empty, check `use_pearson_correlation` and `use_log_joint_prob` settings.

5. **Loss NaNs**: Typically indicates numerical instability; try lower learning rate (`lr`) or batch normalization in GNN.
