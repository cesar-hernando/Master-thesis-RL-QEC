# Master's Thesis: *Adaptive Quantum Error Decoding Under Drift Noise via Graph Reinforcement Learning*

This repository contains the codebase for my master's thesis project in the Applied Quantum Algorithms Group (Leiden University), which is part of the joint Quantum Information Science and Technology program (TU Delft & Leiden University).

## Overview

This project develops an adaptive quantum error correction decoder that combines Graph Neural Networks (GNN) with Soft Actor-Critic (SAC) reinforcement learning to reweight Minimum-Weight Perfect Matching (MWPM) decoders. The approach handles time-drifting error rates and short-range correlations in surface codes, inspired by the [DGR paper](https://arxiv.org/abs/2411.04585).

### Problem Statement

Standard MWPM decoders assume static, independent error models with fixed global edge weights. Real quantum devices experience:
- Slowly drifting error rates over time
- Local correlations between errors
- Spatially/temporally varying error behavior

Fixed global weights cannot capture these dynamics, degrading decoder performance.

### Solution

We use a **GNN-SAC hybrid decoder** that:
1. **Preserves efficiency**: Builds on proven MWPM matching algorithm
2. **Adapts locally**: Uses GNN to predict edge reweightings only for edges correlated with errors
3. **Learns online**: SAC agent adapts to drift and correlations during deployment
4. **Maintains interpretability**: Outputs continuous reweightings (vs. black-box neural decoders)

---

## Installation

### Prerequisites
- **Python 3.9+**
- **pip** or **conda**

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/cesar-hernando/Master-thesis-RL-QEC.git
   cd Master-thesis-RL-QEC
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   # Using venv
   python -m venv .venv
   source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

   # OR using conda
   conda create -n qec-rl python=3.10
   conda activate qec-rl
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

   This installs the `adaptiveQRL` package and all dependencies:
   - **Quantum**: `stim`, `pymatching`
   - **ML/RL**: `torch`, `torch_geometric`, `gymnasium`
   - **Utilities**: `numpy`, `scipy`, `matplotlib`, `plotly`

### Verify Installation
To verify everything is installed correctly:
```bash
python -c "import adaptiveQRL; print('Installation successful!')"
python scripts/test_env.py    # Run environment tests
```

---

## Usage

### Configuration
All execution modes are controlled via the `CONFIG` dictionary in [scripts/main.py](scripts/main.py). Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODE` | Execution mode: `'train'`, `'test'`, or `'analyze_policy'` | `'train'` |
| `distance` | Surface code distance | 5 |
| `n_rounds` | Number of syndrome extraction rounds | 5 |
| `p` | Physical error rate | 0.004 |
| `mismatch` | Drift simulation factor (log-uniform range) | 30.0 |
| `n_shots` | Syndrome samples per episode | 65,000 |
| `local_action_hops` | Local patch radius (edges within N hops of selected edges) | 1 |
| `train_episodes` | Training episodes | 250 |
| `model_path` | Path to pretrained model (for testing) | `'models/sac_gnn_29.pth'` |

### Running the Code

#### 1. **Training a New Model**
```bash
# In scripts/main.py, set:
CONFIG['MODE'] = 'train'
CONFIG['train_episodes'] = 100  # Adjust as needed

# Then run:
python scripts/main.py
```

Trained models are saved to `models/sac_gnn_<N>.pth`.

#### 2. **Testing a Pretrained Model**
```bash
# In scripts/main.py, set:
CONFIG['MODE'] = 'test'
CONFIG['model_path'] = 'models/sac_gnn_29.pth'  # Choose a model
CONFIG['test_episodes'] = 3

# Then run:
python scripts/main.py
```

Returns logical error rates and comparison with standard MWPM.

#### 3. **Analyzing a Trained Policy**
```bash
# In scripts/main.py, set:
CONFIG['MODE'] = 'analyze_policy'
CONFIG['model_path'] = 'models/sac_gnn_29.pth'

# Then run:
python scripts/main.py
```

Generates visualizations of learned weight distributions and edge importance.

---

## Project Architecture

### Core Components

| Module | Purpose |
|--------|---------|
| [`surface_code_stim.py`](src/adaptiveQRL/surface_code_stim.py) | Rotated surface-code circuit builder; extracts DEM & decoding graph |
| [`syndrome_data_generation.py`](src/adaptiveQRL/syndrome_data_generation.py) | Simulates drift, generates syndrome data & MWPM predictions |
| [`drifted_matching_env.py`](src/adaptiveQRL/drifted_matching_env.py) | Gymnasium-compatible environment; applies local reweighting actions |
| [`gnn_sac_agent.py`](src/adaptiveQRL/gnn_sac_agent.py) | GNN encoder + SAC agent + replay buffer |
| [`engine.py`](src/adaptiveQRL/engine.py) | Training, testing, and analysis pipelines |

### Project Pipeline

1. **Circuit & DEM Generation**: Build Stim rotated surface code, extract Detector Error Model (DEM) and decoding graph
2. **Drift Simulation**: Multiply error probabilities by log-uniformly sampled mismatch factors (1/N to N)
3. **Syndrome Sampling**: Run quantum memory experiment → syndrome volume + logical observable label
4. **First MWPM Pass**: Identify activated edges in decoding graph
5. **Correlation Graph**: Build meta-graph where:
   - Nodes = decoding graph edges
   - Node attributes = edge weight + label (activated?)
   - Edges = DEM-determined correlations between decoding edges
   - Edge attributes = co-occurrence statistics
6. **GNN Prediction**: Agent predicts local weight reweightings via GNN encoder
7. **Masked Actions**: Apply reweightings only to edges within `local_action_hops` of activated edges
8. **Second MWPM Pass**: Re-decode with adjusted weights
9. **Reward & Update**: Compute rewards based on logical errors; update statistics for next step
10. **Episode Loop**: Repeat steps 3–9 for n_shots (constitutes one MDP episode)
11. **Training Loop**: Repeat episodes to learn optimal policy

### Directory Structure

```
.
├── src/adaptiveQRL/              # Main package
│   ├── surface_code_stim.py       # Surface code simulator
│   ├── syndrome_data_generation.py # Data generation
│   ├── drifted_matching_env.py    # RL environment
│   ├── gnn_sac_agent.py           # Agent & replay buffer
│   ├── engine.py                  # Training pipelines
│   └── plot_utils.py              # Visualization utilities
├── scripts/
│   ├── main.py                    # Main entry point (training/testing/analysis)
│   ├── test_env.py                # Environment integration tests
│   ├── test_env_profiling.py      # Performance profiling
│   ├── analyze_rl_reweighting.py  # Policy analysis
│   ├── parse_logs.py              # Log parsing utilities
│   └── scaling_decoding_graph.py  # Scalability studies
├── models/                        # Pretrained SAC-GNN checkpoints
│   └── sac_gnn_*.pth
├── plots/                         # Experiment outputs & visualizations
├── notebooks/
│   └── decoding_graph.ipynb       # Interactive exploration & plotting
└── requirements.txt               # Dependencies

```

---

## Pretrained Models

Multiple trained models are available in the `models/` directory:
- `sac_gnn_1.pth` through `sac_gnn_34.pth` — Models trained on various configurations
- `sac_gnn_dim256.pth` — Variant with larger hidden dimension (256)

Load any model via:
```python
CONFIG['model_path'] = 'models/sac_gnn_29.pth'
```

---

## Important Notes

### Performance Optimization
- By default, the environment computes reference MWPM metrics (`compute_reference_decoders=False`). This adds overhead.
- For training, keep this disabled to avoid 2× decoding cost per shot.
- For testing/analysis, enable it to compare against MWPM baseline.

### Key Hyperparameter: Contextual Bandit Mode
- The agent is trained with `gamma=0.0`, treating the problem as a **contextual bandit**.
- This means the agent maximizes immediate reward (logical error rate) without discounting future rewards.
- **Critical for QEC**: Surface code decoding has no "future value"; corrections must happen per shot.

### Local Action Masking
- `local_action_only=True` + `local_action_hops=1` restricts reweighting to edges close to detected errors.
- This improves generalization and reduces action space from O(d²) to O(d) locally selected edges.

---

## Analysis & Visualization

### Jupyter Notebook
Interactive exploration:
```bash
jupyter notebook notebooks/decoding_graph.ipynb
```

### Plotting & Analysis Scripts
```bash
# Analyze learned reweighting patterns
python scripts/analyze_rl_reweighting.py

# Benchmark reweighting performance
python scripts/scaling_decoding_graph.py

# Profile environment performance
python scripts/test_env_profiling.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{hernando2026adaptive,
  author = {Hernando, Cesar},
  title = {Adaptive Quantum Error Decoding Under Drift Noise via Graph Reinforcement Learning},
  school = {Leiden University},
  year = {2026}
}
```

Related work: [DGR: Tackling Drifted and Correlated Noise in Quantum Error Correction via Decoding Graph Re-weighting](https://arxiv.org/abs/2411.04585)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors for `stim` or `pymatching` | Run `pip install -e .` again to ensure all dependencies are installed |
| Out of memory during training | Reduce `n_shots`, `buffer_capacity`, or `batch_size` in `CONFIG` |
| Model not found | Ensure the `model_path` in `CONFIG` exists in the `models/` directory |
| Slow environment reset | Check `compute_reference_decoders` setting; disable for training |

---

## License

Apache 2.0

## Contact

**Author**: Cesar Hernando  
**Email**: chernandodelaf@tudelft.nl  
