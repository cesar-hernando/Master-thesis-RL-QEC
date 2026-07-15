# Master's Thesis: *Adaptive Quantum Error Decoding Under Drift Noise via Graph Reinforcement Learning*

This repository contains the codebase for my master's thesis project in the Applied Quantum Algorithms Group (Leiden University), which is part of the joint Quantum Information Science and Technology program (TU Delft & Leiden University).

## Overview

This project develops an adaptive and correlation-aware quantum error correction decoder that combines Graph Neural Networks (GNN) with Soft Actor-Critic (SAC) reinforcement learning to reweight Minimum-Weight Perfect Matching (MWPM) decoders. 

### Problem Statement

Standard MWPM decoders assume static, independent error models with fixed global edge weights. Real quantum devices experience slowly drifting error rates and correlations between errors.

### Solution

We use a **GNN-SAC hybrid decoder** that:
1. **Preserves efficiency**: Builds on proven MWPM matching algorithm.
2. **Adapts locally**: Uses GNN to predict edge reweightings only for edges correlated with error edges predicted in the 1st MWPM pass.
3. **Learns online**: SAC agent adapts to drift and correlations during deployment.
4. **Maintains interpretability**: Outputs continuous reweightings.)

---

## Installation

**Prerequisites:**
* **Python Version:**
  * **Windows Users:** Python 3.11 is strictly required to install the pre-compiled backend wheel.
  * **Linux / macOS Users:** Python 3.9 or newer is supported.
* **C++ Compiler:** GCC 11+ or Clang is required for Linux and macOS users to build the custom backend from source.

### 1. Clone this repository
```bash
# Replace <project-dir> with your preferred folder name (e.g., qec-thesis)
git clone https://github.com/cesar-hernando/Master-thesis-RL-QEC.git <project-dir>
cd <project-dir>
```

### 2. Create and activate a virtual environment (Highly Recommended)
```bash
# Create the environment
python -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (Windows Command Prompt)
.venv\Scripts\activate.bat
```

### 3. Install Custom PyMatching Backend

This project depends on a **custom fork** of PyMatching that adds correlated matching
(`enable_correlations=True`) and a regularized reweight strength `alpha`:

- Fork: `https://github.com/cesar-hernando/PyMatching.git`
- Branch: `feature/decode-to-edges-edge-reweights`, pinned to commit `69d7c049`
  ("Transform Pearl's CM to Regularized CM").

> **This is not a wheel from PyPI — it is a C++ extension built from source with CMake.**
> You need a C++ toolchain: on Linux/macOS a recent gcc/clang + CMake; on Windows the
> **Visual Studio Build Tools** (Desktop development with C++) + **CMake**.

**For Mac and Linux Users**
```bash
pip install --upgrade pip setuptools wheel
pip install "pymatching @ git+https://github.com/cesar-hernando/PyMatching.git@69d7c049"
```

**For Windows Users**
The direct `git+` build can fail on Windows due to compiler path-length limits (the deep
temporary build directory overflows `MAX_PATH`). Build from a **short-path local clone**
instead:
```bash
# Clone the fork somewhere with a short path (e.g. C:\pm), then:
git clone --branch feature/decode-to-edges-edge-reweights https://github.com/cesar-hernando/PyMatching.git C:\pm
git -C C:\pm checkout 69d7c049
pip install --upgrade pip setuptools wheel
pip wheel C:\pm --no-deps -w C:\pm\dist
pip install --force-reinstall --no-deps C:\pm\dist\PyMatching-2.3.1-cp311-cp311-win_amd64.whl
```
(Enable long paths — `git config --system core.longpaths true` and the Windows
`LongPathsEnabled` registry key — if cloning/building still hits path-length errors.)

### 4. Install NeuralCM
```bash
pip install -e .
```

### 5. Verify installation
To verify that the custom reweighting logic compiled and linked correctly, run the following command:
```bash
python -c "import pymatching; import numpy as np; m = pymatching.Matching(); m.add_edge(0, 1); m.decode_to_edges_array(np.array([1, 1]), edge_reweights=np.array([[0, 1, 0.5]])); print('SUCCESS: Backend is working!')"
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
| [`surface_code_stim.py`](src/NeuralCM/surface_code_stim.py) | Rotated surface-code circuit builder; extracts DEM & decoding graph |
| [`syndrome_data_generation.py`](src/NeuralCM/syndrome_data_generation.py) | Simulates drift, generates syndrome data & MWPM predictions |
| [`drifted_matching_env.py`](src/NeuralCM/drifted_matching_env.py) | Gymnasium-compatible environment; applies local reweighting actions |
| [`gnn_sac_agent.py`](src/NeuralCM/gnn_sac_agent.py) | GNN encoder + SAC agent + replay buffer |
| [`engine.py`](src/NeuralCM/engine.py) | Training, testing, and analysis pipelines |

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
├── src/NeuralCM/              # Main package
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

## Contact

**Author**: Cesar Hernando  
**Email**: chernandodelaf@tudelft.nl  
