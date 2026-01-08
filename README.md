# Master Thesis: Reinforcement Learning for Quantum Error Correction (RL-QEC) 

This repository contains the code for a Master's thesis project that investigates Reinforcement Learning (RL) methods for decoding quantum errors in the rotated surface code. The codebase provides a simulated environment for the rotated surface code lattice, visualization utilities, RL experiment scripts and logs.

## Project structure 

- `surface_code_env.py` — Core rotated surface code environment. Implements lattice geometry, error simulation, syndrome computation, an RL-style observation space and a `render()` visualization.
- `RL_surface_code.py` — Example RL training / experiment script that interacts with the environment and logs training runs (writes TensorBoard events under `logs/`).
- `neural_network.py` — Neural network model(s) used for RL agents.
- `utils.py` — Utility functions used across the project (helpers for plotting, metrics, etc.).
- `logs/` — TensorBoard event directories for experiments (e.g. `DQN_1/`, `DQN_2/` ...).
- `plots/` — Saved static visualizations produced by experiments.
- `README.md` — This file.

## Core ideas & architecture 

- The environment uses a rotated surface code represented on a `(2*d + 1) × (2*d + 1)` grid where data qubits are located on odd indices and stabilizers on even indices.
- Errors on data qubits are represented using ±1 encoding (easy product-based syndrome calculation):
	- `hidden_state[..., 0]` — X-component (bit flips)
	- `hidden_state[..., 1]` — Z-component (phase flips)
- Syndrome values are computed as products of support data qubits; syndromes are ±1 with `-1` indicating a triggered stabilizer.
- The environment exposes a multi-channel `visible_state` observation combining masks, syndrome channels and action history to the RL agent.

See `surface_code_env.py` for implementation details: `_assign_qubit_coordinates`, `_simulate_errors`, `_obtain_support_qubits`, `_stack_syndrome_and_history`, `_decode_action`, `_encode_action`, and `render()`.


