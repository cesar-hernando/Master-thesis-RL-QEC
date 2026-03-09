**Master's thesis: *Adaptive quantum error decoding under drift noise via Graph Reinforcement Learning***

This repository contains the codebase of my master's thesis project in the Applied Quantum Algorithms Group (AQA) in Leiden University. It is part of the master programme in Quantum Information Science and Technology jointly organized by TU Delft and Leiden University.

**Problem Statement & Approach**

- **Problem:** Quantum decoders based on Minimum-Weight Perfect Matching (MWPM) assume a static, independent error model. Real quantum devices often experience slowly drifting error rates and local correlations, which degrade MWPM performance because fixed global edge weights cannot capture evolving correlations or spatially/temporally varying error behavior.

- **Goal:** Build a decoder that adapts online to slow noise drift and can handle short-range correlations by adjusting how MWPM perceives the decoding graph, while retaining the efficiency and interpretability of matching-based decoders.

- **Approach:** We use a graph-based Soft Actor-Critic (SAC) agent with a Graph Neural Network (GNN) encoder to output continuous reweightings for decoding-graph edges. The environment applies those actions only on a local subset of edges determined by an `action_mask` and the `local_action_hops` parameter (see `drifted_matching_env.py`). After applying the learned, local edge reweighting, a second MWPM pass is performed. This hybrid workflow preserves MWPM's strengths while letting a learned, local policy adapt edge weights to compensate for drift and capture short-range correlations that global weights miss.

**Status:** research prototype — tools and notebooks for reproducible experiments.

**Quick Links**
- Stim wrappers: `surface_code_stim.py`, `syndrome_data_generation.py`
- Environment: `drifted_matching_env.py`
- Test: `test_env.py`

**Project layout (high level)**
- `surface_code_stim.py` — Stim-based rotated surface-code circuit builder used throughout experiments.
- `syndrome_data_generation.py` — Generate drifted circuits, sample detector syndrome volumes, and extract MWPM-selected edges.
- `drifted_matching_env.py` — Gym-compatible environment exposing graph observations (node/edge features) and applying local edge reweighting actions. 
- `decoding_graph.ipynb` — interactive experiments and visualizations.

**Core pipeline**

1. Build a Stim rotated-surface-code memory circuit and extract its Detector Error Model (DEM).
2. Build a PyMatching `Matching` from the DEM and run a first-pass MWPM decode.
3. Convert the matching → NetworkX graph → graph tensors for a GNN.
4. The policy outputs a continuous action vector over decoding-graph edges (shape = number of decoding edges). The environment masks these actions to a local subset (via `action_mask` and `local_action_hops`) and scales them to produce edge-weight deltas.
5. Reweight decoding edges using `w'(e) = clip(w(e) + scale * delta_e)` for masked edges and build a reweighted `Matching`.
6. Run second-pass MWPM on the reweighted graph and compute reward (logical success).
