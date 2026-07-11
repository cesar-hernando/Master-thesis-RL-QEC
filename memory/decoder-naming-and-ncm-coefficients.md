---
name: decoder-naming-and-ncm-coefficients
description: What CM / NCM / Neural mean across the scripts, and the learned linear-PPO coefficients
metadata:
  type: project
---

Decoder terminology used in this project (resolved from scripts/compare_linear_vs_cm.py):

- **CM** = correlated matching = PyMatching NATIVE `enable_correlations=True`
  (`from_detector_error_model(dem, enable_correlations=True)`). In compare_linear_vs_cm.py the
  "CM" series is this native decoder, NOT the linear (1,-1,-1,0) surrogate.
- **NCM / Neural / "Neural CM (linear)"** = the linear PPO agent
  `models/linear/linear_model_ppo_0_best.pth`, a `LinearCMActor` (gnn_sac_agent.py:184) whose
  per-edge discount is exactly
  delta_w_mu = sum_nu s_nu (c_joint*w_munu + c_self*w_mu + c_nbr*w_nu + bias),
  with squash=False, action_scale=1.0. Checkpoint keys live under ck['actor'].
  Learned coefficients (this model): c_joint=0.326, c_self=-0.395, c_nbr=-0.390, bias=-0.605
  -> a heavily DAMPED CM (~1/3 strength) with a -0.6 bias = the "residual weight" policy that
  beats native CM at low p. Analytical CM target is (1,-1,-1,0).
- Edge feature with use_log_joint_prob=True is `-log(corr_tracer + 1e-10)` = w_munu
  (drifted_matching_env.py:683).

ARCHITECTURE: the static decoding-graph structure (edge indexing, H, fault map, line graph, k-hop adjacency, endpoint maps) plus its structural ops (compute_action_mask, build_edge_reweights, analytical_cm_second_pass_weights, pearson_correlations, bidirectional_adjacency) live in `src/adaptiveQRL/decoding_graph.py` `DecodingGraph` (build with `DecodingGraph.from_dem(dem)`). `DriftedMatchingEnv` composes one as `env.graph` and aliases its arrays (env.H, env.line_edge_index, env.k_hop_adj_mat, ... still work); env keeps only dynamic per-episode state (current_weights, drifted tracers) and its `_compute_action_mask`/`_build_edge_reweights`/`compute_analytical_correlated_matching_action` are thin delegators to env.graph. `TwoPassCorrelatedMatching`/`LinearCorrelatedMatching` compose a `DecodingGraph` directly (no fake training env); `NeuralCorrelatedMatching` still builds an env (needs observation-building internals + GNN agent).

SRC DECODER CLASSES (all expose `decode_batch(shots)`; don't reimplement CM/NCM in notebooks/scripts — use these): `TwoPassCorrelatedMatching(dem, alpha=...)` = analytical CM / Pearl; `LinearCorrelatedMatching(dem, c_joint,c_self,c_nbr,bias)` = the linear NCM decoder (also `.from_checkpoint(dem, path)` to read the 4 coeffs); `NeuralCorrelatedMatching(dem, model_path)` = GNN agent. For analysis code that runs its own first/second pass and needs per-edge weights, both `LinearCorrelatedMatching` and `TwoPassCorrelatedMatching` have `.from_env(env, ...)` + public `.new_weights(sel, action_mask)`. `LinearCorrelatedMatching` also has `.set_coefficients(...)` for coefficient searches (cache the first pass, swap coeffs per eval). Reproduces the old hand-rolled `_ncm_new`/`DistProbe`/`ParametricCM` bit-for-bit.

VALIDATED: the hand-rolled two-pass conditional CM (implied_p=min(0.5, joint/marginal),
w=log((1-p)/p), min-discount across selected neighbours) reproduces native CM EXACTLY on
reweighted shots (agreement 1.0000 at p=3e-3, identical error counts). So the inspectable two-pass
form is a faithful proxy for native CM when you need per-edge discounts/updated weights.
See notebooks/cm_failure_mechanism_low_p.ipynb. Related: [[cm-weight-clipping]].
