---
name: cm-weight-clipping
description: How the two-pass CM weight discount is clipped, and that native CM ignores those bounds
metadata:
  type: project
---

The hand-rolled two-pass correlated matching (CM) discount is bounded by TWO clips, and the
binding one is NOT `min_weight`/`max_weight`:

- `DriftedMatchingEnv` defaults `min_weight=1e-6`, `max_weight=50.0` (drifted_matching_env.py:79-80),
  applied as `np.clip(current_weights + delta, min_weight, max_weight)` (drifted_matching_env.py:836;
  also two_pass_correlated_matching.py:122-123 and the notebook ParametricCM).
- The actually-binding clip is inside `compute_analytical_correlated_matching_action`
  (drifted_matching_env.py:1236-1240): `implied_p = clip(P(A&B)/P(B), 1e-6, 0.499999)`,
  `w = log((1-implied_p)/implied_p)`. The 0.499999 cap floors the discounted weight at ~0
  (a "free" edge) BEFORE min_weight matters. Measured: at p=3e-3 the implied_p cap binds ~3% of
  correlated pairs while min_weight binds 0%; the cap-bind fraction grows as p drops.
- `max_weight=50` never binds for p>=~2e-22 (base weight log((1-p)/p) < 50); it only caps
  structurally-absent edges (oracle init = max_weight).
- PyMatching NATIVE CM (`from_detector_error_model(dem, enable_correlations=True)`,
  syndrome_data_generation.py:168, neural_correlated_matching.py:41) does NOT use the env's
  min_weight=1e-6 / max_weight=50, BUT it applies the SAME implied-probability <= 0.5 cap. Verified
  in the fork's C++ (oscarhiggott/PyMatching master, src/pymatching/sparse_blossom/driver/
  user_graph.cc): `populate_implied_edge_weights` does
  `implied_probability = std::min(0.5, joint/marginal)`, then
  `to_weight_for_correlations(p) = std::log((1-p)/p)` -> implied weight floors at log(1)=0 (free
  edge), same as the hand-rolled implied_p=0.499999 cap. Sign changes are forbidden ("Edge weight
  rewrite rules that change the sign of an edge weight are not currently supported"). The only hard
  ceiling is MAX_USER_EDGE_WEIGHT = NUM_DISTINCT_WEIGHTS-1 = (1<<24)-1 = 16,777,215 in log-odds
  units (never hit). Weights are normalized to ~2^24 integer levels via
  (NUM_DISTINCT_WEIGHTS-1)/max_abs_weight. So on the low-p free-edge floor, hand-rolled CM and
  native CM AGREE.

The floor is intrinsic: MWPM needs non-negative weights, so you cannot discount past a free edge.
Both your CM and native CM enforce it identically (implied prob capped at 0.5 -> weight 0).
This is why a damped policy (residual positive weight) beats full CM at low p — see notebook
notebooks/investigate_cm_weight_clipping.ipynb. Related: [[notebooks-optimize-cm-coefficients]].
