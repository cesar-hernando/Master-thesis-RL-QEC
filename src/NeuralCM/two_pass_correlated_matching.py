"""
Two-pass correlated matching with the analytical conditional-probability rule.

This is the *hand-rolled* correlated matcher used as the "CM" baseline in engine.py,
exposed with a BeliefMatching-style decode_batch() API so it can be benchmarked against
PyMatching's built-in correlated matching (enable_correlations=True) on equal footing.

Pipeline (per shot, mirrors DriftedMatchingEnv.step with a CM action):
    1. First-pass MWPM (batched over all shots).
    2. Shots with <= bypass_threshold detection events keep the first-pass result.
    3. For each remaining shot, reweight edges that neighbour a first-pass-selected
       edge using the conditional-probability rule
           P(A | B) = P(A and B) / P(B),  w = log((1 - P) / P)
       taking the largest discount (lowest weight) across selected neighbours.
       This is DecodingGraph.analytical_cm_second_pass_weights().
    4. Second-pass MWPM with those edge reweights.

The static structure and the analytical reweighting math live in DecodingGraph; this
class only holds the dynamic decoder state (current weights, occurrence/correlation
statistics, alpha) and drives the two passes. The static correlation statistics come
from the base DEM: occ = base_p (marginal edge probabilities), corr = initial_corr_tracer
(pairwise joint probabilities).
"""

import numpy as np
import pymatching

from NeuralCM.decoding_graph import DecodingGraph


class TwoPassCorrelatedMatching:
    """Analytical two-pass correlated matcher driven from a DEM."""

    def __init__(self, dem, action_scale=5.0, bypass_threshold=0,
                 local_action_hops=1, chunk_size=50_000, alpha=1.0,
                 graph=None, weights=None, occ=None, corr=None,
                 min_weight=1e-6, max_weight=50.0, num_observables=None):
        # Static decoding-graph structure (built from the DEM, or supplied via from_env).
        if graph is None:
            graph = DecodingGraph.from_dem(dem, local_action_hops=local_action_hops)
        self.graph = graph

        # Dynamic decoder state.
        self.current_weights = (weights.astype(np.float32).copy() if weights is not None
                                else graph.initial_weights.copy())
        self.occ = (occ.copy() if occ is not None else graph.base_p.copy())
        self.corr = (corr.copy() if corr is not None else graph.initial_corr_tracer.copy())
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Single matching object reused for both passes (returns selected edges).
        self._matching = pymatching.Matching.from_check_matrix(
            graph.H, weights=self.current_weights
        )

        self.action_scale = action_scale   # kept for API compatibility
        self.bypass_threshold = bypass_threshold
        self.chunk_size = chunk_size
        # alpha=1.0 -> ordinary (hard-evidence) CM; alpha<1 -> Pearl soft evidence.
        self.alpha = alpha
        if num_observables is not None:
            self.num_observables = num_observables
        elif dem is not None:
            self.num_observables = dem.num_observables
        else:
            self.num_observables = 1  # single-observable memory experiment

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env, alpha=1.0, **kwargs):
        """Wrap an already-built DriftedMatchingEnv: share its DecodingGraph and current
        (drifted) weights/tracers, so analysis code can reuse one env."""
        return cls(None, alpha=alpha, graph=env.graph, weights=env.current_weights,
                   occ=env.occ_tracer, corr=env.corr_tracer,
                   min_weight=env.min_weight, max_weight=env.max_weight,
                   action_scale=env.action_scale,
                   num_observables=env.base_dem.num_observables, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_weights(self, sel, action_mask):
        """Analytical CM (Pearl-alpha) second-pass weights for one shot.

        Given the first-pass selection mask `sel` (bool, per decoding edge) and the env
        action mask, returns the clipped reweighted weight vector (largest discount across
        selected neighbours, tempered by alpha)."""
        spw = self.graph.analytical_cm_second_pass_weights(
            self.current_weights, np.asarray(sel, dtype=bool), action_mask,
            self.occ, self.corr, alpha=self.alpha,
        )
        applied = (spw - self.current_weights) * action_mask
        return np.clip(self.current_weights + applied,
                       self.min_weight, self.max_weight).astype(np.float32)

    def decode(self, shot):
        return self.decode_batch(shot[np.newaxis, :])[0]

    def decode_batch(self, shots):
        """shots: (n_shots, n_detectors). Returns (n_shots, n_observables) uint8."""
        shots = np.asarray(shots)
        n_shots = shots.shape[0]
        if n_shots == 0:
            return np.zeros((0, self.num_observables), dtype=np.uint8)
        if n_shots <= self.chunk_size:
            return self._decode_chunk(shots)
        out = np.empty((n_shots, self.num_observables), dtype=np.uint8)
        for start in range(0, n_shots, self.chunk_size):
            end = min(start + self.chunk_size, n_shots)
            out[start:end] = self._decode_chunk(shots[start:end])
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decode_chunk(self, shots):
        g = self.graph
        n_shots = shots.shape[0]

        # 1. Batched first-pass MWPM.
        edges_first = np.asarray(
            self._matching.decode_batch(shots, enable_correlations=False), dtype=bool
        )
        # Observable parity = sum of selected fault edges (mod 2). Index the fault
        # columns directly instead of casting the whole (n_shots, n_edges) array to
        # int64 -- the latter allocates ~n_shots * n_edges * 8 bytes for nothing.
        first_obs = edges_first[:, g.fault_array].sum(axis=1) % 2
        first_obs = np.atleast_2d(first_obs).reshape(n_shots, -1).astype(np.uint8)

        # 2. Bypass split: trivial shots keep the first-pass result.
        non_bypass = np.flatnonzero(
            shots.astype(np.int64).sum(axis=1) > self.bypass_threshold
        )
        result = first_obs.copy()

        # 3+4. Per-shot analytical reweighting + second-pass MWPM.
        for s in non_bypass:
            sel = edges_first[s]
            selected_idx = np.flatnonzero(sel)
            if selected_idx.size == 0:
                continue

            action_mask = g.compute_action_mask(selected_idx)
            new_w = self.new_weights(sel, action_mask)
            if np.array_equal(new_w, self.current_weights):
                continue

            reweights = g.build_edge_reweights(new_w, self.current_weights)
            edges = self._matching.decode(
                shots[s], enable_correlations=False, edge_reweights=reweights
            )
            pred = (np.asarray(edges, dtype=np.int64) @ g.fault_array) % 2
            result[s] = np.atleast_1d(pred).astype(np.uint8)

        return result
