"""
Linear correlated matching: the ``linear_cm`` actor's functional form applied as a
two-pass MWPM decoder, driven from a DEM and exposed with the same ``decode_batch()`` API
as :class:`TwoPassCorrelatedMatching` and :class:`NeuralCorrelatedMatching`.

This is the analytical, weights-only equivalent of running ``LinearCMActor`` inside
``DriftedMatchingEnv``: it reweights edge ``mu`` by

    delta_w_mu = sum_{nu in N(mu), nu selected} ( c_joint * w_{mu,nu}
                                                + c_self  * w_mu
                                                + c_nbr   * w_nu
                                                + bias )

summed over the first-pass-selected line-graph neighbours ``nu`` of ``mu`` (with ``mu``
restricted to the local-action mask), where

    w_mu      = current MWPM weight of edge mu
    w_nu      = current MWPM weight of neighbour edge nu
    w_{mu,nu} = -log P(e_mu, e_nu)   (the use_log_joint_prob edge feature)

so coefficients ``(c_joint, c_self, c_nbr, bias) = (1, -1, -1, 0)`` reproduce ordinary
correlated matching. Byte-for-byte identical to ``LinearCMActor._delta_w`` followed by the
env's masking/clipping.

The static structure lives in :class:`DecodingGraph`; this class holds only the dynamic
state (current weights, the joint-probability edge feature, the 4 coefficients).
"""

import numpy as np
import pymatching

from adaptiveQRL.decoding_graph import DecodingGraph


class LinearCorrelatedMatching:
    """Analytical two-pass matcher with the linear (4-coefficient) CM rule."""

    def __init__(self, dem, coef_joint=1.0, coef_self=-1.0, coef_nbr=-1.0, bias=0.0,
                 action_scale=1.0, bypass_threshold=2, local_action_hops=1,
                 chunk_size=50_000, weights=None, joint_prob_eps=1e-10,
                 graph=None, corr=None, min_weight=1e-6, max_weight=50.0,
                 num_observables=None):
        if graph is None:
            graph = DecodingGraph.from_dem(dem, local_action_hops=local_action_hops)
        self.graph = graph

        self.current_weights = (weights.astype(np.float32).copy() if weights is not None
                                else graph.initial_weights.copy())
        self.min_weight = min_weight
        self.max_weight = max_weight

        self._matching = pymatching.Matching.from_check_matrix(
            graph.H, weights=self.current_weights
        )

        # Symmetrised line graph + the -log joint-probability edge feature w_{mu,nu}.
        li = graph.line_edge_index
        self._src = np.concatenate([li[0], li[1]])
        self._dst = np.concatenate([li[1], li[0]])
        corr_arr = corr if corr is not None else graph.initial_corr_tracer
        joint = np.concatenate([corr_arr, corr_arr])
        self._w_joint = -np.log(joint.astype(np.float64) + joint_prob_eps)

        self.coef_joint = float(coef_joint)
        self.coef_self = float(coef_self)
        self.coef_nbr = float(coef_nbr)
        self.bias = float(bias)
        self.action_scale = action_scale   # kept for API compatibility
        self.bypass_threshold = bypass_threshold
        self.chunk_size = chunk_size
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
    def from_env(cls, env, coef_joint=1.0, coef_self=-1.0, coef_nbr=-1.0, bias=0.0, **kwargs):
        """Wrap an already-built DriftedMatchingEnv (shares its DecodingGraph and current
        weights/correlation tracer). For analysis code that wants only ``new_weights``."""
        return cls(None, coef_joint=coef_joint, coef_self=coef_self, coef_nbr=coef_nbr,
                   bias=bias, graph=env.graph, weights=env.current_weights,
                   corr=env.corr_tracer, min_weight=env.min_weight, max_weight=env.max_weight,
                   num_observables=env.base_dem.num_observables, **kwargs)

    @classmethod
    def from_checkpoint(cls, dem, model_path, map_location="cpu", **kwargs):
        """Build from a trained ``linear_cm`` checkpoint (reads the actor coefficients)."""
        import torch
        ck = torch.load(model_path, map_location=map_location, weights_only=False)["actor"]
        return cls(dem,
                   coef_joint=float(ck["coef_joint"]), coef_self=float(ck["coef_self"]),
                   coef_nbr=float(ck["coef_nbr"]), bias=float(ck["bias"]), **kwargs)

    def set_coefficients(self, coef_joint, coef_self, coef_nbr, bias):
        """Update the 4 coefficients in place (e.g. for a coefficient search)."""
        self.coef_joint = float(coef_joint)
        self.coef_self = float(coef_self)
        self.coef_nbr = float(coef_nbr)
        self.bias = float(bias)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_weights(self, sel, action_mask):
        """Linear-CM reweighted weights for one shot (numpy mirror of LinearCMActor).

        Given the first-pass selection mask `sel` (bool, per decoding edge) and the env
        action mask, returns the clipped second-pass weight vector."""
        return self._new_weights(sel, action_mask)

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

    def _new_weights(self, sel, action_mask):
        """Linear-CM reweighted weights for one shot (numpy mirror of LinearCMActor)."""
        w = self.current_weights
        src, dst = self._src, self._dst
        # Keep edges whose neighbour nu was selected AND whose target mu is unmasked.
        keep = sel[src] & action_mask[dst]
        delta = np.zeros(self.graph.n_dec_edges, dtype=np.float64)
        if keep.any():
            d_keep, s_keep = dst[keep], src[keep]
            msg = (self.coef_joint * self._w_joint[keep]
                   + self.coef_self * w[d_keep]
                   + self.coef_nbr * w[s_keep]
                   + self.bias)
            np.add.at(delta, d_keep, msg)
        return np.clip(w + delta, self.min_weight, self.max_weight).astype(np.float32)

    def _decode_chunk(self, shots):
        g = self.graph
        n_shots = shots.shape[0]

        # 1. Batched first-pass MWPM.
        edges_first = np.asarray(
            self._matching.decode_batch(shots, enable_correlations=False), dtype=bool
        )
        # Observable parity = sum of selected fault edges (mod 2). Index the fault
        # columns directly instead of casting the whole (n_shots, n_edges) array to int64.
        first_obs = edges_first[:, g.fault_array].sum(axis=1) % 2
        first_obs = np.atleast_2d(first_obs).reshape(n_shots, -1).astype(np.uint8)

        # 2. Bypass split: trivial shots keep the first-pass result.
        non_bypass = np.flatnonzero(
            shots.astype(np.int64).sum(axis=1) > self.bypass_threshold
        )
        result = first_obs.copy()

        # 3+4. Per-shot linear reweighting + second-pass MWPM.
        for s in non_bypass:
            sel = edges_first[s]
            selected_idx = np.flatnonzero(sel)
            if selected_idx.size == 0:
                continue

            action_mask = g.compute_action_mask(selected_idx)
            new_w = self._new_weights(sel, action_mask)
            if np.array_equal(new_w, self.current_weights):
                continue

            reweights = g.build_edge_reweights(new_w, self.current_weights)
            edges = self._matching.decode(
                shots[s], enable_correlations=False, edge_reweights=reweights
            )
            pred = (np.asarray(edges, dtype=np.int64) @ g.fault_array) % 2
            result[s] = np.atleast_1d(pred).astype(np.uint8)

        return result
