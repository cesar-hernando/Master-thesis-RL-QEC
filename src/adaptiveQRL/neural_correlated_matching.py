"""
Neural Correlated Matching: trained SAC-GNN policy on top of two-pass MWPM,
exposed with a BeliefMatching-style decode_batch() API.

The decode_batch pipeline:
    1. First-pass MWPM on all shots (batched).
    2. Shots with n_flashes <= bypass_threshold skip the agent entirely.
    3. GNN forward pass on the remaining shots in a single batched call.
    4. Per-shot second-pass MWPM with the agent's edge_reweights.

Step 3 is the main speed-up over per-shot env loops.
"""

import numpy as np
import pymatching
import torch
from torch_geometric.data import Batch, Data

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


class _DemBackedGenerator(SyndromeDataGenerator):
    """Hands a DEM to DriftedMatchingEnv without ever simulating shots.

    The env's __init__ asks the generator for a circuit object; we return a tiny
    stub that exposes only what __init__ reads (num_detectors plus an empty
    detector-coordinate dict that the rendering geometry pass tolerates).
    reset() and step() are never called from NeuralCorrelatedMatching, so
    simulate_syndrome_data is a no-op.
    """

    class _Stub:
        def __init__(self, n): self.num_detectors = n
        def get_detector_coordinates(self): return {}

    def __init__(self, dem):
        self._dem = dem
        self._m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
        self._m_corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        # Fields the SyndromeDataGenerator base class expects to exist.
        self.distance = self.n_rounds = 1
        self.mismatch = 1.0
        self.noise_model = {"version": "ncm"}
        self.memory_type = "z"
        self.qec_code = "surface_code"
        self.n_shots = 1

    def generate_base_circuit(self):
        return self._Stub(self._m.num_detectors), self._dem, self._m

    def generate_drifted_circuit(self, base_circuit, seed=42):
        return base_circuit, self._dem, self._m_corr

    def simulate_syndrome_data(self, drifted_circuit, seed=42, n_shots=None):
        n = n_shots or 0
        return (np.zeros((n, self._m.num_detectors), dtype=np.uint8),
                np.zeros(n, dtype=np.uint8))


class NeuralCorrelatedMatching:
    """SAC-GNN policy on top of two-pass MWPM. BeliefMatching-style API."""

    def __init__(self, dem, model_path, hidden_dim=256, n_layers=1, alpha=0.01,
                 action_scale=5.0, bypass_threshold=2, local_action_hops=1,
                 use_endpoint_firing=False, use_pearson_correlation=True,
                 weights=None, device=None,
                 chunk_size=50_000, actor_batch_size=256):
        # Build the env to reuse its tested structure builders (H, fault_array,
        # line_edge_index, k_hop_adj_mat, ...). reset() / step() are never called,
        # so all the CMA-update / drift / training parameters are irrelevant; the
        # only knob with init-time effect we need to set is start_from_oracle=True,
        # which is the env's own flag for "no alignment reweighting".
        self.env = DriftedMatchingEnv(
            syndrome_data_generator=_DemBackedGenerator(dem),
            local_action_only=True,
            local_action_hops=local_action_hops,
            action_scale=action_scale,
            use_pearson_correlation=use_pearson_correlation,
            use_endpoint_firing=use_endpoint_firing,
            start_from_oracle=True,
        )
        if weights is not None:
            self.env.current_weights = weights.astype(np.float32).copy()

        # Single matching object reused for both passes (returns selected edges).
        self._matching = pymatching.Matching.from_check_matrix(
            self.env.H, weights=self.env.current_weights
        )

        # Agent (architecture must match what was trained).
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent = SACAgent(
            node_dim=self.env.node_feats.shape[1], hidden_dim=hidden_dim,
            static_edge_index=self.env.line_edge_index, n_layers=n_layers,
            lr=1e-4, gamma=0.0, tau=0.005, alpha=alpha,
        )
        self.agent.load_models(model_path)
        self.agent.actor.eval()

        self.action_scale = action_scale
        self.bypass_threshold = bypass_threshold
        self.use_endpoint_firing = use_endpoint_firing
        self.chunk_size = chunk_size
        self.actor_batch_size = actor_batch_size
        self.num_observables = dem.num_observables

        # Static tensors reused every batched forward pass.
        self._edge_index_t = torch.from_numpy(self.env.line_edge_index).long()
        self._edge_attr_t = torch.from_numpy(
            self.env.initial_pearson_corr.astype(np.float32).reshape(-1, 1)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, shot):
        return self.decode_batch(shot[np.newaxis, :])[0]

    def decode_batch(self, shots):
        """shots: (n_shots, n_detectors). Returns (n_shots, n_observables) uint8.

        Internally chunks the input into self.chunk_size pieces so peak RAM stays
        bounded regardless of how many shots are passed in.
        """
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

    def _decode_chunk(self, shots):
        """Decode a chunk of shots that comfortably fits in memory."""
        n_shots = shots.shape[0]

        # 1. Batched first-pass MWPM.
        edges_first = np.asarray(
            self._matching.decode_batch(shots, enable_correlations=False), dtype=bool
        )
        first_obs = (edges_first.astype(np.int64) @ self.env.fault_array) % 2
        first_obs = np.atleast_2d(first_obs).reshape(n_shots, -1).astype(np.uint8)

        # 2. Bypass split.
        non_bypass = np.flatnonzero(
            shots.astype(np.int64).sum(axis=1) > self.bypass_threshold
        )
        if non_bypass.size == 0:
            return first_obs

        # 3. Build per-shot observations and run the GNN in batched sub-calls.
        feats, masks = self._build_observations(shots, edges_first, non_bypass)
        actions = self._batched_actor(feats, masks)

        # 4. Per-shot second-pass MWPM with edge reweights.
        result = first_obs.copy()
        for i, s in enumerate(non_bypass):
            delta = actions[i] * masks[i].astype(np.float32) * self.action_scale
            if not np.any(delta):
                continue
            new_w = np.clip(self.env.current_weights + delta,
                            self.env.min_weight, self.env.max_weight)
            reweights = self._edge_reweights(new_w)
            edges = self._matching.decode(
                shots[s], enable_correlations=False, edge_reweights=reweights
            )
            pred = (np.asarray(edges, dtype=np.int64) @ self.env.fault_array) % 2
            result[s] = np.atleast_1d(pred).astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_observations(self, shots, edges_first, idx):
        n = idx.size
        node_dim = self.env.node_feats.shape[1]
        feats = np.zeros((n, self.env.n_dec_edges, node_dim), dtype=np.float32)
        masks = np.zeros((n, self.env.n_dec_edges), dtype=bool)
        feats[:, :, 0] = self.env.current_weights
        for i, s in enumerate(idx):
            sel = edges_first[s]
            feats[i, :, 1] = sel.astype(np.float32)
            masks[i] = self.env._compute_action_mask(np.flatnonzero(sel))
        if self.use_endpoint_firing:
            d1c, d2c, bc = (self.env._endpoint_d1_col,
                            self.env._endpoint_d2_col,
                            self.env._endpoint_b_col)
            d1_idx, d2_idx = self.env.edge_d1_idx, self.env.edge_d2_idx_safe
            is_bdy = self.env.edge_is_boundary
            for i, s in enumerate(idx):
                syn = shots[s]
                feats[i, :, d1c] = syn[d1_idx]
                feats[i, :, d2c] = np.where(is_bdy > 0, 0.0, syn[d2_idx])
                feats[i, :, bc] = is_bdy
        return feats, masks

    def _batched_actor(self, feats, masks):
        """Run the GNN actor over all non-bypass shots, in sub-batches of size
        actor_batch_size so a large `feats` doesn't blow up RAM in the linear
        layers of the actor (peak alloc ~ actor_batch_size * n_dec_edges * hidden_dim).
        """
        n = feats.shape[0]
        out = np.empty((n, self.env.n_dec_edges), dtype=np.float32)
        bs = max(1, self.actor_batch_size)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            data = [
                Data(
                    x=torch.from_numpy(feats[i]),
                    edge_index=self._edge_index_t,
                    edge_attr=self._edge_attr_t,
                    action_mask=torch.from_numpy(masks[i]).float().unsqueeze(-1),
                )
                for i in range(start, end)
            ]
            batch = Batch.from_data_list(data).to(self._device, non_blocking=True)
            with torch.no_grad():
                actions, _ = self.agent.actor(
                    batch.x, batch.edge_index, batch.edge_attr, batch.action_mask,
                    evaluate=True,
                )
            out[start:end] = (
                actions.cpu().numpy().reshape(end - start, self.env.n_dec_edges)
            )
        return out

    def _edge_reweights(self, new_weights):
        changed = np.flatnonzero(np.abs(new_weights - self.env.current_weights) > 0.0)
        if changed.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        edges = self.env.dec_edge_arr[changed]
        u, v = edges[:, 0].copy(), edges[:, 1].copy()
        bdy = (u == -1) & (v != -1)
        u[bdy], v[bdy] = v[bdy], -1
        return np.column_stack((u, v, new_weights[changed]))
