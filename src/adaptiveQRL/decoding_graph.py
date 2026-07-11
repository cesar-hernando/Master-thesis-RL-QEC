"""
DecodingGraph: the STATIC structure of a surface-code matching/decoding graph extracted
from a DEM, plus the structural operations built on it.

It owns the decoding-edge indexing, sparse parity-check matrix ``H``, fault map, the line
graph of pairwise correlations, the k-hop adjacency used for action masking, the
endpoint-firing maps, and the operations that read only this structure (+ caller-supplied
dynamic weights/tracers): the action mask, the ``edge_reweights`` builder, the analytical
correlated-matching reweighting, and the Pearson-correlation transform.

This used to live inside ``DriftedMatchingEnv``. Extracting it means (a) the env is a
thinner Gym wrapper that composes one of these, and (b) the two-pass decoders compose a
``DecodingGraph`` directly (``DecodingGraph.from_dem(dem)``) instead of building a throwaway
training env just to reach the graph builders. The math lives in one tested place.

Nothing here holds mutable per-episode state: dynamic quantities (current weights, drifted
occurrence/correlation tracers, the first-pass selection) are passed in as arguments.
"""

import numpy as np
import scipy.sparse as sp
import pymatching


class DecodingGraph:
    def __init__(self, matching, dem, n_detectors, local_action_hops=1):
        self.n_detectors = n_detectors
        self.local_action_hops = local_action_hops

        (self.dec_edge_list,
         self.pair_to_idx_matrix,
         self.initial_weights,
         self.base_p,
         self.fault_array,
         self.H) = self._index_decoding_graph_edges(matching)
        self.n_dec_edges = len(self.dec_edge_list)
        self.dec_edge_arr = np.array(self.dec_edge_list, dtype=np.float64)

        (self.line_edge_index,
         self.initial_corr_tracer,
         self.k_hop_adj_mat) = self._build_line_graph_edges(dem)
        self.n_line_edges = self.line_edge_index.shape[1]

        # Per-edge endpoint detector indices for the endpoint-firing features. Boundary
        # edges store the real detector in d1; the d2 slot is a dummy index 0 zeroed at
        # fill-time via the is_boundary mask.
        edge_d1_idx = np.zeros(self.n_dec_edges, dtype=np.int64)
        edge_d2_idx_safe = np.zeros(self.n_dec_edges, dtype=np.int64)
        edge_is_boundary = np.zeros(self.n_dec_edges, dtype=np.float32)
        for i, (u, v) in enumerate(self.dec_edge_list):
            if u == -1 and v == -1:
                edge_is_boundary[i] = 1.0
            elif u == -1:
                edge_d1_idx[i] = v
                edge_is_boundary[i] = 1.0
            elif v == -1:
                edge_d1_idx[i] = u
                edge_is_boundary[i] = 1.0
            else:
                edge_d1_idx[i] = u
                edge_d2_idx_safe[i] = v
        self.edge_d1_idx = edge_d1_idx
        self.edge_d2_idx_safe = edge_d2_idx_safe
        self.edge_is_boundary = edge_is_boundary

        self._adjacency = None  # lazily built bidirectional line-graph adjacency

    @classmethod
    def from_dem(cls, dem, local_action_hops=1):
        """Build the structure straight from a (decomposed) DEM."""
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
        return cls(matching, dem, dem.num_detectors, local_action_hops=local_action_hops)

    # ------------------------------------------------------------------
    # Static builders
    # ------------------------------------------------------------------

    def _index_decoding_graph_edges(self, matching: pymatching.Matching):
        """Index the decoding graph and build the sparse parity-check matrix H.

        Returns (dec_edge_list, pair_to_idx_matrix, weights, error_probs, fault_array, H).
        """
        dec_edge_list = []
        weights = []
        error_probs = []
        fault_ids = []

        matrix_size = self.n_detectors + 1
        pair_to_idx_matrix = np.full((matrix_size, matrix_size), -1, dtype=np.int32)

        h_rows = []
        h_cols = []

        idx = 0
        for u, v, data in matching.edges():
            u = -1 if u is None else u
            v = -1 if v is None else v
            key = (u, v) if u <= v else (v, u)

            dec_edge_list.append(key)
            weights.append(data["weight"])
            error_probs.append(data["error_probability"])

            pair_to_idx_matrix[u, v] = idx
            pair_to_idx_matrix[v, u] = idx

            if u != -1:
                h_rows.append(u)
                h_cols.append(idx)
            if v != -1:
                h_rows.append(v)
                h_cols.append(idx)

            f_ids = data.get("fault_ids", set())
            if isinstance(f_ids, (int, float)):
                f_ids = {int(f_ids)}
            else:
                f_ids = set(int(f) for f in f_ids)
            fault_ids.append(f_ids)
            idx += 1

        fault_array = np.array([len(f) > 0 for f in fault_ids], dtype=bool)

        h_data = np.ones(len(h_rows), dtype=np.int8)
        H = sp.csc_matrix((h_data, (h_rows, h_cols)), shape=(self.n_detectors, idx))

        return (
            dec_edge_list,
            pair_to_idx_matrix,
            np.asarray(weights, dtype=np.float32),
            np.asarray(error_probs, dtype=np.float32),
            fault_array,
            H,
        )

    def _build_line_graph_edges(self, dem, return_k_hop_adj_mat=True):
        """Build line-graph connectivity from correlated hyperedges in the DEM.

        Returns (line_edge_index [2, M], initial_correlations [M], k_hop_adj_mat [N, N]).
        """
        correlation_dict = {}

        for inst in dem.flattened():
            if inst.type != "error":
                continue

            p = inst.args_copy()[0]
            targets = inst.targets_copy()

            components = []
            current_component = []
            for t in targets:
                if t.is_separator():
                    if current_component:
                        components.append(current_component)
                        current_component = []
                elif t.is_relative_detector_id():
                    current_component.append(t.val)
            if current_component:
                components.append(current_component)

            if len(components) < 2:
                continue

            node_indices = []
            for comp in components:
                u = comp[0] if len(comp) > 0 else -1
                v = comp[1] if len(comp) > 1 else -1
                idx = self.pair_to_idx_matrix[u, v]
                node_indices.append(idx)

            for a in range(len(node_indices)):
                for b in range(a + 1, len(node_indices)):
                    na, nb = node_indices[a], node_indices[b]
                    if na == nb:
                        continue
                    edge_key = (na, nb) if na <= nb else (nb, na)
                    if edge_key in correlation_dict:
                        existing_p = correlation_dict[edge_key]
                        correlation_dict[edge_key] = existing_p * (1 - p) + p * (1 - existing_p)
                    else:
                        correlation_dict[edge_key] = p

        if not correlation_dict:
            dummy_adj = np.eye(self.n_dec_edges, dtype=bool)
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32), dummy_adj

        sorted_edges = sorted(correlation_dict.keys())
        line_edges = np.array(sorted_edges, dtype=np.int64).T
        src, dst = line_edges[0], line_edges[1]

        initial_correlations = np.array([correlation_dict[e] for e in sorted_edges], dtype=np.float32)

        if not return_k_hop_adj_mat:
            return line_edges, initial_correlations

        diag_idx = np.arange(self.n_dec_edges)
        rows = np.concatenate([src, dst, diag_idx])
        cols = np.concatenate([dst, src, diag_idx])
        data = np.ones(len(rows), dtype=np.int8)
        adj_mat = sp.csr_matrix((data, (rows, cols)), shape=(self.n_dec_edges, self.n_dec_edges))

        k_hop_sparse = adj_mat ** self.local_action_hops
        if sp.issparse(k_hop_sparse):
            k_hop_adj_mat = k_hop_sparse.toarray() > 0
        else:
            k_hop_adj_mat = k_hop_sparse > 0

        return line_edges, initial_correlations, k_hop_adj_mat

    def build_line_graph(self, dem, return_k_hop_adj_mat=False):
        """Public re-run of the line-graph builder for another DEM that shares this
        graph's detector/edge layout (e.g. a drifted DEM), mapping its joint probabilities
        onto the same edge indexing. Returns (line_edge_index, joint_probs[, k_hop])."""
        return self._build_line_graph_edges(dem, return_k_hop_adj_mat=return_k_hop_adj_mat)

    # ------------------------------------------------------------------
    # Structural operations (read structure + caller-supplied dynamic state)
    # ------------------------------------------------------------------

    @property
    def bidirectional_adjacency(self):
        """Cached {node: [(neighbour, line_edge_idx), ...]} from the line graph."""
        if self._adjacency is None:
            adj = {}
            src = self.line_edge_index[0]
            dst = self.line_edge_index[1]
            for e_idx in range(len(src)):
                u, v = src[e_idx], dst[e_idx]
                if u not in adj:
                    adj[u] = []
                if v not in adj:
                    adj[v] = []
                adj[u].append((v, e_idx))
                adj[v].append((u, e_idx))
            self._adjacency = adj
        return self._adjacency

    def compute_action_mask(self, selected_idx: np.ndarray, local_action_only: bool = True):
        """Boolean [N] mask of edges within k hops of any selected edge (isolated selected
        edges excluded). If ``local_action_only`` is False, every edge is unmasked."""
        if not local_action_only:
            return np.ones(self.n_dec_edges, dtype=np.bool_)
        if selected_idx.size == 0:
            return np.zeros(self.n_dec_edges, dtype=np.bool_)

        selected_rows = self.k_hop_adj_mat[selected_idx]
        mask = selected_rows.any(axis=0)

        selected_subgraph = selected_rows[:, selected_idx]
        selected_neighbor_counts = selected_subgraph.sum(axis=1)
        isolated_selected_idx = selected_idx[selected_neighbor_counts == 1]
        if isolated_selected_idx.size > 0:
            mask[isolated_selected_idx] = False
        return mask

    def build_edge_reweights(self, second_pass_weights: np.ndarray, reference_weights: np.ndarray):
        """Pymatching ``edge_reweights`` rows (u, v, new_weight) for edges whose weight
        changed relative to ``reference_weights``."""
        changed = np.flatnonzero(np.abs(second_pass_weights - reference_weights) > 0.0)
        if changed.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        edges = self.dec_edge_arr[changed]
        u = edges[:, 0].copy()
        v = edges[:, 1].copy()

        boundary_mask = (u == -1) & (v != -1)
        u[boundary_mask] = v[boundary_mask]
        v[boundary_mask] = -1

        return np.column_stack((u, v, second_pass_weights[changed]))

    def pearson_correlations(self, occ_array: np.ndarray, corr_array: np.ndarray):
        """Pearson correlation per line-graph edge, clipped to [-1, 1]."""
        if self.n_line_edges == 0:
            return np.zeros(0, dtype=np.float32)

        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]
        p_src = occ_array[src]
        p_dst = occ_array[dst]

        covariance = corr_array - (p_src * p_dst)
        std_src = np.sqrt(p_src * (1.0 - p_src))
        std_dst = np.sqrt(p_dst * (1.0 - p_dst))
        denom = std_src * std_dst
        safe_denom = np.where(denom > 1e-9, denom, 1.0)
        return np.clip(covariance / safe_denom, -1.0, 1.0)

    def analytical_cm_second_pass_weights(self, current_weights, selected_mask, action_mask,
                                          occ_tracer, corr_tracer, alpha=1.0):
        """Analytical correlated-matching second-pass weights.

        Returns a [N] array that is 0 on non-active edges and, on each active edge mu, the
        largest discount (lowest weight) across mu's first-pass-selected neighbours nu:
            implied_p = P(mu | nu) (alpha=1) or Pearl soft-evidence mix (alpha<1)
            w = log((1 - implied_p) / implied_p)
        This is the partial vector the env turns into an action and the decoders turn into
        clipped second-pass weights. (Caller applies any min/max weight clipping.)
        """
        second_pass = np.zeros(self.n_dec_edges, dtype=np.float32)
        if self.n_line_edges == 0:
            return second_pass

        active_nodes = np.where(action_mask)[0]
        adj = self.bidirectional_adjacency

        for node in active_nodes:
            new_weight = current_weights[node]
            for nbr, e_idx in adj.get(node, []):
                if selected_mask[nbr]:
                    implied_p = corr_tracer[e_idx] / occ_tracer[nbr]   # P(mu | nu), hard evidence
                    if alpha != 1.0:
                        denom = max(1.0 - occ_tracer[nbr], 1e-12)
                        implied_p_neg = max((occ_tracer[node] - corr_tracer[e_idx]) / denom, 0.0)
                        implied_p = alpha * implied_p + (1.0 - alpha) * implied_p_neg
                    implied_p = np.clip(implied_p, 1e-6, 0.499999)
                    corr_mat_weight = np.log((1.0 - implied_p) / implied_p)
                    new_weight = min(new_weight, corr_mat_weight)
            second_pass[node] = new_weight

        return second_pass
