'''
In this file, we build a Gymnasium environment for reweighting the decoding graph based on the edge correlations 
statistics and the first MWPM pass selected edges.
'''

from typing import Dict, Any, Optional, Tuple, List, Set

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

import pymatching

from syndrome_data_generation import SyndromeDataGenerator



class DriftedMatchingEnv(gym.Env):
    """
    Graph-structured RL environment for learning MWPM edge reweighting in presence of drift noise.

    Oservation (graph in array form)
    ---------------------------------
    A fixed-size dictionary (good for Gym + GNN training pipelines):
      - node_features: [N_edges, 2]
          [:, 0] = current MWPM edge weight (base + occurrence-tracer bias)
          [:, 1] = selected by first MWPM pass (0/1)
      - edge_index: [2, M_line]
          Line-graph connectivity (nodes = decoding-graph edges)
      - edge_attr: [M_line, 1]
          Correlation tracer on line-graph edges
      - action_mask: [N_edges]
          0/1 mask; if local_action_only=True, action is applied only where mask=1

    Action
    ------
    Continuous vector of shape [N_edges], each component in [-1, 1].
    The env scales it by action_scale and applies it as a small weight delta.

    By default (for scalability), the env only applies actions on the local
    neighborhood of first-pass selected edges (controlled by local_action_only
    and local_action_hops). The action_mask is included in the observation.

    Transition (one env.step)
    -------------------------
    1) Use current shot + first-pass MWPM info prepared by the env
    2) Build second-pass weights = current_weights + masked_action_delta
    3) Run second-pass MWPM
    4) Update occurrence and correlation tracers using second-pass selected edges
    5) Compute reward:
         - logical reward: based on predicted vs true observable
         - optional oracle imitation reward (DGR-like auxiliary signal)
    6) Sample next shot and prepare next observation

    Episode semantics
    -----------------
    Each episode samples a new drifted circuit using SyndromeDataGenerator, 
    and the agent interacts with a sequence of shots from that circuit.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        syndrome_data_generator: SyndromeDataGenerator,
        action_scale: float = 0.25,
        local_action_only: bool = True,
        local_action_hops: int = 1,
        xz_crosstalk_radius: float = 2.1,
        occ_ema_alpha: float = 0.05,
        corr_ema_alpha: float = 0.05,
        occ_to_weight_scale: float = 0.5,
        min_weight: float = 1e-6,
        max_weight: float = 50.0,
        oracle_reward_coef: float = 0.0,
    ):
        super().__init__()

        # Use the provided syndrome data generator to create drifted circuits for each episode
        self.syndrome_data_generator = syndrome_data_generator

        # Build the base circuit and matching
        self.base_circuit, self.current_matching = self.syndrome_data_generator.generate_base_circuit()

        # Capture the number of detectors
        self.n_detectors = self.base_circuit.num_detectors

        # Store a template decoding graph
        self.decoder_template_graph = self.current_matching.to_networkx()

        # Construct the simplified GNN graph
        self.dec_graph = self._matching_to_simple_nx(self.current_matching)

        # Config / hyperparams
        self.max_steps = self.syndrome_data_generator.n_shots
        self.action_scale = action_scale
        self.local_action_only = local_action_only
        self.local_action_hops = local_action_hops # Number of hops in line graph for local action masking

        self.occ_ema_alpha = occ_ema_alpha
        self.corr_ema_alpha = corr_ema_alpha
        self.occ_to_weight_scale = occ_to_weight_scale

        self.min_weight = min_weight
        self.max_weight = max_weight

        self.oracle_reward_coef = oracle_reward_coef

        # Detector coordinates (used only to identify "real detectors" vs boundary nodes)
        self.detector_coords = self.base_circuit.get_detector_coordinates()
        self.real_detectors = set(int(k) for k in self.detector_coords.keys())

        # Fix decoding-edge indexing
        self.dec_edge_list, self.dec_edge_to_idx, self.current_weights = self._index_decoding_graph_edges(self.dec_graph)
        self.n_dec_edges = len(self.dec_edge_list)
        self.base_p = 1.0 / (1.0 + np.exp(self.current_weights))
        self.initial_base_weights = self.current_weights.copy()

        # Build the line graph, adding edges between geometrically close X and Z edges
        self.line_edge_index = self._build_line_graph_edges(
            self.dec_edge_list, 
            self.real_detectors,
            xz_crosstalk_radius
            )
        
        self.n_line_edges = self.line_edge_index.shape[1]

        # Construct an adjacency list for fast local action masking
        self.line_neighbors = self._build_line_neighbors(self.line_edge_index, self.n_dec_edges)

        # Define the Gym Observation and Action Spaces (fixed size for a given base circuit / distance)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_dec_edges, 2),
                dtype=np.float32
            ),
            "edge_index": spaces.Box(
                low=0,
                high=max(0, self.n_dec_edges - 1),
                shape=(2, self.n_line_edges),
                dtype=np.int64
            ),
            "edge_attr": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_line_edges, 1),
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_dec_edges,),
                dtype=np.float32
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_dec_edges,),
            dtype=np.float32
        )

        # Initialize dynamic tracers
        self.occ_ema = np.zeros(self.n_dec_edges, dtype=np.float32)
        self.corr_ema = np.zeros(self.n_line_edges, dtype=np.float32)

        # Cached current shot (prepared during reset and after each step)
        self.current_syndrome: Optional[np.ndarray] = None
        self.current_true_obs: Optional[np.ndarray] = None
        self.current_first_pass_selected_idx: Optional[np.ndarray] = None
        self.current_action_mask: Optional[np.ndarray] = None

    
    @staticmethod
    def _matching_to_simple_nx(matching: pymatching.Matching) -> nx.Graph:
        """
        Export PyMatching graph and collapse MultiGraph (if present) into a simple graph.
        Normalizes all node IDs to native Python integers.
        """
        # 1. Get the raw graph (likely a MultiGraph)
        G = matching.to_networkx()
        
        # 2. Collapse to simple Graph (automatically handles parallel edges & attributes)
        H = nx.Graph(G)
        
        # 3. Relabel all nodes to native Python ints to ensure compatibility with Gym/GNN
        # This preserves the 'is_boundary' attribute which is CRITICAL for reconstruction
        mapping = {n: int(n) for n in H.nodes()}
        H = nx.relabel_nodes(H, mapping)
        
        return H
    

    @staticmethod
    def _index_decoding_graph_edges(
        G: nx.Graph
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int], np.ndarray]:
        """
        Create fixed indexing of decoding-graph edges.

        Args
        ----
        G: NetworkX graph of the decoding graph (nodes = detectors, edges = possible
        error mechanisms).

        Returns
        -------
        dec_edge_list : list[(u,v)]   canonical sorted endpoints
        dec_edge_to_idx : dict[(u,v)] -> idx
        base_edge_weight : np.ndarray shape [N]
        """
        dec_edge_list: List[Tuple[int, int]] = []
        dec_edge_to_idx: Dict[Tuple[int, int], int] = {}
        weights: List[float] = []

        idx = 0
        for u, v, data in G.edges(data=True):
            key = (u, v) if u <= v else (v, u)
            dec_edge_list.append(key)
            dec_edge_to_idx[key] = idx
            weights.append(float(data.get("weight", 1.0)))
            idx += 1

        return dec_edge_list, dec_edge_to_idx, np.asarray(weights, dtype=np.float32)
    

    def _build_line_graph_edges(
        self,
        dec_edge_list: List[Tuple[int, int]],
        real_detectors: Set[int],
        xz_crosstalk_radius: float
    ) -> np.ndarray:
        """
        Build line-graph connectivity using:
        1. Shared real detector endpoints (captures all local X-X and Z-Z connections)
        2. Geometric proximity (captures overlapping X-Z errors like Y-errors, and local crosstalk)
        """
        line_edges_set: Set[Tuple[int, int]] = set()

        ###############################################################
        # 1) Topological Connections (Shared Detectors for X-X & Z-Z) #
        ###############################################################

        incident: Dict[int, List[int]] = {}
        for i, (u, v) in enumerate(dec_edge_list):
            if u in real_detectors:
                incident.setdefault(u, []).append(i)
            if v in real_detectors:
                incident.setdefault(v, []).append(i)

        for _, edge_ids in incident.items():
            m = len(edge_ids)
            for a in range(m):
                for b in range(a + 1, m):
                    ia, ib = edge_ids[a], edge_ids[b]
                    if ia != ib:
                        e = (ia, ib) if ia < ib else (ib, ia)
                        line_edges_set.add(e)

        #########################################################
        # 2. Geometric Connections (Midpoint Proximity for X-Z) #
        #########################################################
        midpoints = np.zeros((len(dec_edge_list), 3))
        edge_types = []

        for i, (u, v) in enumerate(dec_edge_list):
            cu = self.detector_coords.get(u, None)
            cv = self.detector_coords.get(v, None)
            
            # Helper to ensure 3D (x, y, t)
            def pad3(c):
                if c is None: return None
                lst = list(c)
                while len(lst) < 3: lst.append(0.0)
                return np.array(lst[:3])
            
            cu, cv = pad3(cu), pad3(cv)
            
            # Calculate geometric midpoint of the edge
            if cu is not None and cv is not None:
                midpoints[i] = (cu + cv) / 2.0
            elif cu is not None:
                midpoints[i] = cu
            elif cv is not None:
                midpoints[i] = cv
            else:
                midpoints[i] = np.zeros(3)
                
            # Infer edge type (X vs Z) using the first valid real detector
            etype = "Unknown"
            real_node = u if cu is not None else (v if cv is not None else None)
            if real_node is not None:
                c = self.detector_coords[real_node]
                j, i = int(round(c[0])), int(round(c[1])) # j=x, i=y
                if (i % 4 == 0 and j % 4 == 0) or (i % 4 == 2 and j % 4 == 2):
                    etype = "X"
                elif (i % 4 == 0 and j % 4 == 2) or (i % 4 == 2 and j % 4 == 0):
                    etype = "Z"
            edge_types.append(etype)
            
        # Extract indices of X and Z edges
        x_indices = [i for i, t in enumerate(edge_types) if t == "X"]
        z_indices = [i for i, t in enumerate(edge_types) if t == "Z"]
        
        if len(x_indices) > 0 and len(z_indices) > 0:
            x_mids = midpoints[x_indices]
            z_mids = midpoints[z_indices]
            
            # Fast vectorized calculation of all X-Z pairwise distances in 3D
            # Matrix shape: [Num_X_Edges, Num_Z_Edges]
            dist_matrix = np.linalg.norm(x_mids[:, None, :] - z_mids[None, :, :], axis=-1)
            
            # Find all pairs within the local crosstalk threshold
            close_pairs = np.argwhere(dist_matrix <= xz_crosstalk_radius)
            
            for px, pz in close_pairs:
                ix, iz = x_indices[px], z_indices[pz]
                e = (ix, iz) if ix < iz else (iz, ix)
                line_edges_set.add(e)

        if not line_edges_set:
            return np.zeros((2, 0), dtype=np.int64)

        line_edges = np.array(sorted(line_edges_set), dtype=np.int64)
        
        return line_edges.T
    

    def _build_line_neighbors(self, line_edge_index: np.ndarray, n_nodes: int) -> List[List[int]]:
        """
        Adjacency list of line graph.
        """
        neighbors = [[] for _ in range(n_nodes)]
        if line_edge_index.shape[1] == 0:
            return neighbors
        src = line_edge_index[0]
        dst = line_edge_index[1]
        for i, j in zip(src.tolist(), dst.tolist()):
            neighbors[i].append(j)
            neighbors[j].append(i)
        return neighbors


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Generate a new drifted circuit and corresponding decoding graph
        self.drifted_circuit, self.drifted_matching = self.syndrome_data_generator.generate_drifted_circuit(
            base_circuit=self.base_circuit,
            seed=seed
            )
        
        # Pre-generate syndrome data and true observable for the entire episode (all shots under the same drift)
        self.syndrome_batch, self.true_obs_batch = self.syndrome_data_generator.simulate_syndrome_data(self.drifted_circuit)

        # Pre-generate the oracle solution edges and predicted observable for each shot in the episode using the drifted matching
        self.oracle_solution_edges_batch, self.oracle_predicted_obs_batch = self.syndrome_data_generator.get_solution_edges_batch(
            self.drifted_matching, 
            self.syndrome_batch, 
            enable_correlations=True, 
            return_predicted_obs=True)
        
        # Reset the step count
        self.step_count = 0

        # Reset decoder weights/matching to the original base graph each episode
        self.current_weights = self.initial_base_weights.copy()
        self.current_matching = self._build_matching_with_custom_weights(self.current_weights)

        # Reset tracers
        self.occ_ema = self.base_p.copy()
        self.corr_ema.fill(0.0)

        # Prepare the first shot and build the initial observation
        obs = self._prepare_next_observation()

        info = {
            "n_decoding_edges": self.n_dec_edges,
            "n_line_edges": self.n_line_edges,
        }
        return obs, info
    

    def _prepare_next_observation(self) -> Dict[str, np.ndarray]:
        # Pull the pre-generated syndrome for the current step
        syndrome = self.syndrome_batch[self.step_count]
        true_obs = self.true_obs_batch[self.step_count]

        # First pass MWPM still uses base matching (no drift knowledge)
        selected_edges_1 = self.syndrome_data_generator.get_solution_edges(
            matching=self.current_matching, 
            syndrome_volume=syndrome, 
            enable_correlations=False,
            return_predicted_obs=False)
        
        selected_idx_1 = self._selected_edge_indices_from_pairs(selected_edges_1)

        # Action mask around first-pass selected edges
        action_mask = self._compute_action_mask(selected_idx_1)

        # Cache
        self.current_syndrome = syndrome
        self.current_true_obs = true_obs
        self.current_first_pass_selected_idx = selected_idx_1
        self.current_action_mask = action_mask

        return self._build_observation_from_cached_state()
    

    def _selected_edge_indices_from_pairs(self, edge_pairs: np.ndarray) -> np.ndarray:
        """
        Convert decoding-graph node pairs [[u,v], ...] to our fixed decoding-edge indices.

        Edges that are not in the base graph indexing are ignored (rare if topology matches).
        """
        if edge_pairs.size == 0:
            return np.zeros((0,), dtype=np.int64)

        idxs = []
        for uv in edge_pairs:
            u, v = int(uv[0]), int(uv[1])
            key = (u, v) if u <= v else (v, u)
            idx = self.dec_edge_to_idx.get(key, None)
            if idx is not None:
                idxs.append(idx)

        if not idxs:
            return np.zeros((0,), dtype=np.int64)

        # unique sorted
        return np.asarray(sorted(set(idxs)), dtype=np.int64)
    

    def _compute_action_mask(self, selected_idx: np.ndarray) -> np.ndarray:
        """
        Build a local action mask on line-graph nodes.

        If local_action_only=False -> all ones.
        Else -> selected edges + k-hop line-graph neighborhood.
        """
        if not self.local_action_only:
            return np.ones(self.n_dec_edges, dtype=np.float32)

        mask = np.zeros(self.n_dec_edges, dtype=np.float32)
        if selected_idx.size == 0:
            return mask  # no fired/selected edges -> no action region (reasonable default)

        frontier = set(int(i) for i in selected_idx.tolist())
        visited = set(frontier)

        for i in frontier:
            mask[i] = 1.0

        for _ in range(self.local_action_hops):
            new_frontier = set()
            for i in frontier:
                for j in self.line_neighbors[i]:
                    if j not in visited:
                        visited.add(j)
                        new_frontier.add(j)
            for j in new_frontier:
                mask[j] = 1.0
            frontier = new_frontier
            if not frontier:
                break

        return mask
    

    def _build_observation_from_cached_state(self) -> Dict[str, np.ndarray]:
        """
        Builds observation arrays from current tracers + cached first-pass selection.
        """

        selected_flag = np.zeros(self.n_dec_edges, dtype=np.float32)
        if self.current_first_pass_selected_idx is not None and len(self.current_first_pass_selected_idx) > 0:
            selected_flag[self.current_first_pass_selected_idx] = 1.0

        node_features = np.stack([self.current_weights, selected_flag], axis=1).astype(np.float32)

        edge_attr = self.corr_ema.reshape(-1, 1).astype(np.float32)

        action_mask = (
            self.current_action_mask.astype(np.float32)
            if self.current_action_mask is not None
            else np.ones(self.n_dec_edges, dtype=np.float32)
        )

        obs = {
            "node_features": node_features,
            "edge_index": self.line_edge_index.astype(np.int64),
            "edge_attr": edge_attr,
            "action_mask": action_mask,
        }
        return obs


    def step(self, action: np.ndarray):
        
        assert self.current_syndrome is not None, "Call reset() before step()."

        ###############################################
        # 1) Apply action (masked locally if enabled) #
        ###############################################

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_dec_edges:
            raise ValueError(f"Action has shape {action.shape}; expected ({self.n_dec_edges},)")

        if self.current_action_mask is None:
            mask = np.ones(self.n_dec_edges, dtype=np.float32)
        else:
            mask = self.current_action_mask.astype(np.float32)

        if self.local_action_only:
            applied_delta = self.action_scale * action * mask
        else:
            applied_delta = self.action_scale * action

        second_pass_weights = np.clip(self.current_weights + applied_delta, self.min_weight, self.max_weight)


        #######################################################
        # 2) Build second-pass matching with reweighted graph #
        #######################################################

        second_pass_matching = self._build_matching_with_custom_weights(second_pass_weights)

        # Run MWPM to get the selected edges and predicted observable for the current shot
        selected_edges_2, pred_obs = self.syndrome_data_generator.get_solution_edges(
            matching=second_pass_matching,
            syndrome_volume=self.current_syndrome,
            enable_correlations=False,
            return_predicted_obs=True)
        
        selected_idx_2 = self._selected_edge_indices_from_pairs(selected_edges_2)

        ######################################################
        # 3) Update tracers using SECOND-pass selected edges #
        ######################################################

        self._update_occurrence_tracer(selected_idx_2)
        self._update_correlation_tracer(selected_idx_2)
        self._update_base_matching_from_occurrence()


        #########################
        # 4) Compute the reward #
        #########################

        logical_error = pred_obs != self.current_true_obs
        logical_reward = 1.0 if logical_error == 0 else -1.0

        reward = logical_reward

        oracle_similarity = None
        if self.oracle_reward_coef > 0.0:
            # Pull the pre-calculated oracle edges for this exact shot
            oracle_edges = self.oracle_solution_edges_batch[self.step_count]
            oracle_idx = self._selected_edge_indices_from_pairs(oracle_edges)
            
            # Compute Jaccard similarity between the second-pass selected edges and the oracle solution edges
            oracle_similarity = self._edge_set_jaccard(selected_idx_2, oracle_idx)
            oracle_reward = 2.0 * oracle_similarity - 1.0
            reward += self.oracle_reward_coef * oracle_reward

        # Increment step count after processing the shot
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps # Episode ends when batch is empty


        ############################################################
        # 5) Prepare next observation (next shot under same drift) #
        ############################################################

        if not truncated:
            next_obs = self._prepare_next_observation()
        else:
            # Still return a valid obs (same as last prepared style), but no new shot is needed.
            # Here we return the last state-shaped view after tracer update.
            next_obs = self._build_observation_from_cached_state()
            # (This is okay because the episode ends right away.)

        info = {
            "logical_error": logical_error,
            "true_obs": self.current_true_obs,
            "pred_obs": pred_obs,
            "reward_logical": logical_reward,
            "reward_total": float(reward),
            "oracle_similarity_jaccard": float(oracle_similarity) if oracle_similarity is not None else None,
            "selected_edges_first_pass_idx": self.current_first_pass_selected_idx.copy() if self.current_first_pass_selected_idx is not None else None,
            "selected_edges_second_pass_idx": selected_idx_2.copy(),
            "action_mask": self.current_action_mask.copy() if self.current_action_mask is not None else None,
        }

        return next_obs, float(reward), terminated, truncated, info    
    

    def _update_occurrence_tracer(self, selected_idx: np.ndarray):
        """
        EMA occurrence tracer on line-graph nodes (decoding edges).
        """
        alpha = self.occ_ema_alpha
        active = np.zeros(self.n_dec_edges, dtype=np.float32)
        if selected_idx.size > 0:
            active[selected_idx] = 1.0

        self.occ_ema = (1.0 - alpha) * self.occ_ema + alpha * active


    def _update_correlation_tracer(self, selected_idx: np.ndarray):
        """
        EMA correlation tracer on line-graph edges.

        A line-graph edge (i,j) is "co-active" if both decoding edges i and j were selected
        in the SECOND-pass MWPM solution.
        """
        if self.n_line_edges == 0:
            return

        alpha = self.corr_ema_alpha

        active = np.zeros(self.n_dec_edges, dtype=np.bool_)
        if selected_idx.size > 0:
            active[selected_idx] = True

        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]
        co_active = (active[src] & active[dst]).astype(np.float32)

        self.corr_ema = (1.0 - alpha) * self.corr_ema + alpha * co_active


    def _update_base_matching_from_occurrence(self):
        """
        Periodically update the 'base' decoder to catch up with the drift.
        Use this sparingly (e.g., once per episode in reset) to keep the RL stable.
        """
        p = np.clip(self.occ_ema, 1e-6, 0.499999)
        
        # Apply the exact formula from the paper
        w = np.log((1.0 - p) / p)
        
        # Clip to your environment's safe min/max weight boundaries
        w = np.clip(w, self.min_weight, self.max_weight)

        # Update the current weights for observation building and next steps
        self.current_weights = w.copy()
        
        # Update the base matching object
        self.current_matching = self._build_matching_with_custom_weights(w)


    def _build_matching_with_custom_weights(self, custom_weights: np.ndarray) -> pymatching.Matching:
        """
        Rebuild a PyMatching object using the FULL topology template.
        """
        # 1. Copy the RAW template graph
        G = self.decoder_template_graph.copy()

        # 2. Iterate and update weights
        is_multigraph = G.is_multigraph()
        edges_iterator = G.edges(keys=True, data=True) if is_multigraph else G.edges(data=True)

        for item in edges_iterator:
            if is_multigraph:
                u, v, k, d = item
            else:
                u, v, d = item

            # Normalize key to int for lookup (just for the dictionary)
            u_int, v_int = int(u), int(v)
            key = (u_int, v_int) if u_int <= v_int else (v_int, u_int)
            
            # Lookup and update weight
            idx = self.dec_edge_to_idx.get(key, None)
            if idx is not None:
                d["weight"] = float(custom_weights[idx])

        # 3. Use the constructor with explicit n_detectors
        # This prevents PyMatching from incorrectly inferring boundary nodes as regular nodes.
        m = pymatching.Matching(G, num_detectors=self.n_detectors)
        
        return m
    

    @staticmethod
    def _edge_set_jaccard(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
        """
        Jaccard similarity between two selected-edge index sets.
        """
        a = set(int(x) for x in a_idx.tolist())
        b = set(int(x) for x in b_idx.tolist())
        if not a and not b:
            return 1.0
        inter = len(a & b)
        uni = len(a | b)
        return float(inter / uni) if uni > 0 else 1.0


    def get_base_graph_info(self) -> Dict[str, Any]:
        """
        Useful for notebook inspection / building GNN models.
        """
        return {
            "n_decoding_edges": self.n_dec_edges,
            "n_line_edges": self.n_line_edges,
            "dec_edge_list": list(self.dec_edge_list),
            "base_edge_weight": self.current_weights.copy(),
            "line_edge_index": self.line_edge_index.copy(),
        }