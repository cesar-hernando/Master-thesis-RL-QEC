'''
In this file, we build a Gymnasium environment for reweighting the decoding graph based on the edge correlations 
statistics and the first MWPM pass selected edges.
'''

from typing import Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import plotly.graph_objects as go
import pymatching
import scipy.sparse as sp

from syndrome_data_generation import SyndromeDataGenerator



class DriftedMatchingEnv(gym.Env):
    """
    Graph-structured RL environment for learning MWPM edge reweighting in presence 
    of drift noise.

    Observation (graph in array form)
    ---------------------------------
    A fixed-size dictionary (good for Gym + GNN training pipelines):
      - node_features: [N_edges, 2]
          [:, 0] = current MWPM edge weight (base updated with the occurrence tracer)
          [:, 1] = selected by first MWPM pass (0/1)
      - edge_index: [2, M_line]
          Line-graph connectivity (nodes = decoding-graph edges)
      - edge_attr: [M_line, 1]
          Correlation tracer on line-graph edges 
      - action_mask: [N_edges]
          0/1 mask; if local_action_only=True, action is applied only where mask=1

    Action
    ------
    Vector of length N_edges, that modifies each of the edge weights by a number
    in the range action_scale*[-1, 1].

    For scalability and generalization, the env only applies actions on the local neighborhood 
    of the first-pass selected edges (controlled by local_action_only and local_action_hops). 
    The action_mask is included in the observation.

    Transition (one env.step)
    -------------------------
    1) Build second-pass weights = current_weights + masked_action_delta
    2) Run second-pass MWPM
    3) Update occurrence and correlation tracers using second-pass selected edges. Periodically,
    the first pass MWPM decoding graph is updated based on the occurrence tracer.
    4) Compute reward:
         - logical reward: based on predicted vs true observable
         - optional oracle imitation reward (DGR-like auxiliary signal)
    5) Retrieve next shot from episode cache and prepare next observation (1st MWPM pass)

    Episode semantics
    -----------------
    Each episode samples a new drifted circuit using SyndromeDataGenerator, and the agent 
    interacts with a sequence of shots from that circuit. Thus, the number of shots is a 
    hyperparameter that determines the length of the episode. It plays a simlar role as 
    the number of trials in the DGR paper. 
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        syndrome_data_generator: SyndromeDataGenerator,
        local_action_only: bool = True,
        local_action_hops: int = 1,
        action_scale: float = 1.0,
        update_period: int = 1000,
        prior_shots: int = 1000,
        min_weight: float = 1e-6,
        max_weight: float = 50.0,
        oracle_reward_coef: float = 0.0,
        use_pearson_correlation: bool = True,
        use_syndrome_features: bool = False,
        update_with: str = 'DGR',
        render_mode = None
    ):
        super().__init__()
        self.render_mode = render_mode

        # Use the provided syndrome data generator to create drifted circuits for each episode, sample
        # syndrome data and run MWPM via pymatching
        self.syndrome_data_generator = syndrome_data_generator

        # Build the base circuit and base matching graph, which are kept fixed through the episodes
        self.base_circuit, self.base_dem, self.base_matching = self.syndrome_data_generator.generate_base_circuit()

        # Store the number of detectors
        self.n_detectors = self.base_circuit.num_detectors

        # Config / hyperparams
        self.max_steps = self.syndrome_data_generator.n_shots
        self.local_action_only = local_action_only
        self.local_action_hops = local_action_hops
        self.action_scale = action_scale
        self.update_period = update_period
        self.prior_shots = prior_shots
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.oracle_reward_coef = oracle_reward_coef
        self.use_pearson_correlation = use_pearson_correlation
        self.use_syndrome_features = use_syndrome_features
        self.update_with = update_with

        # Define a dictionary that maps the index of each detector to its 3D coordinates
        self.detector_coords = self.base_circuit.get_detector_coordinates()

        # Extract the decoding edge weights, fault ids array, and fast lookup matrix natively
        (
            self.dec_edge_list, 
            self.pair_to_idx_matrix, 
            self.current_weights, 
            self.base_p,  
            self.fault_array,
            self.H
        ) = self._index_decoding_graph_edges(self.base_matching)
        self.n_dec_edges = len(self.dec_edge_list)

        # Store the original weights obtained from the DEM
        self.initial_base_weights = self.current_weights.copy()

        # Make a copy of the base matching that will be updated every step
        self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)

        # Build the line graph adding edges based on the DEM
        self.line_edge_index, self.initial_corr_tracer, self.k_hop_adj_mat = self._build_line_graph_edges(self.base_dem)
        self.n_line_edges = self.line_edge_index.shape[1]

        # Pre-calculate geometry for render method
        self._calculate_rendering_geometry()

        node_feat_dim = 3 if self.use_syndrome_features else 2
        edge_attr_dim = 2 if self.use_syndrome_features else 1

        # Define the Gym Observation and Action Spaces (fixed size for a given base circuit / distance)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_dec_edges, node_feat_dim),
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
                shape=(self.n_line_edges, edge_attr_dim),
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_dec_edges,),
                dtype=np.float32
            ),
        })

        # Define an action_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_dec_edges,),
            dtype=np.float32
        )       
    

    def _index_decoding_graph_edges(self, matching: pymatching.Matching):
        """
        Indexes the decoding graph and constructs the Sparse Parity Check Matrix (H).

        Maps physical detectors to graph edges to define the lattice topology and identify 
        logical fault mechanisms. The H matrix uses `h_rows` (detector IDs) 
        and `h_cols` (edge indices) to define the syndrome-to-error relationship.

        Args:
            matching: PyMatching graph object derived from the Stim noise model.

        Returns:
            tuple: (dec_edge_list, pair_to_idx, weights, error_probs, fault_array, H)
                - dec_edge_list: List of edge pairs (u, v), with -1 as boundary.
                - fault_array: Boolean mask for edges flipping a logical observable.
                - H: Sparse CSC incidence matrix of shape (n_detectors, n_edges).
        """
        dec_edge_list = []
        weights = []
        error_probs = []
        fault_ids = []

        matrix_size = self.n_detectors + 1
        pair_to_idx_matrix = np.full((matrix_size, matrix_size), -1, dtype=np.int32)

        # Arrays to hold the Sparse Check Matrix (H) coordinates
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

            # Map the edges to the Check Matrix (ignoring the -1 boundary)
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

        # Build the ultra-fast array
        fault_array = np.array([len(f) > 0 for f in fault_ids], dtype=bool)
    
        # Build the actual sparse C-matrix
        h_data = np.ones(len(h_rows), dtype=np.int8)
        H = sp.csc_matrix((h_data, (h_rows, h_cols)), shape=(self.n_detectors, idx))

        return (
            dec_edge_list, 
            pair_to_idx_matrix, 
            np.asarray(weights, dtype=np.float32), 
            np.asarray(error_probs, dtype=np.float32), 
            fault_array,
            H
        )
    

    def _build_line_graph_edges(self, dem, return_k_hop_adj_mat=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build line-graph connectivity exclusively from correlated hyperedges 
        in the Stim Detector Error Model (DEM).
        
        Returns:
            line_edge_index: [2, M_line] array of edges
            initial_correlations: [M_line] array of baseline raw joint probabilities from the DEM
            k_hop_adj_mat: [N_edges, N_edges] boolean adjacency mask
        """

        ##########################################################
        # 1. Parse the DEM to define the line graph connectivity #
        # and extract the joint probabilities                    #
        ##########################################################

        correlation_dict = {}
        
        for inst in dem.flattened():
            if inst.type != "error":
                continue
                
            p = inst.args_copy()[0]
            targets = inst.targets_copy()
            
            # Parse targets separated by '^' (Stim's correlation separator)
            components = []
            current_component = []
            
            for t in targets:
                if t.is_separator(): # Found a '^'
                    if current_component:
                        components.append(current_component)
                        current_component = []
                elif t.is_relative_detector_id():
                    current_component.append(t.val)
            
            if current_component:
                components.append(current_component)
                
            # If there is only 1 component, it's an independent error (no correlation)
            if len(components) < 2:
                continue
                
            # Map components to our decoding graph edges (GNN nodes)
            node_indices = []
            for comp in components:
                # Resolve the endpoints. If 1 detector, the other is the boundary
                u = comp[0] if len(comp) > 0 else -1
                v = comp[1] if len(comp) > 1 else -1
                
                # Fetch the canonical GNN node ID for this physical edge
                idx = self.pair_to_idx_matrix[u, v]
                if idx != -1:
                    node_indices.append(idx)
                    
            # Create pairwise connections in the line graph for this hyperedge
            for a in range(len(node_indices)):
                for b in range(a + 1, len(node_indices)):
                    na, nb = node_indices[a], node_indices[b]
                    if na == nb:
                        continue
                        
                    edge_key = (na, nb) if na <= nb else (nb, na)
                    
                    # Combine probabilities using logical OR: P(A ∪ B) = P(A) + P(B) - 2*P(A)*P(B)
                    if edge_key in correlation_dict:
                        existing_p = correlation_dict[edge_key]
                        correlation_dict[edge_key] = existing_p * (1 - p) + p * (1 - existing_p)
                    else:
                        correlation_dict[edge_key] = p
        
        # Ensure we return all 3 expected items if there are no correlations
        if not correlation_dict:
            dummy_adj = np.eye(self.n_dec_edges, dtype=bool)
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32), dummy_adj
            
        # Format into PyTorch Geometric expected arrays
        sorted_edges = sorted(correlation_dict.keys())
        line_edges = np.array(sorted_edges, dtype=np.int64).T
        src, dst = line_edges[0], line_edges[1]

        # Extract joint probabilities into a flat array
        initial_correlations = np.array([correlation_dict[e] for e in sorted_edges], dtype=np.float32)

        #########################################################
        # 2. Construct a k-hops adjacency matrix to compute the #
        # action mask fast                                      #
        #########################################################

        if not(return_k_hop_adj_mat):
            return line_edges, initial_correlations
        
        # Efficiently build the sparse matrix using coordinate arrays (avoids SparseEfficiencyWarning)
        # We concatenate the forward edges, backward edges, and the self-loops (diagonal)
        diag_idx = np.arange(self.n_dec_edges)
        rows = np.concatenate([src, dst, diag_idx])
        cols = np.concatenate([dst, src, diag_idx])
        data = np.ones(len(rows), dtype=np.int8)
        
        # Create the CSR matrix instantly in one shot
        adj_mat = sp.csr_matrix((data, (rows, cols)), shape=(self.n_dec_edges, self.n_dec_edges))
            
        # Compute K-hops instantly using sparse matrix power
        k_hop_sparse = adj_mat ** self.local_action_hops
        
        # Safely convert to a dense boolean mask
        # (Checks if SciPy kept it sparse or secretly converted it to a dense array)
        if sp.issparse(k_hop_sparse):
            k_hop_adj_mat = k_hop_sparse.toarray() > 0
        else:
            k_hop_adj_mat = k_hop_sparse > 0
        
        return line_edges, initial_correlations, k_hop_adj_mat
    

    def _calculate_rendering_geometry(self):
        """Helper method to compute midpoints and X/Z types for Plotly rendering."""

        def pad3(c):
                if c is None: return None
                lst = list(c)
                while len(lst) < 3: lst.append(0.0)
                return np.array(lst[:3])
        
        midpoints = np.zeros((len(self.dec_edge_list), 3))
        edge_types = []

        for i, (u, v) in enumerate(self.dec_edge_list):
            cu = self.detector_coords.get(u, None)
            cv = self.detector_coords.get(v, None)            
            cu, cv = pad3(cu), pad3(cv)
            
            if cu is not None and cv is not None: midpoints[i] = (cu + cv) / 2.0
            elif cu is not None: midpoints[i] = cu
            elif cv is not None: midpoints[i] = cv
            else: midpoints[i] = np.zeros(3)
                
            etype = "Unknown"
            real_node = u if cu is not None else (v if cv is not None else None)
            if real_node is not None:
                c = self.detector_coords[real_node]
                j, y = int(round(c[0])), int(round(c[1])) # j=x, y=y
                if (y % 4 == 0 and j % 4 == 0) or (y % 4 == 2 and j % 4 == 2): etype = "Z"
                elif (y % 4 == 0 and j % 4 == 2) or (y % 4 == 2 and j % 4 == 0): etype = "X"
            edge_types.append(etype)

            # ----------------------------------------------------------------
            # DETERMINISTIC VISUAL SEPARATION
            # Shift X and Z subgraphs slightly apart to reveal Y-correlations
            # ----------------------------------------------------------------
            visual_offset = 0.15 # Tweak this if the gap is too large or too small
            
            if etype == "X":
                # Shift X-nodes slightly in the +x, +y direction
                midpoints[i] += np.array([visual_offset, visual_offset, 0.0])
            elif etype == "Z":
                # Shift Z-nodes slightly in the -x, -y direction
                midpoints[i] += np.array([-visual_offset, -visual_offset, 0.0])

        self.edge_midpoints = midpoints
        self.edge_types = edge_types


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Generate a new drifted circuit and the corresponding decoding graph
        drifted_circuit, drifted_dem, drifted_matching = self.syndrome_data_generator.generate_drifted_circuit(
            base_circuit=self.base_circuit,
            seed=seed
            )
        
        # Retrieve the weights of the oracle decoding graph
        _, _, self.oracle_weights, oracle_probs, _, _ = self._index_decoding_graph_edges(drifted_matching)

        # Calculate the joint probabilities and Pearson correlations between the oracle edge weights
        _, oracle_joint_probs = self._build_line_graph_edges(drifted_dem, return_k_hop_adj_mat=False)
        self.oracle_correlations = self.compute_pearson_correlations(oracle_probs, oracle_joint_probs)
        
        # Pre-generate syndrome data and true observable for the entire episode (all shots under the same drift)
        self.syndrome_batch, self.true_obs_batch = self.syndrome_data_generator.simulate_syndrome_data(drifted_circuit)

        # Pre-compute physics chunks and initialize trackers
        if self.use_syndrome_features:
            self._precompute_all_syndrome_statistics()
            self.spitz_tracer = self.initial_base_weights.copy()
            self.remm_tracer = self.initial_corr_tracer.copy()

        # Pre-generate the oracle solution edges and predicted observable for each shot in the episode using the drifted matching
        self.oracle_solution_edges_batch, self.oracle_predicted_obs_batch = self.syndrome_data_generator.get_solution_edges_batch(
            matching=drifted_matching, 
            syndrome_volume_batch=self.syndrome_batch, 
            enable_correlations=True, 
            return_predicted_obs=True,
            pair_to_idx_matrix=self.pair_to_idx_matrix,
            fault_array=self.fault_array
        )
        
        # Pre-generate the static decoder solution edge predicted observable
        self.static_predicted_obs_batch = self.base_matching.decode_batch(
            self.syndrome_batch, 
            enable_correlations=False
            ).flatten()
        
        # Reset the step count
        self.step_count = 0

        # Reset decoder weights/matching to the original base graph each episode
        self.current_weights = self.initial_base_weights.copy()
        self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)

        # Calculate the initial weights mse error between oracle and base
        weights_mse_error = np.mean((self.current_weights - self.oracle_weights)**2)

        # Initialize absolute counters for the actual observed MWPM shots
        self.shots_since_update = 0
        self.occ_batch_counts = np.zeros(self.n_dec_edges, dtype=np.float32)
        self.corr_batch_counts = np.zeros(self.n_line_edges, dtype=np.float32)

        # Reset tracers
        self.occ_tracer = self.base_p.copy()
        self.corr_tracer = self.initial_corr_tracer.copy()

        # Compute initial Pearson correlations of our adaptive decoder
        self.pearson_correlations = self.compute_pearson_correlations(self.occ_tracer, self.corr_tracer)

        # Compute the MSE error between the initial Pearson correlations of our decoder and the oracle decoder
        self.corr_mse_error = np.mean((self.pearson_correlations - self.oracle_correlations)**2)

        # Reset Syndrome Physics Tracers
        if self.use_syndrome_features:
            self.spitz_tracer = self.initial_base_weights.copy()
            self.remm_tracer = self.initial_corr_tracer.copy()

        # Prepare the first shot and build the initial observation
        obs = self._prepare_next_observation()

        info = {
            "n_decoding_edges": self.n_dec_edges,
            "n_line_edges": self.n_line_edges,
            "weights_mse_error": weights_mse_error,
            "corr_mse_error": self.corr_mse_error
        }
        return obs, info
    

    def _prepare_next_observation(self) -> Dict[str, np.ndarray]:
        # Pull the pre-generated syndrome for the current step
        syndrome = self.syndrome_batch[self.step_count]
        true_obs = self.true_obs_batch[self.step_count]

        # First pass MWPM still uses the current matching (no drift knowledge)
        selected_edges_1 = self.syndrome_data_generator.get_solution_edges(
            matching=self.current_matching, 
            syndrome_volume=syndrome, 
            enable_correlations=False,
            return_predicted_obs=False
        )
        
        # Determine the indices of the selected edges
        selected_idx_1 = self._selected_edge_indices_from_pairs(selected_edges_1)

        selected_flag = np.zeros(self.n_dec_edges, dtype=np.float32)
        if selected_idx_1 is not None and len(selected_idx_1) > 0:
            selected_flag[selected_idx_1] = 1.0

        # Action mask around first-pass selected edges
        action_mask = self._compute_action_mask(selected_idx_1)
        action_mask = (
            action_mask.astype(np.float32)
            if action_mask is not None
            else np.ones(self.n_dec_edges, dtype=np.float32)
        )

        # Cache
        self.current_syndrome = syndrome
        self.current_true_obs = true_obs
        self.current_first_pass_selected_idx = selected_idx_1
        self.current_action_mask = action_mask

        # Evaluate the DGR (Decoder) Edge Feature
        if self.use_pearson_correlation:
            if self.n_line_edges > 0:
                src = self.line_edge_index[0]
                dst = self.line_edge_index[1]
                
                p_src = self.occ_tracer[src]
                p_dst = self.occ_tracer[dst]
                
                # Raw covariance
                covariance = self.corr_tracer - (p_src * p_dst)
                
                # Normalize (Pearson Correlation)
                std_src = np.sqrt(p_src * (1.0 - p_src))
                std_dst = np.sqrt(p_dst * (1.0 - p_dst))
                denom = std_src * std_dst
                
                safe_denom = np.where(denom > 1e-9, denom, 1.0)
                correlation = covariance / safe_denom
                
                dgr_edge_feat = np.clip(correlation, 0.0, 1.0)
            else:
                dgr_edge_feat = np.zeros(0, dtype=np.float32)
        else:
            dgr_edge_feat = self.corr_tracer

        # Build Feature Arrays Dynamically based on the Flag
        if self.use_syndrome_features:
            # Stack: [Base Weights, 1st Pass Flag, Spitz Probabilities]
            node_feats = np.stack([self.current_weights, selected_flag, self.spitz_tracer], axis=1)
            
            if self.n_line_edges > 0:
                # Stack: [DGR Tracer (Covariance or Raw), Remm Covariance]
                edge_feats = np.stack([dgr_edge_feat, self.remm_tracer], axis=1)
            else:
                edge_feats = np.zeros((0, 2), dtype=np.float32)
        else:
            # Fallback to the original DGR-only sizes
            node_feats = np.stack([self.current_weights, selected_flag], axis=1)
            edge_feats = dgr_edge_feat.reshape(-1, 1)

        obs = {
            "node_features": node_feats.astype(np.float32),
            "edge_index": self.line_edge_index.astype(np.int64),
            "edge_attr": edge_feats.astype(np.float32),
            "action_mask": action_mask,
        }

        return obs
    

    def _selected_edge_indices_from_pairs(self, edge_pairs: np.ndarray) -> np.ndarray:
        """Vectorized O(1) array lookup taking advantage of NumPy's -1 indexing."""
        if edge_pairs.size == 0:
            return np.zeros((0,), dtype=np.int64)

        # Slice out the u and v columns
        u = edge_pairs[:, 0].astype(np.int32)
        v = edge_pairs[:, 1].astype(np.int32)

        # Instant lookup of all edges in C-memory (NumPy handles the -1 boundary magically!)
        idxs = self.pair_to_idx_matrix[u, v]

        # Filter out invalid edges (-1) and return sorted unique indices
        valid_idxs = idxs[idxs != -1]
        return np.unique(valid_idxs).astype(np.int64)
    

    def _compute_action_mask(self, selected_idx: np.ndarray) -> np.ndarray:
        """Instant vectorized mask using pre-computed K-hop matrix."""
        if not self.local_action_only:
            return np.ones(self.n_dec_edges, dtype=np.float32)

        if selected_idx.size == 0:
            return np.zeros(self.n_dec_edges, dtype=np.float32)

        # Slice the K-hop matrix rows for the selected edges, 
        # collapse them with .any(), and cast to float mask!
        mask = self.k_hop_adj_mat[selected_idx].any(axis=0).astype(np.float32)
        
        return mask
    

    def step(self, action: np.ndarray):
        
        assert self.current_syndrome is not None, "Call reset() before step()."

        ###############################################
        # 1) Apply action (masked locally if enabled) #
        ###############################################

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_dec_edges:
            raise ValueError(f"Action has shape {action.shape}; expected ({self.n_dec_edges},)")

        if self.local_action_only:
            mask = self.current_action_mask.astype(np.float32)
            applied_delta = action * mask * self.action_scale
        else:
            applied_delta = action * self.action_scale
        
        if not np.any(applied_delta):
            second_pass_matching = self.current_matching
        else:
            second_pass_weights = np.clip(self.current_weights + applied_delta, self.min_weight, self.max_weight)
            second_pass_matching = pymatching.Matching.from_check_matrix(self.H, weights=second_pass_weights)

        ##############################################
        # 2) Run 2nd pass MWPM with reweighted edges #
        ##############################################
        
        selected_edges_2, pred_obs = self.syndrome_data_generator.get_solution_edges(
            matching=second_pass_matching,
            syndrome_volume=self.current_syndrome,
            enable_correlations=False,
            return_predicted_obs=True,
            pair_to_idx_matrix=self.pair_to_idx_matrix,
            fault_array=self.fault_array
        )
        
        selected_idx_2 = self._selected_edge_indices_from_pairs(selected_edges_2)

        ######################################################
        # 3) Update tracers using 2nd-pass selected edges #
        ######################################################

        self._accumulate_occurrence(selected_idx_2)
        self._accumulate_correlation(selected_idx_2)
        self.shots_since_update += 1

        if self.shots_since_update >= self.update_period:
            self._apply_cma_and_update_graph()
            self.shots_since_update = 0
            self.corr_mse_error = np.mean((self.pearson_correlations - self.oracle_correlations)**2)
                                                 
        #########################
        # 4) Compute the reward #
        #########################

        # 1. Evaluate Truth
        agent_correct = (pred_obs == self.current_true_obs)
        static_correct = (self.static_predicted_obs_batch[self.step_count] == self.current_true_obs)

        # 2. Differential Logical Reward
        if agent_correct and not static_correct:
            logical_reward = +1.0 
        elif not agent_correct and static_correct:
            logical_reward = -1.0
        else:
            logical_reward = 0.0   # Trivial success or completely uncorrectable. Agent didn't matter.

        reward = logical_reward

        # 3. Oracle Imitation Reward (Dense Shaping)
        oracle_similarity = None
        if self.oracle_reward_coef > 0.0:
            # Pull the pre-calculated oracle edges for this exact shot
            oracle_edges = self.oracle_solution_edges_batch[self.step_count]
            oracle_idx = self._selected_edge_indices_from_pairs(oracle_edges)
            
            # Compute Jaccard similarity (0.0 to 1.0)
            oracle_similarity = self._edge_set_jaccard(selected_idx_2, oracle_idx)
            
            # Map to [-1, 1] for zero-centered neural network stability
            oracle_reward = (2.0 * oracle_similarity) - 1.0
            
            # Combine the rewards
            reward += self.oracle_reward_coef * oracle_reward

        # Optional: Clip the final reward to [-1, 1] to keep Q-values extremely stable during early training
        # reward = np.clip(reward, -1.0, 1.0)

        ############################################################
        # 5) Prepare information about current step to be returned #
        ############################################################

        weights_mse_error = np.mean((self.current_weights - self.oracle_weights)**2)

        # Increment step count after processing the shot
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps # Episode ends when batch is empty

        info = {
            "logical_error": not(agent_correct),
            "true_obs": self.current_true_obs,
            "pred_obs": pred_obs,
            "oracle_pred_obs": self.oracle_predicted_obs_batch[self.step_count - 1],
            "static_pred_obs":self.static_predicted_obs_batch[self.step_count - 1],
            "reward_logical": logical_reward,
            "reward_total": float(reward),
            "oracle_similarity_jaccard": float(oracle_similarity) if oracle_similarity is not None else None,
            "selected_edges_first_pass_idx": self.current_first_pass_selected_idx.copy() if self.current_first_pass_selected_idx is not None else None,
            "selected_edges_second_pass_idx": selected_idx_2.copy(),
            "action_mask": self.current_action_mask.copy() if self.current_action_mask is not None else None,
            "weights_mse_error": weights_mse_error,
            "corr_mse_error": self.corr_mse_error
        }

        ############################################################
        # 6) Prepare next observation (next shot under same drift) #
        ############################################################

        if not truncated:
            next_obs = self._prepare_next_observation()
        else:
            next_obs = None
                    

        return next_obs, float(reward), terminated, truncated, info   
    

    def _accumulate_occurrence(self, selected_idx: np.ndarray):
        """Add 1 to the occurrence batch count for selected edges."""
        if selected_idx.size > 0:
            self.occ_batch_counts[selected_idx] += 1.0


    def _accumulate_correlation(self, selected_idx: np.ndarray):
        """Add 1 to the co-occurrence batch count for line graph edges."""
        if self.n_line_edges == 0 or selected_idx.size == 0:
            return
        active = np.zeros(self.n_dec_edges, dtype=np.bool_)
        active[selected_idx] = True
        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]
        self.corr_batch_counts += (active[src] & active[dst]).astype(np.float32)

    
    def _compute_raw_syndrome_statistics(self, recent_syndromes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Maps Spitz and Remm analytical syndrome formulas to the GNN.
        Calculates exact physical probabilities for a specific time window.
        """
        n_shots = recent_syndromes.shape[0]
        spitz_probs = np.zeros(self.n_dec_edges, dtype=np.float32)
        v_mean = np.mean(recent_syndromes, axis=0) 
        
        E_matrix = np.zeros((self.n_dec_edges, n_shots), dtype=bool)

        for idx, (u, v) in enumerate(self.dec_edge_list):
            # Handle boundary edges
            if u == -1 or v == -1:
                real_node = v if u == -1 else u
                p = np.clip(v_mean[real_node] / 2.0, 1e-6, 0.499)
                spitz_probs[idx] = p
                E_matrix[idx] = recent_syndromes[:, real_node] == 1
                continue
            
            # Extract boolean columns for the two detectors
            v_u = recent_syndromes[:, u]
            v_v = recent_syndromes[:, v]
            E_matrix[idx] = v_u & v_v
            
            # Spitz Formula Variables
            mean_u = v_mean[u]
            mean_v = v_mean[v]
            mean_uv = np.mean(E_matrix[idx])
            mean_xor = np.mean(v_u ^ v_v)
            
            # Calculate the Spitz probability
            denom = 1.0 - 2.0 * mean_xor
            if abs(denom) < 1e-9:
                p = 0.499 
            else:
                cov = mean_uv - (mean_u * mean_v)
                root_term = max(0.0, 0.25 - (cov / denom))
                p = np.clip(0.5 - np.sqrt(root_term), 1e-6, 0.499)
                
            spitz_probs[idx] = p

        remm_covariances = np.zeros(self.n_line_edges, dtype=np.float32)

        if self.n_line_edges > 0:
            src = self.line_edge_index[0]
            dst = self.line_edge_index[1]
            mean_E = E_matrix.mean(axis=1)

            for i in range(self.n_line_edges):
                u_idx, v_idx = src[i], dst[i]
                edge1 = self.dec_edge_list[u_idx]
                edge2 = self.dec_edge_list[v_idx]

                # Symmetric difference to handle syndrome cancellation natively (modulo-2 arithmetic)
                nodes1 = set([n for n in edge1 if n != -1])
                nodes2 = set([n for n in edge2 if n != -1])
                signature_nodes = list(nodes1.symmetric_difference(nodes2))

                # If the edges completely cancel each other out (empty signature), covariance is 0
                if not signature_nodes:
                    continue

                # The event H is true ONLY if the exact uncanceled signature flashed
                H_active = np.ones(n_shots, dtype=bool)
                for node in signature_nodes:
                    H_active &= (recent_syndromes[:, node] == 1)

                mean_H = np.mean(H_active)
                accidental_overlap = mean_E[u_idx] * mean_E[v_idx]
                
                # Covariance: <H> - <E_1><E_2>
                remm_covariances[i] = max(0.0, mean_H - accidental_overlap)

        return spitz_probs, remm_covariances


    def _precompute_all_syndrome_statistics(self):
        """
        Pre-computes the Spitz and Remm statistics for every update_period chunk 
        in the episode to maximize step() performance and avoid lookahead bias.
        """
        num_chunks = int(np.ceil(self.max_steps / self.update_period))
        
        self.precomputed_spitz = np.zeros((num_chunks, self.n_dec_edges), dtype=np.float32)
        self.precomputed_remm = np.zeros((num_chunks, self.n_line_edges), dtype=np.float32)
        
        for i in range(num_chunks):
            start_idx = i * self.update_period
            end_idx = min(start_idx + self.update_period, self.max_steps)
            
            # Slice the exact window of shots for this specific chunk
            chunk = self.syndrome_batch[start_idx:end_idx]
            
            # Calculate the physics statistics for this chunk
            spitz, remm = self._compute_raw_syndrome_statistics(chunk)
            
            # Store them in the cache arrays
            self.precomputed_spitz[i] = spitz
            self.precomputed_remm[i] = remm

    
    def compute_pearson_correlations(self, occ_array: np.ndarray, corr_array: np.ndarray) -> np.ndarray:
        """
        Computes Pearson correlations for a given set of marginal and joint probabilities
        using the environment's fixed line graph topology.
        """
        if not self.use_pearson_correlation:
            return corr_array.copy()

        if self.n_line_edges == 0:
            return np.zeros(0, dtype=np.float32)

        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]
        
        p_src = occ_array[src]
        p_dst = occ_array[dst]
        
        # Raw covariance
        covariance = corr_array - (p_src * p_dst)
        
        # Normalize (Pearson Correlation)
        std_src = np.sqrt(p_src * (1.0 - p_src))
        std_dst = np.sqrt(p_dst * (1.0 - p_dst))
        denom = std_src * std_dst
        
        safe_denom = np.where(denom > 1e-9, denom, 1.0)
        return np.clip(covariance / safe_denom, 0.0, 1.0).astype(np.float32)


    def _apply_cma_and_update_graph(self):
        # 1. Calculate the decaying CMA learning rate (alpha)
        alpha = self.update_period / (self.prior_shots + self.step_count)

        # 2. DGR TRACERS 
        batch_occ_rate = self.occ_batch_counts / self.update_period
        self.occ_tracer = self.occ_tracer + alpha * (batch_occ_rate - self.occ_tracer)
        
        if self.n_line_edges > 0:
            batch_corr_rate = self.corr_batch_counts / self.update_period
            self.corr_tracer = self.corr_tracer + alpha * (batch_corr_rate - self.corr_tracer)

        # 3. SYNDROME TRACERS
        if self.use_syndrome_features:
            # Figure out which chunk we just finished 
            chunk_idx = (self.step_count - 1) // self.update_period
            
            # O(1) Array Lookup
            batch_spitz = self.precomputed_spitz[chunk_idx]
            batch_remm = self.precomputed_remm[chunk_idx]
            
            # Apply the exact same CMA update rule
            self.spitz_tracer = self.spitz_tracer + alpha * (batch_spitz - self.spitz_tracer)
            if self.n_line_edges > 0:
                self.remm_tracer = self.remm_tracer + alpha * (batch_remm - self.remm_tracer)

        # 4. Update MWPM weights 
        if self.update_with == 'DGR':
            p = np.clip(self.occ_tracer, 1e-6, 0.499999)
        elif self.update_with == 'Spitz':
            p = np.clip(self.spitz_tracer, 1e-6, 0.499999)

        self.current_weights = np.clip(np.log((1.0 - p) / p), self.min_weight, self.max_weight)
        self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)
        
        self.occ_batch_counts.fill(0.0)
        self.corr_batch_counts.fill(0.0)

        # 5. Update the new Pearson correlations from the updated tracers
        self.pearson_correlations = self.compute_pearson_correlations(self.occ_tracer, self.corr_tracer)
    

    @staticmethod
    def _edge_set_jaccard(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
        """Pure NumPy Jaccard similarity without Python object creation."""
        if a_idx.size == 0 and b_idx.size == 0:
            return 1.0
            
        # assume_unique=True skips a sorting step, making this blazing fast
        inter = np.intersect1d(a_idx, b_idx, assume_unique=True).size
        uni = a_idx.size + b_idx.size - inter
        
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
    

    def render(self):
        """
        Renders the Line Graph topology in 3D using Plotly.
        Nodes represent the original decoding edges, and edges represent message-passing paths.
        Includes GNN node features and edge features (correlations) directly on the lines.
        """
        if self.render_mode != "human":
            return

        print("Rendering Line Graph Topology & GNN Features...")

        midpoints = self.edge_midpoints
        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]

        # ---------------------------------------------------------
        # 1. Build Edge Geometry and Features directly on the lines
        # ---------------------------------------------------------
        edge_x, edge_y, edge_z = [], [], []
        edge_colors = []
        edge_hover_text = []

        for i in range(self.n_line_edges):
            u_idx, v_idx = src[i], dst[i]
            x0, y0, z0 = midpoints[u_idx]
            x1, y1, z1 = midpoints[v_idx]
            
            # Geometry for the continuous line trace (with None to break segments)
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            # Fetch Edge Feature (Pearson Correlation instead of raw joint prob)
            pearson_val = self.pearson_correlations[i]
            raw_joint = self.corr_tracer[i]
            
            txt = (
                f"<b>Line Edge ID:</b> {i}<br>"
                f"<b>Connects Nodes:</b> {u_idx} &mdash; {v_idx}<br>"
                f"<b>Pearson Correlation (GNN Input):</b> {pearson_val:.4f}<br>"
                f"<b>Raw Joint Prob:</b> {raw_joint:.5f}"
            )
            
            # Apply the color and text to both ends of the line segment
            # The third value (0.0 / "") corresponds to the 'None' coordinate gap
            edge_colors.extend([pearson_val, pearson_val, 0.0])
            edge_hover_text.extend([txt, txt, ""])

        fig = go.Figure()

        # Add the Edges (Colored dynamically based on Pearson correlation)
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            opacity=0.5,
            line=dict(
                color=edge_colors,
                colorscale='Viridis', 
                width=4,              
                showscale=True,
                colorbar=dict(title="Pearson<br>Correlation", x=0.85, thickness=15, len=0.5)
            ),
            text=edge_hover_text,
            hoverinfo='text',
            name='Message Passing Paths'
        ))

        # ---------------------------------------------------------
        # 2. Build Node Geometry and Node Features
        # ---------------------------------------------------------
        # Deep Red for X, Deep Blue for Z, Gold for Unknown/Ancilla
        color_map = {'X': '#b22222', 'Z': '#005f87', 'Unknown': '#d4af37'}
        node_colors = [color_map.get(t, '#d4af37') for t in self.edge_types]

        first_pass_set = set()
        if self.current_first_pass_selected_idx is not None:
            first_pass_set = set(self.current_first_pass_selected_idx.tolist())

        node_hover_text = []
        for i in range(self.n_dec_edges):
            weight = self.current_weights[i]
            occ = self.occ_tracer[i]
            fired = "Yes" if i in first_pass_set else "No"
            
            node_hover_text.append(
                f"<b>Node ID:</b> {i} ({self.edge_types[i]})<br>"
                f"<b>Current Weight:</b> {weight:.3f}<br>"
                f"<b>Occurrence prob:</b> {occ:.4f}<br>"
                f"<b>1st Pass Fired:</b> {fired}"
            )

        # Add Nodes (Original decoding edges)
        fig.add_trace(go.Scatter3d(
            x=midpoints[:, 0], y=midpoints[:, 1], z=midpoints[:, 2],
            mode='markers',
            marker=dict(
                size=6, 
                color=node_colors, 
                line=dict(width=1, color='white'),
                opacity=1.0
            ),
            text=node_hover_text,
            hoverinfo='text',
            name='GNN Nodes'
        ))

        # ---------------------------------------------------------
        # 3. Layout Formatting
        # ---------------------------------------------------------
        fig.update_layout(
            title="<b>GNN Line Graph Topology & Features</b><br><sup>Hover over nodes for state features, hover over lines for Pearson correlations</sup>",
            scene=dict(
                xaxis_title="X (Space)",
                yaxis_title="Y (Space)",
                zaxis_title="T (Time)",
                aspectmode="data",
                bgcolor='rgb(245, 245, 245)' 
            ),
            legend=dict(itemsizing="constant", yanchor="top", y=0.9, xanchor="left", x=0.05),
            margin=dict(l=0, r=0, b=0, t=60)
        )

        # Save silently to an HTML file instead of opening a new tab
        filename = "live_graph.html"
        fig.write_html(filename)