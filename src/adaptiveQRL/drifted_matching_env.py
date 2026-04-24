'''
In this script, we build a Gymnasium environment to train a RL agent to reweight the 
decoding graph based on edge correlations and first-pass selected edges, in the 
presence of drifted noise.
'''

from typing import Dict, Any
import inspect
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import plotly.graph_objects as go
import pymatching
import scipy.sparse as sp

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator



class DriftedMatchingEnv(gym.Env):
    """
    Graph-structured RL environment for learning MWPM edge reweighting in presence 
    of drift noise and correlated errors.

    Observation (graph in array form)
    ---------------------------------
    A fixed-size dictionary (good for PyTorch Geometric) containing:
      - node_features: [N_edges, 2]
          [:, 0] = current MWPM edge weight (base updated with the occurrence tracer)
          [:, 1] = selected by first MWPM pass (0/1)
      - edge_index: [2, M_line]
          Line-graph connectivity (nodes = decoding-graph edges)
      - edge_attr: [M_line, 1]
          Correlation tracer on line-graph edges 
      - action_mask: [N_edges]
          0/1 mask; if local_action_only=True, action is applied only where mask = 1

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
        - logical reward: based on the success of the second-pass predicted observable compared
            compared to the success of the first-pass predicted observable (relative improvement)
    5) Retrieve next shot from episode cache and prepare next observation (1st MWPM pass)

    Episode semantics
    -----------------
    Each episode samples a new drifted circuit using SyndromeDataGenerator, and the agent 
    interacts with a sequence of shots from that circuit. Thus, the number of shots is a 
    hyperparameter that determines the length of the episode. It plays a simlar role as 
    the number of trials in the paper:  
    Wang, H., Liu, P. et al (2023). Dgr: Tackling drifted and correlated noise in quantum 
    error correction via decoding graph re-weighting. arXiv:2311.16214.. 
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
        n_test_shots: int = 10_000,
        min_weight: float = 1e-6,
        max_weight: float = 50.0,
        use_pearson_correlation: bool = True,
        use_syndrome_features: bool = False,
        use_log_joint_prob: bool = False,
        update_with: str = 'DGR',
        dynamic_drift: bool = False,
        train_mode: bool = True,
        render_mode = None
    ):
        super().__init__()

        self.render_mode = render_mode
        self.train_mode = train_mode

        # Use the provided syndrome data generator to create drifted circuits for each episode, sample
        # syndrome data and run MWPM via pymatching
        self.syndrome_data_generator = syndrome_data_generator

        # Configurable parameters for the environment dynamics
        self.max_steps = self.syndrome_data_generator.n_shots
        self.local_action_only = local_action_only
        self.local_action_hops = local_action_hops
        self.action_scale = action_scale
        self.update_period = update_period
        self.prior_shots = prior_shots
        self.n_test_shots = n_test_shots
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_pearson_correlation = use_pearson_correlation
        self.use_syndrome_features = use_syndrome_features
        self.use_log_joint_prob = use_log_joint_prob
        self.update_with = update_with
        self.dynamic_drift = dynamic_drift

        # Build the base circuit and base matching graph, which are kept fixed through the episodes
        self.base_circuit, self.base_dem, self.base_matching = self.syndrome_data_generator.generate_base_circuit()

        # Store the number of detectors
        self.n_detectors = self.base_circuit.num_detectors

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
        self.dec_edge_arr = np.array(self.dec_edge_list, dtype=np.float64)

        # Store the original weights obtained from the DEM
        self.initial_base_weights = self.current_weights.copy()

        # Make a copy of the base matching that will be updated every step
        self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)

        # Build the line graph adding edges based on the DEM
        self.line_edge_index, self.initial_corr_tracer, self.k_hop_adj_mat = self._build_line_graph_edges(self.base_dem)
        self.n_line_edges = self.line_edge_index.shape[1]
        self.initial_pearson_corr = self._compute_pearson_correlations(self.base_p, self.initial_corr_tracer)

        # Pre-allocate the node and edge features
        if self.use_syndrome_features:
            self.node_feats = np.zeros((self.n_dec_edges, 3), dtype=np.float32)
            if self.n_line_edges > 0:
                self.edge_feats = np.zeros((self.n_line_edges, 2), dtype=np.float32)
            else:
                self.edge_feats = np.zeros((0, 2), dtype=np.float32)
        else:
            self.node_feats = np.zeros((self.n_dec_edges, 2), dtype=np.float32)
            if self.n_line_edges > 0:
                self.edge_feats = np.zeros((self.n_line_edges, 1), dtype=np.float32)
            else:
                self.edge_feats = np.zeros((0, 1), dtype=np.float32)

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
        Maps physical detectors to graph edges to define the lattice topology and 
        identify logical fault mechanisms. The H matrix uses h_rows (detector IDs) 
        and h_cols (edge indices) to define the syndrome-to-error relationship.

        Args:
            matching: PyMatching graph object derived from the Stim noise model.

        Returns:
            - dec_edge_list: List of (u, v) pairs for each decoding graph edge 
                (u and v are detector IDs or -1 for boundary)
            - pair_to_idx: 2D array mapping (u, v) pairs to edge indices in C-memory 
                (shape: [num_detectors+1, num_detectors+1])
            - weights: Array of edge weights corresponding to the decoding graph edges
            - error_probs: Array of error probabilities corresponding to the decoding graph edges
            - fault_array: Boolean array indicating whether each edge corresponds to a logical fault 
                (derived from fault_ids)
            - H: Sparse parity check matrix in CSC format (shape: [num_detectors, num_edges])
        """

        dec_edge_list = []
        weights = []
        error_probs = []
        fault_ids = []

        matrix_size = self.n_detectors + 1
        pair_to_idx_matrix = np.full((matrix_size, matrix_size), -1, dtype=np.int32)

        # Arrays to hold the Sparse Check Matrix (H) coordinates
        # h_rows correspond to detector IDs
        # h_cols correspond to edge indices in the decoding graph
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
    

    def _build_line_graph_edges(self, dem, return_k_hop_adj_mat=True):
        """
        Build line-graph connectivity exclusively from correlated hyperedges 
        in the Stim Detector Error Model (DEM).
        
        Args:
            dem: Stim Detector Error Model object containing the error mechanisms and 
                their correlations.
            return_k_hop_adj_mat: If True, also compute and return the K-hop adjacency 
                matrix for action masking.

        Returns:
            line_edge_index: [2, M_line] array of edges
            initial_correlations: [M_line] array of baseline raw joint probabilities 
                from the DEM
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
                node_indices.append(idx)
                    
            # Create pairwise connections in the line graph for this hyperedge
            for a in range(len(node_indices)):
                for b in range(a + 1, len(node_indices)):
                    na, nb = node_indices[a], node_indices[b]
                    if na == nb:
                        continue
                        
                    edge_key = (na, nb) if na <= nb else (nb, na)
                    
                    # Combine probabilities using logical XOR: P(A ∪ B) = P(A) + P(B) - 2*P(A)*P(B)
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
        """
        Helper method to compute midpoints and X/Z types for Plotly rendering.
        """

        # Helper function to pad coordinates to 3D (if missing) for consistent 
        # midpoint calculation
        def pad3(c):
                if c is None: return None
                lst = list(c)
                while len(lst) < 3: lst.append(0.0)
                return np.array(lst[:3])
        
        # Get the physical coordinates of the detectors from the base circuit 
        # to compute edge midpoints
        detector_coords = self.base_circuit.get_detector_coordinates()

        # Calculate midpoints and edge types for rendering
        midpoints = np.zeros((len(self.dec_edge_list), 3))
        edge_types = []
        for i, (u, v) in enumerate(self.dec_edge_list):
            cu = detector_coords.get(u, None)
            cv = detector_coords.get(v, None)            
            cu, cv = pad3(cu), pad3(cv)
            
            if cu is not None and cv is not None: midpoints[i] = (cu + cv) / 2.0
            elif cu is not None: midpoints[i] = cu
            elif cv is not None: midpoints[i] = cv
            else: midpoints[i] = np.zeros(3)
                
            etype = "Unknown"
            real_node = u if cu is not None else (v if cv is not None else None)
            if real_node is not None:
                c = detector_coords[real_node]
                j, y = int(round(c[0])), int(round(c[1])) # j=x, y=y
                if (y % 4 == 0 and j % 4 == 0) or (y % 4 == 2 and j % 4 == 2): etype = "Z"
                elif (y % 4 == 0 and j % 4 == 2) or (y % 4 == 2 and j % 4 == 0): etype = "X"
            edge_types.append(etype)

            # DETERMINISTIC VISUAL SEPARATION
            # Shift X and Z subgraphs slightly apart to reveal Y-correlations
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
        """
        Reset the environment for a new episode. This involves generating a new drifted circuit,
        pre-generating the syndrome data and true observables for the entire episode, and resetting
        all trackers and counters. The first observation is prepared based on the first shot's syndrome
        and the initial MWPM decoding graph (which starts as the base graph with no drift knowledge).

        Args:
            seed: Optional random seed for reproducibility of the drifted circuit and syndrome data.
            options: Optional dictionary for additional reset configurations (not used currently).

        Returns:
            obs: The initial observation for the first step of the episode, based on the first shot's syndrome.
            info: A dictionary containing metadata about the environment and episode initialization.
        """

        super().reset(seed=seed)

        # Reset the step count
        self.step_count = 0

        # Reset decoder weights/matching to the original base graph each episode
        self.current_weights = self.initial_base_weights.copy()
        self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)

        # Initialize the dynamic LER trackers (calculated periodically every time the decoding graph is 
        # updated with new drift knowledge)
        self.ler = np.zeros(self.max_steps//self.update_period + 1, dtype=np.float32)

        # Initialize absolute counters for the actual observed MWPM shots
        self.shots_since_update = 0
        self.occ_batch_counts = np.zeros(self.n_dec_edges, dtype=np.float32)
        self.corr_batch_counts = np.zeros(self.n_line_edges, dtype=np.float32)

        # Reset tracers
        self.occ_tracer = self.base_p.copy()
        self.corr_tracer = self.initial_corr_tracer.copy()

        # Compute initial Pearson correlations of our adaptive decoder
        self.pearson_correlations = self.initial_pearson_corr.copy()

        # Generate a new drifted circuit and the corresponding decoding graph
        drifted_circuit, drifted_dem, drifted_matching = self.syndrome_data_generator.generate_drifted_circuit(
            base_circuit=self.base_circuit,
            seed=seed
        )     

        '''        
        # Generate a several test shots with the drifted circuit to evaluate the LER
        self.test_syndrome_batch, self.test_true_obs_batch = self.syndrome_data_generator.simulate_syndrome_data(
            drifted_circuit, seed=2002, n_shots=self.n_test_shots)
        
        # Test the initial LER of our decoder
        test_pred_edges_batch = self.current_matching.decode_batch(
            self.test_syndrome_batch, enable_correlations=False
        )
        test_pred_obs_batch = (test_pred_edges_batch @ self.fault_array) % 2
        self.test_ler = np.mean(test_pred_obs_batch != self.test_true_obs_batch)

        # Test the oracle decoder LER
        oracle_pred_obs_batch = drifted_matching.decode_batch(
            self.test_syndrome_batch, enable_correlations=False
        ).flatten()
        self.oracle_ler = np.mean(oracle_pred_obs_batch != self.test_true_obs_batch)
        '''

        # Pre-generate syndrome data and true observable for the entire episode (all shots under the same drift)
        self.syndrome_batch, self.true_obs_batch = self.syndrome_data_generator.simulate_syndrome_data(drifted_circuit, seed)
        #self.syndrome_batch, self.true_obs_batch = self.syndrome_data_generator.simulate_syndrome_data_from_dem(drifted_dem, seed
        
        # Retrieve the weights of the oracle decoding graph
        # Safely align Oracle weights to the Base Topology
        self.oracle_weights = np.full(self.n_dec_edges, self.max_weight, dtype=np.float32)
        oracle_probs = np.zeros(self.n_dec_edges, dtype=np.float32)
        for u, v, data in drifted_matching.edges():
            u = -1 if u is None else u
            v = -1 if v is None else v
            
            # Use your lookup matrix to find where this edge lives in OUR arrays
            idx = self.pair_to_idx_matrix[u, v]
            self.oracle_weights[idx] = data["weight"]
            oracle_probs[idx] = data["error_probability"]

        # Calculate the joint probabilities and Pearson correlations between the oracle edge weights
        _, oracle_joint_probs = self._build_line_graph_edges(drifted_dem, return_k_hop_adj_mat=False)
        self.oracle_correlations = self._compute_pearson_correlations(oracle_probs, oracle_joint_probs)

        # Calculate the initial weights mse error between adaptive (base initially) and oracle decoder
        self.weights_mse_error = np.mean((self.current_weights - self.oracle_weights)**2)

        # Compute the MSE error between the Pearson correlations of our decoder and the oracle decoder
        self.corr_mse_error = np.mean((self.pearson_correlations - self.oracle_correlations)**2)

        # Calculate the MSE error between the static decoder and the adaptive decoder (which starts as the 
        # base decoder with no drift knowledge)    
        self.weights_mse_error_static = 0
        self.corr_mse_error_static = 0

        if not self.train_mode:
            # Pre-generate the oracle predicted observable for each shot in the episode using the drifted matching
            self.oracle_predicted_obs_batch = drifted_matching.decode_batch(
                self.syndrome_batch, 
                enable_correlations=True, # remove, set as true
            ).flatten()

             # Pre-generate the static decoder solution edge predicted observable
            self.static_predicted_obs_batch = self.base_matching.decode_batch(
                self.syndrome_batch, 
                enable_correlations=False
                ).flatten()        

        # Pre-compute physics chunks and initialize trackers
        if self.use_syndrome_features:
            self._precompute_all_syndrome_statistics()
            self.spitz_tracer = self.base_p.copy()
            self.remm_tracer = self.initial_corr_tracer.copy()            

        # Prepare the first shot and build the initial observation
        obs = self._prepare_next_observation()

        info = {
            "n_decoding_edges": self.n_dec_edges,
            "n_line_edges": self.n_line_edges,
            "weights_mse_error": self.weights_mse_error,
            "corr_mse_error": self.corr_mse_error,
            #"initial_test_ler": self.test_ler,
            #"oracle_ler": self.oracle_ler,
        }
        return obs, info
    

    def _prepare_next_observation(self):
        """
        Prepares the observation for the current step by running the first-pass MWPM on the
        current syndrome and extracting the relevant features for the GNN.

        Returns:
            obs: A dictionary containing the node features, edge indices, edge attributes, and action mask
                for the current step, based on the first-pass MWPM solution and the current drifted syndrome.
        """

        # Pull the pre-generated syndrome for the current step
        syndrome = self.syndrome_batch[self.step_count]
        true_obs = self.true_obs_batch[self.step_count]

        # First pass MWPM still uses the current matching (no drift knowledge)
        selected_idx_1, first_pass_pred_obs = self.syndrome_data_generator.get_solution_edges(
            matching=self.current_matching, 
            syndrome_volume=syndrome, 
            enable_correlations=False,
            return_predicted_obs=True,
            fault_array=self.fault_array
        )
        
        selected_flag = np.zeros(self.n_dec_edges, dtype=np.float32)
        if selected_idx_1 is not None and len(selected_idx_1) > 0:
            selected_flag[selected_idx_1] = 1.0

        # Action mask around first-pass selected edges
        action_mask = self._compute_action_mask(selected_idx_1)

        # Cache
        self.current_syndrome = syndrome
        self.current_true_obs = true_obs
        self.current_first_pass_pred_obs = first_pass_pred_obs
        self.current_first_pass_selected_idx = selected_idx_1
        self.current_action_mask = action_mask

        # Evaluate the DGR (Decoder) Edge Feature
        if self.use_pearson_correlation and not self.use_log_joint_prob:
            dgr_edge_feat = self.pearson_correlations
        elif not self.use_log_joint_prob and not self.use_pearson_correlation:
            dgr_edge_feat = self.corr_tracer
        elif self.use_log_joint_prob and not self.use_pearson_correlation:
            dgr_edge_feat = -np.log(self.corr_tracer + 1e-10) # Add small epsilon to avoid log(0)
        else:
            raise ValueError("Invalid edge feature configuration. Please choose one of the following: \n"
                             "1) use_pearson_correlation=True and use_log_joint_prob=False \n"
                             "2) use_pearson_correlation=False and use_log_joint_prob=False \n"
                             "3) use_pearson_correlation=False and use_log_joint_prob=True \n"
                             )

        # Build Feature Arrays Dynamically based on the Flag
        if self.use_syndrome_features:
            self.node_feats[:, 0] = self.current_weights
            self.node_feats[:, 1] = selected_flag
            self.node_feats[:, 2] = self.spitz_tracer
            
            if self.n_line_edges > 0:
                self.edge_feats[:, 0] = dgr_edge_feat
                self.edge_feats[:, 1] = self.remm_tracer
        else:
            self.node_feats[:, 0] = self.current_weights
            self.node_feats[:, 1] = selected_flag
            if self.n_line_edges > 0:
                self.edge_feats[:, 0] = dgr_edge_feat

        obs = {
            "node_features": self.node_feats.copy(),
            "edge_attr": self.edge_feats.copy(),
            "action_mask": action_mask,
        }

        return obs


    def _build_edge_reweights(self, second_pass_weights: np.ndarray) -> np.ndarray:
        changed = np.flatnonzero(np.abs(second_pass_weights - self.current_weights) > 0.0)
        if changed.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Pull all changed edges at once
        edges = self.dec_edge_arr[changed]
        u = edges[:, 0]
        v = edges[:, 1]

        # Fast boolean masking for the boundary edge rule
        boundary_mask = (u == -1) & (v != -1)
        u[boundary_mask] = v[boundary_mask]
        v[boundary_mask] = -1

        # Stack them together instantly
        return np.column_stack((u, v, second_pass_weights[changed]))


    def _compute_action_mask(self, selected_idx: np.ndarray):
        """
        Instant vectorized mask using pre-computed K-hop matrix.
        Optimized to use single-pass memory extraction and pure boolean logic.

        Args:
            selected_idx: An array of shape [S,] containing the indices of the first-pass selected edges.

        Returns:
            action_mask: A boolean array of shape [N_edges,] where True indicates that the action can be 
                applied to that edge (within K hops of any selected edge and not isolated), and False 
                indicates that the action is masked out for that edge.
        """
        if not self.local_action_only:
            # Return pure boolean array, no float cast!
            return np.ones(self.n_dec_edges, dtype=np.bool_)

        if selected_idx.size == 0:
            return np.zeros(self.n_dec_edges, dtype=np.bool_)

        # Extract the S x N slice exactly ONCE.
        # This saves a massive redundant memory allocation.
        selected_rows = self.k_hop_adj_mat[selected_idx]
        
        # Base mask: Stay in pure boolean space
        mask = selected_rows.any(axis=0)
        
        # Extract the S x S subgraph from the CACHED rows
        selected_subgraph = selected_rows[:, selected_idx]
        
        # Sum the connections (fast integer sum in C)
        selected_neighbor_counts = selected_subgraph.sum(axis=1)
        
        # Identify the isolated selected edges
        isolated_selected_idx = selected_idx[selected_neighbor_counts == 1]
        
        # Mask them out
        if isolated_selected_idx.size > 0:
            mask[isolated_selected_idx] = False
            
        # Return the pure boolean mask.
        # The CPU will handle the float conversion during action multiplication.
        return mask
    

    def step(self, action: np.ndarray):
        """
        Executes one step of the environment's dynamics based on the agent's action. 
        The action is applied to modify the edge weights for a second-pass MWPM decoding, 
        which then produces a new predicted observable. The environment updates its 
        internal trackers based on the second-pass selected edges, computes the reward 
        based on the improvement over the first-pass prediction, and prepares the next 
        observation for the following step.

        Args:
            action: A numpy array of shape [N_edges] containing values in the range [-1, 1], 
                which represent the agent's proposed modifications to the edge weights. The actual 
                weight modifications are computed as action * action_scale, and may be masked 
                locally based on the first-pass selected edges.

        Returns:
            next_obs: The observation for the next step, based on the next shot's syndrome and the 
                updated decoding graph.
            reward: A scalar value representing the reward obtained from this action, based on the 
                improvement in logical prediction.
            terminated: A boolean indicating whether the episode has ended (e.g., all shots have 
                been processed).
            truncated: A boolean indicating whether the episode was truncated due to reaching max steps.
            info: A dictionary containing additional information about the step, such as the true 
                observable, predicted observables, selected edges, and error metrics.
        """
        
        assert self.current_syndrome is not None, "Call reset() before step()."

        ###############################################
        # 1) Apply action (masked locally if enabled) #
        ###############################################

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_dec_edges:
            raise ValueError(f"Action has shape {action.shape}; expected ({self.n_dec_edges},)")

        if self.local_action_only:
            applied_delta = action * self.current_action_mask * self.action_scale
        else:
            applied_delta = action * self.action_scale

        second_pass_edge_reweights = None
        
        if not np.any(applied_delta):
            selected_idx_2 = self.current_first_pass_selected_idx
            pred_obs = self.current_first_pass_pred_obs
        else:
            second_pass_weights = np.clip(self.current_weights + applied_delta, self.min_weight, self.max_weight)
            second_pass_matching = self.current_matching
            second_pass_edge_reweights = self._build_edge_reweights(second_pass_weights)
            

            ##############################################
            # 2) Run 2nd pass MWPM with reweighted edges #
            ##############################################
            
            selected_idx_2, pred_obs = self.syndrome_data_generator.get_solution_edges(
                matching=second_pass_matching,
                syndrome_volume=self.current_syndrome,
                enable_correlations=False,
                edge_reweights=second_pass_edge_reweights,
                return_predicted_obs=True,
                fault_array=self.fault_array,
            )
            
        
        ######################################################
        # 3) Update tracers using 2nd-pass selected edges    #
        ######################################################

        self._accumulate_occurrence(selected_idx_2)
        self._accumulate_correlation(selected_idx_2)
        self.shots_since_update += 1

        if self.shots_since_update >= self.update_period:
            self._apply_cma_and_update_graph()
            self.shots_since_update = 0

            # Compute the MSE error between the weights and correlations of our adaptive decoder
            # and the oracle decoder 
            self.corr_mse_error = np.mean((self.pearson_correlations - self.oracle_correlations)**2)
            self.corr_mse_error_static = np.mean((self.pearson_correlations - self.initial_pearson_corr)**2)
            self.weights_mse_error = np.mean((self.current_weights - self.oracle_weights)**2)
            self.weights_mse_error_static = np.mean((self.current_weights - self.initial_base_weights)**2)

            '''
            # 1. Get the physical edges from PyMatching's batch decoder
            # Guaranteed output shape: (10000, 502)
            test_pred_edges_batch = self.current_matching.decode_batch(
                self.test_syndrome_batch, enable_correlations=False
            )
            
            # 2. Convert physical edges to logical observable via dot product parity
            # Shape: (10000, 502) @ (502,) -> (10000,)
            test_pred_obs_batch = (test_pred_edges_batch @ self.fault_array) % 2
            
            # 3. Calculate LER
            self.test_ler = np.mean(test_pred_obs_batch != self.test_true_obs_batch)
           ''' 
           

                                                 
        #########################
        # 4) Compute the reward #
        #########################

        # 1. Evaluate Truth
        agent_correct = (pred_obs == self.current_true_obs)
        first_pass_correct = (self.current_first_pass_pred_obs == self.current_true_obs)

        # 2. Differential Logical Reward
        if agent_correct and not first_pass_correct:
            reward = +1.0 
        elif not agent_correct and first_pass_correct:
            reward = -1.0
        else:
            reward = 0.0   # Trivial success or completely uncorrectable. Agent didn't matter.

        ############################################################
        # 5) Prepare information about current step to be returned #
        ############################################################

        # Increment step count after processing the shot
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps # Episode ends when batch is empty

        info = {
            "logical_error": not(agent_correct),
            "true_obs": self.current_true_obs,
            "pred_obs": pred_obs,
            "first_pass_obs":self.current_first_pass_pred_obs,
            "oracle_pred_obs": self.oracle_predicted_obs_batch[self.step_count - 1] if not self.train_mode else None,
            "static_pred_obs":self.static_predicted_obs_batch[self.step_count - 1] if not self.train_mode else None,
            "reward": reward,
            "selected_edges_first_pass_idx": self.current_first_pass_selected_idx.copy() if self.current_first_pass_selected_idx is not None else None,
            "selected_edges_second_pass_idx": selected_idx_2.copy(),
            "action_mask": self.current_action_mask.copy() if self.current_action_mask is not None else None,
            "weights_mse_error": self.weights_mse_error,
            "corr_mse_error": self.corr_mse_error,
            "weights_mse_error_static": self.weights_mse_error_static,
            "corr_mse_error_static": self.corr_mse_error_static,
            #"test_ler": self.test_ler
        }

        ############################################################
        # 6) Prepare next observation (next shot under same drift) #
        ############################################################

        if not truncated:
            next_obs = self._prepare_next_observation()
        else:
            next_obs = None
                    
        return next_obs, reward, terminated, truncated, info   
    

    def _accumulate_occurrence(self, selected_idx: np.ndarray):
        """
        Add 1 to the occurrence batch count for selected edges.
        
        Args:
            selected_idx: Array of indices of the edges selected by the second-pass MWPM.
        """

        if selected_idx.size > 0:
            self.occ_batch_counts[selected_idx] += 1.0


    def _accumulate_correlation(self, selected_idx: np.ndarray):
        """
        Add 1 to the co-occurrence batch count for line graph edges.

        Args:
            selected_idx: Array of indices of the edges selected by the second-pass MWPM.
        """

        if self.n_line_edges == 0 or selected_idx.size == 0:
            return
        
        active = np.zeros(self.n_dec_edges, dtype=np.bool_)
        active[selected_idx] = True
        src = self.line_edge_index[0]
        dst = self.line_edge_index[1]
        self.corr_batch_counts += (active[src] & active[dst])

    
    def _compute_raw_syndrome_statistics(self, recent_syndromes: np.ndarray):
        """
        Maps Spitz and Remm analytical syndrome formulas to the GNN.
        Calculates exact physical probabilities for a specific time window.

        Args:
            recent_syndromes: A 2D boolean array of shape [n_shots, n_detectors] containing 
                the raw syndrome data for a recent window of shots.

        Returns:
            spitz_probs: An array of shape [n_dec_edges] containing the Spitz probabilities for
                each decoding graph edge, calculated from the recent syndrome data.
            remm_covariances: An array of shape [n_line_edges] containing the REMM covariance values
                for each line graph edge, calculated from the recent syndrome data.
        """

        n_shots = recent_syndromes.shape[0]
        spitz_probs = np.zeros(self.n_dec_edges, dtype=np.float32)
        v_mean = np.mean(recent_syndromes, axis=0) 
        
        E_matrix = np.zeros((self.n_dec_edges, n_shots), dtype=bool)
        boundary_edges = []

        #########################################################
        # 1. Calculate probabilities for bulk edges #
        #########################################################

        for idx, (u, v) in enumerate(self.dec_edge_list):
            # Defer boundary edges to the second pass
            if u == -1 or v == -1:
                boundary_edges.append((idx, u, v))
                real_node = v if u == -1 else u
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
            
            # Calculate the Spitz probability for bulk edges
            denom = 1.0 - 2.0 * mean_xor
            if abs(denom) < 1e-9:
                p = 0.499 
            else:
                cov = mean_uv - (mean_u * mean_v)
                root_term = max(0.0, 0.25 - (cov / denom))
                p = np.clip(0.5 - np.sqrt(root_term), 1e-6, 0.499)
                
            spitz_probs[idx] = p

        ##############################################################
        # 2. Calculate probabilities for boundary edges #
        ##############################################################

        for idx, u, v in boundary_edges:
            real_node = v if u == -1 else u
            node_mean = v_mean[real_node]
            
            # Calculate the product of (1 - 2*p_ij) for all other edges connected to real_node
            prod_term = 1.0
            for j_idx, (j_u, j_v) in enumerate(self.dec_edge_list):
                if j_idx == idx: 
                    continue # Skip itself
                
                # If the edge connects to real_node and is a bulk edge
                if (j_u == real_node or j_v == real_node) and (j_u != -1 and j_v != -1):
                    prod_term *= (1.0 - 2.0 * spitz_probs[j_idx])
            
            # Apply exact boundary formula 
            if abs(prod_term) < 1e-9:
                p = 0.499
            else:
                p = 0.5 + (node_mean - 0.5) / prod_term
                p = np.clip(p, 1e-6, 0.499)
                
            spitz_probs[idx] = p

        #################################
        # 3. Calculate Remm Covariances #
        #################################
        
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
            #chunk = self.syndrome_batch[start_idx:end_idx]
            chunk = self.syndrome_batch[start_idx:end_idx]  # Cumulative window from the start up to the end of this chunk
            
            # Calculate the physics statistics for this chunk
            spitz, remm = self._compute_raw_syndrome_statistics(chunk)
            
            # Store them in the cache arrays
            self.precomputed_spitz[i] = spitz
            self.precomputed_remm[i] = remm

    
    def _compute_pearson_correlations(self, occ_array: np.ndarray, corr_array: np.ndarray):
        """
        Computes Pearson correlations for a given set of marginal and joint probabilities
        using the environment's fixed line graph topology.

        Args:
            occ_array: An array of shape [n_dec_edges] containing the marginal probabilities (occurrence rates) for each decoding edge.
            corr_array: An array of shape [n_line_edges] containing the joint probabilities (co-occurrence rates) for each line graph edge.

        Returns:
            pearson_corrs: An array of shape [n_line_edges] containing the Pearson correlation values for each line graph edge, clipped to the range [0, 1].
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

        return np.clip(covariance / safe_denom, 0.0, 1.0)


    def _apply_cma_and_update_graph(self):
        """
        Applies the CMA update rule to the occurrence and correlation tracers 
        based on the accumulated batch counts, and then updates the MWPM graph 
        weights accordingly.
        """
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
        #self.current_matching = pymatching.Matching.from_check_matrix(self.H, weights=self.current_weights)
        self.current_matching = pymatching.Matching.from_check_matrix(
            self.H, 
            weights=self.current_weights
        )
        
        self.occ_batch_counts.fill(0.0)
        self.corr_batch_counts.fill(0.0)

        # 5. Update the new Pearson correlations from the updated tracers
        self.pearson_correlations = self._compute_pearson_correlations(self.occ_tracer, self.corr_tracer)


    def get_base_graph_info(self):
        """
        Returns a dictionary containing the base graph information, 
        including the number of decoding edges, number of line graph edges, 
        the list of decoding edges, the current weights of the decoding 
        edges, and the line graph edge indices.
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
        filename = ".\plots\live_graph.html"
        fig.write_html(filename)