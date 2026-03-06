'''
In this file, we create a class to simulate drifted-noise syndrome extraction circuits,
generate syndrome volume data and true logical error labels, and non-drifted decoding
graphs that serve as input to the decoder.
'''

import numpy as np
import stim
import pymatching
import plotly.graph_objects as go

from surface_code_stim import SurfaceCode


class SyndromeDataGenerator:
    """
    A class to simulate syndrome data for a given code distance, number of rounds, 
    drift mismatch, noise model, memory_type and quantum error correction code.
    """

    def __init__(self, distance, n_rounds, mismatch, noise_model, memory_type, n_shots, qec_code):
        self.distance = distance
        self.n_rounds = n_rounds
        self.mismatch = mismatch
        self.noise_model = noise_model
        self.memory_type = memory_type
        self.n_shots = n_shots
        self.qec_code = qec_code


    def generate_base_circuit(self):
        """
        Generate a base Stim circuit for the specified code distance, number of rounds, noise
        model and memory type. We will use the edge weights from the decoding graph of this base
        circuit.
        """

        if self.qec_code == 'surface_code':
            if self.noise_model["version"] == 'built-in':
                base_circuit = stim.Circuit.generated(
                        f"surface_code:rotated_memory_{self.memory_type.lower()}",
                        distance=self.distance,
                        rounds=self.n_rounds,
                        after_clifford_depolarization=self.noise_model["after_clifford_depolarization"],
                        before_measure_flip_probability=self.noise_model["before_measure_flip_probability"],
                        after_reset_flip_probability=self.noise_model["after_reset_flip_probability"],
                        before_round_data_depolarization=self.noise_model["before_round_data_depolarization"],
                    )
            elif self.noise_model["version"] == 'custom':
                sc = SurfaceCode(
                    hardware_params=self.noise_model["hardware_params"],
                    distance=self.distance,
                    n_rounds=self.n_rounds,
                    ad=self.noise_model["ad_dephasing"],
                    crosstalk=self.noise_model["crosstalk"],
                    default_2q_gate=self.noise_model["default_2q_gate"],
                    memory_type=self.memory_type,
                    multiplier=1.0,
                    idle_depol=self.noise_model["idle_depol"]
                )
                base_circuit = sc.build_surface_code_circuit()
        else:
            raise ValueError(f"Unsupported QEC code: {self.qec_code}. Currently only 'surface_code' is supported.")
        
        base_dem = base_circuit.detector_error_model(decompose_errors=True)
        base_matching = pymatching.Matching.from_detector_error_model(base_dem, enable_correlations=False)

        return base_circuit, base_dem, base_matching


    def generate_drifted_circuit(self, base_circuit: stim.Circuit, seed: int=42):
        """
        Generate a Stim circuit with drifted noise parameters based on the specified mismatch.

        For simplicity, we simulate drift by scaling the noise probabilities in the circuit by 
        a certain mismatch factor. In a real implementation, drift varies over time in a more 
        complex manner.
        """
        # Initialize a seeded random generator for reproducibility
        np.random.seed(seed)

        error_list = ["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "E",
                      "M", "MX", "MY", "MZ", "R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]

        drifted_circuit = stim.Circuit()
        for inst in base_circuit:
            if inst.name in error_list:
                base_args = inst.gate_args_copy()
                
                # Step size is 1 for 1-qubit gates, 2 for 2-qubit gates
                step = 2 if inst.name in ["DEPOLARIZE2", "E"] else 1
                targets = inst.targets_copy()
                
                for i in range(0, len(targets), step):
                    # Calculate unique drift for this specific qubit (or pair)
                    f = np.exp(np.random.uniform(-np.log(self.mismatch), np.log(self.mismatch)))
                    
                    # Apply drift to the probabilities (handling PAULI_CHANNEL_1's 3 args)
                    new_args = [min(p * f, 1.0) for p in base_args] 
                    
                    # Append single drifted target
                    drifted_circuit.append(inst.name, targets[i:i+step], new_args)
            else:
                drifted_circuit.append(inst)

        drifted_dem = drifted_circuit.detector_error_model(decompose_errors=True)
        drifted_matching = pymatching.Matching.from_detector_error_model(drifted_dem, enable_correlations=False) # Use correlated-matching for ground truth decoding

        return drifted_circuit, drifted_matching
    

    def simulate_syndrome_data(self, drifted_circuit: stim.Circuit):
        """
        Simulate syndrome data by sampling from the drifted circuit and storing the true logical error labels
        """

        # Sample syndromes from the drifted circuit
        sampler = drifted_circuit.compile_detector_sampler()
        syndrome_volume_batch, true_obs_batch = sampler.sample(shots=self.n_shots, separate_observables=True)
        true_obs_batch = true_obs_batch.flatten()

        return np.asarray(syndrome_volume_batch, dtype=np.uint8), true_obs_batch

    @staticmethod
    def _predict_obs_from_selected_edges(
        selected_edges: np.ndarray,
        pair_to_idx_matrix: np.ndarray,
        fault_array: np.ndarray
    ) -> bool:
        """Fully vectorized observable prediction using native -1 boundary indexing."""
        if selected_edges.size == 0:
            return False

        u = selected_edges[:, 0].astype(np.int32)
        v = selected_edges[:, 1].astype(np.int32)

        # Fast matrix lookup (NumPy dynamically maps -1 to the boundary row/col)
        idxs = pair_to_idx_matrix[u, v]
        valid_idxs = idxs[idxs != -1]

        # Check parity (If the sum of crossed boundaries is odd, it flipped)
        if valid_idxs.size > 0:
            fault_count = np.sum(fault_array[valid_idxs])
            return bool(fault_count % 2 != 0)
            
        return False


    def get_solution_edges(
            self, 
            matching: pymatching.Matching, 
            syndrome_volume: np.ndarray, 
            enable_correlations: bool=False,
            return_predicted_obs: bool=False,
            pair_to_idx_matrix=None,
            fault_array=None
        ):
        """
        Get the oracle solution edges for a given syndrome volume/shot by decoding with the provided matching.
        """

        solution_edges = matching.decode_to_edges_array(syndrome_volume, enable_correlations=enable_correlations)
        
        if return_predicted_obs:
            if pair_to_idx_matrix is None or fault_array is None:
                raise ValueError("pair_to_idx_matrix and fault_array must be provided if return_predicted_obs = True")
            
            predicted_obs = self._predict_obs_from_selected_edges( 
                selected_edges=solution_edges, 
                pair_to_idx_matrix=pair_to_idx_matrix, 
                fault_array=fault_array
            )

            return solution_edges, predicted_obs
        
        return solution_edges


    def get_solution_edges_batch(
            self, 
            matching: pymatching.Matching, 
            syndrome_volume_batch: np.ndarray,
            enable_correlations: bool=True, 
            return_predicted_obs: bool=False,
            pair_to_idx_matrix=None,
            fault_array=None
        ):
        """
        Get the oracle solution edges for each syndrome volume/shot by decoding with the drifted matching.
        """
        n_shots = len(syndrome_volume_batch)
        solution_edges_batch = []
        
        if return_predicted_obs:
            if pair_to_idx_matrix is None or fault_array is None:
                raise ValueError("pair_to_idx_matrix and fault_array must be provided if return_predicted_obs = True")
            
            predicted_obs_batch = np.zeros(n_shots, dtype=bool)
            
            # Decode AND Predict the Obs in one pass
            for shot_idx, syndrome_volume in enumerate(syndrome_volume_batch):
                edges = matching.decode_to_edges_array(syndrome_volume, enable_correlations=enable_correlations)
                solution_edges_batch.append(edges)
                predicted_obs_batch[shot_idx] = self._predict_obs_from_selected_edges( 
                    selected_edges=edges,
                    pair_to_idx_matrix=pair_to_idx_matrix, 
                    fault_array=fault_array
                )
            return solution_edges_batch, predicted_obs_batch     
        else:
            # Just Decode
            for shot_idx, syndrome_volume in enumerate(syndrome_volume_batch):
                edges = matching.decode_to_edges_array(syndrome_volume, enable_correlations=enable_correlations)
                solution_edges_batch.append(edges)
            return solution_edges_batch
        

    @staticmethod
    def plot_mwpm_solution_3d(circuit, matching, syndrome, true_obs, pred_obs, solution_edges):
        """
        3D plot of the PyMatching decoding graph with updated aesthetics.
        - Boundary/aux nodes and their edges are excluded.
        - Detectors are colored by stabilizer type (X vs Z).
        - MWPM selected edges are highlighted in Gold.
        - Hover text is applied directly to the line segments.
        """
        # Matched Aesthetics to the GNN Line Graph
        x_color = "#b22222"        # Deep Red (X)
        z_color = "#005f87"        # Deep Blue (Z)
        fired_x_color = "#ff4d4d"  # Bright Red (Fired X)
        fired_z_color = "#33aaff"  # Bright Blue (Fired Z)
        edge_color = "rgba(150, 150, 150, 0.4)" # Faint Grey for base graph
        sol_edge_color = "#d4af37" # Thick Gold for MWPM Solution

        # Graph export
        G = matching.to_networkx()

        # Fired detector set from syndrome 
        if hasattr(syndrome, "ndim") and syndrome.ndim == 2:
            fired = set(np.where(syndrome[0] == 1)[0].tolist())
        else:
            fired = set(np.where(syndrome == 1)[0].tolist())

        # Detector coordinates (real detectors only)
        det_xyz_raw = circuit.get_detector_coordinates()

        det_xyz = {}
        for k, c in det_xyz_raw.items():
            try:
                ki = int(k)
            except Exception:
                continue
            c = list(c)
            while len(c) < 3:
                c.append(0.0)
            det_xyz[ki] = (float(c[0]), float(c[1]), float(c[2]))

        # Keep only detector nodes that have coords
        node_int = {}
        for n in G.nodes():
            try:
                node_int[n] = int(n)
            except Exception:
                node_int[n] = None

        det_nodes = [n for n in G.nodes() if node_int.get(n, None) in det_xyz]
        det_node_set = set(det_nodes)
        pos = {n: det_xyz[node_int[n]] for n in det_nodes}

        # Filtered edge list
        det_edges = []
        for u, v, data in G.edges(data=True):
            if u in det_node_set and v in det_node_set:
                det_edges.append((u, v, data))

        # MWPM solution edges
        sol_set_int = {tuple(sorted((int(e[0]), int(e[1])))) for e in solution_edges}
        int_to_node = {node_int[n]: n for n in det_nodes}

        sol_set_nodes = set()
        for a, b in sol_set_int:
            if a in int_to_node and b in int_to_node:
                na, nb = int_to_node[a], int_to_node[b]
                sol_set_nodes.add(tuple(sorted((na, nb), key=lambda x: node_int[x])))

        # Infer detector type with the CORRECTED logic for Z-memory
        def infer_stab_type(n):
            x, y, _ = pos[n]
            j, i = int(round(x)), int(round(y))
            if (i % 4 == 0 and j % 4 == 0) or (i % 4 == 2 and j % 4 == 2):
                return "Z"
            if (i % 4 == 0 and j % 4 == 2) or (i % 4 == 2 and j % 4 == 0):
                return "X"
            return "Unknown"
            
        # 1. Build edge traces directly on the lines
        all_x, all_y, all_z, all_txt = [], [], [], []
        sel_x, sel_y, sel_z, sel_txt = [], [], [], []

        for u, v, data in det_edges:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]

            w = data.get("weight", "N/A")
            p = data.get("error_probability", data.get("p", "N/A"))
            
            # Formatting numbers
            w_str = f"{w:.3f}" if isinstance(w, float) else str(w)
            p_str = f"{p:.4e}" if isinstance(p, float) else str(p)

            txt = (
                f"<b>Nodes:</b> D{node_int[u]} &mdash; D{node_int[v]}<br>"
                f"<b>Weight:</b> {w_str}<br>"
                f"<b>Probability:</b> {p_str}"
            )

            e_nodes = tuple(sorted((u, v), key=lambda x: node_int[x]))
            if e_nodes in sol_set_nodes:
                sel_x.extend([x0, x1, None])
                sel_y.extend([y0, y1, None])
                sel_z.extend([z0, z1, None])
                sel_txt.extend([txt, txt, ""])
            else:
                all_x.extend([x0, x1, None])
                all_y.extend([y0, y1, None])
                all_z.extend([z0, z1, None])
                all_txt.extend([txt, txt, ""])

        # 2. Build node traces
        x_x, x_y, x_z, x_txt = [], [], [], []
        z_x, z_y, z_z, z_txt = [], [], [], []
        fx_x, fx_y, fx_z, fx_txt = [], [], [], []
        fz_x, fz_y, fz_z, fz_txt = [], [], [], []

        for n in det_nodes:
            x, y, z = pos[n]
            t = infer_stab_type(n)
            det_id = node_int[n]
            label = f"<b>D{det_id}</b> ({t})"

            if det_id in fired:
                if t == "X":
                    fx_x.append(x); fx_y.append(y); fx_z.append(z); fx_txt.append(label + " &mdash; <b>Fired</b>")
                else:
                    fz_x.append(x); fz_y.append(y); fz_z.append(z); fz_txt.append(label + " &mdash; <b>Fired</b>")
            else:
                if t == "X":
                    x_x.append(x); x_y.append(y); x_z.append(z); x_txt.append(label)
                else:
                    z_x.append(x); z_y.append(y); z_z.append(z); z_txt.append(label)

        # 3. Plot Configuration
        fig = go.Figure()

        # Base Graph Edges
        fig.add_trace(go.Scatter3d(
            x=all_x, y=all_y, z=all_z,
            mode="lines",
            line=dict(width=2, color=edge_color),
            text=all_txt,
            hoverinfo="text",
            name="Base Graph Edges"
        ))

        # MWPM Selected Edges
        fig.add_trace(go.Scatter3d(
            x=sel_x, y=sel_y, z=sel_z,
            mode="lines",
            line=dict(width=6, color=sol_edge_color),
            text=sel_txt,
            hoverinfo="text",
            name="MWPM Selected Edges"
        ))

        # Base Nodes (White borders applied to all)
        marker_style = dict(size=5, line=dict(width=1, color='white'))
        
        fig.add_trace(go.Scatter3d(
            x=x_x, y=x_y, z=x_z, mode="markers", text=x_txt, hoverinfo="text", name="X Detectors",
            marker=dict(**marker_style, color=x_color)
        ))
        fig.add_trace(go.Scatter3d(
            x=z_x, y=z_y, z=z_z, mode="markers", text=z_txt, hoverinfo="text", name="Z Detectors",
            marker=dict(**marker_style, color=z_color)
        ))

        # Fired Nodes (Slightly larger)
        fired_marker_style = dict(size=8, line=dict(width=1.5, color='white'))
        
        fig.add_trace(go.Scatter3d(
            x=fx_x, y=fx_y, z=fx_z, mode="markers", text=fx_txt, hoverinfo="text", name="Fired X",
            marker=dict(**fired_marker_style, color=fired_x_color)
        ))
        fig.add_trace(go.Scatter3d(
            x=fz_x, y=fz_y, z=fz_z, mode="markers", text=fz_txt, hoverinfo="text", name="Fired Z",
            marker=dict(**fired_marker_style, color=fired_z_color)
        ))

        # Title-safe extraction
        try:
            p_obs = pred_obs[0][0] if hasattr(pred_obs, "ndim") and pred_obs.ndim > 1 else pred_obs[0]
        except Exception:
            p_obs = pred_obs
            
        true_obs_val = 1 if bool(true_obs) else 0
        p_obs_val = 1 if bool(p_obs) else 0
        log_err = "Yes" if true_obs_val != p_obs_val else "No"

        # 4. Apply unified formatting
        fig.update_layout(
            title=f"<b>MWPM Physical Decoding Solution</b><br><sup>True Obs: {true_obs_val} | Pred Obs: {p_obs_val} | Logical Error: {log_err}</sup>",
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

        fig.show()
        return G, fig

