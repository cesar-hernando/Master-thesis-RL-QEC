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

        return base_circuit, base_matching


    def generate_drifted_circuit(self, base_circuit: stim.Circuit):
        """
        Generate a Stim circuit with drifted noise parameters based on the specified mismatch.

        For simplicity, we simulate drift by scaling the noise probabilities in the circuit by 
        a certain mismatch factor. In a real implementation, drift varies over time in a more 
        complex manner.
        """

        drifted_circuit = stim.Circuit()
        for inst in base_circuit:
            if inst.name in ["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "E"]:
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
        drifted_matching = pymatching.Matching.from_detector_error_model(drifted_dem, enable_correlations=True) # Use correlated-matching for ground truth decoding

        return drifted_circuit, drifted_matching
    

    def simulate_syndrome_data(self, drifted_circuit: stim.Circuit):
        """
        Simulate syndrome data by sampling from the drifted circuit and storing the true logical error labels
        """

        # Sample syndromes from the drifted circuit
        sampler = drifted_circuit.compile_detector_sampler()
        syndrome_volume_batch, true_obs_batch = sampler.sample(shots=self.n_shots, separate_observables=True)

        return syndrome_volume_batch, true_obs_batch


    def get_oracle_solution_edges(self, drifted_matching: pymatching.Matching, syndrome_volume_batch: np.ndarray, return_predicted_obs: bool=False):
        """
        Get the oracle solution edges for each syndrome volume/shot by decoding with the drifted matching.
        """

        solution_edges_batch = []
        for shot_idx, syndrome_volume in enumerate(syndrome_volume_batch):
            solution_edges = drifted_matching.decode_to_edges_array(syndrome_volume, enable_correlations=True)
            solution_edges_batch.append(solution_edges)
        
        if return_predicted_obs:
            predicted_obs_batch = np.zeros(self.n_shots, dtype=bool)

            for shot_idx, edges in enumerate(solution_edges_batch):
                obs_flip = False
                
                for u, v in edges:
                    # Fetch edge data depending on whether it hits the boundary (-1)
                    if v == -1:
                        edge_data = drifted_matching.get_boundary_edge_data(u)
                    else:
                        edge_data = drifted_matching.get_edge_data(u, v)
                        
                    # If the edge has fault_ids, it crosses your single observable. Toggle parity!
                    if edge_data and edge_data.get('fault_ids'):
                        obs_flip ^= True 
                        
                predicted_obs_batch[shot_idx] = obs_flip
        else:
            predicted_obs_batch = None

        return solution_edges_batch, predicted_obs_batch
    

    def generate_data(self, return_predicted_obs: bool=False):
        """
        Main method to generate all data: base circuit, drifted circuit, syndrome volumes, true labels, and oracle solution edges.
        """

        base_circuit, base_matching = self.generate_base_circuit()
        drifted_circuit, drifted_matching = self.generate_drifted_circuit(base_circuit)
        syndrome_volume_batch, true_obs_batch = self.simulate_syndrome_data(drifted_circuit)
        solution_oracle_edges_batch, predicted_oracle_obs_batch = self.get_oracle_solution_edges(drifted_matching, syndrome_volume_batch, return_predicted_obs)

        return {
            "base_circuit": base_circuit,
            "base_matching": base_matching,
            "drifted_circuit": drifted_circuit,
            "drifted_matching": drifted_matching,
            "syndrome_volume_batch": syndrome_volume_batch,
            "true_obs_batch": true_obs_batch,
            "solution_oracle_edges_batch": solution_oracle_edges_batch,
            "predicted_oracle_obs_batch": predicted_oracle_obs_batch
        }
    
    @staticmethod
    def plot_mwpm_solution_3d(circuit, matching, syndrome, true_obs, pred_obs, solution_edges):
        """
        3D plot of the PyMatching decoding graph:
        - Boundary/aux nodes and their edges are excluded.
        - Detectors are colored by stabilizer type (X vs Z) inferred from (x,y) mod-4 pattern.
        - Fired detectors are highlighted with different colors for fired X vs fired Z.
        - MWPM selected edges are highlighted.
        """

        x_color = "#8b180f"        # X detectors (purple)
        z_color = "#063170"        # Z detectors (teal)
        fired_x_color = "#ff3131"  # fired X (orange)
        fired_z_color = "#107ffd"  # fired Z (red)
        edge_color = "#636262"     # non-solution edges
        sol_edge_color = "#8E9100" # MWPM selected edges

        # Graph export
        G = matching.to_networkx()

        # Fired detector set from syndrome 
        if hasattr(syndrome, "ndim") and syndrome.ndim == 2:
            fired = set(np.where(syndrome[0] == 1)[0].tolist())
        else:
            fired = set(np.where(syndrome == 1)[0].tolist())

        # Detector coordinates (real detectors only)
        det_xyz_raw = circuit.get_detector_coordinates()  # {det_id: (x,y,t,...)}

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

        # Keep only detector nodes that have coords (drop boundary/aux nodes)
        node_int = {}
        for n in G.nodes():
            try:
                node_int[n] = int(n)
            except Exception:
                node_int[n] = None

        det_nodes = [n for n in G.nodes() if node_int.get(n, None) in det_xyz]
        det_node_set = set(det_nodes)

        # Positions for plotting (only detector nodes)
        pos = {n: det_xyz[node_int[n]] for n in det_nodes}

        # Filtered edge list: both endpoints are detector nodes
        det_edges = []
        for u, v, data in G.edges(data=True):
            if u in det_node_set and v in det_node_set:
                det_edges.append((u, v, data))

        # MWPM solution edges: normalize to detector ids, then map back to graph nodes
        sol_set_int = {tuple(sorted((int(e[0]), int(e[1])))) for e in solution_edges}

        int_to_node = {}
        for n in det_nodes:
            int_to_node[node_int[n]] = n

        sol_set_nodes = set()
        for a, b in sol_set_int:
            if a in int_to_node and b in int_to_node:
                na, nb = int_to_node[a], int_to_node[b]
                sol_set_nodes.add(tuple(sorted((na, nb), key=lambda x: node_int[x])))

        # Infer detector type (X vs Z) using your lattice mod-4 pattern 
        # Your stabilizer placement rules:
        #   X: (i%4==0 and j%4==0) or (i%4==2 and j%4==2)
        #   Z: (i%4==0 and j%4==2) or (i%4==2 and j%4==0)
        # In DETECTOR coords we use [j, i, round] => x=j, y=i
        def infer_stab_type(n):
            x, y, _ = pos[n]
            j = int(round(x))
            i = int(round(y))
            if (i % 4 == 0 and j % 4 == 0) or (i % 4 == 2 and j % 4 == 2):
                return "X"
            if (i % 4 == 0 and j % 4 == 2) or (i % 4 == 2 and j % 4 == 0):
                return "Z"
            
        # Build edge traces 
        all_x, all_y, all_z = [], [], []
        sel_x, sel_y, sel_z = [], [], []
        mid_x, mid_y, mid_z, mid_txt = [], [], [], []

        for u, v, data in det_edges:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]

            w = data.get("weight", None)
            p = data.get("error_probability", data.get("p", None))

            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            mid_z.append((z0 + z1) / 2)
            mid_txt.append(f"{node_int[u]}—{node_int[v]}<br>weight={w}<br>p={p}")

            e_nodes = tuple(sorted((u, v), key=lambda x: node_int[x]))
            if e_nodes in sol_set_nodes:
                sel_x += [x0, x1, None]
                sel_y += [y0, y1, None]
                sel_z += [z0, z1, None]
            else:
                all_x += [x0, x1, None]
                all_y += [y0, y1, None]
                all_z += [z0, z1, None]

        # Build node traces (X, Z, fired X, fired Z)
        x_x, x_y, x_z, x_txt = [], [], [], []
        z_x, z_y, z_z, z_txt = [], [], [], []
        fx_x, fx_y, fx_z, fx_txt = [], [], [], []
        fz_x, fz_y, fz_z, fz_txt = [], [], [], []

        for n in det_nodes:
            x, y, z = pos[n]
            t = infer_stab_type(n)
            det_id = node_int[n]
            label = f"D{det_id} ({t})"

            if det_id in fired:
                if t == "X":
                    fx_x.append(x); fx_y.append(y); fx_z.append(z)
                    fx_txt.append(label + " • fired")
                else:
                    fz_x.append(x); fz_y.append(y); fz_z.append(z)
                    fz_txt.append(label + " • fired")
            else:
                if t == "X":
                    x_x.append(x); x_y.append(y); x_z.append(z); x_txt.append(label)
                else:
                    z_x.append(x); z_y.append(y); z_z.append(z); z_txt.append(label)

        # Plot 
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=all_x, y=all_y, z=all_z,
            mode="lines",
            line=dict(width=2, color=edge_color),
            name="graph edges",
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter3d(
            x=sel_x, y=sel_y, z=sel_z,
            mode="lines",
            line=dict(width=7, color=sol_edge_color),
            name="MWPM selected",
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter3d(
            x=mid_x, y=mid_y, z=mid_z,
            mode="markers",
            marker=dict(size=2, opacity=0.0),
            text=mid_txt,
            hoverinfo="text",
            name="edge info (hover)",
        ))

        fig.add_trace(go.Scatter3d(
            x=x_x, y=x_y, z=x_z,
            mode="markers",
            marker=dict(size=4, color=x_color),
            text=x_txt,
            hoverinfo="text",
            name="X detectors",
        ))

        fig.add_trace(go.Scatter3d(
            x=z_x, y=z_y, z=z_z,
            mode="markers",
            marker=dict(size=4, color=z_color),
            text=z_txt,
            hoverinfo="text",
            name="Z detectors",
        ))

        fig.add_trace(go.Scatter3d(
            x=fx_x, y=fx_y, z=fx_z,
            mode="markers",
            marker=dict(size=8, color=fired_x_color),
            text=fx_txt,
            hoverinfo="text",
            name="fired X",
        ))

        fig.add_trace(go.Scatter3d(
            x=fz_x, y=fz_y, z=fz_z,
            mode="markers",
            marker=dict(size=8, color=fired_z_color),
            text=fz_txt,
            hoverinfo="text",
            name="fired Z",
        ))

        # Title-safe extraction
        try:
            p_obs = pred_obs[0][0] if hasattr(pred_obs, "ndim") and pred_obs.ndim > 1 else pred_obs[0]
        except Exception:
            p_obs = pred_obs

        fig.update_layout(
            title=f"MWPM (True Obs={1 if bool(true_obs) else 0}, Pred Obs={p_obs})",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="t (round)",
                aspectmode="data",
            ),
            legend=dict(itemsizing="constant"),
        )

        fig.show()
        return G, fig


if __name__ == "__main__":
    
    distance = 5
    n_rounds = 5
    mismatch = 30.0
    p = 0.01
    noise_model = {
        "version": "built-in",
        "after_clifford_depolarization": p,
        "before_measure_flip_probability": p,
        "after_reset_flip_probability": p,
        "before_round_data_depolarization": p,
    }
    memory_type = 'z'
    n_shots = 1_000
    qec_code = 'surface_code'

    generator = SyndromeDataGenerator(distance, n_rounds, mismatch, noise_model, memory_type, n_shots, qec_code)
    data_dict = generator.generate_data(return_predicted_obs=True)

    print("\nBase matching graph edges:", data_dict["base_matching"].edges())
    print("\nDrifted matching graph edges:", data_dict["drifted_matching"].edges())
    print("\nSyndrome volume batch shape:", data_dict["syndrome_volume_batch"].shape)
    print("\nTrue observable labels batch shape:", data_dict["true_obs_batch"].shape)
    print("\nPredicted observable labels batch shape:", data_dict["predicted_oracle_obs_batch"].shape)

    print("\nOracle solution edges for first 5 shots:")
    for i in range(5):
        print(f"Shot {i}:\nOracle:\n {data_dict['solution_oracle_edges_batch'][i]}")
        solution_edges = data_dict["base_matching"].decode_to_edges_array(data_dict["syndrome_volume_batch"][i], enable_correlations=False)
        print(f"Current weight selected edges:\n {solution_edges}\n") 

    # Visualize a shot with the base matching graph and MWPM solution
    shot_idx = 0
    G, fig = generator.plot_mwpm_solution_3d(
        circuit=data_dict["drifted_circuit"],
        matching=data_dict["base_matching"],
        syndrome=data_dict["syndrome_volume_batch"][shot_idx],
        true_obs=data_dict["true_obs_batch"][shot_idx],
        pred_obs=data_dict["predicted_oracle_obs_batch"][shot_idx],
        solution_edges=data_dict["solution_oracle_edges_batch"][shot_idx]
    )
