'''
In this file, we create a class to simulate drifted-noise syndrome extraction circuits,
generate syndrome volume data and true logical error labels, and non-drifted decoding
graphs that serve as input to the decoder.
'''

import numpy as np
import stim
import pymatching

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
            "base_matching": base_matching,
            "drifted_matching": drifted_matching,
            "syndrome_volume_batch": syndrome_volume_batch,
            "true_obs_batch": true_obs_batch,
            "solution_oracle_edges_batch": solution_oracle_edges_batch,
            "predicted_oracle_obs_batch": predicted_oracle_obs_batch
        }


if __name__ == "__main__":
    # Example usage
    distance = 3
    n_rounds = 3
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
