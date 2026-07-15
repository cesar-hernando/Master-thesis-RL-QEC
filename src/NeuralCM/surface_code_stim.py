import stim
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path

class SurfaceCode:
    """
        This class simulates using stim a surface code concerning Z memory.
    """

    def __init__(self, hardware_params: dict, distance: int, n_rounds: int, ad: bool, 
                 crosstalk: bool, default_2q_gate: str, 
                 memory_type: str, multiplier: float = 1.0, idle_depol: bool = False):
        
        self.hardware_params = hardware_params.copy()

        self.distance = distance
        self.n_rounds = n_rounds if n_rounds is not None else distance - 1
        self.multiplier = multiplier 

        self.ad = ad
        self.crosstalk = crosstalk
        self.idle_depol = idle_depol

        if default_2q_gate not in ['CX', 'CZ']:
            raise Exception("Default 2q gate not supported")
        if memory_type not in ['x', 'z']:
            raise Exception("Memory type not supported")

        self.default_2q_gate = default_2q_gate
        self.memory_type = memory_type

        self.L = 2 * self.distance + 1  # length of the lattice

        # Calculate decoherence error probabilities associated with amplitude damping and dephasing noise channels.
        self._calculate_decoherence_error_probabilities()
        
        # Define scaled hardware parameters for when multiplier is used
        self.hardware_params['gate_error_1q'] *= self.multiplier
        self.hardware_params['gate_error_2q'] *= self.multiplier
        self.hardware_params['measurement_error'] *= self.multiplier
        self.hardware_params['state_prep_error'] *= self.multiplier
        
        
    def _calculate_decoherence_error_probabilities(self):
        """
        Calculate the probabilities of X and Z errors associated with amplitude damping and dephasing noise channels
        based on the hardware parameters. The probabilities are calculated for both single-qubit and two-qubit gates
        as they have different durations.
        """

        duration_1q_gate = self.hardware_params['duration_1q_gate']
        duration_2q_gate = self.hardware_params['duration_2q_gate']
        T1 = self.hardware_params['T1']
        T2 = self.hardware_params['T2']
        duration_readout = self.hardware_params['duration_readout']

        # Using the Pauli twirl approximation, infer the probabilities of X and Z errors for amplitude damping
        # and dephasing noise channels.

        def p_X(t):
            return 0.25 * (1 - np.exp(-t / T1)) * self.multiplier
        
        def p_Z(t):
            return (0.5 * (1 - np.exp(-t / T2)) - 0.25 * (1 - np.exp(-t / T1))) * self.multiplier
        
        # These probabilities depend on the gate duration and are therefore different for single or two qubit gates.
        self.p_X_1q = p_X(duration_1q_gate)
        self.p_Z_1q = p_Z(duration_1q_gate)
        self.p_X_2q = p_X(duration_2q_gate)
        self.p_Z_2q = p_Z(duration_2q_gate)

        # We now calculate the probabilities associated with the decoherence of data qubits during stabilizer measurement
        self.p_X_readout = p_X(duration_readout)
        self.p_Z_readout = p_Z(duration_readout)


    def _assign_qubit_coordinates(self):
        """
        Assign coordinates to data qubits and stabilizers in the surface code lattice.
        The coordinates are stored in the stim Circuit object for visualization purposes.
        """

        L = self.L

        # Data qubits, x stabilizers, z stabilizers
        self.data_qubits = [i * L + j for i in range(L) for j in range(L) if (i + j) % 2 == 0 and i % 2 != 0]
        self.x_stabilizers = [i * L + j for i in range(L) for j in range(L) if
                              0 < j < L - 1 and ((i % 4 == 0 and j % 4 == 0) or (i % 4 == 2 and j % 4 == 2))]
        self.z_stabilizers = [i * L + j for i in range(L) for j in range(L) if
                              0 < i < L - 1 and ((i % 4 == 0 and j % 4 == 2) or (i % 4 == 2 and j % 4 == 0))]
        self.qubits = self.data_qubits + self.x_stabilizers + self.z_stabilizers

        for i in range(L):
            for j in range(L):
                if (i + j) % 2 == 0 and i % 2 != 0:  # data qubits
                    self.qc.append('QUBIT_COORDS', [i * L + j], [j, i])  # put [j,i] so that it corresponds to the image
                if 0 < j < L - 1: # coordinates for x stabilizers
                    if (i % 4 == 0 and j % 4 == 0) or (i % 4 == 2 and j % 4 == 2):
                        self.qc.append('QUBIT_COORDS', [i * L + j], [j, i])
                if 0 < i < L - 1:
                    if (i % 4 == 0 and j % 4 == 2) or (i % 4 == 2 and j % 4 == 0):
                        self.qc.append('QUBIT_COORDS', [i * L + j], [j, i])


    def _initialization_step_z_memory(self):
        
        self.qc.append('R', self.qubits) # initialization
        self.qc.append('TICK')
        self.qc.append('X_ERROR', self.qubits, self.hardware_params['state_prep_error']) # initialization error
        self.qc.append('TICK')


    def _initialization_step_x_memory(self):

        self.qc.append('R', self.x_stabilizers + self.z_stabilizers) # initialization
        self.qc.append('RX', self.data_qubits)
        self.qc.append('TICK')
        self.qc.append('X_ERROR', self.x_stabilizers + self.z_stabilizers, self.hardware_params['state_prep_error']) # initialization error
        self.qc.append('Z_ERROR', self.data_qubits, self.hardware_params['state_prep_error'])
        self.qc.append('TICK')

        # # Apply Hadamard gates to all data qubits.
        # self.qc.append('H', self.data_qubits)
        # self.qc.append('TICK')

        # # Add the noise caused by the Hadamard gates:
        # # Apply depolarizing noise to data qubits.
        # self.qc.append('DEPOLARIZE1', self.data_qubits, self.hardware_params['gate_error_1q'])
        # # Add AD noise to all qubits while one qubit gates are being applied.
        # if self.ad:
        #     self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q])
        # self.qc.append('TICK')


    def _initialization_step(self):
        """
        Initialize the surface code, adding the corresponding state preparation error.
        """

        self.qc = stim.Circuit()
        self._assign_qubit_coordinates()

        if self.memory_type == 'z':
            self._initialization_step_z_memory()
        elif self.memory_type == 'x':
            self._initialization_step_x_memory()
        else:
            raise Exception("Invalid experiment memory type!") 


    def _start_round_cx(self):

        # Apply Hadamard gates to all data qubits.
        self.qc.append('H', self.x_stabilizers)
        self.qc.append('TICK')

        # Apply depolarizing noise to x stabilizers, corresponding to Hadamard errors.
        self.qc.append('DEPOLARIZE1', self.x_stabilizers, self.hardware_params['gate_error_1q'])
        self.qc.append('TICK')

        # Apply AD noise to all qubits while one qubit gates are being applied.
        if self.ad and not(self.idle_depol):
            self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q])
        # Apply depolarizing error to idle qubits
        elif self.idle_depol and not(self.ad):
            self.qc.append('DEPOLARIZE1', self.data_qubits + self.z_stabilizers, self.hardware_params['gate_error_1q'])
        elif self.ad and self.idle_depol:
            raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
        self.qc.append('TICK')


    def _start_round_cz(self):

        # Apply Hadamard gates to all data qubits and X stabilizers.
        self.qc.append('H', self.data_qubits + self.x_stabilizers)
        self.qc.append('TICK')

        # Apply depolarizing noise to data qubits and x stabilizers, caused by Hadamard gates.
        self.qc.append('DEPOLARIZE1', self.data_qubits + self.x_stabilizers, self.hardware_params['gate_error_1q']) 
        self.qc.append('TICK')

        # Apply AD noise to all qubits while one qubit gates are being applied.
        if self.ad and not(self.idle_depol):
            self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q])
        # Apply depolarizing error to idle qubits
        elif self.idle_depol and not(self.ad):
            self.qc.append('DEPOLARIZE1', self.z_stabilizers, self.hardware_params['gate_error_1q'])
        elif self.ad and self.idle_depol:
            raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
        self.qc.append('TICK')


    def _start_round(self):
        if self.default_2q_gate == 'CX':
            return self._start_round_cx()
        elif self.default_2q_gate == 'CZ':
            return self._start_round_cz()
        else:
            raise Exception("Default 2q gate not supported")


    def _end_round_cx(self):

        # Apply Hadamard gates to x stabilizers qubits
        self.qc.append('H', self.x_stabilizers)
        self.qc.append('TICK')

        # Add the noise caused by the Hadamard gates.
        self.qc.append('DEPOLARIZE1', self.x_stabilizers, self.hardware_params['gate_error_1q'])  # gate noise
        self.qc.append('TICK')

        # Apply AD noise to all qubits while one qubit gates are being applied.
        if self.ad and not(self.idle_depol):
            self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q])
        # Apply depolarizing error to idle qubits
        elif self.idle_depol and not(self.ad):
            self.qc.append('DEPOLARIZE1', self.z_stabilizers + self.data_qubits, self.hardware_params['gate_error_1q']) 
        elif self.idle_depol and self.ad:
            raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
        self.qc.append('TICK')


    def _end_round_cz(self):

        # Apply Hadamard gates to all Z stabilizers (return them to the pre-Z-stabilizers state).
        self.qc.append('H', self.z_stabilizers)
        self.qc.append('TICK')

        # Add the noise caused by the Hadamard gates.
        self.qc.append('DEPOLARIZE1', self.z_stabilizers, self.hardware_params['gate_error_1q']) # 1Q gate
        self.qc.append('TICK')

        # Apply AD noise to all qubits while one qubit gates are being applied.
        if self.ad and not(self.idle_depol):
            self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q]) 
        # Apply depolarizing error to idle qubits
        elif self.idle_depol and not(self.ad):
            self.qc.append('DEPOLARIZE1', self.data_qubits + self.x_stabilizers, self.hardware_params['gate_error_1q']) 
        elif self.idle_depol and self.ad:
            raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
        self.qc.append('TICK')


    def _end_round(self):
        if self.default_2q_gate == 'CX':
            self._end_round_cx()
        elif self.default_2q_gate == 'CZ':
            self._end_round_cz()
        else:
            raise Exception("Default 2q gate not supported")  


    def _add_detectors_x_memory(self, round: int):
        L = self.L
        # Goes through each stabilizer and correlates the measurement outcome of future rounds to previous rounds.
        num_Z_stabs = len(self.z_stabilizers)
        num_X_stabs = len(self.x_stabilizers)


        # Z stabilizers
        if round > 0:
            for k, z_stab_idx in enumerate(self.z_stabilizers):
                i = z_stab_idx // L
                j = z_stab_idx % L
                if round == 0: # add detector for the first round
                    self.qc.append('DETECTOR', [stim.target_rec(-(num_Z_stabs - k))], [j, i, round])
                else:
                    self.qc.append('DETECTOR', [stim.target_rec(-(((num_Z_stabs + num_X_stabs) + num_Z_stabs - k))),
                                                stim.target_rec(-(num_Z_stabs - k))], [j, i, round])
                
        # X stabilizers        
        for k, x_stab_idx in enumerate(self.x_stabilizers):
            i = x_stab_idx // L
            j = x_stab_idx % L
            if round == 0:  # add detector for the first round
                self.qc.append('DETECTOR', [stim.target_rec(-(num_X_stabs + num_Z_stabs - k))], [j, i, round])
            else:
                self.qc.append('DETECTOR', [stim.target_rec(-((2*(num_X_stabs + num_Z_stabs) - k))),
                                            stim.target_rec(-(num_X_stabs + num_Z_stabs - k))], [j, i, round])
                
        self.qc.append('TICK')

    def _add_detectors_z_memory(self, round: int):
        L = self.L

        # Goes through each stabilizer and correlates the measurement outcome of future rounds to previous rounds.
        num_Z_stabs = len(self.z_stabilizers)   
        num_X_stabs = len(self.x_stabilizers)

        # Z stabilizers
        for k, z_stab_idx in enumerate(self.z_stabilizers):
            i = z_stab_idx // L
            j = z_stab_idx % L
            if round == 0: # add detector for the first round
                self.qc.append('DETECTOR', [stim.target_rec(-(num_Z_stabs - k))], [j, i, round])
            else:
                self.qc.append('DETECTOR', [stim.target_rec(-(((num_Z_stabs + num_X_stabs) + num_Z_stabs - k))),
                                            stim.target_rec(-(num_Z_stabs - k))], [j, i, round])
                
        # X stabilizers
        if round > 0:
            for k, x_stab_idx in enumerate(self.x_stabilizers):
                i = x_stab_idx // L
                j = x_stab_idx % L
                self.qc.append('DETECTOR', [stim.target_rec(-((2*(num_Z_stabs + num_X_stabs) - k))),
                                            stim.target_rec(-((num_Z_stabs + num_X_stabs - k)))], [j, i, round])

        self.qc.append('TICK')


    def _add_detectors(self, round: int):
        if self.memory_type == 'x':
            self._add_detectors_x_memory(round)
        elif self.memory_type == 'z':
            self._add_detectors_z_memory(round)
        else:
            raise Exception("Memory type not supported")  



    def _rounds_step(self):
        """
        Perform n_rounds rounds of stabilizer measurements, including gate operations and associated errors.
        """
        residual_zz = self.multiplier*(1 - (1 - 0.0025)**(1/3))
        zz_errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, residual_zz]

        # First, apply the X and Z stabilizers in an interleaved way.
        x_directions = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        z_directions = [(-1, 1), (1, 1), (-1, -1), (1, -1)]

        self.z_stab_support = defaultdict(list)
        self.x_stab_support = defaultdict(list)

        L = self.L

        qubits_set = set(self.qubits)
        active_qubits_per_step = [set() for _ in range(4)]

        for round in range(self.n_rounds):
            self._start_round()
            step = 0

            for (x_di, x_dj), (z_di, z_dj) in zip(x_directions, z_directions):
                # Apply the CX gates for X stabilizers between every ancilla qubit and the data qubit corresponding to the direction (di, dj).
                for q_x in self.x_stabilizers:
                    i = q_x // L
                    j = q_x % L
                    ti = i + x_di
                    tj = j + x_dj
                    if 0 <= ti < L and 0 <= tj < L:
                        target = ti * L + tj
                        # one direction done, add tick, then proceed to the next direction
                        self.qc.append(self.default_2q_gate, [q_x, target])
                        if round == 0:
                            self.x_stab_support[q_x].append(target)
                            active_qubits_per_step[step].update([q_x, target])

                # Apply the CX gates for Z stabilizers between every ancilla qubit and the data qubit corresponding to the direction (di, dj).
                for q_z in self.z_stabilizers:
                    i = q_z // L
                    j = q_z % L
                    ci = i + z_di
                    cj = j + z_dj
                    if 0 <= ci < L and 0 <= cj < L:
                        control = ci * L + cj
                        self.qc.append(self.default_2q_gate, [control, q_z])
                        if round == 0:
                            self.z_stab_support[q_z].append(control)
                            active_qubits_per_step[step].update([q_z, control])
                self.qc.append('TICK')

                # Add crosstalk noise caused by the CX gates for X stabilizers.
                for q_x in self.x_stabilizers:
                    i = q_x // L
                    j = q_x % L
                    ti = i + x_di
                    tj = j + x_dj
                    if 0 <= ti < L and 0 <= tj < L:
                        target = ti * L + tj
                        if self.crosstalk:
                            self.qc.append('E', [f'Z{q_x}', f'Z{target}'], zz_errors[-1])
                        self.qc.append('DEPOLARIZE2', [q_x, target], self.hardware_params['gate_error_2q'])  # 2Q gate error

                # Add crosstalk noise caused by the CX gates for Z stabilizers.
                for q_z in self.z_stabilizers:
                    i = q_z // L
                    j = q_z % L
                    ci = i + z_di
                    cj = j + z_dj
                    if 0 <= ci < L and 0 <= cj < L:
                        control = ci * L + cj
                        if self.crosstalk:
                            self.qc.append('E', [f'Z{control}', f'Z{q_z}'], zz_errors[-1])
                        self.qc.append('DEPOLARIZE2', [control, q_z], self.hardware_params['gate_error_2q'])
                self.qc.append('TICK')

                # Add AD noise corresponding to the duration of the CX gates.
                if self.ad and not(self.idle_depol):
                    self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_2q, self.p_X_2q, self.p_Z_2q])
                elif self.idle_depol and not(self.ad):
                    idle_qubits = qubits_set - active_qubits_per_step[step]
                    self.qc.append('DEPOLARIZE1', idle_qubits, self.hardware_params['gate_error_1q']) # check which qubits do not participate
                elif self.idle_depol and self.ad:
                    raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
                self.qc.append('TICK')

                step += 1

            self._end_round()

            # Apply measurement errors to stabilizers
            self.qc.append('X_ERROR', self.x_stabilizers + self.z_stabilizers, self.hardware_params['measurement_error'])
            self.qc.append('TICK')
            
            # Apply AD+dephasing noise to data qubits if it's not the last round
            if round < self.n_rounds - 1:
                if self.ad and not(self.idle_depol):
                    self.qc.append('PAULI_CHANNEL_1', self.data_qubits, [self.p_X_readout, self.p_X_readout, self.p_Z_readout])
                # Apply depolarizing error on idle qubits (data qubits)
                elif self.idle_depol and not(self.ad):  
                    self.qc.append('DEPOLARIZE1', self.data_qubits, self.hardware_params['gate_error_1q'])
                elif self.ad and self.idle_depol:
                    raise Exception("Selecting both AD + dephasing (Pauli twirl) and depolarizing error to idle qubits is incompatible")
                self.qc.append('TICK')
                self.qc.append('MR', self.x_stabilizers + self.z_stabilizers)
            else:
                self.qc.append('M', self.x_stabilizers + self.z_stabilizers)

            self._add_detectors(round)

            # Reset error to stabilizers if it's not the last round
            if round < self.n_rounds - 1:
                self.qc.append('X_ERROR', self.x_stabilizers + self.z_stabilizers, self.hardware_params['state_prep_error']) 
                self.qc.append('TICK')


    def _final_step_x_memory(self):
        L = self.L

        # # Apply Hadamard gates to all data qubits.
        # self.qc.append('H', self.data_qubits)
        # self.qc.append('TICK')
        # # Add the noise caused by the Hadamard gates.
        # # Apply depolarizing noise to data qubits.
        # self.qc.append('DEPOLARIZE1', self.data_qubits, self.hardware_params['gate_error_1q'])
        # # Add AD noise to all qubits while one qubit gates are being applied.
        # if self.ad:
        #     self.qc.append('PAULI_CHANNEL_1', self.qubits, [self.p_X_1q, self.p_X_1q, self.p_Z_1q])
        # self.qc.append('TICK')

        self.qc.append('Z_ERROR', self.data_qubits, self.hardware_params['measurement_error'])  # measurement error
        self.qc.append('TICK')
        self.qc.append('MX', self.data_qubits)  # measure data qubits in X basis

        num_Z_stabs = len(self.z_stabilizers)
        num_X_stabs = len(self.x_stabilizers)
        x_stabs_index = []
        for k in range(num_X_stabs):
            x_stabs_index.append(-(((num_X_stabs + num_Z_stabs) + num_X_stabs+ num_Z_stabs - k) + 1))

        data_meas_indices = {
            q: -(((num_X_stabs + num_Z_stabs) - k) + 1)
            for k, q in enumerate(self.data_qubits)}

        for meas_idx, stab_qubit in zip(x_stabs_index, self.x_stabilizers):
            i_stab = stab_qubit // L
            j_stab = stab_qubit % L
            support = self.x_stab_support[stab_qubit]

            # Create a list of all target records for this detector
            detector_targets = [stim.target_rec(meas_idx)]

            # Add all data qubit measurements from the support
            for dq in support:
                detector_targets.append(stim.target_rec(data_meas_indices[dq]))

            # Create a single detector with all targets
            self.qc.append("DETECTOR", detector_targets, [j_stab, i_stab, self.n_rounds])

        column_j = 1  
        observable_targets = [
            stim.target_rec(-(len(self.data_qubits) - k))
            for k, dq in enumerate(self.data_qubits)
            if dq % L == column_j  
        ]
        self.qc.append("OBSERVABLE_INCLUDE", observable_targets, 0)

    def _final_step_z_memory(self):
        L = self.L

        self.qc.append('X_ERROR', self.data_qubits, self.hardware_params['measurement_error'])
        self.qc.append('TICK')
        self.qc.append('M', self.data_qubits)  # measure data qubits

        num_Z_stabs = len(self.z_stabilizers)
        num_X_stabs = len(self.x_stabilizers)
        z_stabs_index = []
        for k in range(num_Z_stabs):
            z_stabs_index.append(-(((num_Z_stabs + num_X_stabs) + num_Z_stabs - k) + 1))

        data_meas_indices = {
            q: -(((num_Z_stabs + num_X_stabs) - k) + 1)
            for k, q in enumerate(self.data_qubits)}

        for meas_idx, stab_qubit in zip(z_stabs_index, self.z_stabilizers):
            i_stab = stab_qubit // L
            j_stab = stab_qubit % L
            support = self.z_stab_support[stab_qubit]

            # Create a list of all target records for this detector
            detector_targets = [stim.target_rec(meas_idx)]

            # Add all data qubit measurements from the support
            for dq in support:
                detector_targets.append(stim.target_rec(data_meas_indices[dq]))

            # Create a single detector with all targets
            self.qc.append("DETECTOR", detector_targets, [j_stab, i_stab, self.n_rounds])

        observable_targets = [stim.target_rec(-(len(self.data_qubits) - i)) for i in range(self.distance)]
        self.qc.append("OBSERVABLE_INCLUDE", observable_targets, 0)


    def _final_step(self):
        """
        Final round of measurements to complete the surface code cycle.
        """
        if self.memory_type == 'x':
            self._final_step_x_memory()
        elif self.memory_type == 'z':
            self._final_step_z_memory()
        else:
            raise Exception("Memory type not supported")

    def build_surface_code_circuit(self):
        """
        Build the complete surface code circuit by executing initialization, rounds, and final steps.
        """
        self._initialization_step()
        self._rounds_step()
        self._final_step()

        return self.qc
    



if __name__ == "__main__":
    ###############################################################################################
    ################### Define hardware parameters and simulation settings ########################
    ###############################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--T1", type=float, default=250e-6)
    parser.add_argument("--T2", type=float, default=250e-6)
    parser.add_argument("--duration_1q_gate", type=float, default=20e-9)
    parser.add_argument("--duration_2q_gate", type=float, default=30e-9)
    parser.add_argument("--state_prep_error", type=float, default=2.5e-3)
    parser.add_argument("--measurement_error", type=float, default=3e-3)
    parser.add_argument("--gate_error_1q", type=float, default=2e-4)
    parser.add_argument("--gate_error_2q", type=float, default=5e-4)
    parser.add_argument("--duration_readout", type=float, default=500e-9)

    parser.add_argument("--ad", type=bool, default=False)
    parser.add_argument("--crosstalk", type=bool, default=False)
    parser.add_argument("--idle_depol", type=bool, default=True)

    parser.add_argument("--default_2q_gate", type=str, default='CX')
    parser.add_argument("--memory_type", type=str, default='z')


    args = parser.parse_args()

    hardware_params = {
        'T1': args.T1,  # in seconds
        'T2': args.T2,  # in seconds
        'duration_1q_gate': args.duration_1q_gate,  # in seconds
        'duration_2q_gate': args.duration_2q_gate,  # in seconds
        'state_prep_error': args.state_prep_error,
        'measurement_error': args.measurement_error,
        'gate_error_1q': args.gate_error_1q,
        'gate_error_2q': args.gate_error_2q,
        'duration_readout': args.duration_readout
    }

    ad = args.ad
    crosstalk = args.crosstalk
    idle_depol = args.idle_depol

    default_2q_gate = args.default_2q_gate
    memory_type = args.memory_type

    sc = SurfaceCode(
        hardware_params=hardware_params,
        distance=3,
        n_rounds=3,
        ad=ad,
        crosstalk=crosstalk,
        default_2q_gate=default_2q_gate,
        memory_type=memory_type,
        multiplier=1.0,
        idle_depol=idle_depol
    )

    qc = sc.build_surface_code_circuit()

    d_timeslice = qc.diagram("timeslice-svg")
    d_timeline = qc.diagram("timeline-svg")
    d_destlice = qc.diagram("detslice-with-ops-svg")
    d_timeslice_without_noise = qc.without_noise().diagram("timeslice-svg")

    # Robust conversion to SVG text
    svg_timeslice = d_timeslice._repr_svg_() if hasattr(d_timeslice, "_repr_svg_") else str(d_timeslice)
    svg_timeline = d_timeline._repr_svg_() if hasattr(d_timeline, "_repr_svg_") else str(d_timeline)
    svg_detslice = d_destlice._repr_svg_() if hasattr(d_destlice, "_repr_svg_") else str(d_destlice)
    svg_timeslice_without_noise = d_timeslice_without_noise._repr_svg_() if hasattr(d_timeslice_without_noise, "_repr_svg_") else str(d_timeslice_without_noise)

    if idle_depol and not(ad) and not(crosstalk):
        text = "_paper"
    else:
        text = ""

    Path(f"reg_sc_circuit_timeslice{text}.svg").write_text(svg_timeslice, encoding="utf-8")
    Path(f"reg_sc_circuit_timeline{text}.svg").write_text(svg_timeline, encoding="utf-8")
    Path(f"reg_sc_circuit_detslice{text}.svg").write_text(svg_detslice, encoding="utf-8")
    Path(f"reg_sc_circuit_timeslice_without_noise{text}.svg").write_text(svg_timeslice_without_noise, encoding="utf-8")

    html = qc.diagram("matchgraph-3d-html")  # interactive
    Path("matchgraph.html").write_text(str(html), encoding="utf-8")

    dem = qc.detector_error_model(decompose_errors=False)
    print(dem)
    print("num_detectors:", dem.num_detectors)
    print("num_observables:", dem.num_observables)
