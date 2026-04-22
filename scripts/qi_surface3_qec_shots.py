import os
import time
import socket
import aiohttp
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_quantuminspire.qi_provider import QIProvider

def build_d3_surface_code(rounds=3):
    """
    Builds a Distance-3 Rotated Surface Code memory experiment.
    Uses a single flat QuantumRegister to avoid OpenSquirrel translation bugs.
    Initialized in the logical |0> state.
    """
    # 1. FLAT REGISTER FIX: 17 Qubits total
    q = QuantumRegister(17, 'q')
    
    # Separate classical registers. 
    # Added in this order so QI puts syndromes on the left of the output string!
    syn_cr = ClassicalRegister(8 * rounds, 'syndromes')
    data_cr = ClassicalRegister(9, 'final_readout')
    
    qc = QuantumCircuit(q, syn_cr, data_cr)
    
    # Map the flat register to semantic lists for clean circuit building
    dq = [q[i] for i in range(9)]        # Data Qubits: 0 to 8
    xq = [q[i] for i in range(9, 13)]    # X-Ancillas: 9 to 12
    zq = [q[i] for i in range(13, 17)]   # Z-Ancillas: 13 to 16
    
    # ==========================================
    # 1. GEOMETRY & STABILIZER DEFINITION
    # ==========================================
    # Z-Checks (Measure Z, detects X bit-flips). Target is the Z-ancilla.
    Z_CHECKS = [
        [1, 2],          # Z0 (Top Right Boundary)
        [0, 1, 3, 4],    # Z1 (Top Left Plaquette)
        [4, 5, 7, 8],    # Z2 (Bottom Right Plaquette)
        [6, 7]           # Z3 (Bottom Left Boundary)
    ]
    
    # X-Checks (Measure X, detects Z phase-flips). Control is the X-ancilla.
    X_CHECKS = [
        [0, 1],          # X0 (Top Left Boundary)
        [1, 2, 4, 5],    # X1 (Top Right Plaquette)
        [3, 4, 6, 7],    # X2 (Bottom Left Plaquette)
        [7, 8]           # X3 (Bottom Right Boundary)
    ]
    
    # ==========================================
    # 2. STATE PREPARATION
    # ==========================================
    qc.barrier()
    
    # ==========================================
    # 3. SYNDROME EXTRACTION CYCLES
    # ==========================================
    for r in range(rounds):
        if r > 0:
            qc.reset(xq)
            qc.reset(zq)
            
        qc.h(xq)
        
        for anc_idx, data_indices in enumerate(Z_CHECKS):
            for d in data_indices:
                qc.cx(dq[d], zq[anc_idx])
                
        for anc_idx, data_indices in enumerate(X_CHECKS):
            for d in data_indices:
                qc.cx(xq[anc_idx], dq[d])
                
        qc.h(xq)
        
        for i in range(4):
            qc.measure(xq[i], syn_cr[r*8 + i])
            qc.measure(zq[i], syn_cr[r*8 + 4 + i])
            
        qc.barrier()
        
    # ==========================================
    # 4. FINAL DATA MEASUREMENT
    # ==========================================
    qc.measure(dq, data_cr)
    
    return qc


def run_and_parse_surface_code(backend_name="QX emulator", shots=2048, rounds=3):
    """
    Executes the Surface Code on Quantum Inspire and parses the 
    bitstrings into numpy arrays. Features a robust Wi-Fi drop retry loop.
    """
    print(f"[*] Connecting to Quantum Inspire...")
    provider = QIProvider()
    backend = provider.get_backend(backend_name)
    
    qc = build_d3_surface_code(rounds=rounds)
    qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
    
    print(f"[*] Submitting Surface Code job to {backend_name}...")
    job = backend.run(qc_transpiled, shots=shots, memory=True)
    
    # Wait a few seconds to ensure the cloud populates the ID in the object
    time.sleep(3)
    
    job_id = getattr(job, 'batch_job_id', getattr(job, 'job_id', 'Unknown'))
    print(f"[*] Job successfully submitted! Cloud Job ID: {job_id}")
    print(f"[*] Waiting in the hardware queue... (This may take hours)")
    
    # ==========================================
    # ROBUST NETWORK POLLING
    # ==========================================
    result = None
    while result is None:
        try:
            result = job.result(timeout=86400)
        except (socket.gaierror, aiohttp.client_exceptions.ClientConnectorError) as e:
            print(f"\n[!] Network connection lost: {e}")
            print(f"[*] Don't panic! The job is still in the cloud queue.")
            print(f"[*] Retrying connection in 60 seconds...")
            time.sleep(60)
        except Exception as e:
            if "Timeout" in str(e):
                print(f"\n[!] Qiskit timeout reached. Still waiting in queue...")
                time.sleep(30)
            else:
                raise e

    print("[*] Job complete! Downloading data...")
    raw_memory = result.get_memory()
    
    # ==========================================
    # PARSING LOGIC
    # ==========================================
    raw_syndromes = []
    true_observables = []
    num_syn_bits = 8 * rounds
    
    for shot_str in raw_memory:
        shot_str = shot_str.strip()
        syn_str = shot_str[:num_syn_bits]
        data_str = shot_str[num_syn_bits:]
        
        syn_bits = [int(b) for b in syn_str]
        data_bits = [int(b) for b in data_str]
        
        raw_syndromes.append(syn_bits)
        
        logical_flip = (data_bits[0] ^ data_bits[1] ^ data_bits[2]) == 1
        true_observables.append(logical_flip)
        
    return np.array(raw_syndromes, dtype=np.int8), np.array(true_observables, dtype=bool)


if __name__ == "__main__":
    N_SHOTS = 2048 
    N_ROUNDS = 3
    BACKEND = "Tuna-17" 
    
    syndromes, observables = run_and_parse_surface_code(
        backend_name=BACKEND, 
        shots=N_SHOTS, 
        rounds=N_ROUNDS
    )
    
    print("\n[*] Execution Complete!")
    print(f"Syndrome Matrix Shape: {syndromes.shape} ({N_ROUNDS} rounds * 8 ancillas)")
    print(f"Logical Observables Shape: {observables.shape}")
    print(f"Total Logical Errors Observed: {np.sum(observables)}")
    
    # ==========================================
    # DATA STORAGE FOR RL AGENT
    # ==========================================
    os.makedirs("data", exist_ok=True)
    filename = f"data/{BACKEND}_d3_r{N_ROUNDS}_shots.npz"
    
    np.savez_compressed(
        filename, 
        syndromes=syndromes, 
        observables=observables
    )
    print(f"\n[*] Data successfully saved to {filename} for RL training!")