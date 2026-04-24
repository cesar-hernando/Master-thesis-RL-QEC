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
    q = QuantumRegister(17, 'q')
    
    syn_cr = ClassicalRegister(8 * rounds, 'syndromes')
    data_cr = ClassicalRegister(9, 'final_readout')
    
    qc = QuantumCircuit(q, syn_cr, data_cr)
    
    # ==========================================
    # GEOMETRY & 4-STEP SCHEDULE
    # ==========================================
    Z_NW = [(0, 14), (4, 15), (6, 16)]
    X_NW = [(10, 1), (11, 3), (12, 5)]
    
    Z_NE = [(1, 14), (5, 15), (7, 16)]
    X_NE = [(10, 2), (11, 4), (9, 0)]
    
    Z_SW = [(3, 14), (7, 15), (1, 13)]
    X_SW = [(10, 4), (11, 6), (12, 8)]
    
    Z_SE = [(4, 14), (8, 15), (2, 13)]
    X_SE = [(10, 5), (11, 7), (9, 3)]

    qc.barrier()
    
    # ==========================================
    # SYNDROME EXTRACTION CYCLES
    # ==========================================
    for r in range(rounds):
        if r > 0:
            qc.reset([q[i] for i in range(9, 17)])
            
        qc.h([q[i] for i in range(9, 13)])
        
        # NW
        for dq, aq in Z_NW: qc.cx(q[dq], q[aq])
        for aq, dq in X_NW: qc.cx(q[aq], q[dq])
        
        # NE
        for dq, aq in Z_NE: qc.cx(q[dq], q[aq])
        for aq, dq in X_NE: qc.cx(q[aq], q[dq])
        
        # SW
        for dq, aq in Z_SW: qc.cx(q[dq], q[aq])
        for aq, dq in X_SW: qc.cx(q[aq], q[dq])
        
        # SE
        for dq, aq in Z_SE: qc.cx(q[dq], q[aq])
        for aq, dq in X_SE: qc.cx(q[aq], q[dq])
        
        qc.h([q[i] for i in range(9, 13)])
        
        for i in range(4):
            qc.measure(q[9 + i], syn_cr[r*8 + i])
            qc.measure(q[13 + i], syn_cr[r*8 + 4 + i])
            
        qc.barrier()
        
    qc.measure([q[i] for i in range(9)], data_cr)
    
    return qc


def run_and_parse_surface_code(backend_name="QX emulator", shots=2048, rounds=3):
    print(f"[*] Connecting to Quantum Inspire...")
    provider = QIProvider()
    backend = provider.get_backend(backend_name)
    
    qc = build_d3_surface_code(rounds=rounds)
    
    print(f"[*] Submitting Surface Code job to {backend_name}...")
    
    try:
        # Strict zero-SWAP mapping to ensure topological integrity
        # (vf2 string explicitly removed for modern Qiskit compatibility)
        qc_transpiled = transpile(
            qc, 
            backend=backend, 
            optimization_level=3,
            routing_method='none' 
        )
        print(f"[*] Transpiled Circuit Depth: {qc_transpiled.depth()} (0 SWAP gates!)")
    except Exception as e:
        print("\n[!!!] FATAL ERROR: ZERO-SWAP MAPPING FAILED [!!!]")
        raise e
        
    job = backend.run(qc_transpiled, shots=shots, memory=True)
    
    time.sleep(3)
    job_id = getattr(job, 'batch_job_id', getattr(job, 'job_id', 'Unknown'))
    print(f"[*] Job successfully submitted! Cloud Job ID: {job_id}")
    print(f"[*] Waiting in the hardware queue... (This may take hours)")
    
    result = None
    while result is None:
        try:
            result = job.result(timeout=86400)
        except (socket.gaierror, aiohttp.client_exceptions.ClientConnectorError) as e:
            print(f"\n[!] Network connection lost. Retrying in 60s...")
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
    # PARSING LOGIC: True Left-to-Right
    # ==========================================
    raw_syndromes = []
    true_observables = []
    num_syn_bits = 8 * rounds
    
    for shot_str in raw_memory:
        # Strip Quantum Inspire formatting spaces
        clean_str = shot_str.strip().replace(" ", "")
        
        # Read sequentially Left-to-Right (NO REVERSAL)
        syn_str = clean_str[:num_syn_bits]
        data_str = clean_str[num_syn_bits:]
        
        syn_bits = [int(b) for b in syn_str]
        data_bits = [int(b) for b in data_str]
        
        raw_syndromes.append(syn_bits)
        
        # Observables: Parity of the vertical column (D0, D3, D6)
        logical_flip = (data_bits[0] ^ data_bits[3] ^ data_bits[6]) == 1
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
    
    os.makedirs("data", exist_ok=True)
    filename = f"data/{BACKEND}_d3_r{N_ROUNDS}_shots.npz"
    np.savez_compressed(filename, syndromes=syndromes, observables=observables)
    print(f"\n[*] Data successfully saved to {filename} for RL training!")