import numpy as np
import pymatching
import stim
import os

def raw_syndromes_to_detectors(raw_syndromes, rounds=3):
    """
    Converts raw Qiskit absolute measurements into relative 'Detector Events'.
    A detector fires (1) if the measurement changed from the previous round,
    indicating an error occurred in that specific time window.
    """
    N_shots = raw_syndromes.shape[0]
    
    # Total detectors: Round 0 has 4 (Z only). Rounds 1 and 2 have 8 each (X and Z).
    # Total = 4 + 8 + 8 = 20 detectors
    detectors = np.zeros((N_shots, 20), dtype=np.int8)
    
    # --- ROUND 0 ---
    # Z-checks (initialized in |0>, so expected outcome is 0)
    # Raw bits 4, 5, 6, 7
    detectors[:, 0:4] = raw_syndromes[:, 4:8]
    # X-checks are perfectly random in Round 0, so they don't form detectors!
    
    # --- ROUND 1 ---
    # X-detectors (Current ^ Previous) -> Raw bits 8..11 ^ 0..3
    detectors[:, 4:8] = raw_syndromes[:, 8:12] ^ raw_syndromes[:, 0:4]
    # Z-detectors (Current ^ Previous) -> Raw bits 12..15 ^ 4..7
    detectors[:, 8:12] = raw_syndromes[:, 12:16] ^ raw_syndromes[:, 4:8]
    
    # --- ROUND 2 ---
    # X-detectors (Current ^ Previous) -> Raw bits 16..19 ^ 8..11
    detectors[:, 12:16] = raw_syndromes[:, 16:20] ^ raw_syndromes[:, 8:12]
    # Z-detectors (Current ^ Previous) -> Raw bits 20..23 ^ 12..15
    detectors[:, 16:20] = raw_syndromes[:, 20:24] ^ raw_syndromes[:, 12:16]
    
    return detectors

def build_matching_dem(rounds=3, physical_noise=0.02):
    """
    Builds a Stim circuit identical to our 4-Step Qiskit circuit to generate 
    a perfectly commuting, deterministic Detector Error Model.
    """
    c = stim.Circuit()
    
    # Corrected 4-Step Geometry mapping
    Z_NW = [(0, 14), (4, 15), (6, 16)];  X_NW = [(10, 1), (11, 3), (12, 5)]
    Z_NE = [(1, 14), (5, 15), (7, 16)];  X_NE = [(10, 2), (11, 4), (9, 0)]
    Z_SW = [(3, 14), (7, 15), (1, 13)];  X_SW = [(10, 4), (11, 6), (12, 8)]
    Z_SE = [(4, 14), (8, 15), (2, 13)];  X_SE = [(10, 5), (11, 7), (9, 3)]
    
    for r in range(rounds):
        if r > 0:
            c.append("R", range(9, 17))
            c.append("X_ERROR", range(9, 17), physical_noise)
            
        c.append("H", range(9, 13))
        
        # Step 1: NW
        for dq, aq in Z_NW: c.append("CX", [dq, aq])
        for aq, dq in X_NW: c.append("CX", [aq, dq])
        c.append("DEPOLARIZE2", [idx for pair in Z_NW+X_NW for idx in pair], physical_noise)
        
        # Step 2: NE
        for dq, aq in Z_NE: c.append("CX", [dq, aq])
        for aq, dq in X_NE: c.append("CX", [aq, dq])
        c.append("DEPOLARIZE2", [idx for pair in Z_NE+X_NE for idx in pair], physical_noise)
        
        # Step 3: SW
        for dq, aq in Z_SW: c.append("CX", [dq, aq])
        for aq, dq in X_SW: c.append("CX", [aq, dq])
        c.append("DEPOLARIZE2", [idx for pair in Z_SW+X_SW for idx in pair], physical_noise)
        
        # Step 4: SE
        for dq, aq in Z_SE: c.append("CX", [dq, aq])
        for aq, dq in X_SE: c.append("CX", [aq, dq])
        c.append("DEPOLARIZE2", [idx for pair in Z_SE+X_SE for idx in pair], physical_noise)
        
        c.append("H", range(9, 13))
        
        # Measurement
        c.append("M", range(9, 17))
        c.append("X_ERROR", range(9, 17), physical_noise)
        
        # Define Deterministic Detectors for PyMatching
        if r == 0:
            for i in range(4): # Round 0 X measurements are random, so only track Z
                c.append("DETECTOR", [stim.target_rec(-4 + i)])
        else:
            for i in range(4): # Compare X current to X previous
                c.append("DETECTOR", [stim.target_rec(-8 + i), stim.target_rec(-16 + i)])
            for i in range(4): # Compare Z current to Z previous
                c.append("DETECTOR", [stim.target_rec(-4 + i), stim.target_rec(-12 + i)])
                
    # Final Data Measurement
    c.append("M", range(9))
    
    # CORRECTED OBSERVABLE:
    # We must measure the vertical Z_L string connecting the boundaries (D0, D3, D6)
    # In the record of 9 measurements, D0 is -9, D3 is -6, D6 is -3
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-9), stim.target_rec(-6), stim.target_rec(-3)], 0)
    
    # Tell Stim to decompose hyperedges so PyMatching can use correlations!
    return c.detector_error_model(decompose_errors=True)

if __name__ == "__main__":
    # 1. LOAD THE HARDWARE DATA
    filename = "data/Tuna-17_d3_r3_shots.npz"
    if not os.path.exists(filename):
        print(f"[!] Error: Could not find {filename}")
        exit()
        
    data = np.load(filename)
    raw_syndromes = data['syndromes']
    true_observables = data['observables']
    n_shots = raw_syndromes.shape[0]
    
    print(f"[*] Loaded {n_shots} shots from Tuna-17.")
    
    # 2. DATA PRE-PROCESSING
    detector_events = raw_syndromes_to_detectors(raw_syndromes)
    physical_error_rate = detector_events.mean()
    
    # Calculate baseline logical errors
    base_error_count = np.sum(true_observables)
    base_logical_error = base_error_count / n_shots
    
    print("\n--- Pre-Decoding Hardware Baseline ---")
    print(f"Average Physical Detector Fraction: {physical_error_rate:.2%}")
    print(f"Uncorrected Logical Error Rate:     {base_logical_error:.2%} ({base_error_count}/{n_shots} errors)")
    
    # 3. BUILD GRAPH AND DECODE
    print("\n[*] Generating exact DEM layout for PyMatching...")
    
    # Noise prior tuned to 0.02 for the clean, zero-SWAP data
    dem = build_matching_dem(rounds=3, physical_noise=0.02)
    
    # --- Decoder A: Uncorrelated MWPM ---
    matcher_base = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    predictions_base = matcher_base.decode_batch(detector_events)
    
    corrected_errors_base = np.logical_xor(true_observables, predictions_base[:, 0])
    decoded_count_base = np.sum(corrected_errors_base)
    decoded_rate_base = decoded_count_base / n_shots
    
    # --- Decoder B: Correlated MWPM ---
    matcher_corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    predictions_corr = matcher_corr.decode_batch(detector_events)
    
    corrected_errors_corr = np.logical_xor(true_observables, predictions_corr[:, 0])
    decoded_count_corr = np.sum(corrected_errors_corr)
    decoded_rate_corr = decoded_count_corr / n_shots
    
    print("\n--- Decoding Results ---")
    print(f"Uncorrelated MWPM LER: {decoded_rate_base:.2%} ({decoded_count_base} errors) | {base_error_count - decoded_count_base} errors suppressed")
    print(f"Correlated MWPM LER:   {decoded_rate_corr:.2%} ({decoded_count_corr} errors) | {base_error_count - decoded_count_corr} errors suppressed")
    
    if decoded_rate_corr < decoded_rate_base:
        print("\n[*] Conclusion: Enable_Correlations successfully suppressed more physical drift!")