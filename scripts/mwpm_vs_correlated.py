"""
Evaluate and plot the Logical Error Rate (LER) of Standard MWPM vs Correlated Matching
for distances d in {3, 5, 7} using Stim's built-in circuit-level noise model.
"""

import argparse
import os
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords

def evaluate_decoder(d, p, target_errors=200, max_shots=20_000_000, batch_size=100_000):
    """
    Evaluates both standard and correlated matching for a given distance and physical error rate.
    Uses adaptive batching to ensure statistical significance without wasting compute.
    """
    print(f"\n--- Distance {d}, p = {p:.5e} ---")
    
    # 1. Generate the Stim Circuit and DEM
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )
    #dem = circuit.detector_error_model(decompose_errors=True)
    # Generate the RAW dem (tell Stim NOT to decompose)
    dem_raw = circuit.detector_error_model(decompose_errors=False)
    
    # Use Tesseract to perform the geometrically-aware decomposition
    dem = decompose_errors_for_stim_surface_code_coords(dem_raw)
    
    # 2. Initialize PyMatching decoders
    m_standard = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    m_corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    
    # 3. Create a fast C++ sampler
    sampler = circuit.compile_detector_sampler()
    
    total_shots = 0
    err_standard = 0
    err_corr = 0
    
    while total_shots < max_shots:
        # Sample a batch of syndromes and true logical observables
        syndromes, true_obs = sampler.sample(shots=batch_size, separate_observables=True)
        true_obs = true_obs.flatten()
        
        # Decode with Standard MWPM
        pred_standard = m_standard.decode_batch(syndromes)
        err_standard += np.sum(pred_standard[:, 0] != true_obs)
        
        # Decode with Correlated Matching
        pred_corr = m_corr.decode_batch(syndromes, enable_correlations=True)
        err_corr += np.sum(pred_corr[:, 0] != true_obs)
        
        total_shots += batch_size
        
        # Check if we have gathered enough errors to be confident in the LER
        if min(err_standard, err_corr) >= target_errors:
            break

    # Calculate LER and Binomial Standard Error: sqrt(ler * (1 - ler) / N)
    ler_std = err_standard / total_shots
    ler_std_err = np.sqrt((ler_std * (1.0 - ler_std)) / total_shots)
    
    ler_corr = err_corr / total_shots
    ler_corr_err = np.sqrt((ler_corr * (1.0 - ler_corr)) / total_shots)
    
    print(f"Total Shots: {total_shots:,}")
    print(f"Standard MWPM : {err_standard} errors | LER: {ler_std:.4e} ± {ler_std_err:.4e}")
    print(f"Corr Matching : {err_corr} errors | LER: {ler_corr:.4e} ± {ler_corr_err:.4e}")
    
    return {
        'shots': total_shots,
        'ler_std': ler_std,
        'ler_std_err': ler_std_err,
        'ler_corr': ler_corr,
        'ler_corr_err': ler_corr_err
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-png", type=str, default="plots/mwpm_vs_correlated.png")
    parser.add_argument("--out-csv", type=str, default="data/mwpm_vs_correlated.csv")
    parser.add_argument("--max-shots", type=int, default=100_000_000, help="Max shots per data point.")
    parser.add_argument("--target-errors", type=int, default=500, help="Target errors for adaptive batching.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    distances = [3, 5]
    # Log-space ensures the 5 points are evenly distributed visually on the log-log plot
    #p_values = np.logspace(np.log10(1e-4), np.log10(1e-2), 5) 
    p_values = np.array([4e-4])

    # Data structures to store results for plotting
    results = {d: {'p': [], 'std': [], 'std_err': [], 'corr': [], 'corr_err': []} for d in distances}

    # Open CSV for saving
    with open(args.out_csv, "w") as f:
        f.write("distance,p,shots,ler_standard,std_err_standard,ler_correlated,std_err_correlated\n")
        
        for d in distances:
            for p in p_values:
                metrics = evaluate_decoder(d, p, args.target_errors, args.max_shots)
                
                # Save to dictionary
                results[d]['p'].append(p)
                results[d]['std'].append(metrics['ler_std'])
                results[d]['std_err'].append(metrics['ler_std_err'])
                results[d]['corr'].append(metrics['ler_corr'])
                results[d]['corr_err'].append(metrics['ler_corr_err'])
                
                # Write to CSV
                f.write(f"{d},{p},{metrics['shots']},{metrics['ler_std']},{metrics['ler_std_err']},{metrics['ler_corr']},{metrics['ler_corr_err']}\n")
                f.flush()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {3: '#1f77b4', 5: '#ff7f0e'} # Blue, Orange
    
    for d in distances:
        c = colors[d]
        p_arr = np.array(results[d]['p'])
        
        # Plot Standard MWPM (Dashed line, circular markers)
        ax.errorbar(p_arr, results[d]['std'], yerr=results[d]['std_err'], 
                    fmt='o--', color=c, capsize=4, linewidth=2, label=f'Standard MWPM (d={d})')
        
        # Plot Correlated Matching (Solid line, square markers)
        ax.errorbar(p_arr, results[d]['corr'], yerr=results[d]['corr_err'], 
                    fmt='s-', color=c, capsize=4, linewidth=2, label=f'Correlated Matching (d={d})')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Standard MWPM vs. Correlated Matching (Stim Circuit-Level Noise)", fontsize=14)
    ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
    ax.set_ylabel("Logical Error Rate (LER)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Place legend outside or neatly inside
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {args.out_png}")
    print(f"Data saved to {args.out_csv}")

if __name__ == "__main__":
    main()