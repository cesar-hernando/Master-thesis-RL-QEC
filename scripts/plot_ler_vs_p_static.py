"""
Plot logical error rate vs physical error rate from hardcoded data points
(no evaluation; just renders the LER vs p figure for a precomputed sweep).
"""

import os

import matplotlib.pyplot as plt
import numpy as np


DATA = {
    "p":           [0.002,        0.004,        0.006,        0.008,        0.01],
    "static_mean": [2.684615e-04, 4.080000e-03, 1.913500e-02, 5.008000e-02, 1.009200e-01],
    "static_std":  [1.016006e-05, 1.425369e-04, 3.063401e-04, 4.877089e-04, 6.735546e-04],
    "corr_mean":   [1.223077e-04, 2.120000e-03, 1.158000e-02, 3.519000e-02, 7.903000e-02],
    "corr_std":    [6.858255e-06, 1.028471e-04, 2.392269e-04, 4.120174e-04, 6.032589e-04],
    "neural_mean": [1.526923e-04, 2.880000e-03, 1.479500e-02, 4.215000e-02, 8.891500e-02],
    "neural_std":  [7.662822e-06, 1.198271e-04, 2.699640e-04, 4.492960e-04, 6.364319e-04],
}


def make_plot(out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    p = np.array(DATA["p"], dtype=np.float64)

    static_mean = np.array(DATA["static_mean"], dtype=np.float64)
    static_std  = np.array(DATA["static_std"],  dtype=np.float64)
    corr_mean   = np.array(DATA["corr_mean"],   dtype=np.float64)
    corr_std    = np.array(DATA["corr_std"],    dtype=np.float64)
    neural_mean = np.array(DATA["neural_mean"], dtype=np.float64)
    neural_std  = np.array(DATA["neural_std"],  dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(p, static_mean, yerr=static_std, fmt='o-', capsize=4,
                label='Standard MWPM', linewidth=2)
    ax.errorbar(p, corr_mean, yerr=corr_std, fmt='s-', capsize=4,
                label='Correlated Matching', linewidth=2)
    ax.errorbar(p, neural_mean, yerr=neural_std, fmt='^-', capsize=4,
                label='Neural Correlated Matching', linewidth=2)

    ax.set_yscale('log')
    ax.set_xscale('linear')

    ax.set_title("Logical Error Rate vs Physical Error Rate")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    out_png = "plots/ler_vs_p/ler_vs_p_mwpm_corr_neural_static.png"
    make_plot(out_png)
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
