"""
Analyze how the GNN graph size (number of nodes and number of edges) scales
with the surface code distance.

- GNN nodes = number of decoding-graph edges (env.n_dec_edges)
- GNN edges = number of line-graph edges built from DEM correlated hyperedges
              (env.n_line_edges)

For each distance d we instantiate DriftedMatchingEnv (with n_rounds = d, as
used in main.py), read both counts directly off the environment, and then fit
degree-3 polynomials of the form

    f(d) = c3 * d^3 + c2 * d^2 + c1 * d + c0

to both series. Coefficients are printed and the fits are plotted on top of
the measured points.
"""

import numpy as np
import matplotlib.pyplot as plt

from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv


def build_env(distance, n_rounds, p=0.004, mismatch=30.0):
    """Build a minimal DriftedMatchingEnv to read graph sizes."""
    generator = SyndromeDataGenerator(
        distance=distance,
        n_rounds=n_rounds,
        mismatch=mismatch,
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": p,
            "before_measure_flip_probability": p,
            "after_reset_flip_probability": p,
            "before_round_data_depolarization": p,
            "p_gate_zz": 0.0,
        },
        memory_type="z",
        n_shots=1,
        qec_code="surface_code",
    )

    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True,
        local_action_hops=1,
        action_scale=1.0,
        update_period=1_000,
        prior_shots=1_000,
        n_test_shots=0,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        use_endpoint_firing=False,
        use_log_joint_prob=False,
        start_from_oracle=False,
        update_with="DGR",
        train_mode=False,
    )
    return env


def scan_distances(distances, p=0.004, mismatch=30.0):
    """Return arrays (distances, n_nodes, n_edges) over the requested distances."""
    n_nodes = []
    n_edges = []
    for d in distances:
        n_rounds = d  # Matches the convention in scripts/main.py
        print(f"Building env for distance={d}, rounds={n_rounds}...")
        env = build_env(d, n_rounds, p=p, mismatch=mismatch)
        nodes = int(env.n_dec_edges)
        edges = int(env.n_line_edges)
        print(f"  GNN nodes (dec edges) = {nodes}")
        print(f"  GNN edges (line graph) = {edges}")
        n_nodes.append(nodes)
        n_edges.append(edges)
    return np.array(distances), np.array(n_nodes), np.array(n_edges)


def fit_cubic(x, y):
    """Fit y = c3*x^3 + c2*x^2 + c1*x + c0. Returns (coefs_high_to_low, r_squared)."""
    coefs = np.polyfit(x, y, deg=3)  # highest power first
    y_pred = np.polyval(coefs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, r2


def format_poly(coefs):
    c3, c2, c1, c0 = coefs
    return f"{c3:.4f} d^3 + {c2:.4f} d^2 + {c1:.4f} d + {c0:.4f}"


def plot_results(distances, n_nodes, n_edges, node_coefs, edge_coefs, out_path):
    d_fine = np.linspace(distances.min(), distances.max(), 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GNN nodes
    axes[0].plot(distances, n_nodes, "o", color="C0", label="Measured")
    axes[0].plot(d_fine, np.polyval(node_coefs, d_fine), "--", color="C0",
                 label=f"Cubic fit: {format_poly(node_coefs)}")
    axes[0].set_xlabel("Code distance d (n_rounds = d)")
    axes[0].set_ylabel("Number of GNN nodes (= decoding-graph edges)")
    axes[0].set_title("GNN nodes vs distance")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    # GNN edges
    axes[1].plot(distances, n_edges, "o", color="C1", label="Measured")
    axes[1].plot(d_fine, np.polyval(edge_coefs, d_fine), "--", color="C1",
                 label=f"Cubic fit: {format_poly(edge_coefs)}")
    axes[1].set_xlabel("Code distance d (n_rounds = d)")
    axes[1].set_ylabel("Number of GNN edges (= line-graph edges)")
    axes[1].set_title("GNN edges vs distance")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


def main():
    distances = [3, 5, 7, 9, 11, 13, 15, 17]
    p = 0.004
    mismatch = 30.0
    out_path = "plots/gnn_graph_scaling.png"

    distances, n_nodes, n_edges = scan_distances(distances, p=p, mismatch=mismatch)

    node_coefs, node_r2 = fit_cubic(distances, n_nodes)
    edge_coefs, edge_r2 = fit_cubic(distances, n_edges)

    print("\n=== Cubic fit coefficients (highest power first: c3, c2, c1, c0) ===")
    print(f"GNN nodes: {node_coefs}  (R^2 = {node_r2:.5f})")
    print(f"           {format_poly(node_coefs)}")
    print(f"GNN edges: {edge_coefs}  (R^2 = {edge_r2:.5f})")
    print(f"           {format_poly(edge_coefs)}")

    plot_results(distances, n_nodes, n_edges, node_coefs, edge_coefs, out_path)


if __name__ == "__main__":
    main()
