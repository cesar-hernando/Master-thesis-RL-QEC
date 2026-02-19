'''
In this file, we analyze how the number of edges in the decoding graph scales 
with the code distance and the number of rounds. 
'''

import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching


def count_matching_edges(distance, rounds, enable_correlations=False):
	"""Build a Stim rotated memory circuit and return number of edges in pymatching Matching.

	Uses Stim's built-in generator `surface_code:rotated_memory_z`.
	"""
	# Use small default noise rates — these don't affect topology/number of nodes,
	# but can influence how DEM expands (kept small and fixed here).
	circ = stim.Circuit.generated(
		"surface_code:rotated_memory_z",
		distance=distance,
		rounds=rounds,
		after_clifford_depolarization=0.01,
		before_measure_flip_probability=0.01,
		after_reset_flip_probability=0.01,
		before_round_data_depolarization=0.01,
	)

	# Build matching from the stim circuit
	matching = pymatching.Matching.from_stim_circuit(circ, enable_correlations=enable_correlations)

	# Count edges (matching.edges() yields (u,v,data) triples)
	edge_count = len(matching.edges())
	return edge_count


def scan_distances(distances, rounds_factor=1, enable_correlations=False):
	"""Scan a list of distances and return a dict distance->edge_count.

	rounds_factor: rounds = distance * rounds_factor (use 1 for rounds==distance)
	"""
	results = {}
	for d in distances:
		rounds = max(1, int(d * rounds_factor))
		print(f"Building matching for distance={d}, rounds={rounds}...")
		n_edges = count_matching_edges(d, rounds, enable_correlations=enable_correlations)
		print(f"  edges = {n_edges}")
		results[d] = n_edges
	return results


def plot_results(results, out_path=None):
	distances = sorted(results.keys())
	edges = [results[d] for d in distances]

	plt.figure(figsize=(6,4))
	plt.plot(distances, edges, marker='o')
	plt.xlabel('Code distance (d)')
	plt.ylabel('Number of edges in decoding graph')
	plt.title('Decoding-graph edges vs code distance (Stim built-in Z memory)')
	plt.grid(True, linestyle='--', alpha=0.4)

	if out_path:
		plt.savefig(out_path, dpi=150, bbox_inches='tight')
		print(f"Saved figure to {out_path}")
	plt.show()


def main():
	# Simple configuration (edit these lists/values directly)
	distances = [3, 5, 7, 9, 11, 13, 15]  # List of code distances to analyze
	rounds_factor = 1.0
	out_path = 'edges_vs_distance.png'
	enable_correlations = False

	results = scan_distances(distances, rounds_factor=rounds_factor, enable_correlations=enable_correlations)
	plot_results(results, out_path=out_path)


if __name__ == '__main__':
	main()

