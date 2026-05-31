"""
Evaluate three decoders on Google's qec3v5 experimental dataset
(Sycamore "Suppressing quantum errors by scaling a surface code logical qubit",
Nature 614, 676-681, 2023):
  1) Standard MWPM with calibrated weights.
  2) Correlated Matching (manual two-pass with calibrated joints when calibration
     produces them, otherwise pymatching's native enable_correlations=True path).
  3) Neural Correlated Matching (trained SAC-GNN agent), seeded at calibrated weights.

Calibration: DGR (Wang et al. 2023) or Spitz (2018) alignment reweighting on a
held-out slice of the data.

Important difference from Willow: this dataset uses the **standard rotated
surface code** (CSS, with separate X-type and Z-type stabilizers) -- NOT XZZX.
So the project's `decompose_errors_for_stim_surface_code_coords` works directly
and no built-in fallback is needed.

Data layout:
    <data-root>/surface_code_b{X|Z}_d{D}_r{NN}_center_{row}_{col}/
        circuit_ideal.stim
        circuit_noisy.stim                          <- used to build the analytical DEM
        circuit_detector_error_model.dem            <- analytical DEM from the noisy circuit
        pij_from_even_for_odd.dem
        pij_from_odd_for_even.dem                   <- DEMs already calibrated from data
        detection_events.b8
        obs_flips_actual.01                         <- 1 bit per shot, text 01 format
        obs_flips_predicted_by_pymatching.01
        obs_flips_predicted_by_correlated_matching.01
        obs_flips_predicted_by_belief_matching.01
        obs_flips_predicted_by_tensor_network_contraction.01
        measurements.b8, sweep.b8, properties.yml, layout.svg

Available combinations:
    d=3: centers {3_5, 5_3, 5_7, 7_5}; rounds 1, 3, 5, ..., 25 (odd, every 2).
    d=5: center {5_5} only;            rounds 1, 3, 5, ..., 25 (odd, every 2).
Each leaf folder contains 50000 shots.

Writes:
    data/experimental/qec3v5_results.csv             (one row per evaluation)
    data/experimental/qec3v5_calibration_trace.csv   (one row per calibration iter)
The CSV schema matches evaluate_on_willow.py so the same
plot_experimental_results.py script consumes both.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from glob import glob
from typing import Optional

import numpy as np
import pymatching
import stim

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords
from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator


DATASET_NAME = "google_qec3v5"
DEFAULT_CENTER = {3: "5_5", 5: "5_5"}   # d=3 also has 3_5, 5_3, 5_7, 7_5
AVAILABLE_ROUNDS = list(range(1, 26, 2))  # 1, 3, 5, ..., 25


# =============================================================================
# Experimental data generator: reads pre-recorded shots from disk.
# =============================================================================

class ExperimentalDataGenerator(SyndromeDataGenerator):
    """Loads pre-recorded experimental detection events + observable flips.

    Subclasses SyndromeDataGenerator so it can be plugged into DriftedMatchingEnv
    without modifying the env. Auto-detects detector / observable file format
    by extension (.b8 vs .01).
    """

    def __init__(
        self,
        circuit_path: str,
        det_events_path: str,
        obs_flips_path: str,
        distance: int,
        n_rounds: int,
        test_slice: Optional[slice] = None,
    ):
        self.circuit_path = circuit_path
        self.det_events_path = det_events_path
        self.obs_flips_path = obs_flips_path
        self.distance = distance
        self.n_rounds = n_rounds
        self.mismatch = 1.0
        self.noise_model = {"version": "experimental"}
        self.memory_type = "z"
        self.qec_code = "surface_code"

        with open(circuit_path, "r") as f:
            self._circuit = stim.Circuit(f.read())

        self._dets_all, self._obs_all = self._load_data()
        if test_slice is None:
            test_slice = slice(0, len(self._obs_all))
        self.test_dets = self._dets_all[test_slice]
        self.test_obs = self._obs_all[test_slice]
        self.n_shots = len(self.test_obs)

    def _load_data(self):
        n_det = self._circuit.num_detectors
        n_obs = self._circuit.num_observables
        det_fmt = "b8" if self.det_events_path.endswith(".b8") else "01"
        obs_fmt = "b8" if self.obs_flips_path.endswith(".b8") else "01"

        dets = stim.read_shot_data_file(
            path=self.det_events_path,
            format=det_fmt,
            num_detectors=n_det,
            num_observables=0,
        )
        obs = stim.read_shot_data_file(
            path=self.obs_flips_path,
            format=obs_fmt,
            num_detectors=0,
            num_observables=n_obs,
        )
        n = min(len(dets), len(obs))
        return dets[:n].astype(np.uint8), obs[:n].astype(np.uint8).reshape(-1)

    def _build_dem(self):
        """Decomposed DEM. qec3v5 uses the standard rotated surface code so the
        project decomposer succeeds; fall back to Stim's built-in only if it
        doesn't (defensive)."""
        try:
            return decompose_errors_for_stim_surface_code_coords(
                self._circuit.detector_error_model(decompose_errors=False)
            )
        except Exception as exc:
            print(f"  [info] project decomposer failed ({type(exc).__name__}); "
                  "falling back to Stim's built-in decompose_errors=True.")
            return self._circuit.detector_error_model(decompose_errors=True)

    def generate_base_circuit(self):
        base_dem = self._build_dem()
        base_matching = pymatching.Matching.from_detector_error_model(
            base_dem, enable_correlations=False
        )
        return self._circuit, base_dem, base_matching

    def generate_drifted_circuit(self, base_circuit, seed=42):
        drifted_dem = self._build_dem()
        drifted_matching = pymatching.Matching.from_detector_error_model(
            drifted_dem, enable_correlations=True
        )
        return base_circuit, drifted_dem, drifted_matching

    def simulate_syndrome_data(self, drifted_circuit, seed=42, n_shots=None):
        n = n_shots if n_shots is not None else self.n_shots
        n = min(n, len(self.test_obs))
        return self.test_dets[:n], self.test_obs[:n]


# =============================================================================
# Alignment reweighting: DGR and Spitz
# =============================================================================

def probs_to_weights(p: np.ndarray, min_w: float = 1e-6, max_w: float = 50.0) -> np.ndarray:
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
    w = np.log((1.0 - p_safe) / p_safe)
    return np.clip(w, min_w, max_w).astype(np.float32)


def _compute_p_joint_from_decodes(edges_batch: np.ndarray, line_edge_index: np.ndarray) -> np.ndarray:
    if line_edge_index.size == 0:
        return np.zeros(0, dtype=np.float32)
    src = line_edge_index[0]
    dst = line_edge_index[1]
    co = edges_batch[:, src] & edges_batch[:, dst]
    return co.mean(axis=0).astype(np.float32)


def dgr_calibrate_probabilities(
    det_events: np.ndarray,
    check_matrix,
    initial_weights: np.ndarray,
    line_edge_index: np.ndarray,
    n_iterations: int = 1,
    min_p: float = 1e-6,
    max_p: float = 0.499,
):
    """DGR (Wang et al. 2023). Returns (p_single, p_joint, trace)."""
    n_shots = det_events.shape[0]
    weights = initial_weights.astype(np.float32).copy()
    edges_batch = None
    trace = []

    for it in range(n_iterations):
        matching = pymatching.Matching.from_check_matrix(check_matrix, weights=weights)
        edges_batch = np.asarray(
            matching.decode_batch(det_events, enable_correlations=False), dtype=bool
        )
        occurrence_counts = edges_batch.sum(axis=0)
        p = occurrence_counts / float(n_shots)
        p = np.clip(p, min_p, max_p).astype(np.float32)
        weights = probs_to_weights(p)
        row = {
            "iteration": it + 1,
            "mean_p": float(p.mean()),
            "max_p": float(p.max()),
            "n_active_edges": int((occurrence_counts > 0).sum()),
            "n_edges": int(len(p)),
        }
        trace.append(row)
        print(f"  DGR iter {it + 1}/{n_iterations}: "
              f"mean p={row['mean_p']:.4e}  max p={row['max_p']:.4e}  "
              f"n_active_edges={row['n_active_edges']}/{row['n_edges']}")

    p_joint = _compute_p_joint_from_decodes(edges_batch, line_edge_index)
    return p, p_joint, trace


def spitz_calibrate_probabilities(
    det_events: np.ndarray,
    dec_edge_list,
    base_p: np.ndarray,
    check_matrix=None,
    line_edge_index: Optional[np.ndarray] = None,
    min_p: float = 1e-6,
    max_p: float = 0.499,
):
    """Spitz (2018). Mirrors env._compute_raw_syndrome_statistics.

    Returns (p_single, p_joint, trace).
    """
    n_shots, n_det = det_events.shape
    dets = det_events.astype(np.float64)
    v_mean = dets.mean(axis=0)

    p_cal = base_p.astype(np.float64).copy()
    boundary_edges = []

    for idx, (a, b) in enumerate(dec_edge_list):
        if a == -1 or b == -1:
            boundary_edges.append((idx, a, b))
            continue
        if a < 0 or b < 0 or a >= n_det or b >= n_det:
            continue
        va = dets[:, a]
        vb = dets[:, b]
        mean_uv = float((va * vb).mean())
        mean_xor = float((va.astype(np.int8) ^ vb.astype(np.int8)).mean())
        denom = 1.0 - 2.0 * mean_xor
        if abs(denom) < 1e-9:
            p = 0.499
        else:
            cov = mean_uv - v_mean[a] * v_mean[b]
            root_term = max(0.0, 0.25 - (cov / denom))
            p = 0.5 - np.sqrt(root_term)
        p_cal[idx] = np.clip(p, min_p, max_p)

    for idx, a, b in boundary_edges:
        real_node = b if a == -1 else a
        if real_node < 0 or real_node >= n_det:
            continue
        prod_term = 1.0
        for j_idx, (ja, jb) in enumerate(dec_edge_list):
            if j_idx == idx:
                continue
            if (ja == real_node or jb == real_node) and ja != -1 and jb != -1:
                prod_term *= 1.0 - 2.0 * p_cal[j_idx]
        if abs(prod_term) < 1e-9:
            p = 0.499
        else:
            p = 0.5 + (v_mean[real_node] - 0.5) / prod_term
        p_cal[idx] = np.clip(p, min_p, max_p)

    p_single = p_cal.astype(np.float32)

    if check_matrix is not None and line_edge_index is not None and line_edge_index.size > 0:
        weights = probs_to_weights(p_single)
        matching = pymatching.Matching.from_check_matrix(check_matrix, weights=weights)
        edges_batch = np.asarray(
            matching.decode_batch(det_events, enable_correlations=False), dtype=bool
        )
        p_joint = _compute_p_joint_from_decodes(edges_batch, line_edge_index)
    else:
        p_joint = np.zeros(0, dtype=np.float32)

    trace = [{
        "iteration": 1,
        "mean_p": float(p_single.mean()),
        "max_p": float(p_single.max()),
        "n_active_edges": int((p_single > min_p).sum()),
        "n_edges": int(len(p_single)),
    }]
    return p_single, p_joint, trace


# =============================================================================
# Decoders
# =============================================================================

def decode_matching(matching, syndromes, fault_array, enable_correlations, edge_reweights=None):
    """Run pymatching.decode_batch and return observable predictions.

    enable_correlations=False -> returns (n_shots, n_edges) edges; fault-multiply.
    enable_correlations=True  -> the DEM-aware path returns observable predictions
                                 directly; just flatten.
    """
    kw = {"enable_correlations": enable_correlations}
    if edge_reweights is not None:
        kw["edge_reweights"] = edge_reweights
    result = np.asarray(matching.decode_batch(syndromes, **kw))
    if enable_correlations:
        return result.flatten().astype(np.uint8)
    pred_obs = (result @ fault_array) % 2
    return pred_obs.flatten().astype(np.uint8)


def build_conditional_adjacency(
    p_single: np.ndarray,
    p_joint: np.ndarray,
    line_edge_index: np.ndarray,
    n_dec_edges: int,
    eps: float = 1e-15,
):
    """For each decoding edge K, list correlated edges L with P(L | K) = p_joint / p_K."""
    adj = [[] for _ in range(n_dec_edges)]
    if line_edge_index.size == 0:
        return adj
    src = line_edge_index[0]
    dst = line_edge_index[1]
    for i in range(len(src)):
        k = int(src[i]); l = int(dst[i])
        pj = float(p_joint[i])
        if pj <= eps:
            continue
        if p_single[k] > eps:
            adj[k].append((l, pj / float(p_single[k])))
        if p_single[l] > eps:
            adj[l].append((k, pj / float(p_single[l])))
    return adj


def decode_manual_correlated_matching(
    check_matrix,
    w_calibrated: np.ndarray,
    dec_edge_list,
    adj,
    syndromes: np.ndarray,
    fault_array: np.ndarray,
    min_w: float = 1e-6,
    max_w: float = 50.0,
):
    """Manual Correlated Matching (Fowler 2013) with calibrated joints.

    Per shot:
      1) First pass: MWPM with calibrated weights.
      2) Reweight every edge L correlated with at least one selected K via
         w_L <- -log(max_K P(L | K)).
      3) Second pass: MWPM with the per-shot updated weights.
    """
    n_shots = syndromes.shape[0]
    matching = pymatching.Matching.from_check_matrix(check_matrix, weights=w_calibrated)

    dec_arr = np.array(dec_edge_list, dtype=np.int64)
    u_col = dec_arr[:, 0].copy()
    v_col = dec_arr[:, 1].copy()
    boundary_mask = (u_col == -1) & (v_col != -1)
    u_col[boundary_mask] = v_col[boundary_mask]
    v_col[boundary_mask] = -1

    pred_obs = np.zeros(n_shots, dtype=np.uint8)
    for shot in range(n_shots):
        syndrome = syndromes[shot]
        edges_bin = matching.decode(syndrome, enable_correlations=False)
        selected_idx = np.flatnonzero(edges_bin)
        if selected_idx.size == 0:
            pred_obs[shot] = 0
            continue

        max_cond = {}
        for k in selected_idx:
            for (l, p_cond) in adj[k]:
                if edges_bin[l]:
                    continue
                prev = max_cond.get(l, 0.0)
                if p_cond > prev:
                    max_cond[l] = p_cond

        if not max_cond:
            pred_obs[shot] = int((edges_bin @ fault_array) % 2)
            continue

        ids = np.fromiter(max_cond.keys(), dtype=np.int64)
        probs = np.fromiter(max_cond.values(), dtype=np.float64)
        new_w = np.clip(-np.log(np.clip(probs, 1e-15, 1.0)), min_w, max_w)
        reweights = np.column_stack((u_col[ids], v_col[ids], new_w))

        edges_bin2 = matching.decode(
            syndrome, enable_correlations=False, edge_reweights=reweights
        )
        pred_obs[shot] = int((edges_bin2 @ fault_array) % 2)

    return pred_obs


def decode_neural(env, agent, bypass_threshold: int):
    obs, _ = env.reset(seed=0)
    preds, truths = [], []
    done = False
    while not done:
        n_flashes = int(np.sum(env.current_syndrome != 0))
        if n_flashes <= bypass_threshold:
            action = np.zeros(env.n_dec_edges, dtype=np.float32)
        else:
            action = agent.select_action(obs, evaluate=True)
        obs, _, terminated, truncated, info = env.step(action)
        preds.append(int(info["pred_obs"]))
        truths.append(int(info["true_obs"]))
        done = terminated or truncated
    return np.array(preds, dtype=np.uint8), np.array(truths, dtype=np.uint8)


# =============================================================================
# Path resolution
# =============================================================================

def list_centers(data_root: str, distance: int):
    pat = os.path.join(data_root, f"surface_code_b*_d{distance}_r*_center_*")
    hits = glob(pat)
    centers = set()
    for h in hits:
        name = os.path.basename(h)
        # "..._center_{row}_{col}"
        try:
            parts = name.split("_center_")[1]
            centers.add(parts)  # e.g. "5_5"
        except IndexError:
            continue
    return sorted(centers)


def find_experiment_dir(data_root: str, distance: int, rounds: int,
                        basis: str, center: str) -> str:
    folder = (f"surface_code_b{basis.upper()}_d{distance}_r{rounds:02d}"
              f"_center_{center}")
    path = os.path.join(data_root, folder)
    if not os.path.isdir(path):
        centers = list_centers(data_root, distance)
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            f"  Available centers for d={distance}: {centers}\n"
            f"  Available rounds: {AVAILABLE_ROUNDS}."
        )
    return path


# =============================================================================
# CSV outputs (schema identical to evaluate_on_willow.py)
# =============================================================================

RESULTS_FIELDS = [
    "timestamp", "dataset", "distance", "rounds", "basis", "patch",
    "calibration", "n_calibration_requested", "dgr_iterations",
    "n_total", "n_cal", "n_test",
    "err_mwpm", "ler_mwpm", "se_mwpm",
    "err_corr", "ler_corr", "se_corr", "corr_method",
    "err_neural", "ler_neural", "se_neural",
    "model_path", "elapsed_seconds", "notes",
]

CALIB_TRACE_FIELDS = [
    "timestamp", "dataset", "distance", "rounds", "basis", "patch",
    "calibration", "iteration", "mean_p", "max_p", "n_active_edges", "n_edges",
]


def append_results_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for k in RESULTS_FIELDS:
            row.setdefault(k, "")
        w.writerow(row)


def append_calibration_trace_csv(csv_path: str, base_row: dict, trace: list):
    if not trace:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CALIB_TRACE_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in trace:
            full = {**base_row, **r}
            for k in CALIB_TRACE_FIELDS:
                full.setdefault(k, "")
            w.writerow(full)


def _binom_se(p, n):
    return float(np.sqrt(p * (1.0 - p) / n)) if n > 0 else 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate three decoders on Google qec3v5 experimental data."
    )
    parser.add_argument("--data-root", default="google_qec3v5_experiment_data",
                        help="Root of the qec3v5 dataset.")
    parser.add_argument("--distance", type=int, default=5, choices=[3, 5])
    parser.add_argument("--rounds", type=int, default=5,
                        help=f"One of {AVAILABLE_ROUNDS}.")
    parser.add_argument("--basis", choices=["x", "z", "X", "Z"], default="Z")
    parser.add_argument("--center", default=None,
                        help="Patch center, e.g. '5_5'. Default depends on distance "
                             "(d=5 only has 5_5; d=3 also has 3_5, 5_3, 5_7, 7_5).")
    parser.add_argument("--list-centers", action="store_true",
                        help="Print available centers for the chosen distance and exit.")
    parser.add_argument("--circuit-path", default=None)
    parser.add_argument("--dets-path", default=None)
    parser.add_argument("--obs-path", default=None)

    parser.add_argument("--calibration", choices=["dgr", "spitz", "none"], default="dgr")
    parser.add_argument("--n-calibration", type=int, default=25_000)
    parser.add_argument("--dgr-iterations", type=int, default=3)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Alias for --calibration none.")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--action-scale", type=float, default=5.0)
    parser.add_argument("--bypass-threshold", type=int, default=2)
    parser.add_argument("--local-action-hops", type=int, default=1)
    parser.add_argument("--use-endpoint-firing", action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument("--results-csv", default="data/experimental/qec3v5_results.csv")
    parser.add_argument("--trace-csv", default="data/experimental/qec3v5_calibration_trace.csv")
    parser.add_argument("--notes", default="")

    args = parser.parse_args()
    basis = args.basis.upper()
    calibration = "none" if args.skip_calibration else args.calibration

    if args.list_centers:
        centers = list_centers(args.data_root, args.distance)
        print(f"Centers available for d={args.distance}: {centers}")
        return

    if args.rounds not in AVAILABLE_ROUNDS:
        raise SystemExit(f"--rounds={args.rounds} not in the dataset. "
                         f"Available: {AVAILABLE_ROUNDS}.")

    # ----- Resolve data paths -----
    if args.circuit_path and args.dets_path and args.obs_path:
        circuit_path = args.circuit_path
        dets_path = args.dets_path
        obs_path = args.obs_path
        center = args.center or "custom"
    else:
        center = args.center or DEFAULT_CENTER[args.distance]
        exp_dir = find_experiment_dir(args.data_root, args.distance, args.rounds,
                                       basis, center)
        circuit_path = args.circuit_path or os.path.join(exp_dir, "circuit_noisy.stim")
        dets_path = args.dets_path or os.path.join(exp_dir, "detection_events.b8")
        obs_path = args.obs_path or os.path.join(exp_dir, "obs_flips_actual.01")

    print(f"[qec3v5] d={args.distance}  rounds={args.rounds}  basis={basis}  "
          f"center={center}")
    print(f"  circuit : {circuit_path}")
    print(f"  dets    : {dets_path}")
    print(f"  obs     : {obs_path}")

    # ----- Load full dataset -----
    t_start = time.time()
    full = ExperimentalDataGenerator(
        circuit_path=circuit_path,
        det_events_path=dets_path,
        obs_flips_path=obs_path,
        distance=args.distance,
        n_rounds=args.rounds,
    )
    n_total = len(full.test_obs)
    n_cal = 0 if calibration == "none" else min(args.n_calibration, n_total // 2)
    n_test = n_total - n_cal
    print(f"  total={n_total:,}  calibration={n_cal:,}  test={n_test:,}")

    cal_dets = full._dets_all[:n_cal] if n_cal > 0 else None
    test_slice = slice(n_cal, n_total)

    # ----- Build the env (needs graph structure derived from the DEM) -----
    test_gen = ExperimentalDataGenerator(
        circuit_path=circuit_path,
        det_events_path=dets_path,
        obs_flips_path=obs_path,
        distance=args.distance,
        n_rounds=args.rounds,
        test_slice=test_slice,
    )
    env = DriftedMatchingEnv(
        syndrome_data_generator=test_gen,
        local_action_only=True,
        local_action_hops=args.local_action_hops,
        action_scale=args.action_scale,
        update_period=n_test + 1,
        prior_shots=0,
        n_test_shots=0,
        use_pearson_correlation=True,
        use_syndrome_features=False,
        use_log_joint_prob=False,
        start_from_oracle=True,
        use_endpoint_firing=args.use_endpoint_firing,
        update_with="DGR",
        train_mode=False,
    )

    # ----- Alignment reweighting -----
    p_joint = None
    trace = []
    if calibration == "dgr" and cal_dets is not None:
        print(f"Running DGR alignment reweighting ({args.dgr_iterations} iter)...")
        p_calibrated, p_joint, trace = dgr_calibrate_probabilities(
            det_events=cal_dets,
            check_matrix=env.H,
            initial_weights=env.initial_base_weights,
            line_edge_index=env.line_edge_index,
            n_iterations=args.dgr_iterations,
        )
    elif calibration == "spitz" and cal_dets is not None:
        print("Running Spitz alignment reweighting...")
        p_calibrated, p_joint, trace = spitz_calibrate_probabilities(
            det_events=cal_dets,
            dec_edge_list=env.dec_edge_list,
            base_p=env.base_p,
            check_matrix=env.H,
            line_edge_index=env.line_edge_index,
        )
    else:
        print("Skipping calibration: using analytical DEM weights.")
        p_calibrated = env.base_p.astype(np.float32).copy()
    w_calibrated = probs_to_weights(p_calibrated)
    print(f"  edges: {len(w_calibrated)}  "
          f"weight range: [{w_calibrated.min():.3f}, {w_calibrated.max():.3f}]  "
          f"mean: {w_calibrated.mean():.3f}")

    # ----- Decode test set -----
    test_dets = test_gen.test_dets
    test_obs = test_gen.test_obs

    print("Decoding: Standard MWPM (calibrated)...")
    matching_mwpm = pymatching.Matching.from_check_matrix(env.H, weights=w_calibrated)
    pred_static = decode_matching(matching_mwpm, test_dets, env.fault_array, False)

    if p_joint is not None:
        print(f"Decoding: Correlated Matching (manual two-pass, "
              f"n_line_edges={env.line_edge_index.shape[1]})...")
        adj = build_conditional_adjacency(
            p_single=p_calibrated, p_joint=p_joint,
            line_edge_index=env.line_edge_index, n_dec_edges=env.n_dec_edges,
        )
        pred_corr = decode_manual_correlated_matching(
            check_matrix=env.H, w_calibrated=w_calibrated,
            dec_edge_list=env.dec_edge_list, adj=adj,
            syndromes=test_dets, fault_array=env.fault_array,
        )
        corr_method = "manual_calibrated"
    else:
        print("Decoding: Correlated Matching (pymatching enable_correlations=True)...")
        _, dem_decomposed, _ = test_gen.generate_drifted_circuit(env.base_circuit)
        matching_corr = pymatching.Matching.from_detector_error_model(
            dem_decomposed, enable_correlations=True
        )
        pred_corr = decode_matching(matching_corr, test_dets, env.fault_array, True)
        corr_method = "analytical_dem"

    print("Decoding: Neural Correlated Matching...")
    sample_obs, _ = env.reset(seed=0)
    env.current_weights = w_calibrated.copy()
    env.oracle_weights = w_calibrated.copy()
    env.current_matching = pymatching.Matching.from_check_matrix(
        env.H, weights=env.current_weights
    )
    node_dim = sample_obs["node_features"].shape[1]
    agent = SACAgent(
        node_dim=node_dim, hidden_dim=args.hidden_dim,
        static_edge_index=env.line_edge_index, n_layers=args.n_layers,
        lr=1e-4, gamma=0.0, tau=0.005, alpha=args.alpha,
    )
    agent.load_models(args.model_path)
    pred_neural, truths = decode_neural(env, agent, bypass_threshold=args.bypass_threshold)

    # ----- Report -----
    def stats(pred, truth):
        n = len(truth)
        err = int(np.sum(pred != truth))
        ler = err / n if n > 0 else 0.0
        return n, err, ler, _binom_se(ler, n)

    n_s, e_s, l_s, se_s = stats(pred_static, test_obs)
    n_c, e_c, l_c, se_c = stats(pred_corr, test_obs)
    n_n, e_n, l_n, se_n = stats(pred_neural, truths)
    elapsed = time.time() - t_start

    print("\n" + "=" * 72)
    print(f"Results  d={args.distance}  rounds={args.rounds}  basis={basis}  "
          f"center={center}")
    print(f"Calibration shots: {n_cal:,}    Test shots: {n_test:,}")
    print("-" * 72)
    print(f"  Standard MWPM                   errors={e_s:>6d}  LER={l_s:.4e} +/- {se_s:.2e}")
    print(f"  Correlated Matching ({corr_method:<18}) errors={e_c:>6d}  LER={l_c:.4e} +/- {se_c:.2e}")
    print(f"  Neural Correlated Matching      errors={e_n:>6d}  LER={l_n:.4e} +/- {se_n:.2e}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 72)

    # ----- Write CSVs -----
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    base_row = {
        "timestamp": timestamp,
        "dataset": DATASET_NAME,
        "distance": args.distance, "rounds": args.rounds,
        "basis": basis, "patch": center,
        "calibration": calibration,
        "n_calibration_requested": args.n_calibration,
        "dgr_iterations": args.dgr_iterations if calibration == "dgr" else 0,
        "model_path": args.model_path,
        "notes": args.notes,
    }
    full_row = {
        **base_row,
        "n_total": n_total, "n_cal": n_cal, "n_test": n_test,
        "err_mwpm": e_s, "ler_mwpm": l_s, "se_mwpm": se_s,
        "err_corr": e_c, "ler_corr": l_c, "se_corr": se_c, "corr_method": corr_method,
        "err_neural": e_n, "ler_neural": l_n, "se_neural": se_n,
        "elapsed_seconds": elapsed,
    }
    append_results_csv(args.results_csv, full_row)
    append_calibration_trace_csv(args.trace_csv, base_row, trace)
    print(f"Wrote: {args.results_csv}")
    if trace:
        print(f"Wrote: {args.trace_csv}")


if __name__ == "__main__":
    main()
