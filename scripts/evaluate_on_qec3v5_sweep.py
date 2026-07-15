"""
evaluate_on_qec3v5_sweep.py
---------------------------

Single-invocation sweep across the qec3v5 dataset:

    for distance in {3, 5}:
        for calibration in {none, spitz, dgr}:
            evaluate Standard MWPM + Correlated Matching + Neural CM

Produces three families of plots:

  1. **Calibration convergence**  (DGR + Spitz only)
     Per iteration, how the calibrated weights and joint probabilities evolve,
     and how the held-out validation LER tracks. Uses a small slice of the
     calibration shots as the held-out validator -- the calibration only ever
     reads detector firings, never observables, so there's no leakage.

  2. **Decoder comparison**
     Grouped bar chart of LER per decoder, per calibration method, faceted
     by distance. Tells you whether DGR/Spitz buys anything over no-calibration
     for each decoder on the qec3v5 hardware.

  3. **CM vs NCM disagreement by n_flashes**
     For each (distance, calibration), bar chart of NCM rescues vs regressions
     bucketed by syndrome weight. Same diagnostic as the strategy-analyzer and
     decoders-comparison notebook, but applied to real device shots.

This script imports the heavy lifting (ExperimentalDataGenerator, the two
calibration functions, the three decoders) from evaluate_on_qec3v5.py and only
adds the sweep orchestration and the new plots.

Usage:
    python scripts/evaluate_on_qec3v5_sweep.py \\
        --data-root google_qec3v5_experiment_data \\
        --rounds 5 --basis Z \\
        --n-calibration 25000 --dgr-iterations 3 \\
        --model-path-d5 models/qec_graph_optuna_run_d5_trial_0000_best.pth \\
        --out-dir plots/qec3v5_sweep/

Memory: per distance we hold (n_total, n_detectors) bool shots ~50000*detectors
bytes. For d=5/r=5/center_5_5 that's ~6 MB. Per-shot predictions: ~50k bytes.
Easy.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import seaborn as sns

from NeuralCM.drifted_matching_env import DriftedMatchingEnv
from NeuralCM.gnn_sac_agent import SACAgent

# Reuse everything from the single-eval script.
from evaluate_on_qec3v5 import (
    AVAILABLE_ROUNDS,
    DATASET_NAME,
    DEFAULT_CENTER,
    ExperimentalDataGenerator,
    _compute_p_joint_from_decodes,
    build_conditional_adjacency,
    decode_manual_correlated_matching,
    decode_matching,
    decode_neural,
    dgr_calibrate_probabilities,
    find_experiment_dir,
    probs_to_weights,
    spitz_calibrate_probabilities,
)


CALIBRATIONS = ("none", "spitz", "dgr")
DISTANCES    = [5]


# =============================================================================
# Per-cell evaluation: load data once per distance, then sweep calibrations
# =============================================================================

def _build_env_for_distance(args, distance):
    """Locate data, instantiate generator + DriftedMatchingEnv, and split into
    calibration / validation / test slices. Returns a dict of arrays + the env.
    """
    center = args.center or DEFAULT_CENTER[distance]
    exp_dir = find_experiment_dir(args.data_root, distance, args.rounds, args.basis, center)
    circuit_path = os.path.join(exp_dir, "circuit_noisy.stim")
    dets_path    = os.path.join(exp_dir, "detection_events.b8")
    obs_path     = os.path.join(exp_dir, "obs_flips_actual.01")

    print(f"\n=== distance={distance}  center={center}  rounds={args.rounds}  basis={args.basis} ===")
    print(f"  circuit : {circuit_path}")
    print(f"  dets    : {dets_path}")
    print(f"  obs     : {obs_path}")

    full = ExperimentalDataGenerator(
        circuit_path=circuit_path, det_events_path=dets_path, obs_flips_path=obs_path,
        distance=distance, n_rounds=args.rounds,
    )
    n_total = len(full.test_obs)
    n_cal   = min(args.n_calibration, n_total // 2)
    n_val   = min(args.n_val, n_cal // 4)   # validation slice carved off the END of the cal pool
    n_test  = n_total - n_cal
    cal_dets = full._dets_all[:n_cal - n_val]
    val_dets = full._dets_all[n_cal - n_val:n_cal]
    val_obs  = full._obs_all [n_cal - n_val:n_cal]
    print(f"  shots total={n_total:,}  cal={len(cal_dets):,}  val={n_val:,}  test={n_test:,}")

    test_gen = ExperimentalDataGenerator(
        circuit_path=circuit_path, det_events_path=dets_path, obs_flips_path=obs_path,
        distance=distance, n_rounds=args.rounds,
        test_slice=slice(n_cal, n_total),
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

    return {
        "distance": distance,
        "center": center,
        "n_total": n_total,
        "n_cal":   len(cal_dets),
        "n_val":   n_val,
        "n_test":  n_test,
        "cal_dets": cal_dets,
        "val_dets": val_dets,
        "val_obs":  val_obs,
        "test_dets": test_gen.test_dets,
        "test_obs":  test_gen.test_obs,
        "env": env,
        "test_gen": test_gen,
    }


def _run_one(distance, calibration, env_pack, model_path, args) -> Dict:
    """Run one (distance, calibration) cell of the sweep."""
    env       = env_pack["env"]
    cal_dets  = env_pack["cal_dets"]
    val_dets  = env_pack["val_dets"]
    val_obs   = env_pack["val_obs"]
    test_dets = env_pack["test_dets"]
    test_obs  = env_pack["test_obs"]

    print(f"\n--- distance={distance}  calibration={calibration} ---")
    t0 = time.time()

    # Calibration -----------------------------------------------------------
    if calibration == "dgr":
        p_calibrated, p_joint, trace = dgr_calibrate_probabilities(
            det_events=cal_dets,
            check_matrix=env.H,
            initial_weights=env.initial_base_weights,
            line_edge_index=env.line_edge_index,
            n_iterations=args.dgr_iterations,
            val_dets=val_dets, val_obs=val_obs, fault_array=env.fault_array,
        )
    elif calibration == "spitz":
        p_calibrated, p_joint, trace = spitz_calibrate_probabilities(
            det_events=cal_dets,
            dec_edge_list=env.dec_edge_list,
            base_p=env.base_p,
            check_matrix=env.H,
            line_edge_index=env.line_edge_index,
            val_dets=val_dets, val_obs=val_obs, fault_array=env.fault_array,
        )
    else:
        p_calibrated = env.base_p.astype(np.float32).copy()
        p_joint = None
        trace = []
    w_calibrated = probs_to_weights(p_calibrated)

    # Standard MWPM ---------------------------------------------------------
    matching_mwpm = pymatching.Matching.from_check_matrix(env.H, weights=w_calibrated)
    pred_mwpm = decode_matching(matching_mwpm, test_dets, env.fault_array, False)

    # Correlated Matching ---------------------------------------------------
    if p_joint is not None:
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
        _, dem_decomposed, _ = env_pack["test_gen"].generate_drifted_circuit(env.base_circuit)
        matching_corr = pymatching.Matching.from_detector_error_model(
            dem_decomposed, enable_correlations=True,
        )
        pred_corr = decode_matching(matching_corr, test_dets, env.fault_array, True)
        corr_method = "analytical_dem"

    # Neural Correlated Matching -------------------------------------------
    sample_obs, _ = env.reset(seed=0)
    env.current_weights  = w_calibrated.copy()
    env.oracle_weights   = w_calibrated.copy()
    env.current_matching = pymatching.Matching.from_check_matrix(env.H, weights=env.current_weights)

    agent = SACAgent(
        node_dim=sample_obs["node_features"].shape[1],
        hidden_dim=args.hidden_dim, static_edge_index=env.line_edge_index,
        n_layers=args.n_layers, lr=1e-4, gamma=0.0, tau=0.005, alpha=args.alpha,
    )
    agent.load_models(model_path)
    pred_ncm, truths = decode_neural(env, agent, bypass_threshold=args.bypass_threshold)

    # Stats / artefacts -----------------------------------------------------
    def stats(pred, truth):
        n = len(truth)
        err = int(np.sum(pred != truth))
        ler = err / n if n > 0 else 0.0
        se  = float(np.sqrt(ler * (1 - ler) / n)) if n > 0 else 0.0
        return n, err, ler, se

    n_t, err_mwpm,   ler_mwpm,   se_mwpm   = stats(pred_mwpm,  test_obs)
    _,   err_corr,   ler_corr,   se_corr   = stats(pred_corr,  test_obs)
    _,   err_neural, ler_neural, se_neural = stats(pred_ncm,   truths)
    elapsed = time.time() - t0

    print(f"  MWPM   : {err_mwpm:>6,d} / {n_t:,}   LER {ler_mwpm:.4e} +/- {se_mwpm:.2e}")
    print(f"  Corr   : {err_corr:>6,d}            LER {ler_corr:.4e} +/- {se_corr:.2e}  ({corr_method})")
    print(f"  Neural : {err_neural:>6,d}            LER {ler_neural:.4e} +/- {se_neural:.2e}")
    print(f"  elapsed {elapsed:.1f}s")

    # Number of flashes per test shot (for disagreement plot)
    n_flashes_test = test_dets.astype(np.int64).sum(axis=1)

    return {
        "distance":     distance,
        "calibration":  calibration,
        "trace":        trace,
        "corr_method":  corr_method,
        "pred_mwpm":    pred_mwpm,
        "pred_corr":    pred_corr,
        "pred_ncm":     pred_ncm,
        "test_obs":     test_obs,
        "truths_ncm":   truths,
        "n_flashes":    n_flashes_test,
        "n_test":       n_t,
        "err_mwpm":     err_mwpm,   "ler_mwpm":   ler_mwpm,   "se_mwpm":   se_mwpm,
        "err_corr":     err_corr,   "ler_corr":   ler_corr,   "se_corr":   se_corr,
        "err_neural":   err_neural, "ler_neural": ler_neural, "se_neural": se_neural,
        "elapsed":      elapsed,
    }


# =============================================================================
# Plot 1: calibration convergence
# =============================================================================

def plot_calibration_convergence(results: Dict[Tuple[int, str], dict], out_path: str):
    """4 subplots: mean weight, std weight, mean joint probability, val LER."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("mean_weight",   axes[0, 0], "mean edge weight"),
        ("std_weight",    axes[0, 1], "std of edge weights"),
        ("mean_p_joint",  axes[1, 0], "mean joint probability"),
        ("val_ler",       axes[1, 1], "validation LER (held-out cal slice)"),
    ]
    # Colour distinguishes the calibration method; line style distinguishes
    # the code distance. A single point marker ('.') is used throughout.
    color = {"dgr": "tab:blue", "spitz": "tab:red"}
    style = {3: "-", 5: "--"}

    for key, ax, title in panels:
        for (d, cal), res in results.items():
            if cal not in ("dgr", "spitz"):
                continue
            trace = res["trace"]
            if not trace or key not in trace[0]:
                continue
            xs = [r["iteration"] for r in trace]
            ys = [r[key]         for r in trace]
            ax.plot(xs, ys,
                    linestyle=style.get(d, "-"), marker=".",
                    color=color[cal], linewidth=1.5,
                    label=f"d={d}, {cal}")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        if key == "val_ler":
            ax.set_yscale("log")

    fig.suptitle("Calibration convergence per iteration "
                 "(DGR: multiple iters, Spitz: single)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[*] Wrote {out_path}")


# =============================================================================
# Plot 2: decoder comparison across calibrations and distances
# =============================================================================

def plot_decoder_comparison(results: Dict[Tuple[int, str], dict], out_path: str):
    """Two panels (one per distance). Grouped bar chart: x = decoder,
    bars colored by calibration method."""
    decoders = ["MWPM", "Correlated Matching", "Neural Correlated Matching"]
    key_ler  = {"MWPM": "ler_mwpm", "Correlated Matching": "ler_corr",
                "Neural Correlated Matching": "ler_neural"}
    key_se   = {"MWPM": "se_mwpm",  "Correlated Matching": "se_corr",
                "Neural Correlated Matching": "se_neural"}
    cal_colors = {"none": "tab:gray", "spitz": "tab:blue", "dgr": "tab:orange"}

    distances = sorted({d for d, _ in results.keys()})
    fig, axes = plt.subplots(1, len(distances), figsize=(6 * len(distances), 5), sharey=True)
    if len(distances) == 1:
        axes = [axes]

    for ax, d in zip(axes, distances):
        x = np.arange(len(decoders))
        width = 0.27
        for i, cal in enumerate(CALIBRATIONS):
            res = results.get((d, cal))
            if res is None:
                continue
            lers = [res[key_ler[dec]] for dec in decoders]
            ses  = [res[key_se [dec]] for dec in decoders]
            offset = (i - 1) * width
            ax.bar(x + offset, lers, width=width, yerr=ses, capsize=3,
                   color=cal_colors[cal], label=cal, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(decoders, rotation=15, ha="right")
        ax.set_title(f"d = {d}")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
        if d == distances[0]:
            ax.set_ylabel("Logical Error Rate (LER)")
        ax.legend(title="calibration", fontsize=8)

    fig.suptitle("Decoder LER on qec3v5 test set, across calibration methods and distances",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[*] Wrote {out_path}")


# =============================================================================
# Plot 3: disagreement between CM and NCM by n_flashes
# =============================================================================

def plot_disagreement_by_nflashes(results: Dict[Tuple[int, str], dict],
                                  out_path: str, max_nf: int = 25):
    """One sub-plot per (distance, calibration). Bars show NCM rescues (CM wrong,
    NCM right) and NCM regressions (CM right, NCM wrong) bucketed by syndrome
    weight."""
    distances = sorted({d for d, _ in results.keys()})
    n_d = len(distances)
    n_c = len(CALIBRATIONS)

    fig, axes = plt.subplots(n_d, n_c, figsize=(4.2 * n_c, 3.2 * n_d),
                             sharex=True, sharey=True, squeeze=False)
    nf_axis = np.arange(max_nf + 1)
    for r, d in enumerate(distances):
        for c, cal in enumerate(CALIBRATIONS):
            ax = axes[r, c]
            res = results.get((d, cal))
            if res is None:
                ax.set_visible(False); continue

            obs = res["test_obs"]
            cm_correct  = (res["pred_corr"] == obs)
            ncm_correct = (res["pred_ncm"]  == res["truths_ncm"])
            # NCM and CM may have ordered shots identically (both came from
            # the same test slice), so cm_correct and ncm_correct align by index.
            nf = np.clip(res["n_flashes"], 0, max_nf)
            rescues_per_nf     = np.zeros(max_nf + 1, dtype=np.int64)
            regressions_per_nf = np.zeros(max_nf + 1, dtype=np.int64)
            rescue_mask  = (~cm_correct) &  ncm_correct
            regress_mask =  cm_correct & ~ncm_correct
            for v in range(max_nf + 1):
                bin_mask = (nf == v)
                rescues_per_nf[v]     = int(np.sum(rescue_mask  & bin_mask))
                regressions_per_nf[v] = int(np.sum(regress_mask & bin_mask))

            width = 0.4
            ax.bar(nf_axis - width/2, rescues_per_nf,     width=width,
                   color="seagreen",
                   label=f"rescues ({int(rescues_per_nf.sum())})")
            ax.bar(nf_axis + width/2, regressions_per_nf, width=width,
                   color="indianred",
                   label=f"regressions ({int(regressions_per_nf.sum())})")
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(f"d={d}  cal={cal}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            if r == n_d - 1:
                ax.set_xlabel("n_flashes")
            if c == 0:
                ax.set_ylabel("count (symlog)")

    fig.suptitle("CM vs NCM disagreement bucketed by syndrome weight\n"
                 "(rescues: CM wrong, NCM right     regressions: CM right, NCM wrong)",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[*] Wrote {out_path}")


# =============================================================================
# CSV summary
# =============================================================================

def write_summary_csv(results: Dict[Tuple[int, str], dict], csv_path: str, args):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = ["timestamp", "distance", "calibration", "n_test",
              "err_mwpm",  "ler_mwpm",  "se_mwpm",
              "err_corr",  "ler_corr",  "se_corr", "corr_method",
              "err_neural","ler_neural","se_neural",
              "elapsed_seconds"]
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (d, cal), res in sorted(results.items()):
            w.writerow({
                "timestamp": stamp,
                "distance":  d, "calibration": cal, "n_test": res["n_test"],
                "err_mwpm":  res["err_mwpm"],  "ler_mwpm":  res["ler_mwpm"],  "se_mwpm":  res["se_mwpm"],
                "err_corr":  res["err_corr"],  "ler_corr":  res["ler_corr"],  "se_corr":  res["se_corr"],
                "corr_method": res["corr_method"],
                "err_neural":res["err_neural"],"ler_neural":res["ler_neural"],"se_neural":res["se_neural"],
                "elapsed_seconds": res["elapsed"],
            })
    print(f"[*] Wrote {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Sweep + analysis on Google qec3v5.")
    p.add_argument("--data-root", default="google_qec3v5_experiment_data")
    p.add_argument("--rounds", type=int, default=5,
                   help=f"One of {AVAILABLE_ROUNDS}.")
    p.add_argument("--basis", choices=["x", "z", "X", "Z"], default="Z")
    p.add_argument("--center", default=None,
                   help="Patch center id, e.g. '5_5'. Defaults to DEFAULT_CENTER per distance.")
    p.add_argument("--distances", type=int, default=DISTANCES,
                   help=f"Distances to sweep over. Default: {DISTANCES}.")

    # Calibration
    p.add_argument("--n-calibration", type=int, default=30_000)
    p.add_argument("--n-val",         type=int, default=2_000,
                   help="Held-out slice (taken from the end of the cal pool) used for "
                        "the per-iteration validation LER plot.")
    p.add_argument("--dgr-iterations", type=int, default=300)

    # SAC-GNN is graph-agnostic (GCNConv + MLP heads work on any line graph),
    # so one checkpoint is reused for every distance in the sweep.
    p.add_argument("--model-path", default="models/qec_graph_optuna_run_d5_trial_0000_best.pth")
    p.add_argument("--hidden-dim",        type=int,   default=256)
    p.add_argument("--n-layers",          type=int,   default=1)
    p.add_argument("--alpha",             type=float, default=0.01)
    p.add_argument("--action-scale",      type=float, default=5.0)
    p.add_argument("--bypass-threshold",  type=int,   default=2)
    p.add_argument("--local-action-hops", type=int,   default=1)
    p.add_argument("--use-endpoint-firing", action=argparse.BooleanOptionalAction, default=False)

    # Output
    p.add_argument("--out-dir", default="plots/qec3v5_sweep/")
    p.add_argument("--csv",     default="data/experimental/qec3v5_sweep_results.csv")

    args = p.parse_args()
    args.basis = args.basis.upper()
    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    if args.rounds not in AVAILABLE_ROUNDS:
        raise SystemExit(f"--rounds={args.rounds} not available. Pick one of {AVAILABLE_ROUNDS}.")

    # Run the sweep. The same checkpoint is fed to every (distance, calibration)
    # cell because the GNN is graph-agnostic.
    results: Dict[Tuple[int, str], dict] = {}
    for distance in args.distances:
        env_pack = _build_env_for_distance(args, distance)
        for cal in CALIBRATIONS:
            results[(distance, cal)] = _run_one(
                distance=distance, calibration=cal, env_pack=env_pack,
                model_path=args.model_path, args=args,
            )

    # Persist and plot.
    write_summary_csv(results, args.csv, args)
    plot_calibration_convergence(results, os.path.join(args.out_dir, "calibration_convergence.png"))
    plot_decoder_comparison    (results, os.path.join(args.out_dir, "decoder_comparison.png"))
    plot_disagreement_by_nflashes(results, os.path.join(args.out_dir, "cm_vs_ncm_disagreement.png"))

    print("\n[*] Sweep complete.")


if __name__ == "__main__":
    main()
