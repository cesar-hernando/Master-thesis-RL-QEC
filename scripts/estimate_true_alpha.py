#!/usr/bin/env python
"""Rigorous estimate of the *true* trust  alpha = P(e_nu fired | e_nu selected by first-pass MWPM).

This is the quantity Correlated Matching implicitly assumes to be 1 (hard evidence), and that
Pearl-CM treats as a tunable alpha. We estimate it from ground truth, for a range of p, and also
restricted to the shots where the correlated reweight is *pivotal* (it changes the logical decode).

METHOD (see the summary printed at the end of a run, and the module docstring):

  1. Ground truth of which faults fired: Stim DEM sampler with return_errors=True gives, per shot,
     the boolean vector of which DEM error instructions fired.
  2. Map each DEM error -> decoding-graph edge(s): every error decomposes (Tesseract) into <=2-detector
     components; a 2-detector component {a,b} is edge (a,b), a 1-detector component {a} is the boundary
     edge (a, boundary). Building the (n_errors x n_edges) parity matrix M, the true fired-edge support
     of a shot is  true_parity = (errs @ M) mod 2   (mod 2 = matching-relevant parity).
     "edge nu fired"  <=>  true_parity[nu] == 1.
  3. First-pass selection: uncorrelated MWPM on the same graph, decode_batch(enable_correlations=False)
     returns the per-edge boolean selection.  "edge nu selected" <=> that entry is True.
  4. alpha_hat = P(fired | selected) = (# (shot,edge): selected AND fired) / (# (shot,edge): selected),
     pooled and per-edge.
  5. We condition alpha on two nested "the reweight actually did something" shot-sets, and report
     alpha_hat (restricted to first-pass-selected edges on those shots), each with a bootstrap CI:
       - alpha_EDGE_CHANGED : shots where hard CM's *edge selection* differs from the first pass
                              (the reweight re-routed at least one edge; a broad set).
       - alpha_LOGICAL_CHANGED : shots where hard CM's *logical observable* differs from the first pass
                              (the reweight flipped the decode; a strict subset of edge-changed).
     CM's edge selection is read with decode_to_edges_array(enable_correlations=True) and mapped back to
     the shared edge basis via pair_to_idx_matrix.

Edge/detector indexing is shared through one DecodingGraph so M, the first-pass edges, and the fault
vector are all aligned. Detector order matches the DEM sampler (same DEM).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords
from adaptiveQRL.decoding_graph import DecodingGraph


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--p-values", type=str, default="2e-4,4e-4,7e-4,1e-3,2e-3,5e-3,1e-2")
    ap.add_argument("--target-pivotal", type=int, default=300,
                    help="keep sampling a p until this many LOGICAL-changed shots (or max-shots)")
    ap.add_argument("--target-edge-changed", type=int, default=300,
                    help="collect the edge-changed alpha until this many edge-changed shots (per p)")
    ap.add_argument("--max-shots", type=int, default=10000_000_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--min-sel", type=int, default=50, help="min selections for a per-edge estimate")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the pivotal CI")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-csv", type=str, default="data/true_alpha_estimate.csv")
    return ap.parse_args()


def build(p, args):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=args.distance, rounds=args.rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = decompose_errors_for_stim_surface_code_coords(circ.detector_error_model(decompose_errors=False))
    g = DecodingGraph.from_dem(dem)
    fp = pymatching.Matching.from_check_matrix(g.H, weights=np.asarray(g.initial_weights))  # first-pass MWPM (edges)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)         # hard CM (observable)
    P2I = g.pair_to_idx_matrix
    n_edges = g.n_dec_edges

    # error -> edge parity matrix
    M = np.zeros((0, n_edges), dtype=np.int8)
    rows = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps, cur = [[]], None
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append([])
            elif t.is_relative_detector_id():
                comps[-1].append(t.val)
        row = np.zeros(n_edges, dtype=np.int8)
        for comp in comps:
            d = sorted(set(comp))
            e = (P2I[d[0], -1] if len(d) == 1 else (P2I[d[0], d[1]] if len(d) == 2 else -1))
            if e >= 0:
                row[e] ^= 1
        rows.append(row)
    M = np.asarray(rows, dtype=np.int8)
    return dict(circ=circ, dem=dem, g=g, fp=fp, cm=cm, fault=g.fault_array.astype(np.int64),
                M=M, sampler=dem.compile_sampler())


def cm_edge_selection(edge_array, P2I, n_edges, boundary):
    """Map a CM decode_to_edges_array output (k,2 detector-pairs, boundary=-1) to a boolean edge vector."""
    ee = np.where(edge_array == -1, boundary, edge_array)
    lo = ee.min(1); hi = ee.max(1)
    idx = P2I[lo, hi]
    cs = np.zeros(n_edges, dtype=bool)
    v = idx >= 0
    cs[idx[v]] = True
    return cs


def boot_ci(shot_sel, shot_fired, rng, n_boot):
    """95% bootstrap CI for pooled alpha = sum(fired)/sum(sel), resampling whole shots."""
    if not shot_sel:
        return float("nan"), float("nan")
    s = np.concatenate(shot_sel).astype(np.float64)
    f = np.concatenate(shot_fired).astype(np.float64)
    if s.sum() <= 0 or len(s) < 5:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    bs = f[idx].sum(1) / np.maximum(s[idx].sum(1), 1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def main():
    args = parse_args()
    p_values = [float(x) for x in args.p_values.split(",")]
    print("Estimating true alpha = P(e_nu fired | e_nu selected by first-pass MWPM)")
    print(f"d={args.distance} rounds={args.rounds} | p={p_values}")
    print(f"target_pivotal={args.target_pivotal} max_shots={args.max_shots:,} chunk={args.chunk:,}\n")
    rows = []
    t_all = time.time()
    rng = np.random.default_rng(args.seed)

    for p in p_values:
        B = build(p, args)
        M, fault, fp, cm, P2I = B["M"], B["fault"], B["fp"], B["cm"], B["g"].pair_to_idx_matrix
        n_edges = fault.size
        boundary = P2I.shape[0] - 1                   # boundary node index (= n_detectors)
        n_sel = n_sel_fired = 0
        e_sel = np.zeros(n_edges, np.int64)          # per-edge selection count
        e_selfired = np.zeros(n_edges, np.int64)
        # LOGICAL-changed: CM flips the observable (strict subset of edge-changed)
        n_sel_log = n_sel_fired_log = n_log = 0
        log_sel, log_fired = [], []
        # EDGE-changed: CM re-routes at least one edge (broad set); collected until target, then stop
        n_sel_ec = n_sel_fired_ec = n_ec = shots_ec = 0
        ec_sel, ec_fired = [], []
        done_ec = False
        shots = 0
        e_mwpm = e_cm = 0
        t0 = time.time()

        while shots < args.max_shots and n_log < args.target_pivotal:
            det, obs, errs = B["sampler"].sample(shots=args.chunk, return_errors=True)
            obs = obs.reshape(len(det), -1)[:, 0].astype(np.int64)
            true_parity = (np.asarray(sp.csr_matrix(errs.astype(np.int8)) @ M) % 2).astype(bool)
            sel = np.asarray(fp.decode_batch(det, enable_correlations=False), dtype=bool)
            fp_pred = (sel.astype(np.int64) @ fault) % 2
            cm_pred = np.asarray(cm.decode_batch(det, enable_correlations=True), dtype=np.int64)[:, 0]
            sf = sel & true_parity                    # selected AND actually fired

            n_sel += int(sel.sum()); n_sel_fired += int(sf.sum())
            e_sel += sel.sum(0); e_selfired += sf.sum(0)
            e_mwpm += int((fp_pred != obs).sum()); e_cm += int((cm_pred != obs).sum())

            log = cm_pred != fp_pred                   # reweight changed the LOGICAL decode
            if log.any():
                log_sel.append(sel[log].sum(1)); log_fired.append(sf[log].sum(1))
                n_sel_log += int(sel[log].sum()); n_sel_fired_log += int(sf[log].sum())
            n_log += int(log.sum())

            if not done_ec:                            # EDGE-changed: needs CM's per-shot edge set
                ec = np.zeros(len(det), dtype=bool)
                for i in range(len(det)):
                    cs = cm_edge_selection(cm.decode_to_edges_array(det[i], enable_correlations=True),
                                           P2I, n_edges, boundary)
                    ec[i] = np.any(cs != sel[i])
                if ec.any():
                    ec_sel.append(sel[ec].sum(1)); ec_fired.append(sf[ec].sum(1))
                    n_sel_ec += int(sel[ec].sum()); n_sel_fired_ec += int(sf[ec].sum())
                n_ec += int(ec.sum()); shots_ec += len(det)
                if n_ec >= args.target_edge_changed:
                    done_ec = True

            shots += args.chunk

        a_all = n_sel_fired / n_sel
        a_ec = n_sel_fired_ec / n_sel_ec if n_sel_ec else float("nan")
        a_log = n_sel_fired_log / n_sel_log if n_sel_log else float("nan")
        ec_lo, ec_hi = boot_ci(ec_sel, ec_fired, rng, args.boot)
        log_lo, log_hi = boot_ci(log_sel, log_fired, rng, args.boot)
        # per-edge distribution (edges with enough selections)
        m = e_sel >= args.min_sel
        per_edge = e_selfired[m] / e_sel[m]

        print(f"p={p:<7g} shots={shots:>11,} | "
              f"alpha_ALL={a_all:.4f} (n_sel={n_sel:,}) | "
              f"alpha_EDGE_CHANGED={a_ec:.4f} [{ec_lo:.3f},{ec_hi:.3f}] "
              f"(shots={n_ec}, frac={n_ec/max(shots_ec,1):.2e}) | "
              f"alpha_LOGICAL_CHANGED={a_log:.4f} [{log_lo:.3f},{log_hi:.3f}] "
              f"(shots={n_log}, frac={n_log/shots:.2e}) | "
              f"per-edge median={np.median(per_edge):.3f} min={per_edge.min():.3f} | "
              f"[sanity LER: MWPM={e_mwpm/shots:.2e} CM={e_cm/shots:.2e}] | {time.time()-t0:.0f}s", flush=True)

        rows.append(dict(p=p, shots=shots,
                         alpha_all=a_all, n_sel_all=n_sel,
                         alpha_edge_changed=a_ec, alpha_ec_lo=ec_lo, alpha_ec_hi=ec_hi,
                         edge_changed_shots=n_ec, edge_changed_frac=n_ec / max(shots_ec, 1),
                         edge_changed_denom_shots=shots_ec, n_sel_edge_changed=n_sel_ec,
                         alpha_logical_changed=a_log, alpha_lc_lo=log_lo, alpha_lc_hi=log_hi,
                         logical_changed_shots=n_log, logical_changed_frac=n_log / shots,
                         n_sel_logical_changed=n_sel_log,
                         per_edge_median=float(np.median(per_edge)), per_edge_min=float(per_edge.min()),
                         mwpm_ler=e_mwpm / shots, cm_ler=e_cm / shots))
        with open(os.path.join(ROOT, args.out_csv), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\nsaved {args.out_csv} | total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
