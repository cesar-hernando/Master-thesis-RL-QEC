"""
analyze_neural_vs_correlated_policy.py
======================================

Interpret the policy learned by a trained SAC-GNN ("Neural Correlated Matching",
NCM) and compare it, shot for shot, against the *analytical* conditional-probability
policy used by the Correlated Matching (CM) decoder.

The CM policy is the one implemented in
``DriftedMatchingEnv.compute_analytical_correlated_matching_action``: for an
unselected edge A that neighbours a first-pass-selected edge B in the DEM line
graph, discount A's weight to the conditional log-likelihood

        w_A|B = log( (1 - p_A|B) / p_A|B ),   p_A|B = P(A & B) / P(B)
                                                     = corr_tracer[AB] / occ_tracer[B]

taking the largest discount (min weight) over all selected neighbours B, and
never *penalising* (the new weight is capped at the current weight). NCM is the
trained GNN producing a per-edge reweight in action_scale * [-1, 1].

Both decoders sit on top of the SAME static first-pass MWPM graph (the base DEM
graph), so the comparison isolates *what each policy does with the correlation
structure* — exactly the "neural vs analytical correlated matching" question.

What it produces
----------------
Reusing the streaming histograms + plots of ``strategy_analyzer_v2`` (p1-p8):
  p1  NCM-vs-CM disagreement by syndrome weight (rescues vs regressions)
  p2  action (weight-discount) distributions, split by is_selected AND by whether
      the second pass changed the logical answer (fixed / broken / unchanged)
  p3  joint distribution of NCM action vs CM action per edge
  p4  mean action vs each input feature (base weight, max Pearson corr to a
      selected neighbour, weight of highest-corr neighbour, # selected neighbours),
      GNN curve overlaid on the analytical CM curve
  p5  3x3 outcome confusion (NCM rows x CM cols), incl. "unchanged"
  p6  outcome confusion bucketed by n_flashes
  p7  2D reweight vs max Pearson corr to a selected neighbour
  p8  2D reweight vs partner (selected neighbour) weight

Plus one analysis v2 lacks, answering "performance vs number of syndromes fired":
  p9  per-decoder logical error rate (MWPM first pass / CM / NCM) as a function of
      n_flashes, with shot counts.

Design for scale (default 50M shots)
------------------------------------
* Syndromes are simulated and decoded in CHUNKS so peak RAM stays bounded.
* First-pass MWPM is batched (decode_batch).
* The GNN actor runs in batched sub-calls (no per-shot forward).
* The analytical CM action and all per-edge input features are computed
  VECTORISED across the active shots of a chunk via the line-graph edge list.
* Only the unavoidable second-pass MWPM (one decode per active shot per decoder,
  skipped when the proposed delta is all-zero) stays a per-shot loop.

Usage
-----
    # full run (collect + analyze) at the chosen deployment regime
    python scripts/analyze_neural_vs_correlated_policy.py --mode both \
        --shots 50000000 --chunk 100000 \
        --p 0.001 --mismatch 1.0 --distance 5 --n-rounds 5 \
        --model-path models/hyperparam_tun_v2/slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.pth \
        --out-prefix data/ncm_vs_cm_trial40_p1e-3 \
        --plot-dir plots/ncm_vs_cm_trial40_p1e-3/

    # re-make plots only (after a collect run)
    python scripts/analyze_neural_vs_correlated_policy.py --mode analyze \
        --out-prefix data/ncm_vs_cm_trial40_p1e-3 \
        --plot-dir plots/ncm_vs_cm_trial40_p1e-3/
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from adaptiveQRL.drifted_matching_env import DriftedMatchingEnv
from adaptiveQRL.gnn_sac_agent import SACAgent
from adaptiveQRL.syndrome_data_generation import SyndromeDataGenerator

# Reuse the tested accumulator + plotting from strategy_analyzer_v2.
from strategy_analyzer_v2 import (
    N_FLASHES_MAX,
    StrategyAccumulator,
    run_analysis,
)


# ============================================================================
# Static topology helpers
# ============================================================================

def precompute_cm_constants(env):
    """Per line-graph edge, the analytical CM target weight each endpoint would
    receive if its *other* endpoint were the selected partner.

    Returns dict of float arrays indexed by line-edge id:
        w_src_given_dst[i] : new weight for node src[i] if dst[i] is selected
        w_dst_given_src[i] : new weight for node dst[i] if src[i] is selected
    plus the per-line-edge Pearson correlation (the GNN's edge feature).
    """
    src = env.line_edge_index[0]
    dst = env.line_edge_index[1]
    joint = env.initial_corr_tracer.astype(np.float64)      # P(A & B)
    occ = env.base_p.astype(np.float64)                     # P(edge)

    def cond_weight(partner_idx):
        p = joint / np.clip(occ[partner_idx], 1e-12, None)  # P(cand | partner)
        p = np.clip(p, 1e-6, 0.499999)
        return np.log((1.0 - p) / p)

    return {
        "src": src,
        "dst": dst,
        "pearson": env.initial_pearson_corr.astype(np.float64),
        "w_src_given_dst": cond_weight(dst),    # candidate = src, partner = dst
        "w_dst_given_src": cond_weight(src),    # candidate = dst, partner = src
    }


def precompute_node_static_features(env):
    """Per decoding-graph edge (GNN node): weight of the neighbour with the
    highest Pearson correlation (regardless of selection). Static across shots."""
    n = env.n_dec_edges
    src = env.line_edge_index[0]
    dst = env.line_edge_index[1]
    pearson = env.initial_pearson_corr
    w = env.current_weights

    best_corr = np.full(n, -1.0, dtype=np.float64)
    hc_nbr_w = np.zeros(n, dtype=np.float64)
    for i in range(env.n_line_edges):
        a, b, c = int(src[i]), int(dst[i]), float(pearson[i])
        if c > best_corr[a]:
            best_corr[a] = c
            hc_nbr_w[a] = w[b]
        if c > best_corr[b]:
            best_corr[b] = c
            hc_nbr_w[b] = w[a]
    return hc_nbr_w


def compute_action_masks(sel, k_hop_int):
    """Vectorised replica of DriftedMatchingEnv._compute_action_mask for a batch
    of first-pass selections.

    sel        : (B, N) bool  -- first-pass selected edges
    k_hop_int  : (N, N) int8  -- k-hop adjacency incl. self loops
    Returns (B, N) bool action mask matching the env exactly (including the
    "drop isolated selected edges" rule).
    """
    nbr_count_sel = sel.astype(np.int32) @ k_hop_int            # (B, N)
    mask = nbr_count_sel > 0
    isolated = sel & (nbr_count_sel == 1)                       # selected w/ no selected k-hop neighbour
    mask[isolated] = False
    return mask


# ============================================================================
# Vectorised per-active-shot features + analytical CM weights
# ============================================================================

def compute_cm_and_features(sel, cmc, current_weights, hc_nbr_w):
    """Vectorised over the active shots of a chunk.

    sel             : (A, N) bool  first-pass selections for the A active shots
    cmc             : dict from precompute_cm_constants
    current_weights : (N,) float   base graph weights
    hc_nbr_w        : (N,) float   static highest-corr-neighbour weight

    Returns:
        cm_new_w     (A, N) float   analytical CM target weight per node (min over
                                    selected partners, capped at current weight)
        max_corr_sel (A, N) float   max Pearson corr to a *selected* neighbour
        w_hc_sel     (A, N) float   weight of the selected neighbour with max corr
        num_sel_nbr  (A, N) int32   number of selected neighbours
    """
    A, N = sel.shape
    cm_new_w = np.broadcast_to(current_weights, (A, N)).astype(np.float64).copy()
    max_corr_sel = np.full((A, N), -1.0, dtype=np.float64)
    w_hc_sel = np.zeros((A, N), dtype=np.float64)
    num_sel_nbr = np.zeros((A, N), dtype=np.int32)

    src, dst = cmc["src"], cmc["dst"]
    pear = cmc["pearson"]
    w_s_d = cmc["w_src_given_dst"]
    w_d_s = cmc["w_dst_given_src"]
    cur = current_weights

    n_line = len(src)
    for i in range(n_line):
        a, b = int(src[i]), int(dst[i])
        c = pear[i]
        sel_a = sel[:, a]          # is endpoint a selected? (A,)
        sel_b = sel[:, b]

        # candidate = a, partner = b (b selected)
        if w_s_d[i] < cur[a]:      # CM only discounts; skip non-discounting partners
            rows = np.flatnonzero(sel_b)
            if rows.size:
                # Advanced indexing (array rows, scalar a) returns a copy, so an
                # `out=cm_new_w[rows, a]` would write to a throwaway and leave
                # cm_new_w untouched. Assign back explicitly instead.
                cm_new_w[rows, a] = np.minimum(cm_new_w[rows, a], w_s_d[i])
        # feature bookkeeping for candidate = a (any selected partner b)
        rows_b = np.flatnonzero(sel_b)
        if rows_b.size:
            num_sel_nbr[rows_b, a] += 1
            better = c > max_corr_sel[rows_b, a]
            br = rows_b[better]
            max_corr_sel[br, a] = c
            w_hc_sel[br, a] = cur[b]

        # candidate = b, partner = a (a selected)
        if w_d_s[i] < cur[b]:
            rows = np.flatnonzero(sel_a)
            if rows.size:
                cm_new_w[rows, b] = np.minimum(cm_new_w[rows, b], w_d_s[i])
        rows_a = np.flatnonzero(sel_a)
        if rows_a.size:
            num_sel_nbr[rows_a, b] += 1
            better = c > max_corr_sel[rows_a, b]
            br = rows_a[better]
            max_corr_sel[br, b] = c
            w_hc_sel[br, b] = cur[a]

    max_corr_sel[max_corr_sel < 0] = 0.0
    return cm_new_w, max_corr_sel, w_hc_sel, num_sel_nbr


# ============================================================================
# Vectorised feed into a StrategyAccumulator (per chunk, no per-shot Python).
# ============================================================================

def _outcome_vec(p1, p2):
    """0=fixed (p1 wrong, p2 right), 1=broken (p1 right, p2 wrong), 2=unchanged."""
    return np.where(~p1 & p2, 0, np.where(p1 & ~p2, 1, 2)).astype(np.int64)


def feed_shot_level(acc, nf_clip, mwpm_ok, cm_ok, ncm_ok, n_active):
    """Update per-shot histograms for every shot in the chunk (bypass + active)."""
    acc.total_shots += int(nf_clip.size)
    acc.active_shots += int(n_active)
    acc.bypass_shots += int(nf_clip.size - n_active)

    # confusion_by_nf axes: [nf, cm_correct, ncm_correct]
    np.add.at(acc.confusion_by_nf, (nf_clip, cm_ok.astype(np.int64), ncm_ok.astype(np.int64)), 1)
    # outcome_by_nf axes: [nf, gnn_outcome, cm_outcome] vs first pass
    gnn_o = _outcome_vec(mwpm_ok, ncm_ok)
    cm_o = _outcome_vec(mwpm_ok, cm_ok)
    np.add.at(acc.outcome_by_nf, (nf_clip, gnn_o, cm_o), 1)


def feed_active_edges(acc, mask, sel, gnn_action, cm_action,
                      current_weights, hc_nbr_w, max_corr_sel, w_hc_sel,
                      num_sel_nbr, gnn_o_act, cm_o_act):
    """Vectorised replica of StrategyAccumulator.record_active_edges applied to
    ALL active (shot, node) pairs of a chunk at once."""
    rows, cols = np.nonzero(mask)               # active (shot, node) pairs
    if rows.size == 0:
        return
    s_val = sel[rows, cols].astype(np.int64)    # is_selected per edge
    gnn_a = gnn_action[rows, cols]
    cm_a = cm_action[rows, cols]
    gnn_o = gnn_o_act[rows]
    cm_o = cm_o_act[rows]

    gnn_bin = np.clip(np.digitize(gnn_a, acc.action_edges) - 1, 0, acc.gnn_action_hist.shape[-1] - 1)
    cm_bin = np.clip(np.digitize(cm_a, acc.action_edges) - 1, 0, acc.cm_action_hist.shape[-1] - 1)

    # 1D action histograms split by is_selected and shot-level outcome
    np.add.at(acc.gnn_action_hist, (s_val, gnn_o, gnn_bin), 1)
    np.add.at(acc.cm_action_hist, (s_val, cm_o, cm_bin), 1)
    # 2D joint GNN-vs-CM
    np.add.at(acc.gnn_vs_cm_hist, (s_val, cm_bin, gnn_bin), 1)

    # per-edge feature values
    feats = {
        "base_weight": current_weights[cols],
        "max_corr_to_selected": max_corr_sel[rows, cols],
        "weight_of_highest_corr_neighbor": hc_nbr_w[cols],
        "weight_of_highest_corr_selected_neighbor": w_hc_sel[rows, cols],
        "num_selected_neighbors": num_sel_nbr[rows, cols].astype(np.float64),
    }
    # mean-intensity accumulators
    for fname, vals in feats.items():
        if fname not in acc.feat_edges:
            continue
        edges = acc.feat_edges[fname]
        nbins = len(edges) - 1
        fbin = np.clip(np.digitize(vals, edges) - 1, 0, nbins - 1)
        store = acc.feature_action[fname]
        np.add.at(store["gnn_sum"], fbin, gnn_a)
        np.add.at(store["gnn_count"], fbin, 1)
        np.add.at(store["cm_sum"], fbin, cm_a)
        np.add.at(store["cm_count"], fbin, 1)

    # 2D feature-vs-action histograms for the headline-comparison features
    for fname in acc.feat_action_2d:
        edges = acc.feat_edges[fname]
        nbins_f = len(edges) - 1
        fbin = np.clip(np.digitize(feats[fname], edges) - 1, 0, nbins_f - 1)
        store2d = acc.feat_action_2d[fname]
        np.add.at(store2d["gnn"], (s_val, fbin, gnn_bin), 1)
        np.add.at(store2d["cm"], (s_val, fbin, cm_bin), 1)


# ============================================================================
# GNN actor, batched.
# ============================================================================

class BatchedActor:
    def __init__(self, agent, env, device, actor_batch_size=512):
        self.agent = agent
        self.env = env
        self.device = device
        self.bs = actor_batch_size
        self.edge_index_t = torch.from_numpy(env.line_edge_index).long()
        self.edge_attr_t = torch.from_numpy(
            env.initial_pearson_corr.astype(np.float32).reshape(-1, 1)
        )

    def __call__(self, feats, masks):
        """feats: (A, N, D) float32, masks: (A, N) bool -> (A, N) float32 actions."""
        n = feats.shape[0]
        out = np.empty((n, self.env.n_dec_edges), dtype=np.float32)
        for start in range(0, n, self.bs):
            end = min(start + self.bs, n)
            data = [
                Data(
                    x=torch.from_numpy(feats[i]),
                    edge_index=self.edge_index_t,
                    edge_attr=self.edge_attr_t,
                    action_mask=torch.from_numpy(masks[i]).float().unsqueeze(-1),
                )
                for i in range(start, end)
            ]
            batch = Batch.from_data_list(data).to(self.device, non_blocking=True)
            with torch.no_grad():
                actions, _ = self.agent.actor(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.action_mask, evaluate=True,
                )
            out[start:end] = actions.cpu().numpy().reshape(end - start, self.env.n_dec_edges)
        return out


# ============================================================================
# Collection
# ============================================================================

def run_collection(args):
    print("[*] Building environment / decoder topology...")
    generator = SyndromeDataGenerator(
        distance=args.distance, n_rounds=args.n_rounds, mismatch=args.mismatch,
        noise_model={
            "version": "built-in",
            "after_clifford_depolarization": args.p,
            "before_measure_flip_probability": args.p,
            "after_reset_flip_probability": args.p,
            "before_round_data_depolarization": args.p,
            "p_gate_zz": args.p_gate_zz,
        },
        memory_type="z", n_shots=args.chunk, qec_code="surface_code",
    )
    # __init__ alone builds H, fault_array, line graph, k-hop adjacency, base
    # weights, Pearson/joint tracers -- we never call reset()/step(), so no large
    # episode is ever materialised.
    env = DriftedMatchingEnv(
        syndrome_data_generator=generator,
        local_action_only=True, local_action_hops=args.local_action_hops,
        action_scale=args.action_scale, use_pearson_correlation=True,
        use_syndrome_features=False, use_endpoint_firing=False,
        start_from_oracle=False, update_with="DGR", train_mode=False,
    )

    node_dim = env.node_feats.shape[1]
    assert node_dim == 2, f"Expected node_dim=2 for this checkpoint, got {node_dim}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        node_dim=node_dim, hidden_dim=args.hidden_dim,
        static_edge_index=env.line_edge_index, n_layers=args.n_layers,
        lr=1e-4, gamma=0.0, tau=0.005, alpha=args.alpha, target_entropy=-1.0,
    )
    agent.load_models(args.model_path)
    agent.actor.eval()
    actor = BatchedActor(agent, env, device, actor_batch_size=args.actor_batch_size)

    # Static graph object reused for every decode (base DEM graph).
    matching = env.current_matching
    current_weights = env.current_weights.astype(np.float64)
    fault_array = env.fault_array
    k_hop_int = env.k_hop_adj_mat.astype(np.int32)
    cmc = precompute_cm_constants(env)
    hc_nbr_w = precompute_node_static_features(env)
    action_scale = float(args.action_scale)
    bypass = int(args.bypass_threshold)

    # Build the (drifted == base at M=1) circuit for sampling.
    base_circuit, _, _ = generator.generate_base_circuit()
    drifted_circuit, _, _ = generator.generate_drifted_circuit(base_circuit, seed=args.seed)

    acc = StrategyAccumulator()

    # Extra: per-decoder logical error rate vs n_flashes (the analysis v2 lacks).
    nf_total = np.zeros(N_FLASHES_MAX + 1, dtype=np.int64)
    nf_mwpm_err = np.zeros(N_FLASHES_MAX + 1, dtype=np.int64)
    nf_cm_err = np.zeros(N_FLASHES_MAX + 1, dtype=np.int64)
    nf_ncm_err = np.zeros(N_FLASHES_MAX + 1, dtype=np.int64)

    n_chunks = (args.shots + args.chunk - 1) // args.chunk
    print(f"[*] Streaming {args.shots:,} shots in {n_chunks} chunk(s) of {args.chunk:,} "
          f"(device={device.type})...")
    t0 = time.time()
    done_shots = 0
    pbar = tqdm(total=args.shots)
    for ci in range(n_chunks):
        this_chunk = min(args.chunk, args.shots - done_shots)
        if this_chunk <= 0:
            break
        # Unique seed per chunk for independent samples.
        shots, true_obs = generator.simulate_syndrome_data(
            drifted_circuit, seed=args.seed + 1 + ci, n_shots=this_chunk
        )
        true_obs = true_obs.astype(np.uint8)

        # First-pass MWPM (batched).
        edges_first = np.asarray(matching.decode_batch(shots, enable_correlations=False), dtype=bool)
        first_obs = (edges_first.astype(np.int64) @ fault_array.astype(np.int64)) % 2
        first_obs = first_obs.reshape(-1).astype(np.uint8)

        n_flashes = shots.sum(axis=1).astype(np.int64)
        nf_clip = np.minimum(n_flashes, N_FLASHES_MAX)
        act_idx = np.flatnonzero(n_flashes > bypass)
        n_active = act_idx.size

        # Decoder predictions default to the first pass (bypass shots & no-op).
        cm_obs = first_obs.copy()
        ncm_obs = first_obs.copy()

        if n_active > 0:
            sel = edges_first[act_idx]                          # (A, N) bool
            mask = compute_action_masks(sel, k_hop_int)         # (A, N) bool

            # GNN actions (batched).
            feats = np.zeros((n_active, env.n_dec_edges, node_dim), dtype=np.float32)
            feats[:, :, 0] = env.current_weights
            feats[:, :, 1] = sel.astype(np.float32)
            gnn_action = actor(feats, mask)                     # (A, N) in [-1,1], 0 off-mask

            # Analytical CM target weights + per-edge features (vectorised).
            cm_new_w, max_corr_sel, w_hc_sel, num_sel_nbr = compute_cm_and_features(
                sel, cmc, current_weights, hc_nbr_w
            )
            # CM action in the same units as the GNN action (delta / action_scale).
            cm_action = ((cm_new_w - current_weights) / action_scale).astype(np.float32)
            cm_action *= mask                                   # only act on masked nodes

            # Per-shot second passes (the only unavoidable per-shot loop).
            gnn_delta_full = gnn_action * mask * action_scale
            cm_delta_full = (cm_new_w - current_weights) * mask
            for j in range(n_active):
                s = act_idx[j]
                gd = gnn_delta_full[j]
                if np.any(gd):
                    nw = np.clip(current_weights + gd, env.min_weight, env.max_weight)
                    rw = env._build_edge_reweights(nw)
                    e = matching.decode(shots[s], enable_correlations=False, edge_reweights=rw)
                    ncm_obs[s] = int((np.asarray(e, dtype=np.int64) @ fault_array.astype(np.int64)) % 2)
                cd = cm_delta_full[j]
                if np.any(cd):
                    nw = np.clip(current_weights + cd, env.min_weight, env.max_weight)
                    rw = env._build_edge_reweights(nw)
                    e = matching.decode(shots[s], enable_correlations=False, edge_reweights=rw)
                    cm_obs[s] = int((np.asarray(e, dtype=np.int64) @ fault_array.astype(np.int64)) % 2)

            # Per-edge histograms (vectorised feed).
            mwpm_ok_act = first_obs[act_idx] == true_obs[act_idx]
            ncm_ok_act = ncm_obs[act_idx] == true_obs[act_idx]
            cm_ok_act = cm_obs[act_idx] == true_obs[act_idx]
            gnn_o_act = _outcome_vec(mwpm_ok_act, ncm_ok_act)
            cm_o_act = _outcome_vec(mwpm_ok_act, cm_ok_act)
            feed_active_edges(
                acc, mask, sel, gnn_action, cm_action,
                current_weights, hc_nbr_w, max_corr_sel, w_hc_sel, num_sel_nbr,
                gnn_o_act, cm_o_act,
            )

        # Per-shot stats for every shot in the chunk.
        mwpm_ok = first_obs == true_obs
        cm_ok = cm_obs == true_obs
        ncm_ok = ncm_obs == true_obs
        feed_shot_level(acc, nf_clip, mwpm_ok, cm_ok, ncm_ok, n_active)

        # Extra per-decoder error-rate-vs-n_flashes counters.
        np.add.at(nf_total, nf_clip, 1)
        np.add.at(nf_mwpm_err, nf_clip, (~mwpm_ok).astype(np.int64))
        np.add.at(nf_cm_err, nf_clip, (~cm_ok).astype(np.int64))
        np.add.at(nf_ncm_err, nf_clip, (~ncm_ok).astype(np.int64))

        done_shots += this_chunk
        pbar.update(this_chunk)
    pbar.close()

    dt = time.time() - t0
    print(f"[*] Collected {done_shots:,} shots in {dt:.1f}s "
          f"({done_shots / max(dt, 1e-9):,.0f} shots/s).")

    v2_path = args.out_prefix + "_v2hist.npz"
    extra_path = args.out_prefix + "_perf.npz"
    acc.save(v2_path)
    os.makedirs(os.path.dirname(extra_path) or ".", exist_ok=True)
    np.savez_compressed(
        extra_path,
        nf_total=nf_total, nf_mwpm_err=nf_mwpm_err,
        nf_cm_err=nf_cm_err, nf_ncm_err=nf_ncm_err,
        n_flashes_max=np.int64(N_FLASHES_MAX),
    )
    print(f"[*] Saved v2 histograms to {v2_path}")
    print(f"[*] Saved performance-vs-n_flashes counts to {extra_path}")


# ============================================================================
# Extra analysis (p9): per-decoder LER vs n_flashes
# ============================================================================

def plot_performance_vs_nflashes(extra_path, out_dir, max_nf_plot=15):
    import matplotlib.pyplot as plt

    z = np.load(extra_path)
    total = z["nf_total"].astype(np.float64)
    mwpm = z["nf_mwpm_err"].astype(np.float64)
    cm = z["nf_cm_err"].astype(np.float64)
    ncm = z["nf_ncm_err"].astype(np.float64)

    nf = np.arange(total.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        ler_mwpm = np.where(total > 0, mwpm / total, np.nan)
        ler_cm = np.where(total > 0, cm / total, np.nan)
        ler_ncm = np.where(total > 0, ncm / total, np.nan)

    # Console table.
    print("\n" + "=" * 78)
    print("  PER-DECODER LOGICAL ERROR RATE vs n_flashes")
    print("=" * 78)
    print(f"  {'n_flashes':>9} {'shots':>14} {'MWPM_err':>10} {'CM_err':>10} {'NCM_err':>10}"
          f" {'MWPM_LER':>10} {'CM_LER':>10} {'NCM_LER':>10}")
    print("  " + "-" * 90)
    upto = min(max_nf_plot, total.size - 1)
    for k in range(upto + 1):
        if total[k] == 0:
            continue
        print(f"  {nf[k]:>9} {int(total[k]):>14,} {int(mwpm[k]):>10,} {int(cm[k]):>10,} "
              f"{int(ncm[k]):>10,} {ler_mwpm[k]:>10.2e} {ler_cm[k]:>10.2e} {ler_ncm[k]:>10.2e}")
    tot_all = total.sum()
    print("  " + "-" * 90)
    print(f"  {'TOTAL':>9} {int(tot_all):>14,} {int(mwpm.sum()):>10,} {int(cm.sum()):>10,} "
          f"{int(ncm.sum()):>10,} {mwpm.sum()/tot_all:>10.2e} {cm.sum()/tot_all:>10.2e} "
          f"{ncm.sum()/tot_all:>10.2e}")
    print("=" * 78)

    sel = (nf <= max_nf_plot) & (total > 0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(nf[sel], ler_mwpm[sel], "o-", color="dimgray", label="MWPM (first pass)")
    ax1.plot(nf[sel], ler_cm[sel], "s--", color="tab:orange", label="Correlated Matching")
    ax1.plot(nf[sel], ler_ncm[sel], "^-", color="tab:blue", label="Neural Correlated Matching")
    ax1.set_yscale("log")
    ax1.set_ylabel("conditional logical error rate  P(error | n_flashes)")
    ax1.set_title("Per-decoder performance vs syndrome weight (number of detectors fired)")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2.bar(nf[sel], total[sel], color="gray", alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("shots (log)")
    ax2.set_xlabel("n_flashes (detectors fired in the shot)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "p9_performance_vs_nflashes.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[*] Wrote {out}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["collect", "analyze", "both"], default="both")
    ap.add_argument("--shots", type=int, default=50_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)

    # Circuit / regime (deployment defaults: low noise, no drift).
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--p-gate-zz", type=float, default=0.0)
    ap.add_argument("--mismatch", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2024)

    # Decoder / action.
    ap.add_argument("--action-scale", type=float, default=5.0)
    ap.add_argument("--bypass-threshold", type=int, default=2)
    ap.add_argument("--local-action-hops", type=int, default=1)

    # Agent architecture (matches the trial-0040 checkpoint).
    ap.add_argument("--model-path", type=str,
                    default="models/hyperparam_tun_v2/"
                            "slowgpu_v2_qec_graph_optuna_run_d5_trial_0040_best.pth")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--actor-batch-size", type=int, default=512)

    # I/O.
    ap.add_argument("--out-prefix", type=str,
                    default="data/ncm_vs_cm_trial40_p1e-3")
    ap.add_argument("--plot-dir", type=str,
                    default="plots/ncm_vs_cm_trial40_p1e-3/")

    args = ap.parse_args()

    if args.mode in ("collect", "both"):
        run_collection(args)

    if args.mode in ("analyze", "both"):
        v2_path = args.out_prefix + "_v2hist.npz"
        extra_path = args.out_prefix + "_perf.npz"
        if not os.path.exists(v2_path):
            print(f"[!] {v2_path} not found -- run --mode collect first.")
            return
        run_analysis(v2_path, args.plot_dir)          # p1-p8
        plot_performance_vs_nflashes(extra_path, args.plot_dir)  # p9


if __name__ == "__main__":
    main()
