#!/usr/bin/env python
"""Damped CM vs Pearl CM across high p: per-p alpha sweep for BOTH rules.

  Damped Bayes:  implied_p = alpha * P(mu|nu) = alpha * P(mu,nu)/P(nu)
  Pearl:         implied_p = alpha * P(mu|nu) + (1-alpha) * P(mu|not nu)

Both are run through the SAME hand-rolled two-pass (common random numbers,
bypass_threshold=2), so the comparison is exactly apples-to-apples. (Damped is not in
the native C++ decoder, so both go through Python here.) At alpha=1.0 the two rules
coincide (= hard CM), which is used as the stopping reference.

Output:
  * CSV  data/damped_vs_pearl_highp.csv : one row per (p, rule, alpha).
  * stdout: for each p, the ratio  LER(best-alpha Damped) / LER(best-alpha Pearl)
    (the two best alphas may differ).

Usage:
  python scripts/damped_vs_pearl_highp.py
  python scripts/damped_vs_pearl_highp.py --p-values 0.1,0.01 --alphas 0.1,0.5,0.9,1.0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords
from NeuralCM.two_pass_correlated_matching import TwoPassCorrelatedMatching

CLO, CHI = 1e-6, 0.499999


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bypass", type=int, default=0)
    ap.add_argument("--p-values", type=str,
                    default="0.01,0.005,0.001")
    ap.add_argument("--alphas", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--target-errors", type=int, default=1000,
                    help="stop a p once hard CM (alpha=1.0) reaches this many logical errors")
    ap.add_argument("--max-shots", type=int, default=1000_000_000, help="per-p shot cap")
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-csv", type=str, default="data/damped_vs_pearl_highp.csv")
    return ap.parse_args()


def build(p, args):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=args.distance, rounds=args.rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = decompose_errors_for_stim_surface_code_coords(circ.detector_error_model(decompose_errors=False))
    dec = TwoPassCorrelatedMatching(dem, alpha=1.0, bypass_threshold=args.bypass)
    g = dec.graph
    return dict(circ=circ, g=g, m=dec._matching, adj=g.bidirectional_adjacency,
                occ=np.asarray(dec.occ), corr=np.asarray(dec.corr),
                cw=np.asarray(dec.current_weights), fault=g.fault_array)


def decode_chunk(B, syn, alphas, bypass):
    """Return {(rule, alpha): preds[chunk]} for both rules, sharing the first pass."""
    g, m, adj = B["g"], B["m"], B["adj"]
    occ, corr, cw, fault = B["occ"], B["corr"], B["cw"], B["fault"]
    sel_all = np.asarray(m.decode_batch(syn, enable_correlations=False), dtype=bool)
    out = {(r, a): np.empty(len(syn), np.uint8) for r in ("damped", "pearl") for a in alphas}
    for s in range(len(syn)):
        sel = sel_all[s]; first = np.flatnonzero(sel)
        if first.size == 0 or syn[s].sum() <= bypass:
            v = np.uint8((sel @ fault) % 2)
            for key in out:
                out[key][s] = v
            continue
        # Per active edge, gather the (pcond, pneg) over its selected neighbours ONCE.
        epc = []
        for node in np.flatnonzero(g.compute_action_mask(first)):
            pcs, pns = [], []
            for nbr, e in adj.get(node, []):
                if sel[nbr]:
                    pcs.append(corr[e] / occ[nbr])
                    pns.append(max((occ[node] - corr[e]) / max(1 - occ[nbr], 1e-12), 0.0))
            if pcs:
                epc.append((node, np.asarray(pcs), np.asarray(pns)))
        for a in alphas:
            for rule in ("damped", "pearl"):
                nw = cw.copy()
                for node, pc, pn in epc:
                    ip = a * pc if rule == "damped" else a * pc + (1 - a) * pn
                    ip = np.clip(ip, CLO, CHI)
                    wmin = np.log((1 - ip) / ip).min()
                    if wmin < cw[node]:
                        nw[node] = wmin
                ed = np.asarray(m.decode(syn[s], enable_correlations=False,
                                         edge_reweights=g.build_edge_reweights(nw, cw)), dtype=np.int64)
                out[(rule, a)][s] = np.uint8((ed @ fault) % 2)
    return out


def main():
    args = parse_args()
    p_values = [float(x) for x in args.p_values.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    if 1.0 not in alphas:
        alphas = sorted(alphas + [1.0])
    print(f"d={args.distance} rounds={args.rounds} bypass={args.bypass}")
    print(f"p_values={p_values}\nalphas={alphas}")
    print(f"target_errors={args.target_errors} max_shots={args.max_shots:,} chunk={args.chunk:,}\n")

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out_csv)) or ".", exist_ok=True)
    rows = []
    t_all = time.time()

    for p in p_values:
        B = build(p, args)
        sampler = B["circ"].compile_detector_sampler(seed=args.seed)
        err = {(r, a): 0 for r in ("damped", "pearl") for a in alphas}
        shots = 0
        t0 = time.time()
        while shots < args.max_shots and err[("pearl", 1.0)] < args.target_errors:
            syn, obs = sampler.sample(shots=args.chunk, separate_observables=True)
            obs = obs.flatten().astype(np.uint8)
            preds = decode_chunk(B, syn, alphas, args.bypass)
            for key, pr in preds.items():
                err[key] += int(np.sum(pr != obs))
            shots += args.chunk

        def L(r, a):
            return err[(r, a)] / shots
        best_d = min(alphas, key=lambda a: err[("damped", a)])
        best_p = min(alphas, key=lambda a: err[("pearl", a)])
        ld, lp = L("damped", best_d), L("pearl", best_p)
        ratio = ld / lp if lp > 0 else float("nan")
        print(f"p={p:<6g} shots={shots:>9,} | best Damped a={best_d:g} LER={ld:.3e} ({err[('damped',best_d)]} err) | "
              f"best Pearl a={best_p:g} LER={lp:.3e} ({err[('pearl',best_p)]} err) | "
              f"ratio(Damped/Pearl)={ratio:.3f} | {time.time()-t0:.0f}s", flush=True)

        for r in ("damped", "pearl"):
            for a in alphas:
                e = err[(r, a)]; ler = e / shots
                rows.append(dict(p=p, rule=r, alpha=a, shots=shots, errors=e,
                                 ler=ler, ler_std=(ler * (1 - ler) / shots) ** 0.5))
        with open(os.path.join(ROOT, args.out_csv), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\nsaved {args.out_csv} | total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
