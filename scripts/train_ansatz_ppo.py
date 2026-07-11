"""
Train a 4-coefficient reweighting *ansatz* for two-pass correlated matching with
single-step PPO (contextual bandit, no baseline/critic).

Ansatz (applied to every decoding edge mu that is line-graph-correlated to an edge nu
selected in the first MWPM pass):

    p_mu' = p_mu^a * p_nu^b * p_{mu,nu}^c * d           (per selected neighbour nu)
    w_mu' = log((1 - p_mu') / p_mu')                    (reweighted MWPM weight)

with p_mu = marginal of edge mu, p_nu = marginal of the selected neighbour nu,
p_{mu,nu} = joint co-occurrence of (mu, nu).  As in ordinary correlated matching, if mu
has several selected neighbours we take the *strongest* reweight (lowest weight), and the
new weight only *lowers* the current one (min with current).  The coefficients start at
the ordinary-CM point:  a=0, b=-1, c=1, d=1   (=> p_mu' = p_{mu,nu}/p_nu = P(mu|nu)).

Reward: the SAME differential logical reward used in drifted_matching_env.step():
    +1 if the reweighted (2nd-pass) prediction is correct AND the 1st pass was wrong,
    -1 if the 2nd pass is wrong AND the 1st pass was correct, else 0.

Policy / optimisation:
    The "action" is the 4-vector k=(a,b,c,d), drawn from a diagonal Gaussian N(mu, sigma)
    whose mean mu (and log-sigma) are the only trainable parameters.  Each PPO iteration
    samples ONE batch of shots at the current curriculum p, evaluates several k on those
    *common* shots (common random numbers), and does a clipped-PPO update using the raw
    per-batch reward as advantage (single step, NO baseline / value network).

Training regime: distance-5 rotated Z memory, d rounds, Stim built-in circuit-level
depolarizing noise, NO drift.  Curriculum over p in [4e-4, 1e-2] (high p -> low p, then a
consolidation phase sampling p across the range).

Run:   python scripts/train_ansatz_ppo.py            (full training)
       python scripts/train_ansatz_ppo.py --smoke    (quick self-test)
Out:   models/ansatz_ppo_coeffs.json  (+ .csv training log)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import stim
import pymatching
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords
from adaptiveQRL.decoding_graph import DecodingGraph

CFG = dict(
    distance=5, rounds=5,
    p_low=4e-4, p_high=1e-2, n_stages=9,        # curriculum p-grid (geometric)
    iters=400, curric_frac=0.7,                 # fraction of iters spent annealing high->low
    batch_shots=20_000,                         # shots per PPO iteration (shared across k samples)
    rollouts=16,                                # k samples evaluated per iteration
    ppo_epochs=4, clip_eps=0.2, lr=0.02, ent_coef=1e-3,
    bypass_threshold=2,                         # shots with <= this many fired detectors keep 1st pass
    init_coeffs=(0.0, -1.0, 1.0, 1.0),          # CM: a=0, b=-1, c=1, d=1
    init_log_std=np.log(0.15), min_log_std=np.log(0.02), max_log_std=np.log(0.5),
    eval_every=20, eval_shots=200_000,          # deterministic-mean eval vs CM / MWPM
    seed=0, out="models/ansatz_ppo_coeffs.json",
)
CM = np.array([0.0, -1.0, 1.0, 1.0])            # a,b,c,d for ordinary correlated matching


# ── decoding graph + ansatz two-pass ─────────────────────────────────────────
def build_graph(p, D, R):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=R,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = decompose_errors_for_stim_surface_code_coords(circ.detector_error_model(decompose_errors=False))
    g = DecodingGraph.from_dem(dem)
    fp = pymatching.Matching.from_check_matrix(g.H, weights=np.asarray(g.initial_weights))
    base_p = np.clip(np.asarray(g.base_p, dtype=np.float64), 1e-12, 0.5)
    corr = np.clip(np.asarray(g.initial_corr_tracer, dtype=np.float64), 1e-12, 0.5)
    src, dst = g.line_edge_index
    tgt = np.r_[src, dst].astype(np.int64)       # edge mu being reweighted
    srcn = np.r_[dst, src].astype(np.int64)      # its correlated neighbour nu
    pj = np.r_[corr, corr]                        # joint p_{mu,nu}
    return dict(circ=circ, g=g, fp=fp, cw=np.asarray(g.initial_weights, dtype=np.float64),
                fault=np.asarray(g.fault_array, dtype=np.int64), tgt=tgt, srcn=srcn,
                pmu=base_p[tgt], pnu=base_p[srcn], pj=pj,
                sampler=circ.compile_detector_sampler())


def first_pass(G, syn, bypass=2):
    sel = np.asarray(G["fp"].decode_batch(syn, enable_correlations=False), dtype=bool)
    fp_pred = (sel.astype(np.int64) @ G["fault"]) % 2
    src_sel = sel[:, G["srcn"]]                   # (B, K): candidate's neighbour selected?
    det_count = np.asarray(syn, dtype=np.int64).sum(axis=1)   # fired detectors per shot
    # 2nd pass only for shots with a reweightable edge AND more than `bypass` fired detectors
    active = np.flatnonzero(src_sel.any(1) & (det_count > bypass))
    return sel, fp_pred, src_sel, active


def candidate_weights(G, coeffs):
    a, b, c, d = coeffs
    implied = np.clip((G["pmu"] ** a) * (G["pnu"] ** b) * (G["pj"] ** c) * d, 1e-9, 0.499999)
    return np.log((1.0 - implied) / implied)      # w_mu' per directed candidate


def second_pass_pred(G, syn, fp_pred, src_sel, active, coeffs):
    wc = candidate_weights(G, coeffs)
    cw, tgt, fault = G["cw"], G["tgt"], G["fault"]
    pred = fp_pred.copy()
    if active.size:
        rw = []
        for s in active:
            act = src_sel[s]
            nw = cw.copy()
            np.minimum.at(nw, tgt[act], wc[act])   # min over selected neighbours AND with current
            rw.append(G["g"].build_edge_reweights(nw, cw))
        ed = np.asarray(G["fp"].decode_batch(syn[active], enable_correlations=False, edge_reweights=rw),
                        dtype=np.int64)
        pred[active] = (ed @ fault) % 2
    return pred


def differential_reward(fp_pred, ag_pred, obs):
    ag, fpc = ag_pred == obs, fp_pred == obs
    return np.where(ag & ~fpc, 1.0, np.where(~ag & fpc, -1.0, 0.0))


# ── PPO ───────────────────────────────────────────────────────────────────────
def gaussian_logp(k, mean, log_std):
    var = torch.exp(2 * log_std)
    return (-0.5 * (((k - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))).sum(-1)


def curriculum_p(it, iters, curric_frac, stages_hi_lo):
    n = len(stages_hi_lo)
    phase1 = int(curric_frac * iters)
    if it < phase1:
        return stages_hi_lo[min(n - 1, int(it / max(1, phase1 / n)))]   # step high -> low
    return stages_hi_lo[np.random.randint(n)]                            # consolidation


def evaluate(G, coeffs_list, labels, n_shots, chunk=50_000):
    """Absolute LER for each coeff set (deterministic) on a fresh batch, vs first-pass MWPM."""
    errs = {lab: 0 for lab in labels}
    errs["mwpm"] = 0
    done = 0
    while done < n_shots:
        n = min(chunk, n_shots - done)
        syn, obs = G["sampler"].sample(shots=n, separate_observables=True)
        obs0 = obs[:, 0].astype(np.int64)
        sel, fp_pred, src_sel, active = first_pass(G, syn, CFG["bypass_threshold"])
        errs["mwpm"] += int((fp_pred != obs0).sum())
        for coeffs, lab in zip(coeffs_list, labels):
            pred = second_pass_pred(G, syn, fp_pred, src_sel, active, coeffs)
            errs[lab] += int((pred != obs0).sum())
        done += n
    return {k: v / done for k, v in errs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=CFG["iters"])
    ap.add_argument("--batch-shots", type=int, default=CFG["batch_shots"])
    ap.add_argument("--rollouts", type=int, default=CFG["rollouts"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--out", type=str, default=CFG["out"])
    ap.add_argument("--smoke", action="store_true", help="quick self-test (tiny sizes)")
    args = ap.parse_args()
    if args.smoke:
        args.iters, args.batch_shots, args.rollouts = 6, 4000, 6
        CFG["eval_every"], CFG["eval_shots"], CFG["n_stages"] = 3, 20_000, 4

    np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    D, R = CFG["distance"], CFG["rounds"]
    stages = np.geomspace(CFG["p_low"], CFG["p_high"], CFG["n_stages"])
    stages_hi_lo = stages[::-1].copy()
    print(f"Building {len(stages)} decoding graphs (d={D}, {R} rounds) for p in "
          f"[{CFG['p_low']:.0e}, {CFG['p_high']:.0e}] ...", flush=True)
    graphs = {float(p): build_graph(float(p), D, R) for p in stages}

    mean = torch.tensor(CFG["init_coeffs"], dtype=torch.float32, requires_grad=True)
    log_std = torch.full((4,), float(CFG["init_log_std"]), requires_grad=True)
    opt = torch.optim.Adam([mean, log_std], lr=args.lr)

    log_rows = []
    t0 = time.time()
    print("a,b,c,d start = CM =", CFG["init_coeffs"], "\n", flush=True)
    for it in range(args.iters):
        p = float(curriculum_p(it, args.iters, CFG["curric_frac"], stages_hi_lo))
        G = graphs[p]
        syn, obs = G["sampler"].sample(shots=args.batch_shots, separate_observables=True)
        obs0 = obs[:, 0].astype(np.int64)
        sel, fp_pred, src_sel, active = first_pass(G, syn, CFG["bypass_threshold"])  # shared across k

        m = mean.detach().numpy(); s = np.exp(log_std.detach().numpy())
        Ks = m + s * np.random.randn(args.rollouts, 4)
        Rs = np.array([differential_reward(
            fp_pred, second_pass_pred(G, syn, fp_pred, src_sel, active, Ks[i]), obs0).mean()
            for i in range(args.rollouts)])

        # single-step PPO update (advantage = raw reward, no baseline)
        Kt = torch.tensor(Ks, dtype=torch.float32); Rt = torch.tensor(Rs, dtype=torch.float32)
        with torch.no_grad():
            logp_old = gaussian_logp(Kt, mean, log_std)
        for _ in range(CFG["ppo_epochs"]):
            logp = gaussian_logp(Kt, mean, log_std)
            ratio = torch.exp(logp - logp_old)
            surr = torch.min(ratio * Rt, torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"]) * Rt)
            loss = -surr.mean() - CFG["ent_coef"] * log_std.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                log_std.clamp_(CFG["min_log_std"], CFG["max_log_std"])

        mc = mean.detach().numpy()
        log_rows.append(dict(iter=it, p=p, reward_mean=float(Rs.mean()),
                             a=float(mc[0]), b=float(mc[1]), c=float(mc[2]), d=float(mc[3]),
                             std=float(np.exp(log_std.detach().numpy()).mean())))
        if it % 5 == 0 or it == args.iters - 1:
            print(f"it {it:>4} p={p:<7.1e} R={Rs.mean():+.4f} | "
                  f"a={mc[0]:+.3f} b={mc[1]:+.3f} c={mc[2]:+.3f} d={mc[3]:+.3f} "
                  f"sigma={np.exp(log_std.detach().numpy()).mean():.3f} | {time.time()-t0:.0f}s", flush=True)

        if (it + 1) % CFG["eval_every"] == 0 or it == args.iters - 1:
            pe = float(stages[len(stages) // 2])   # mid-p eval point
            res = evaluate(graphs[pe], [mc, CM], ["ansatz", "cm"], CFG["eval_shots"])
            print(f"    [eval p={pe:.1e}] MWPM={res['mwpm']:.3e}  CM={res['cm']:.3e}  "
                  f"ansatz={res['ansatz']:.3e}  (ansatz/CM={res['ansatz']/max(res['cm'],1e-12):.3f})", flush=True)

    coeffs = mean.detach().numpy().tolist()
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    out_path = os.path.join(ROOT, args.out)
    json.dump({"coeffs": {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2], "d": coeffs[3]},
               "cm_coeffs": {"a": 0, "b": -1, "c": 1, "d": 1},
               "ansatz": "p_mu' = p_mu^a * p_nu^b * p_{mu,nu}^c * d ; w = log((1-p')/p')",
               "config": {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in CFG.items()
                          if k not in ("init_coeffs",)}},
              open(out_path, "w"), indent=2)
    import csv
    with open(out_path.replace(".json", "_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)
    print(f"\nlearned coeffs a,b,c,d = {[round(x,4) for x in coeffs]}  (CM = 0,-1,1,1)")
    print(f"saved {args.out} (+ _log.csv) | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
