#!/usr/bin/env python
"""Weight-2 confusing-configuration analysis for the Google qec3v5 d=5 experiment, across the three
DEM models (Google analytical = Stim decompose; decompose_errors.py = 'proj'; Spitz-calibrated pij),
for MWPM / CM(alpha=1) / RCM(alpha*).

This is the thesis Fig 5.6/5.11 analysis applied to the real device DEMs.  A distance-5 code should
correct every weight-2 error; the count of weight-2 configurations a decoder gets WRONG measures its
effective-distance loss.

Method
------
* Physical fault set = the 1677 error mechanisms of the RAW (undecomposed) circuit DEM.  Using the
  SAME physical faults for every model makes the comparison fair (only the decoder's DEM differs).
* A weight-2 configuration = an unordered pair of distinct physical faults.  Its syndrome is the XOR
  of the two faults' detectors and its true logical flip is the XOR of their observables.  All
  C(1677,2)=1,405,326 pairs are enumerated exactly.
* For each DEM model and noise scale s (probs scaled p_i -> s*p_i), decoders are built from the
  scaled DEM and every configuration is decoded.  A configuration is "confusing" for a decoder if
  its predicted logical flip != the true one.
* RCM uses the LER-optimal alpha*(model, s), found by a quick Monte-Carlo LER minimisation.

s=1 is the experimentally-calibrated device noise level ("experimental data"); s<1 are the
scaled-down simulated future-device points.

Writes data/qec3v5_weight2_confusion.csv and plots/figures/qec3v5_weight2_confusion.{png,pdf}.
"""
import csv
import itertools
import os
import sys
import time

import numpy as np
import stim
import pymatching
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_using_detector_assignment  # noqa: E402

BASE = os.path.join(ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")
OUT_CSV = os.path.join(ROOT, "data", "qec3v5_weight2_confusion.csv")
FIG_DIR = os.path.join(ROOT, "plots", "figures")

SCALES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05]
ALPHAS_STAR = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1..0.9 for the alpha* search
DEMS = ["analytical", "proj", "pij"]
DLAB = {"analytical": "analytical (Stim)", "proj": r"decompose\_errors.py", "pij": r"calibrated $p_{ij}$"}
N_ASTAR = 400_000          # shots for the alpha* LER minimisation
DECODE_CHUNK = 350_000     # weight-2 configs decoded per batch


def scale_dem(dem, s):
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            out.append("error", min(inst.args_copy()[0] * s, 0.5), inst.targets_copy())
        elif inst.type == "detector":
            out.append(inst)
    return out


def build_dems():
    circ = stim.Circuit.from_file(os.path.join(BASE, "circuit_noisy.stim"))
    raw = circ.detector_error_model(decompose_errors=False)
    an = stim.DetectorErrorModel.from_file(os.path.join(BASE, "circuit_detector_error_model.dem"))
    pij = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))
    coords = an.get_detector_coordinates()
    proj = decompose_errors_using_detector_assignment(
        raw, lambda d: int(round(coords[d][0])) % 2, strip_undecomposable_errors=True)
    return raw, {"analytical": an, "proj": proj, "pij": pij}


def raw_fault_table(raw):
    """(fault_det [n, ND] uint8, fault_obs [n] uint8) for the raw physical fault mechanisms."""
    nd = raw.num_detectors
    dets, obs = [], []
    for inst in raw.flattened():
        if inst.type != "error":
            continue
        ds, o = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                ds ^= {t.val}
            elif t.is_logical_observable_id():
                o ^= 1
        dets.append(ds); obs.append(o)
    n = len(dets)
    fd = np.zeros((n, nd), np.uint8)
    for i, ds in enumerate(dets):
        if ds:
            fd[i, list(ds)] = 1
    return fd, np.array(obs, np.uint8)


def alpha_star(dem_scaled):
    """LER-optimal damped alpha for a scaled DEM, from N_ASTAR sampled shots."""
    smp = dem_scaled.compile_sampler(seed=7)
    d, o, _ = smp.sample(N_ASTAR)
    d = np.asarray(d, bool); o = np.asarray(o, np.uint8).reshape(-1)
    cm = pymatching.Matching.from_detector_error_model(dem_scaled, enable_correlations=True)
    best_a, best_err = 1.0, np.inf
    for a in ALPHAS_STAR:
        pred = np.asarray(cm.decode_batch(d, enable_correlations=True, alpha=a)).reshape(-1)
        e = int((pred != o).sum())
        if e < best_err:
            best_err, best_a = e, a
    return best_a


def main():
    t0 = time.time()
    raw, dems = build_dems()
    fd, fo = raw_fault_table(raw)
    n = fd.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    print(f"physical faults={n}  weight-2 configs={len(ii):,}  ND={raw.num_detectors}", flush=True)

    # full syndrome + true logical flip for every weight-2 configuration (built once, reused)
    synd = (fd[ii] ^ fd[jj]).astype(bool)              # [Npairs, ND]
    obs_true = (fo[ii] ^ fo[jj]).astype(np.uint8)      # [Npairs]
    del ii, jj
    npairs = synd.shape[0]

    # pre-build decoders and alpha* for every (model, scale)
    MW, CM, AST = {}, {}, {}
    for name in DEMS:
        for s in SCALES:
            ds = scale_dem(dems[name], s)
            MW[(name, s)] = pymatching.Matching.from_detector_error_model(ds, enable_correlations=False)
            CM[(name, s)] = pymatching.Matching.from_detector_error_model(ds, enable_correlations=True)
            AST[(name, s)] = alpha_star(ds)
        print(f"{name}: alpha* = {[AST[(name, s)] for s in SCALES]}", flush=True)

    # decode every configuration with MWPM / CM(1) / RCM(alpha*) for each (model, scale)
    conf = {(name, s): {"mwpm": 0, "cm": 0, "rcm": 0} for name in DEMS for s in SCALES}
    for a in range(0, npairs, DECODE_CHUNK):
        b = min(a + DECODE_CHUNK, npairs)
        d = synd[a:b]; ot = obs_true[a:b]
        for name in DEMS:
            for s in SCALES:
                mw, cm, al = MW[(name, s)], CM[(name, s)], AST[(name, s)]
                p_mw = np.asarray(mw.decode_batch(d)).reshape(-1)
                p_cm = np.asarray(cm.decode_batch(d, enable_correlations=True, alpha=1.0)).reshape(-1)
                p_rc = np.asarray(cm.decode_batch(d, enable_correlations=True, alpha=al)).reshape(-1)
                conf[(name, s)]["mwpm"] += int((p_mw != ot).sum())
                conf[(name, s)]["cm"] += int((p_cm != ot).sum())
                conf[(name, s)]["rcm"] += int((p_rc != ot).sum())
        print(f"  decoded {b:,}/{npairs:,}  ({time.time()-t0:.0f}s)", flush=True)

    # write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["dem", "s", "alpha_star", "n_configs", "mwpm_confuse", "cm_confuse", "rcm_confuse"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name in DEMS:
            for s in SCALES:
                c = conf[(name, s)]
                w.writerow({"dem": name, "s": s, "alpha_star": AST[(name, s)], "n_configs": npairs,
                            "mwpm_confuse": c["mwpm"], "cm_confuse": c["cm"], "rcm_confuse": c["rcm"]})
    print("wrote", OUT_CSV, flush=True)

    # table at s=1 (the experimental / native device noise level)
    print("\n=== weight-2 confusing configs at s=1 (experimental Sycamore noise) ===")
    print(f"{'DEM':<12}{'alpha*':>8}{'MWPM':>10}{'CM':>10}{'RCM':>10}")
    for name in DEMS:
        c = conf[(name, 1.0)]
        print(f"{name:<12}{AST[(name, 1.0)]:>8}{c['mwpm']:>10}{c['cm']:>10}{c['rcm']:>10}")

    plot(conf, AST)


def plot(conf, AST):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "mathtext.fontset": "cm",
                         "axes.grid": True, "grid.alpha": 0.3})
    C_MW, C_CM, C_RCM = "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    for ax, name in zip(axes, DEMS):
        s = np.array(SCALES)
        mw = np.array([conf[(name, x)]["mwpm"] for x in SCALES])
        cm = np.array([conf[(name, x)]["cm"] for x in SCALES])
        rc = np.array([conf[(name, x)]["rcm"] for x in SCALES])
        ax.axvline(1.0, color="0.7", lw=1.0, ls=":")   # s=1 = experimental
        ax.plot(s, cm, "s-", color=C_CM, label=r"CM ($\alpha=1$)")
        ax.plot(s, rc, "D-", color=C_RCM, label=r"RCM ($\alpha^{*}$)")
        ax.plot(s, mw, "o--", color=C_MW, label="MWPM")
        ax.set_xscale("log"); ax.set_xlabel(r"Noise scale $s$  ($s\!=\!1$: experimental)")
        ax.set_title(DLAB[name]); ax.legend()
    axes[0].set_ylabel("Confusing weight-2 configurations")
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.10, 1.02, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"qec3v5_weight2_confusion.{ext}"), bbox_inches="tight")
    print("wrote", os.path.join(FIG_DIR, "qec3v5_weight2_confusion.{png,pdf}"))


if __name__ == "__main__":
    main()
