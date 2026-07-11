"""
LER vs physical error rate for Belief Matching, sweeping BP iteration counts.

Same experiment as the decoders comparison: d=5 rotated surface-code Z-memory,
circuit-level depolarizing noise. Sweeps p and, at each p, decodes with Belief
Matching (one BeliefMatching per max_bp_iters in BP_ITERS) and, for reference,
uncorrelated MWPM on the SAME DEM. All decoders see the SAME shots (sampled once
per p) so the comparison is apples-to-apples. Adaptive shot count: keep sampling
until every BP setting has >= MIN_ERRORS logical errors (or MAX_SHOTS is hit);
the MWPM baseline (always worse) is emitted as a row with bp_iters="mwpm".

Deps:  pip install beliefmatching stim numpy   (beliefmatching pulls in pymatching + ldpc)
       + this repo's package on PATH:  pip install -e .   (for adaptiveQRL.decompose_errors)
Run:   python ler_vs_p_beliefmatching.py
Out:   data/ler_vs_p_beliefmatching_d5.csv   (long format: one row per (p, bp_iters))

DEM construction matches notebooks/decoders_comparison.ipynb exactly: the custom
coordinate-aware decomposer fed to BeliefMatching(dem, max_bp_iters=n).
"""

import csv
import time
import numpy as np
import stim
import pymatching
from beliefmatching import BeliefMatching

from adaptiveQRL.decompose_errors import decompose_errors_for_stim_surface_code_coords

# ── Config ──────────────────────────────────────────────────────────────────
D          = 7
P_LIST     = [0.01,0.007,0.004,0.002,0.001,0.0007,0.0004]    # 1e-2 → 4e-4
BP_ITERS   = [5]
MIN_ERRORS = 1000                            # adaptive target (per iteration setting)
MAX_SHOTS  = 1000000_000_000                      # per-point cap (ample in this p-range)
CHUNK      = 100_000
SEED       = 12345
OUT_CSV    = "data/ler_vs_p_beliefmatching_d7_extended.csv"

FIELDS = ["p", "bp_iters", "ler_mean", "ler_std", "n_shots", "n_errors"]


def make_circuit(p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=D, distance=D,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
    ) 


def se(ler, n):
    return float(np.sqrt(ler * (1.0 - ler) / n)) if n > 0 else float("nan")


def evaluate_p(p, p_index):
    circuit = make_circuit(p)
    # DEM exactly as in the notebook: custom coordinate-aware decomposition.
    dem = decompose_errors_for_stim_surface_code_coords(
        circuit.detector_error_model(decompose_errors=False)
    )
    
    # One Belief-Matching decoder per BP-iteration setting (sharing the same DEM).
    decoders = {n: BeliefMatching(dem, max_bp_iters=n) for n in BP_ITERS}
    # Uncorrelated MWPM on the SAME DEM/shots, as a reference baseline.
    mwpm = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=SEED + p_index)
    n_shots = 0
    errs = {n: 0 for n in BP_ITERS}
    err_mwpm = 0
    t0 = time.time()

    while min(errs.values()) < MIN_ERRORS and n_shots < MAX_SHOTS:
        chunk = min(CHUNK, MAX_SHOTS - n_shots)
        dets, obs = sampler.sample(chunk, separate_observables=True)
        obs = np.asarray(obs, dtype=np.uint8)
        for n, bm in decoders.items():
            pred = np.asarray(bm.decode_batch(dets), dtype=np.uint8)
            errs[n] += int(np.sum(np.any(pred != obs, axis=1)))
        pred_m = np.asarray(mwpm.decode_batch(dets), dtype=np.uint8)
        err_mwpm += int(np.sum(np.any(pred_m != obs, axis=1)))
        n_shots += chunk
        rate = n_shots / (time.time() - t0)
        err_str = " ".join(f"bp{n}={errs[n]}" for n in BP_ITERS)
        print(f"  p={p:<7} {n_shots:>10,} shots | {err_str} mwpm={err_mwpm} | {rate:,.0f} shots/s",
              flush=True)

    rows = []
    for n in BP_ITERS:
        ler = errs[n] / n_shots
        if errs[n] < MIN_ERRORS:
            print(f"  [!] p={p}, bp_iters={n}: only {errs[n]} errors (target {MIN_ERRORS}).",
                  flush=True)
        rows.append({
            "p": p, "bp_iters": n,
            "ler_mean": ler, "ler_std": se(ler, n_shots),
            "n_shots": n_shots, "n_errors": errs[n],
        })
    ler_m = err_mwpm / n_shots
    rows.append({
        "p": p, "bp_iters": "mwpm",
        "ler_mean": ler_m, "ler_std": se(ler_m, n_shots),
        "n_shots": n_shots, "n_errors": err_mwpm,
    })
    return rows


def main():
    print(f"d={D}, p-grid={P_LIST}, bp_iters={BP_ITERS}\n"
          f"MIN_ERRORS={MIN_ERRORS}, MAX_SHOTS={MAX_SHOTS:,}\nWriting -> {OUT_CSV}\n")
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for idx, p in enumerate(P_LIST):
        rows = evaluate_p(p, idx)
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writerows(rows)
        summary = "  ".join(
            f"{'mwpm' if r['bp_iters'] == 'mwpm' else 'bp' + str(r['bp_iters'])}={r['ler_mean']:.3e}"
            for r in rows)
        print(f"  -> p={p}: {summary}  (saved)\n", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
