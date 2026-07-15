"""
LER vs physical error rate for the Tesseract decoder (run under WSL2 / Linux).

Same experiment as data/ler_vs_p_slowgpu_v2_*.csv: d=5 rotated surface-code
Z-memory, circuit-level depolarizing noise, swept over the same p-grid. Adaptive
shot count: keep sampling until >= MIN_ERRORS logical errors OR MAX_SHOTS is hit
(whichever comes first). MWPM (public pymatching) is decoded on the same shots
as a free, directly-comparable baseline.

Deps:  pip install tesseract-decoder stim numpy pymatching
Run:   python ler_vs_p_tesseract.py

⚠ FEASIBILITY: at p=2e-4 the LER ~1e-6, so 300 errors needs ~3e8 shots. Tesseract
  is a slow single-shot decoder — the low-p points will almost certainly hit
  MAX_SHOTS before reaching 300 errors (their error bars will be correspondingly
  wider; the actual error count is recorded). Raise MAX_SHOTS / lower the p-grid /
  run on the cluster if you need the full-precision tail. Rows are written to CSV
  as each point finishes, so a slow point never costs you the earlier ones.
"""

import csv
import time
import numpy as np
import stim
import pymatching
from tesseract_decoder import tesseract

from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords

# ── Config ──────────────────────────────────────────────────────────────────
D          = 5
P_LIST     = [0.0008, 0.0006, 0.0004, 0.0002]   # 1e-2 → 1e-3 (300 errors reachable on a laptop)
MIN_ERRORS = 300              # adaptive target
MAX_SHOTS  = 1000_000_000        # per-point cap — raise for more low-p precision
CHUNK      = 200_000          # sampling/decoding chunk size
DET_BEAM   = 50               # Tesseract beam width (smaller = faster, less optimal)
SEED       = 12345
OUT_CSV    = "data/ler_vs_p_tesseract_d5.csv"

FIELDS = ["p", "tess_mean", "tess_std", "mwpm_mean", "mwpm_std",
          "n_shots", "tess_errors", "mwpm_errors"]


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
    n_obs = circuit.num_observables

    # Tesseract: raw DEM (handles hyperedges directly).
    tess = tesseract.TesseractConfig(
        dem=circuit.detector_error_model(decompose_errors=False), det_beam=DET_BEAM
    ).compile_decoder()
    # MWPM: decomposed (graphlike) DEM for public pymatching. Use the project's
    # coordinate-aware (tesseract-style) decomposition, not Stim's heuristic.
    mwpm = pymatching.Matching.from_detector_error_model(
        decompose_errors_for_stim_surface_code_coords(
            circuit.detector_error_model(decompose_errors=False)
        )
    )

    sampler = circuit.compile_detector_sampler(seed=SEED + p_index)
    n_shots = tess_err = mwpm_err = 0
    t0 = time.time()

    while tess_err < MIN_ERRORS and n_shots < MAX_SHOTS:
        chunk = min(CHUNK, MAX_SHOTS - n_shots)
        dets, obs = sampler.sample(chunk, separate_observables=True)
        dets = np.asarray(dets, dtype=bool)
        obs = np.asarray(obs, dtype=np.uint8)

        # MWPM (batched, ~free).
        pred_m = np.asarray(mwpm.decode_batch(dets), dtype=np.uint8)
        mwpm_err += int(np.sum(np.any(pred_m != obs, axis=1)))

        # Tesseract (single-shot loop — the bottleneck).
        for i in range(chunk):
            pred_t = np.asarray(tess.decode(dets[i]), dtype=np.uint8).reshape(n_obs)
            if np.any(pred_t != obs[i]):
                tess_err += 1

        n_shots += chunk
        rate = n_shots / (time.time() - t0)
        print(f"  p={p:<7} {n_shots:>12,} shots | tess_err={tess_err:>4} "
              f"mwpm_err={mwpm_err:>4} | {rate:,.0f} shots/s", flush=True)

    tess_ler = tess_err / n_shots
    mwpm_ler = mwpm_err / n_shots
    if tess_err < MIN_ERRORS:
        print(f"  [!] p={p}: hit MAX_SHOTS with only {tess_err} tesseract errors "
              f"(target {MIN_ERRORS}) — error bar is wide.", flush=True)
    return {
        "p": p,
        "tess_mean": tess_ler, "tess_std": se(tess_ler, n_shots),
        "mwpm_mean": mwpm_ler, "mwpm_std": se(mwpm_ler, n_shots),
        "n_shots": n_shots, "tess_errors": tess_err, "mwpm_errors": mwpm_err,
    }


def main():
    print(f"d={D}, p-grid={P_LIST}\nMIN_ERRORS={MIN_ERRORS}, MAX_SHOTS={MAX_SHOTS:,}, "
          f"DET_BEAM={DET_BEAM}\nWriting -> {OUT_CSV}\n")
    # Write header immediately; append each row as its point finishes.
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for idx, p in enumerate(P_LIST):
        row = evaluate_p(p, idx)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"  -> p={p}: tess LER={row['tess_mean']:.3e} +/- {row['tess_std']:.1e} "
              f"| mwpm LER={row['mwpm_mean']:.3e}  (saved)\n", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
