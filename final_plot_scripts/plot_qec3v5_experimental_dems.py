#!/usr/bin/env python
"""Decode the RAW 50k Google qec3v5 experimental shots (d=5, Z, r=5) with all 9 combinations of
{MWPM, CM(alpha=1), RCM(alpha*)} x {analytical (Stim), decompose_errors.py, calibrated pij}.

This is the genuine experimental-shots anchor for the s=1 end of the DEM-scaling study: instead of
sampling from the DEM, we decode the actual detection events recorded on the device.

  * analytical / proj DEMs are circuit-derived, so all 50k shots are decoded with the single DEM.
  * pij is calibrated FROM the data, so we cross-validate: odd shots decoded with the DEM calibrated
    on the even shots (pij_from_even_for_odd) and vice-versa -- never decoding shots with a DEM fit
    to them.
  * RCM uses alpha* = 0.8, the s=1 optimum found (independently) by the DEM-scaling sweep for all
    three DEMs.

Figure -> final_plots/qec3v5_experimental_dems.{pdf,png}
"""
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import stim            # noqa: E402
import pymatching      # noqa: E402
from NeuralCM.decompose_errors import decompose_errors_using_detector_assignment  # noqa: E402

BASE = os.path.join(ROOT, "google_qec3v5_experiment_data", "surface_code_bZ_d5_r05_center_5_5")
OUT = os.path.join(ROOT, "final_plots")
os.makedirs(OUT, exist_ok=True)
ALPHA_STAR = 0.8   # s=1 optimum from the DEM-scaling sweep (same for all three DEMs)


def read_shots():
    circ = stim.Circuit.from_file(os.path.join(BASE, "circuit_noisy.stim"))
    nd, no = circ.num_detectors, circ.num_observables
    dets = np.asarray(stim.read_shot_data_file(
        path=os.path.join(BASE, "detection_events.b8"), format="b8",
        num_detectors=nd, num_observables=0)).astype(bool)
    act = np.asarray(stim.read_shot_data_file(
        path=os.path.join(BASE, "obs_flips_actual.01"), format="01",
        num_detectors=0, num_observables=no)).astype(np.uint8).reshape(-1)
    return dets, act, circ


def build_single_dems(circ):
    raw = circ.detector_error_model(decompose_errors=False)
    an = stim.DetectorErrorModel.from_file(os.path.join(BASE, "circuit_detector_error_model.dem"))
    coords = an.get_detector_coordinates()
    proj = decompose_errors_using_detector_assignment(
        raw, lambda d: int(round(coords[d][0])) % 2, strip_undecomposable_errors=True)
    return {"analytical": an, "proj": proj}


def decode(dem, dets, corr, alpha):
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
    kw = {"enable_correlations": True, "alpha": alpha} if corr else {}
    return np.asarray(m.decode_batch(dets, **kw)).reshape(-1)


def ler_direct(dem, dets, act, corr, alpha):
    pred = decode(dem, dets, corr, alpha)
    p = float((pred != act).mean())
    return p, float(np.sqrt(p * (1 - p) / len(act)))


def ler_cv(dem_odd, dem_even, dets, act, corr, alpha):
    """Cross-validated: odd shots <- dem_odd (calibrated on even), even shots <- dem_even."""
    N = len(act); idx = np.arange(N); EVEN = idx % 2 == 0; ODD = ~EVEN
    pred = np.zeros(N, np.uint8)
    for mask, dem in ((ODD, dem_odd), (EVEN, dem_even)):
        pred[mask] = decode(dem, dets[mask], corr, alpha)
    p = float((pred != act).mean())
    return p, float(np.sqrt(p * (1 - p) / N))


def main():
    dets, act, circ = read_shots()
    single = build_single_dems(circ)
    pij_odd = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_even_for_odd.dem"))
    pij_even = stim.DetectorErrorModel.from_file(os.path.join(BASE, "pij_from_odd_for_even.dem"))

    # decoders: (label, corr, alpha)
    DEC = [("MWPM", False, 1.0), (r"CM ($\alpha=1$)", True, 1.0),
           (rf"RCM ($\alpha={ALPHA_STAR}$)", True, ALPHA_STAR)]
    DEMS = ["analytical", "proj", "pij"]
    DLAB = {"analytical": "analytical (Stim)", "proj": "decompose_errors.py", "pij": r"calibrated $p_{ij}$"}

    res = {}   # (dem, dec_label) -> (ler, se)
    for dem in DEMS:
        for lab, corr, a in DEC:
            if dem == "pij":
                res[(dem, lab)] = ler_cv(pij_odd, pij_even, dets, act, corr, a)
            else:
                res[(dem, lab)] = ler_direct(single[dem], dets, act, corr, a)
    print(f"50,000 experimental shots, d=5, Z, r=5   (RCM alpha* = {ALPHA_STAR})")
    print(f"{'DEM':<14}" + "".join(f"{lab:>18}" for lab, _, _ in DEC))
    for dem in DEMS:
        print(f"{dem:<14}" + "".join(f"{res[(dem, lab)][0]*100:>16.2f}%" for lab, _, _ in DEC))

    # ---- grouped bar chart ----
    mpl.rcParams.update({"font.size": 10, "font.family": "sans-serif", "figure.dpi": 130,
                         "axes.linewidth": 0.9, "axes.labelsize": 11, "axes.titlesize": 11.5,
                         "legend.fontsize": 9.5, "mathtext.fontset": "dejavusans",
                         "axes.axisbelow": True, "grid.color": "0.85", "grid.linewidth": 0.6,
                         "savefig.bbox": "tight", "savefig.dpi": 300})
    DCOL = ["#2e86de", "#e67e22", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.grid(axis="y", alpha=0.7)
    x = np.arange(len(DEMS)); w = 0.26
    for j, (lab, _, _) in enumerate(DEC):
        vals = np.array([res[(dem, lab)][0] * 100 for dem in DEMS])
        ses = np.array([res[(dem, lab)][1] * 100 for dem in DEMS])
        bars = ax.bar(x + (j - 1) * w, vals, w, yerr=ses, capsize=3, color=DCOL[j],
                      edgecolor="white", linewidth=0.6, label=lab,
                      error_kw=dict(elinewidth=1.0, ecolor="0.3"))
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=8, color="0.25")
    ax.set_xticks(x); ax.set_xticklabels([DLAB[d] for d in DEMS])
    ax.set_xlabel("DEM (noise model fed to the decoder)")
    ax.set_ylabel(r"Logical error rate  $p_{\mathrm{L}}$  (%)")
    ax.set_title("Google qec3v5 d=5, Z-memory, r=5 — 50 000 experimental shots")
    ax.set_ylim(0, max(res[(d, l)][0] for d in DEMS for l, _, _ in DEC) * 100 * 1.34)
    ax.legend(title="decoder", ncol=3, loc="upper center", framealpha=1.0,
              facecolor="white", edgecolor="0.7")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"qec3v5_experimental_dems.{ext}"))
    print("saved final_plots/qec3v5_experimental_dems.{pdf,png}")


if __name__ == "__main__":
    main()
