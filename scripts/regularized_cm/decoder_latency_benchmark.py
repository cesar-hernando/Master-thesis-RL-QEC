#!/usr/bin/env python
"""Decoder latency benchmark: MWPM, CM, RCM, Belief Matching, Tesseract.

Rotated surface code, Z-memory, d=5, r=5 rounds, circuit-level depolarizing noise at a
single physical error rate p (default 1e-3 -- a representative near-threshold operating
point; see the --p flag to change it). DEM hyperedges are decomposed with the
coordinate-aware decomposer in NeuralCM.decompose_errors, exactly as used for the LER
comparisons in Chapter 5 (MWPM / CM / RCM / Belief Matching all decode the SAME decomposed
DEM; Tesseract decodes the raw undecomposed DEM directly, since operating on the
hypergraph is the whole point of its A* search -- matching scripts/ler_vs_p_tesseract.py).

Latency is measured as the wall-clock time of individual, single-shot decode() calls, not
an amortized batched decode_batch() call. This mirrors how a real-time decoder is actually
invoked in the hardware control loop -- one shot at a time -- rather than the throughput a
batched call can achieve by amortizing Python/array overhead across many shots at once.
(All five decoders here, including Tesseract, expose a decode_batch(); see --also-batched.)

Sampling is ADAPTIVE per decoder: shots are drawn in chunks and timed individually until
at least --min-errors logical errors have been observed for that specific decoder (each
decoder has a different LER, so needs a different number of shots for the same error
count), or a safety cap (--max-shots / --max-seconds) is hit -- matching the adaptive
stopping convention used by the LER sweeps elsewhere in this repo (e.g.
scripts/ler_vs_p_tesseract.py, NeuralCM.mc_collect).

CAVEAT: single-shot Python calls carry a fixed pybind11/argument-marshalling overhead
(~10-70 us on this machine, measured by timing a trivial no-op property access vs. a real
decode() call) that has nothing to do with the matching algorithm itself. For MWPM/CM/RCM,
whose actual C++ work at d=5 is a few microseconds, this overhead dominates the reported
number -- which is why it reads ~100 us/shot here rather than the ~1 us/round figures
quoted for specialised, non-Python real-time decoder hardware (FPGA/ASIC control loops,
which never cross a Python boundary at all, and which report per-round pipelined
throughput rather than a single Python function-call's latency). Because that overhead is
close to a fixed additive constant, it is roughly common across MWPM/CM/RCM, so their
RELATIVE ratios (reported here) are still meaningful; Belief Matching's and Tesseract's
ratios are essentially unaffected by it since their actual computation dominates. For
reference, --also-batched additionally times one amortized decode_batch() call per decoder
(where available, over a capped sub-sample of the already-decoded shots -- --batch-cap) to
show the overhead-free floor alongside the single-shot number.

RCM uses a single damping strength alpha, defaulting to alpha=0.5, the empirically optimal
value at d=5, p=1e-3 found in the RCM alpha scan (data/reg_cm_alpha_scan_new_combined.csv,
see Section 5.1/5.3); pass --alpha to use the right value for a different p.

Tesseract requires the `tesseract-decoder` package, which this repository only builds on
Linux/WSL (see scripts/ler_vs_p_tesseract.py). If it is not importable here, its row is
skipped with a warning and every other decoder still runs to completion. Results are
merged into the existing CSV by decoder name, so re-running this script later (e.g. from
WSL, with --decoders tesseract) fills in the missing row without erasing the rest.

Usage:
    python scripts/regularized_cm/decoder_latency_benchmark.py
    python scripts/regularized_cm/decoder_latency_benchmark.py --p 7e-3 --alpha 0.9
    python scripts/regularized_cm/decoder_latency_benchmark.py --decoders tesseract   # WSL
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import stim
import pymatching

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords  # noqa: E402

ORDER = ["mwpm", "cm", "rcm", "belief", "tesseract"]
LABELS = {"mwpm": "MWPM", "cm": "CM", "rcm": "RCM", "belief": "Belief Matching",
          "tesseract": "Tesseract"}
FIELDS = ["decoder", "d", "rounds", "p", "alpha", "bp_iters", "n_shots",
          "mean_seconds", "std_seconds", "median_seconds", "sem_seconds",
          "mean_normalized", "logical_errors", "target_errors", "stop_reason",
          "batched_seconds_per_shot", "batched_normalized", "seconds",
          "n_outliers_excluded", "raw_mean_seconds",
          "batched_n_repeats", "batched_min_us", "batched_max_us", "batched_std_us"]

# A single decode() call taking longer than this is not real algorithmic cost at d=5 for
# ANY of these decoders (the slowest clean single-shot number observed across this whole
# benchmark, Belief Matching, is ~2 ms) -- it means the OS paused/throttled the process
# (screen lock, sleep, background-app throttling, antivirus scan, ...) while a decode()
# call happened to be in flight. That pause lands on whichever single call was executing,
# so it contaminates a handful of samples out of a run that can be hundreds of thousands to
# millions long; the MEAN is wrecked by it (a single multi-second outlier can dominate a
# sum of tiny numbers) while the MEDIAN is essentially untouched. We exclude such outliers
# before computing summary statistics and report how many were dropped for transparency.
OUTLIER_CUTOFF_SECONDS = 0.05


# --------------------------------------------------------------------------------------- circuit
def make_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


# --------------------------------------------------------------------------------------- timing
def adaptive_time_decode(decode_fn, sampler, *, n_warmup, chunk, min_errors,
                         max_shots, max_seconds, label, fixed_shots=None):
    """Warm up (untimed), then decode freshly-sampled chunks -- timing each individual,
    single-shot decode() call.

    Two stopping modes:
      * adaptive (fixed_shots=None): stop once `min_errors` logical errors are observed for
        THIS decoder, or a safety cap (--max-shots / --max-seconds) is hit. Different
        decoders then use different shot counts, sized to each one's own error rate.
      * fixed (fixed_shots=N): every decoder decodes EXACTLY N shots, ignoring min_errors --
        for a same-sample-size comparison across decoders (see --fixed-shots). N is
        typically chosen as whatever the least-accurate-tolerant decoder (Tesseract) needs
        for ~100 errors, so every decoder gets at least that many errors' worth of
        statistics, just with more headroom for the more error-prone decoders. --max-seconds
        still applies as an emergency safety valve.

    Returns (times, n_err, dets_all, stop_reason, n_outliers)."""
    wdets, _ = sampler.sample(n_warmup, separate_observables=True)
    wdets = np.asarray(wdets, dtype=bool)
    for i in range(n_warmup):
        decode_fn(wdets[i])

    times_chunks, dets_chunks = [], []
    n_err = n_shots = 0
    t_start = time.perf_counter()
    reason = "target"
    while True:
        this_chunk = min(chunk, fixed_shots - n_shots) if fixed_shots is not None else chunk
        dets, obs = sampler.sample(this_chunk, separate_observables=True)
        dets = np.asarray(dets, dtype=bool)
        obs = np.asarray(obs, dtype=np.uint8)[:, 0]
        t = np.empty(this_chunk, dtype=np.float64)
        for i in range(this_chunk):
            t0 = time.perf_counter()
            pred = decode_fn(dets[i])
            t[i] = time.perf_counter() - t0
            if int(np.atleast_1d(pred)[0]) != int(obs[i]):
                n_err += 1
        times_chunks.append(t)
        dets_chunks.append(dets)
        n_shots += this_chunk
        elapsed = time.perf_counter() - t_start
        running_mean = float(np.mean(np.concatenate(times_chunks))) * 1e6
        target_label = f"fixed {fixed_shots:,} shots" if fixed_shots is not None else f"target {min_errors} errors"
        print(f"    [{label}] {n_shots:>10,} shots | {n_err:>4} errors "
              f"({target_label}) | {elapsed:>7.1f}s | {running_mean:.2f} us/shot avg",
              flush=True)
        if fixed_shots is not None:
            if n_shots >= fixed_shots:
                reason = "fixed_shots"; break
        else:
            if n_err >= min_errors:
                reason = "target"; break
            if n_shots >= max_shots:
                reason = "max_shots"; break
        if elapsed >= max_seconds:
            reason = "max_seconds"; break
    if fixed_shots is None and reason != "target":
        print(f"    [!] [{label}] stopped by {reason} with only {n_err}/{min_errors} errors "
              f"-- wider error bar than requested.", flush=True)
    elif fixed_shots is not None and reason == "max_seconds":
        print(f"    [!] [{label}] stopped by max_seconds before reaching the fixed shot count "
              f"({n_shots:,}/{fixed_shots:,}) -- wall-clock safety valve triggered.", flush=True)
    times = np.concatenate(times_chunks)
    dets_all = np.concatenate(dets_chunks, axis=0)
    n_outliers = int(np.count_nonzero(times > OUTLIER_CUTOFF_SECONDS))
    if n_outliers:
        worst = float(np.max(times))
        print(f"    [!] [{label}] excluded {n_outliers} outlier call(s) > "
              f"{OUTLIER_CUTOFF_SECONDS*1e3:.0f} ms (worst: {worst:.1f}s) -- almost certainly "
              f"an OS scheduling stall (sleep/lock/throttle), not real decode cost.",
              flush=True)
    return times, n_err, dets_all, reason, n_outliers


def time_batched(decode_batch_fn, dets_all, cap, n_repeats=21):
    """Overhead-free reference: MEDIAN of `n_repeats` amortized decode_batch() calls over
    up to `cap` of the already-decoded shots (capped so a decoder that needed millions of
    shots to reach min_errors doesn't also cost minutes on this diagnostic). Returns
    (seconds/shot, spread_dict) where spread_dict has min/max/std across repeats in us/shot
    for transparency -- or (None, None) if `decode_batch_fn` is None (i.e. --no-also-batched,
    or a future decoder that genuinely has no batched API; all five decoders currently
    wired into this script do).

    A SINGLE untimed-repeat call is not safe on this machine: a controlled, interleaved
    diagnostic showed the SAME call (alpha=1.0) costing 15.5 us/shot early in a run and
    27.3-27.5 us/shot a few minutes later, on an otherwise idle matching object -- i.e. the
    machine itself drifts slower over time (thermal throttling / background load), and a
    lone measurement can silently land on either side of that drift. 21 repeats (up from an
    original 7) tightens the median against that drift and against occasional short stalls,
    at a cost of a few tens of seconds per decoder -- cheap relative to the risk of another
    misleading batched ratio."""
    if decode_batch_fn is None:
        return None, None
    shots = dets_all[:cap]
    decode_batch_fn(shots)      # warm-up, discarded
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        decode_batch_fn(shots)
        times.append((time.perf_counter() - t0) / shots.shape[0])
    times = np.array(times)
    spread = {"min_us": float(times.min()) * 1e6, "max_us": float(times.max()) * 1e6,
             "std_us": float(times.std()) * 1e6, "n_repeats": n_repeats}
    return float(np.median(times)), spread


def summarize(decoder, d, rounds, p, alpha, bp_iters, times, n_err, min_errors, reason,
             seconds, n_outliers, batched_spshot=None, batched_spread=None):
    raw_mean = float(np.mean(times))
    clean = times[times <= OUTLIER_CUTOFF_SECONDS]
    if clean.size == 0:              # pathological: keep something rather than divide by zero
        clean = times
    sp = batched_spread or {}
    return {
        "decoder": decoder, "d": d, "rounds": rounds, "p": p,
        "alpha": "" if alpha is None else alpha,
        "bp_iters": "" if bp_iters is None else bp_iters,
        "n_shots": len(times),
        "mean_seconds": float(np.mean(clean)),
        "std_seconds": float(np.std(clean)),
        "median_seconds": float(np.median(clean)),
        "sem_seconds": float(np.std(clean) / np.sqrt(len(clean))),
        "mean_normalized": float("nan"),      # filled in once MWPM's mean is known
        "logical_errors": n_err,
        "target_errors": min_errors,
        "stop_reason": reason,
        "batched_seconds_per_shot": "" if batched_spshot is None else batched_spshot,
        "batched_normalized": "",             # filled in once MWPM's batched mean is known
        "seconds": round(seconds, 1),
        "n_outliers_excluded": n_outliers,
        "raw_mean_seconds": raw_mean,
        "batched_n_repeats": sp.get("n_repeats", ""),
        "batched_min_us": sp.get("min_us", ""),
        "batched_max_us": sp.get("max_us", ""),
        "batched_std_us": sp.get("std_us", ""),
    }


# --------------------------------------------------------------------------------------- csv i/o
def load_existing(path):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows[row["decoder"]] = row
    return rows


def write_csv(path, rows_by_decoder):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name in ORDER:
            if name in rows_by_decoder:
                # keep only known fields (older CSV rows may lack newer columns)
                row = {k: rows_by_decoder[name].get(k, "") for k in FIELDS}
                w.writerow(row)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------- plot
# MWPM is the normalisation baseline (excluded from the bars -- its own value is always
# exactly 1.0x MWPM, which is not informative to plot). The compared decoders use this
# project's established Wong/Okabe-Ito colorblind-safe categorical palette (also used in
# figures_thesis_singlecol/_scripts/thesis_style.py:DCOL), reused rather than invented.
PLOT_COLORS = {"cm": "#D55E00", "rcm": "#009E73", "belief": "#0072B2", "tesseract": "#CC79A7"}


def _log_tick_formatter(value, pos):
    """Major-tick label for the log-scale y-axis: plain "1" at 10^0 (the MWPM baseline,
    read more naturally as "1" than "10^0"), the usual "10^n" mathtext for every other
    power of ten."""
    if value <= 0:
        return ""
    n = round(np.log10(value))
    if abs(value - 10.0 ** n) > 1e-9 * value:   # not a clean power of ten -> minor tick
        return ""
    return "1" if n == 0 else rf"$10^{{{n}}}$"


def make_plot(rows_by_decoder, out_plot, d, rounds, p):
    """One bar per decoder (MWPM excluded -- it is the normalisation baseline, always
    exactly 1.0x itself). Every decoder is shown at its overhead-free, amortized
    decode_batch() latency; a decoder whose row lacks one (e.g. an older CSV row from
    before Tesseract's decode_batch() was wired in, or a run with --no-also-batched)
    falls back to its single-shot decode() latency instead, so the plot never breaks --
    but note that number is NOT directly comparable to the batched bars beside it (see
    this script's module docstring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [n for n in ORDER if n in rows_by_decoder and n != "mwpm"]
    x = np.arange(len(present))

    def _value_and_err(n):
        """Error bar = standard error of the mean (std / sqrt(n)), not the raw spread --
        it answers "how precisely do we know the mean latency", which is what a reader
        comparing bars actually wants. Same statistic in both branches: batched rows use
        the spread across the --batch-repeats decode_batch() calls, the single-shot
        fallback uses sem_seconds (already std/sqrt(n_shots)) directly."""
        r = rows_by_decoder[n]
        if r.get("batched_normalized", "") in ("", None):
            v = float(r["mean_normalized"])
            err = float(r["sem_seconds"]) / float(r["mean_seconds"]) * v
        else:
            v = float(r["batched_normalized"])
            std_us = r.get("batched_std_us", "")
            n_rep = r.get("batched_n_repeats", "")
            mean_us = float(r["batched_seconds_per_shot"]) * 1e6
            if std_us not in ("", None) and n_rep not in ("", None) and int(n_rep) > 0:
                sem_us = float(std_us) / np.sqrt(int(n_rep))
                err = sem_us / mean_us * v
            else:
                err = 0.0
        return v, err

    vals, errs = zip(*(_value_and_err(n) for n in present))
    colors = [PLOT_COLORS.get(n, "0.4") for n in present]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor="0.2", linewidth=0.6,
          width=0.6, zorder=3,
          error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "0.25"})
    ax.axhline(1.0, color="0.3", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_log_tick_formatter)
    ax.set_xticks(x, labels=[LABELS[n] for n in present])
    ax.set_ylabel("Decoder latency / MWPM latency")
    ax.set_title(rf"$d={d}$, $r={rounds}$, $p={p:g}$")
    for xi, yi in zip(x, vals):
        ax.text(xi, yi * 1.10, f"{yi:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", which="both", color="0.85", lw=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_plot) or ".", exist_ok=True)
    fig.savefig(out_plot, dpi=200)
    root, _ = os.path.splitext(out_plot)
    fig.savefig(root + ".pdf")
    plt.close(fig)
    print(f"saved {out_plot} (+ .pdf)")


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--p", type=float, default=1e-3,
                    help="physical error rate (default 1e-3: a representative near-"
                         "threshold operating point common across this thesis' benchmarks; "
                         "decode latency is only weakly p-dependent)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="RCM damping strength (default 0.5: optimal at d=5, p=1e-3 per "
                         "data/reg_cm_alpha_scan_new_combined.csv -- pick the right value "
                         "for --p if you change it)")
    ap.add_argument("--bp-iters", type=int, default=5, help="Belief Matching max BP iterations")
    ap.add_argument("--tesseract-det-beam", type=int, default=50)
    ap.add_argument("--min-errors", type=int, default=100,
                    help="adaptive mode: sample until this many logical errors are seen, per "
                         "decoder (ignored if --fixed-shots is set)")
    ap.add_argument("--fixed-shots", type=int, default=0,
                    help="fixed mode: every decoder decodes EXACTLY this many shots instead of "
                         "sampling to --min-errors -- for a same-sample-size comparison across "
                         "decoders. Pick N from a calibration run of the least-accurate-"
                         "tolerant decoder (Tesseract) at --min-errors, so every decoder gets "
                         "at least that many errors' worth of statistics. 0 = adaptive mode "
                         "(default).")
    ap.add_argument("--chunk", type=int, default=20_000, help="shots per sampling round")
    ap.add_argument("--max-shots", type=int, default=20_000_000, help="per-decoder safety cap")
    ap.add_argument("--max-seconds", type=float, default=7200.0,
                    help="per-decoder wall-clock safety cap (default 2h)")
    ap.add_argument("--n-warmup", type=int, default=500, help="untimed warm-up decodes")
    ap.add_argument("--also-batched", action=argparse.BooleanOptionalAction, default=True,
                    help="also time one amortized decode_batch() call per decoder (where "
                         "available) as an overhead-free reference alongside the fair, "
                         "single-shot-call number used for the plot")
    ap.add_argument("--batch-cap", type=int, default=50_000,
                    help="max shots used for the --also-batched reference call")
    ap.add_argument("--batch-repeats", type=int, default=21,
                    help="repeated decode_batch() calls to median over per decoder (up from "
                         "an original 7 -- tightens the estimate against machine drift/stalls)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--decoders", type=str, default="mwpm,cm,rcm,belief,tesseract",
                    help="comma-separated subset of {mwpm,cm,rcm,belief,tesseract} to (re)run")
    ap.add_argument("--out-csv", type=str, default="data/decoder_latency_d5.csv")
    ap.add_argument("--out-plot", type=str, default="plots/decoder_latency_d5.png")
    a = ap.parse_args()

    wanted = [w.strip() for w in a.decoders.split(",") if w.strip()]
    for w in wanted:
        if w not in ORDER:
            sys.exit(f"unknown decoder '{w}'; choose from {ORDER}")

    mode_desc = (f"fixed_shots={a.fixed_shots:,}" if a.fixed_shots > 0
                else f"min_errors={a.min_errors}  max_shots={a.max_shots:,}")
    print(f"d={a.distance} rounds={a.rounds} p={a.p:g} alpha(RCM)={a.alpha} "
          f"bp_iters={a.bp_iters}\n{mode_desc}  chunk={a.chunk}  "
          f"max_seconds={a.max_seconds:g}  decoders={wanted}\n",
          flush=True)

    circ = make_circuit(a.distance, a.rounds, a.p)
    dem_decomp = decompose_errors_for_stim_surface_code_coords(
        circ.detector_error_model(decompose_errors=False))

    rows = load_existing(a.out_csv)

    # -- MWPM / CM / RCM: the custom-fork Matching object exposes enable_correlations/alpha
    # directly, so one pair of Matching objects (one built with correlations off, one on)
    # covers all three decoders. --
    if any(k in wanted for k in ("mwpm", "cm", "rcm")):
        m_plain = pymatching.Matching.from_detector_error_model(dem_decomp,
                                                                 enable_correlations=False)
        m_corr = pymatching.Matching.from_detector_error_model(dem_decomp,
                                                                enable_correlations=True)

    def run(name, fn, alpha, bp_iters, decode_batch_fn):
        sampler = circ.compile_detector_sampler(seed=a.seed + 7919 * (ORDER.index(name) + 1))
        t0 = time.perf_counter()
        t, e, dets_all, reason, n_out = adaptive_time_decode(
            fn, sampler, n_warmup=a.n_warmup, chunk=a.chunk, min_errors=a.min_errors,
            max_shots=a.max_shots, max_seconds=a.max_seconds, label=LABELS[name],
            fixed_shots=(a.fixed_shots if a.fixed_shots > 0 else None))
        seconds = time.perf_counter() - t0
        bspshot, bspread = time_batched(decode_batch_fn if a.also_batched else None,
                                        dets_all, a.batch_cap, n_repeats=a.batch_repeats)
        rows[name] = summarize(name, a.distance, a.rounds, a.p, alpha, bp_iters,
                               t, e, a.min_errors, reason, seconds, n_out, bspshot, bspread)
        if bspshot is not None:
            extra = (f"  |  batched (amortized, median of {bspread['n_repeats']})="
                    f"{bspshot*1e6:.2f} us/shot [{bspread['min_us']:.2f}-{bspread['max_us']:.2f}, "
                    f"std={bspread['std_us']:.2f}]")
        else:
            extra = ""
        outlier_note = f"  ({n_out} outlier(s) excluded)" if n_out else ""
        print(f"  -> {LABELS[name]}: {len(t):,} shots, {e} errors, {seconds:.1f}s, "
              f"single-shot mean={rows[name]['mean_seconds']*1e6:.2f} us/shot{outlier_note}"
              f"{extra}\n", flush=True)

    if "mwpm" in wanted:
        print("timing MWPM...", flush=True)
        run("mwpm", lambda z: m_plain.decode(z, enable_correlations=False), None, None,
            lambda s: m_plain.decode_batch(s, enable_correlations=False))

    if "cm" in wanted:
        print("timing CM (alpha=1.0)...", flush=True)
        run("cm", lambda z: m_corr.decode(z, enable_correlations=True, alpha=1.0), 1.0, None,
            lambda s: m_corr.decode_batch(s, enable_correlations=True, alpha=1.0))

    if "rcm" in wanted:
        print(f"timing RCM (alpha={a.alpha})...", flush=True)
        run("rcm", lambda z: m_corr.decode(z, enable_correlations=True, alpha=a.alpha), a.alpha,
            None, lambda s: m_corr.decode_batch(s, enable_correlations=True, alpha=a.alpha))

    if "belief" in wanted:
        try:
            from beliefmatching import BeliefMatching
        except ImportError:
            print("  [!] beliefmatching not installed -- skipping Belief Matching.")
        else:
            print(f"timing Belief Matching ({a.bp_iters} BP iters)...", flush=True)
            bm = BeliefMatching(dem_decomp, max_bp_iters=a.bp_iters)
            run("belief", lambda z: bm.decode(z), None, a.bp_iters, bm.decode_batch)

    if "tesseract" in wanted:
        try:
            from tesseract_decoder import tesseract
        except ImportError:
            print("  [!] tesseract_decoder not installed (Linux/WSL-only build in this repo "
                  "-- see scripts/ler_vs_p_tesseract.py). Skipping Tesseract; re-run this "
                  "script with --decoders tesseract from WSL/the cluster to fill in that row.")
        else:
            print(f"timing Tesseract (det_beam={a.tesseract_det_beam})...", flush=True)
            dem_raw = circ.detector_error_model(decompose_errors=False)
            n_obs = circ.num_observables
            tess = tesseract.TesseractConfig(
                dem=dem_raw, det_beam=a.tesseract_det_beam).compile_decoder()
            fn = lambda z: np.asarray(tess.decode(z), dtype=np.uint8).reshape(n_obs)
            # Tesseract's compiled decoder DOES expose decode_batch(syndromes) -> predictions
            # (confirmed against its docstring and a direct correctness/timing check -- an
            # earlier version of this script wrongly assumed it didn't exist, because
            # scripts/ler_vs_p_tesseract.py's docstring describes Tesseract as a "slow
            # single-shot decoder", which is a statement about its per-shot cost, not its
            # API surface). Batched and single-shot land close together for Tesseract
            # specifically (unlike MWPM/CM/RCM) because its ~650+ us/shot of real A* search
            # work dwarfs the ~10-70 us of fixed Python-call overhead that dominates the
            # much cheaper matching-based decoders.
            run("tesseract", fn, None, None, tess.decode_batch)

    # Re-merge with whatever is on disk NOW, right before writing: this run may have taken
    # a long time (e.g. Belief Matching / RCM sampling to --min-errors can run for tens of
    # minutes to hours), and a concurrent invocation covering OTHER decoders (e.g. a
    # Tesseract-only run from WSL) could have finished and written its row in the meantime.
    # A blind overwrite from this process's `rows` snapshot -- taken at ITS start -- would
    # silently drop that row. Only decoders THIS run actually computed should override the
    # on-disk version; everything else on disk is kept as-is.
    #
    # NOTE: `rows` also holds stale entries loaded by the `rows = load_existing(...)` call at
    # the top of main() for decoders NOT in `wanted` (kept around only so their normalization
    # below has something to divide by). Those must NOT be pushed back out here -- only the
    # decoders this run actually recomputed (`wanted`) should override on_disk; a prior
    # `on_disk.update(rows)` pushed the whole stale dict and clobbered a freshly-written
    # Tesseract row from a concurrent WSL run with the stale one this process had loaded at
    # its own start.
    on_disk = load_existing(a.out_csv)
    for name in wanted:
        if name in rows:
            on_disk[name] = rows[name]
    rows = on_disk

    if "mwpm" not in rows:
        sys.exit("no MWPM row available (in this run or the existing CSV) to normalise against; "
                 "run with mwpm included at least once.")
    mwpm_mean = float(rows["mwpm"]["mean_seconds"])
    mwpm_batched = rows["mwpm"].get("batched_seconds_per_shot", "")
    mwpm_batched = float(mwpm_batched) if mwpm_batched not in ("", None) else None
    for name in rows:
        rows[name]["mean_normalized"] = float(rows[name]["mean_seconds"]) / mwpm_mean
        bsp = rows[name].get("batched_seconds_per_shot", "")
        if bsp not in ("", None) and mwpm_batched is not None:
            rows[name]["batched_normalized"] = float(bsp) / mwpm_batched
        else:
            rows[name]["batched_normalized"] = ""

    write_csv(a.out_csv, rows)
    print(f"\nsaved {a.out_csv}")

    print("\ndecoder          shots       errors   single-shot(us)  x MWPM    batched(us)  x MWPM")
    for name in ORDER:
        if name in rows:
            r = rows[name]
            bsp = r.get("batched_seconds_per_shot", "")
            bnorm = r.get("batched_normalized", "")
            bsp_str = f"{float(bsp)*1e6:>10.2f}" if bsp not in ("", None) else "       n/a"
            bnorm_str = f"{float(bnorm):>6.2f}x" if bnorm not in ("", None) else "    n/a"
            print(f"  {LABELS[name]:<15} {int(r['n_shots']):>9,}  {int(r['logical_errors']):>6}   "
                  f"{float(r['mean_seconds'])*1e6:>13.2f}   {float(r['mean_normalized']):>6.2f}x   "
                  f"{bsp_str}   {bnorm_str}")

    make_plot(rows, a.out_plot, a.distance, a.rounds, a.p)


if __name__ == "__main__":
    main()