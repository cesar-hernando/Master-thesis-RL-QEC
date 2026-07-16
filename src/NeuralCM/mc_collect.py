"""Parallel Monte-Carlo collection for Regularized-CM sweeps.

A single primitive, `collect`, streams shots and decodes them with MWPM + a set of
regularized-CM alphas, accumulating logical errors until the BEST decoder reaches a
target error count (or a shot / wall-time budget). It optionally spreads the work over
`n_workers` independent-seed processes and pools their counts.

Common random numbers (CRN): within each worker every shot is decoded by *every* alpha
(same `det` chunk), so the per-shot outcomes are paired across alphas. Pooling workers
keeps this property globally (each sampled shot, in whichever worker, was decoded by all
alphas), so the pooled alpha differences retain CRN variance reduction -- seed-splitting
across cores does NOT cost the paired-comparison advantage.
"""
import multiprocessing as mp
import queue as _queue
import time

import numpy as np
import stim
import pymatching

from .decompose_errors import decompose_errors_for_stim_surface_code_coords


# =============================================================================
# Circuit / matchers / one-chunk decode
# =============================================================================
def make_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def build_matchers(d, rounds, p):
    circ = make_circuit(d, rounds, p)
    dem = decompose_errors_for_stim_surface_code_coords(
        circ.detector_error_model(decompose_errors=False))
    mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    return circ, mwpm, cm


def decode_chunk(mwpm, cm, alphas, det, ob):
    """Decode one bit-packed chunk with MWPM and hard/damped CM at each alpha.
    Returns (errors_by_key, seconds_by_key); key = ('mwpm', None) or ('cm', float)."""
    errd, tsd = {}, {}
    ta = time.perf_counter()
    pm = mwpm.decode_batch(det, bit_packed_shots=True)[:, 0]
    tsd[("mwpm", None)] = time.perf_counter() - ta
    errd[("mwpm", None)] = int(np.count_nonzero(pm != ob))
    for a in alphas:
        ta = time.perf_counter()
        pa = cm.decode_batch(det, bit_packed_shots=True,
                             enable_correlations=True, alpha=float(a))[:, 0]
        tsd[("cm", float(a))] = time.perf_counter() - ta
        errd[("cm", float(a))] = int(np.count_nonzero(pa != ob))
    return errd, tsd


def _worker(d, rounds, p, alphas, seed, chunk, stop_event, out_q):
    """Sample+decode chunks with a private RNG seed; push counts until told to stop."""
    circ, mwpm, cm = build_matchers(d, rounds, p)
    sampler = circ.compile_detector_sampler(seed=seed)
    while not stop_event.is_set():
        det, obs = sampler.sample(shots=chunk, separate_observables=True, bit_packed=True)
        ob = obs[:, 0] & 1
        errd, tsd = decode_chunk(mwpm, cm, alphas, det, ob)
        while not stop_event.is_set():
            try:
                out_q.put((chunk, errd, tsd), timeout=0.5)
                break
            except _queue.Full:
                continue


# =============================================================================
# collect
# =============================================================================
def collect(*, d, rounds, p, alphas, target_errors, n_workers=1, seed=12345,
            chunk=1_000_000, max_shots=20_000_000_000, deadline=None, on_update=None):
    """Stream until the best decoder reaches `target_errors` (or max_shots / deadline).

    Parameters
    ----------
    alphas : iterable of float
        Damped CM strengths to test (include 1.0 for hard CM). MWPM is always added.
    n_workers : int
        1 => run inline in this process; >1 => that many independent-seed subprocesses,
        pooled. CRN across alphas is preserved either way.
    deadline : float or None
        Absolute time.time() at which to stop gracefully.
    on_update : callable(err, tsec, shots) or None
        Called after every aggregation step (for incremental CSV / progress).

    Returns
    -------
    (err, tsec, shots, reason) with reason in {"target", "max_shots", "deadline"}.
    """
    alphas = [float(a) for a in alphas]
    keys = [("mwpm", None)] + [("cm", a) for a in alphas]
    err = {k: 0 for k in keys}
    tsec = {k: 0.0 for k in keys}
    shots = 0

    def stop_reason():
        best = min(err, key=lambda k: err[k])
        if err[best] >= target_errors:
            return "target"
        if shots >= max_shots:
            return "max_shots"
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        return None

    # -- single process: simplest, fully reproducible --
    if n_workers <= 1:
        circ, mwpm, cm = build_matchers(d, rounds, p)
        sampler = circ.compile_detector_sampler(seed=seed)
        while True:
            det, obs = sampler.sample(shots=chunk, separate_observables=True, bit_packed=True)
            ob = obs[:, 0] & 1
            errd, tsd = decode_chunk(mwpm, cm, alphas, det, ob)
            shots += chunk
            for k in keys:
                err[k] += errd[k]
                tsec[k] += tsd[k]
            if on_update:
                on_update(err, tsec, shots)
            r = stop_reason()
            if r:
                return err, tsec, shots, r

    # -- multi-process seed split --
    stop_event = mp.Event()
    out_q = mp.Queue(maxsize=4 * n_workers)
    procs = [mp.Process(target=_worker,
                        args=(d, rounds, p, alphas, seed + 7919 * (w + 1),
                              chunk, stop_event, out_q),
                        daemon=True)
             for w in range(n_workers)]
    for pr in procs:
        pr.start()
    reason = "deadline"
    try:
        while True:
            try:
                dchunk, errd, tsd = out_q.get(timeout=1.0)
            except _queue.Empty:
                if deadline is not None and time.time() >= deadline:
                    reason = "deadline"
                    break
                continue
            shots += dchunk
            for k in keys:
                err[k] += errd[k]
                tsec[k] += tsd[k]
            if on_update:
                on_update(err, tsec, shots)
            r = stop_reason()
            if r:
                reason = r
                break
    finally:
        stop_event.set()
        # let any worker blocked on a full queue unblock, then join
        t0 = time.time()
        while any(pr.is_alive() for pr in procs) and time.time() - t0 < 30:
            try:
                out_q.get(timeout=0.2)
            except _queue.Empty:
                pass
        for pr in procs:
            pr.join(timeout=2)
            if pr.is_alive():
                pr.terminate()
    return err, tsec, shots, reason
