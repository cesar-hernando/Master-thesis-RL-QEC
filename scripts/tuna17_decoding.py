"""
Tuna-17 Distance-3 Surface Code Memory Experiment Pipeline
===========================================================
Compatible with:  qiskit-quantuminspire >= 0.12  (QI2 platform SDK)
                  quantuminspire >= 2.0           (QI2 CLI/auth tool)
                  qiskit >= 1.0
"""

import os
import sys
import time
import types
import numpy as np
import pymatching
import stim

# =============================================================================
# QI2 COMPUTE API CLIENT PATCH
# =============================================================================
import compute_api_client as _compute_api_client
import compute_api_client.models as _compute_api_models
from compute_api_client.models.job_patch import JobPatch as _JobPatch
from pydantic import BaseModel, ConfigDict


class _CompatJob(BaseModel):
    job_id: int | None = None
    status: object | None = None
    message: str | None = None
    trace_id: str | None = None
    batch_job_id: int | None = None
    file_id: int | None = None

    model_config = ConfigDict(extra="allow")

    @property
    def id(self):
        return self.job_id

    @classmethod
    def from_dict(cls, obj):
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            return cls.model_validate(getattr(obj, "__dict__", {}))
        normalized = dict(obj)
        if normalized.get("job_id") is None and normalized.get("id") is not None:
            normalized["job_id"] = normalized["id"]
        return cls.model_validate(normalized)

    def to_dict(self):
        return self.model_dump()

_job_module = types.ModuleType("compute_api_client.models.job")
_job_module.Job = _CompatJob
sys.modules.setdefault("compute_api_client.models.job", _job_module)

from compute_api_client.api.backend_types_api import BackendTypesApi as _BackendTypesApi
from compute_api_client.api.batch_jobs_api import BatchJobsApi as _BatchJobsApi
from compute_api_client.api.commits_api import CommitsApi as _CommitsApi
from compute_api_client.api.files_api import FilesApi as _FilesApi
from compute_api_client.api.jobs_api import JobsApi as _JobsApi
from compute_api_client.api.languages_api import LanguagesApi as _LanguagesApi
from compute_api_client.api.projects_api import ProjectsApi as _ProjectsApi
from compute_api_client.api.results_api import ResultsApi as _ResultsApi
from compute_api_client.api.algorithms_api import AlgorithmsApi as _AlgorithmsApi
from compute_api_client.api_client import ApiClient as _ApiClient
from compute_api_client.configuration import Configuration as _Configuration
from compute_api_client.models.algorithm import Algorithm as _Algorithm
from compute_api_client.models.algorithm_in import AlgorithmIn as _AlgorithmIn
from compute_api_client.models.algorithm_type import AlgorithmType as _AlgorithmType
from compute_api_client.models.backend_status import BackendStatus as _BackendStatus
from compute_api_client.models.backend_type import BackendType as _BackendType
from compute_api_client.models.batch_job import BatchJob as _BatchJob
from compute_api_client.models.batch_job_in import BatchJobIn as _BatchJobIn
from compute_api_client.models.batch_job_status import BatchJobStatus as _BatchJobStatus
from compute_api_client.models.commit import Commit as _Commit
from compute_api_client.models.commit_in import CommitIn as _CommitIn
from compute_api_client.models.compile_stage import CompileStage as _CompileStage
from compute_api_client.models.file import File as _File
from compute_api_client.models.file_in import FileIn as _FileIn
from compute_api_client.models.job_in import JobIn as _JobIn
from compute_api_client.models.job_status import JobStatus as _JobStatus
from compute_api_client.models.language import Language as _Language
from compute_api_client.models.page_language import PageLanguage as _PageLanguage
from compute_api_client.models.page_backend_type import PageBackendType as _PageBackendType
from compute_api_client.models.page_batch_job import PageBatchJob as _PageBatchJob
from compute_api_client.models.page_result import PageResult as _PageResult
from compute_api_client.models.project import Project as _Project
from compute_api_client.models.project_in import ProjectIn as _ProjectIn
from compute_api_client.models.result import Result as _Result
from compute_api_client.models.result_in import ResultIn as _ResultIn
from compute_api_client.models.share_type import ShareType as _ShareType

_compute_api_client.ApiClient = _ApiClient
_compute_api_client.Configuration = _Configuration
_compute_api_client.BackendTypesApi = _BackendTypesApi
_compute_api_client.BatchJobsApi = _BatchJobsApi
_compute_api_client.CommitsApi = _CommitsApi
_compute_api_client.FilesApi = _FilesApi
_compute_api_client.JobsApi = _JobsApi
_compute_api_client.LanguagesApi = _LanguagesApi
_compute_api_client.ProjectsApi = _ProjectsApi
_compute_api_client.ResultsApi = _ResultsApi
_compute_api_client.AlgorithmsApi = _AlgorithmsApi
_compute_api_client.Algorithm = _Algorithm
_compute_api_client.AlgorithmIn = _AlgorithmIn
_compute_api_client.AlgorithmType = _AlgorithmType
_compute_api_client.BackendStatus = _BackendStatus
_compute_api_client.BackendType = _BackendType
_compute_api_client.BatchJob = _BatchJob
_compute_api_client.BatchJobIn = _BatchJobIn
_compute_api_client.BatchJobStatus = _BatchJobStatus
_compute_api_client.Commit = _Commit
_compute_api_client.CommitIn = _CommitIn
_compute_api_client.CompileStage = _CompileStage
_compute_api_client.File = _File
_compute_api_client.FileIn = _FileIn
_compute_api_client.Job = _CompatJob
_compute_api_client.JobIn = _JobIn
_compute_api_client.JobStatus = _JobStatus
_compute_api_client.Language = _Language
_compute_api_client.PageBackendType = _PageBackendType
_compute_api_client.PageBatchJob = _PageBatchJob
_compute_api_client.PageResult = _PageResult
_compute_api_client.Project = _Project
_compute_api_client.ProjectIn = _ProjectIn
_compute_api_client.Result = _Result
_compute_api_client.ResultIn = _ResultIn
_compute_api_client.ShareType = _ShareType

_compute_api_models.Algorithm = _Algorithm
_compute_api_models.AlgorithmIn = _AlgorithmIn
_compute_api_models.AlgorithmType = _AlgorithmType
_compute_api_models.BackendStatus = _BackendStatus
_compute_api_models.BackendType = _BackendType
_compute_api_models.BatchJob = _BatchJob
_compute_api_models.BatchJobIn = _BatchJobIn
_compute_api_models.BatchJobStatus = _BatchJobStatus
_compute_api_models.Commit = _Commit
_compute_api_models.CommitIn = _CommitIn
_compute_api_models.CompileStage = _CompileStage
_compute_api_models.File = _File
_compute_api_models.FileIn = _FileIn
_compute_api_models.Job = _CompatJob
_compute_api_models.JobIn = _JobIn
_compute_api_models.JobPatch = _JobPatch
_compute_api_models.JobStatus = _JobStatus
_compute_api_models.Language = _Language
_compute_api_models.PageLanguage = _PageLanguage
_compute_api_models.PageBackendType = _PageBackendType
_compute_api_models.PageBatchJob = _PageBatchJob
_compute_api_models.PageResult = _PageResult
_compute_api_models.PageJob = _PageResult
_compute_api_models.Project = _Project
_compute_api_models.ProjectIn = _ProjectIn
_compute_api_models.Result = _Result
_compute_api_models.ResultIn = _ResultIn
_compute_api_models.ShareType = _ShareType

# =============================================================================
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_quantuminspire.qi_provider import QIProvider


# =============================================================================
# CONFIGURATION
# =============================================================================
N_SHOTS        = 2048
N_ROUNDS       = 3
BACKEND_NAME   = "Tuna-17"
PHYSICAL_NOISE = 0.02
DATA_DIR       = "data"
FILENAME       = f"{DATA_DIR}/{BACKEND_NAME}_d3_r{N_ROUNDS}_shots.npz"


# =============================================================================
# 1. HARDWARE CIRCUIT GENERATION
# =============================================================================

def build_d3_surface_code(rounds: int = 3) -> QuantumCircuit:
    """
    Build a distance-3 rotated surface code Z-memory experiment for Tuna-17.

    Qubit layout (all 17 Tuna-17 physical qubits):
        Data qubits : Q0–Q8   (indices 0–8)
        X ancillas  : Q9–Q12  (indices 9–12)
        Z ancillas  : Q13–Q16 (indices 13–16)

    Classical register layout per shot:
        A single unified register is used to prevent hardware backend truncation.
        Length: 8*rounds + 9.
    """
    q  = QuantumRegister(17, 'q')
    cr = ClassicalRegister(8 * rounds + 9, 'c')
    qc = QuantumCircuit(q, cr)

    # ── 4-step CX schedule ────────────────────────────────────────────────────
    Z_NW = [(0, 14), (4, 15), (6, 16)];  X_NW = [(10, 1), (11, 3), (12, 5)]
    Z_NE = [(1, 14), (5, 15), (7, 16)];  X_NE = [(10, 2), (11, 4), (9,  0)]
    Z_SW = [(3, 14), (7, 15), (1, 13)];  X_SW = [(10, 4), (11, 6), (12, 8)]
    Z_SE = [(4, 14), (8, 15), (2, 13)];  X_SE = [(10, 5), (11, 7), (9,  3)]

    def apply_step(circuit, z_pairs, x_pairs):
        for dq, aq in z_pairs:
            circuit.cx(q[dq], q[aq])
        for aq, dq in x_pairs:
            circuit.cx(q[aq], q[dq])

    qc.barrier()

    for r in range(rounds):
        # Reset ancillas every round (explicit reset mirrors the Stim DEM)
        qc.reset([q[i] for i in range(9, 17)])

        # Prepare X ancillas in |+⟩
        qc.h([q[i] for i in range(9, 13)])

        # 4-step syndrome extraction
        apply_step(qc, Z_NW, X_NW)
        apply_step(qc, Z_NE, X_NE)
        apply_step(qc, Z_SW, X_SW)
        apply_step(qc, Z_SE, X_SE)

        # Un-Hadamard X ancillas
        qc.h([q[i] for i in range(9, 13)])

        # Measure ancillas into the single syndrome register sequentially
        for i in range(4):
            qc.measure(q[9  + i], cr[r * 8 + i])
            qc.measure(q[13 + i], cr[r * 8 + 4 + i])

        qc.barrier()

    # Final data readout mapped to the end of the unified register
    for i in range(9):
        qc.measure(q[i], cr[8 * rounds + i])
        
    return qc


# =============================================================================
# 2. HARDWARE EXECUTION & RESULT PARSING
# =============================================================================

def parse_memory_shot(shot_str: str, rounds: int):
    """
    Parse one raw memory string from result.get_memory() into syndrome bits
    and the logical observable.

    Fixes applied: 
    1. Reverses string to account for Qiskit's MSB-first formatting.
    2. Uses fixed lengths to correctly slice the single unified string.
    """
    clean = shot_str.replace(" ", "").strip()
    
    # Pad to ensure length is correct, though it should be exactly 8*rounds + 9
    total_bits = (8 * rounds) + 9
    clean = clean.zfill(total_bits)
    
    # REVERSE the string! Qiskit prints MSB-first. Reversing it means 
    # rev_str[0] corresponds exactly to cr[0].
    rev_str = clean[::-1]

    # Extract syndromes (first 24 measurements)
    syn_bits = [int(b) for b in rev_str[:8 * rounds]]

    # Extract data (last 9 measurements)
    data_bits = [int(b) for b in rev_str[8 * rounds:]]

    # Because rev_str[0] = cr[0], data_bits[0] corresponds to q[0]
    # Logical Z_L = parity of left column: Q0, Q3, Q6
    q0 = data_bits[0]
    q3 = data_bits[3]
    q6 = data_bits[6]
    logical_flip = bool((q0 ^ q3 ^ q6) == 1)

    return syn_bits, logical_flip


def run_and_parse_surface_code(
    backend_name: str = "Tuna-17",
    shots: int = 2048,
    rounds: int = 3,
):
    """
    Submit the surface code circuit to QI2 hardware and parse raw shot data.
    """
    print("[*] Connecting to Quantum Inspire (QI2 platform)...")
    print("    (Credentials from ~/.quantuminspire/config.json — run `qi login` if missing)")
    provider = QIProvider()

    print(f"[*] Available backends: {[b.name for b in provider.backends()]}")
    backend = provider.get_backend(backend_name)

    qc = build_d3_surface_code(rounds=rounds)
    print(f"[*] Circuit built: {qc.num_qubits} qubits, depth {qc.depth()}")

    # Transpile — try strict zero-SWAP first, fall back to standard routing
    print("[*] Transpiling for Tuna-17 topology...")
    try:
        qc_t = transpile(
            qc,
            backend=backend,
            optimization_level=3,
            routing_method='none',  # enforce zero-SWAP to preserve error model
        )
        print(f"    SWAP-free transpilation succeeded (depth={qc_t.depth()})")
    except Exception as exc:
        print(f"    Zero-SWAP routing failed ({exc}); falling back to standard routing.")
        qc_t = transpile(qc, backend=backend, optimization_level=3)
        print(f"    Standard transpilation (depth={qc_t.depth()})")

    # Submit
    print(f"[*] Submitting job to {backend_name} ({shots} shots) ...")
    job = backend.run(qc_t, shots=shots, memory=True)
    job_id = getattr(job, 'batch_job_id', None)
    if job_id is None:
        job_id = getattr(job, 'id', None)
    if job_id is None:
        job_id = 'unknown'
    print(f"[*] Job submitted. ID: {job_id}")
    print("[*] Waiting for result (may take several hours on hardware queue)...")

    # Poll for result with network-error resilience
    result = None
    poll_interval = 60  # seconds
    while result is None:
        try:
            result = job.result()
        except Exception as exc:
            err_str = str(exc).lower()
            if any(k in err_str for k in ("timeout", "network", "connection", "gaierror")):
                print(f"    [!] Transient error: {exc}. Retrying in {poll_interval}s...")
                time.sleep(poll_interval)
            else:
                raise

    print("[*] Job complete. Parsing raw memory...")
    raw_memory = result.get_memory()
    if not raw_memory:
        raise RuntimeError(
            "result.get_memory() returned no data. "
            "Ensure the circuit was submitted with memory=True and the backend supports raw shot data."
        )

    raw_syndromes_list = []
    true_observables_list = []

    for shot_str in raw_memory:
        syn_bits, logical_flip = parse_memory_shot(shot_str, rounds=rounds)
        raw_syndromes_list.append(syn_bits)
        true_observables_list.append(logical_flip)

    return (
        np.array(raw_syndromes_list,    dtype=np.int8),
        np.array(true_observables_list, dtype=bool),
    )


# =============================================================================
# 3. DETECTOR EVENT CONVERSION
# =============================================================================

def raw_syndromes_to_detectors(raw_syndromes: np.ndarray) -> np.ndarray:
    """
    Convert raw ancilla measurements into relative detector events.

    Bit layout of raw_syndromes (columns):
        0– 3  : Round 0 · X ancillas Q9–Q12
        4– 7  : Round 0 · Z ancillas Q13–Q16
        8–11  : Round 1 · X ancillas Q9–Q12
        12–15 : Round 1 · Z ancillas Q13–Q16
        16–19 : Round 2 · X ancillas Q9–Q12
        20–23 : Round 2 · Z ancillas Q13–Q16

    Detector layout (20 total = 4 + 8 + 8):
        0– 3  : Z detectors round 0 (absolute — no prior round)
        4– 7  : X detectors round 1 (round-1 XOR round-0)
        8–11  : Z detectors round 1 (round-1 XOR round-0)
        12–15 : X detectors round 2 (round-2 XOR round-1)
        16–19 : Z detectors round 2 (round-2 XOR round-1)
    """
    assert raw_syndromes.shape[1] == 24, (
        f"Expected 24 raw syndrome bits (3 rounds × 8 ancillas), "
        f"got {raw_syndromes.shape[1]}"
    )
    n = raw_syndromes.shape[0]
    detectors = np.zeros((n, 20), dtype=np.int8)

    X_r0 = raw_syndromes[:, 0:4];   Z_r0 = raw_syndromes[:, 4:8]
    X_r1 = raw_syndromes[:, 8:12];  Z_r1 = raw_syndromes[:, 12:16]
    X_r2 = raw_syndromes[:, 16:20]; Z_r2 = raw_syndromes[:, 20:24]

    detectors[:, 0:4]   = Z_r0           # round 0: Z only
    detectors[:, 4:8]   = X_r1 ^ X_r0   # round 1: X diff
    detectors[:, 8:12]  = Z_r1 ^ Z_r0   # round 1: Z diff
    detectors[:, 12:16] = X_r2 ^ X_r1   # round 2: X diff
    detectors[:, 16:20] = Z_r2 ^ Z_r1   # round 2: Z diff

    return detectors


# =============================================================================
# 4. STIM DETECTOR ERROR MODEL
# =============================================================================

def build_matching_dem(
    rounds: int = 3,
    physical_noise: float = 0.02,
) -> stim.DetectorErrorModel:
    """
    Build a Stim DEM mirroring the Tuna-17 d=3 Z-memory circuit.

    Noise model:
        DEPOLARIZE2(p) on every two-qubit CX gate
        X_ERROR(p)     on ancilla reset (rounds > 0 only)
        M(p)           on every ancilla measurement  ← readout misclassification

    Observable: logical Z_L = Q0 ⊗ Q3 ⊗ Q6 (left column of data grid).
    """
    c = stim.Circuit()

    DATA    = list(range(9))
    X_ANC   = list(range(9,  13))
    Z_ANC   = list(range(13, 17))
    ANC_ALL = X_ANC + Z_ANC

    Z_NW = [(0, 14), (4, 15), (6, 16)];  X_NW = [(10, 1), (11, 3), (12, 5)]
    Z_NE = [(1, 14), (5, 15), (7, 16)];  X_NE = [(10, 2), (11, 4), (9,  0)]
    Z_SW = [(3, 14), (7, 15), (1, 13)];  X_SW = [(10, 4), (11, 6), (12, 8)]
    Z_SE = [(4, 14), (8, 15), (2, 13)];  X_SE = [(10, 5), (11, 7), (9,  3)]

    def apply_step(circuit, z_pairs, x_pairs):
        for dq, aq in z_pairs:
            circuit.append("CX", [dq, aq])
        for aq, dq in x_pairs:
            circuit.append("CX", [aq, dq])
        flat = [q for pair in z_pairs + [(a, d) for a, d in x_pairs] for q in pair]
        if flat:
            circuit.append("DEPOLARIZE2", flat, physical_noise)

    for r in range(rounds):
        c.append("R", ANC_ALL)
        if r > 0:
            c.append("X_ERROR", ANC_ALL, physical_noise)

        c.append("H", X_ANC)
        apply_step(c, Z_NW, X_NW)
        apply_step(c, Z_NE, X_NE)
        apply_step(c, Z_SW, X_SW)
        apply_step(c, Z_SE, X_SE)
        c.append("H", X_ANC)

        # M(p) correctly models readout bit-flip (unlike X_ERROR after M)
        c.append("M", ANC_ALL, physical_noise)

        if r == 0:
            for i in range(4):
                c.append("DETECTOR", [stim.target_rec(-4 + i)])
        else:
            for i in range(4):
                c.append("DETECTOR", [stim.target_rec(-8 + i), stim.target_rec(-16 + i)])
            for i in range(4):
                c.append("DETECTOR", [stim.target_rec(-4 + i), stim.target_rec(-12 + i)])

    c.append("M", DATA)
    c.append("OBSERVABLE_INCLUDE", [
        stim.target_rec(-9),   # Q0
        stim.target_rec(-6),   # Q3
        stim.target_rec(-3),   # Q6
    ], 0)

    return c.detector_error_model(decompose_errors=True)


# =============================================================================
# 5. UTILITIES
# =============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval."""
    if n == 0:
        return 0.0, 1.0
    p      = k / n
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    print(f"[*] stim version      : {stim.__version__}")
    print(f"[*] pymatching version: {pymatching.__version__}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 6.1 Acquire data ──────────────────────────────────────────────────────
    if os.path.exists(FILENAME):
        print(f"\n[*] Loading cached dataset from {FILENAME}")
        data             = np.load(FILENAME)
        raw_syndromes    = data['syndromes']
        true_observables = data['observables']
    else:
        print(f"\n[*] No cached dataset found — submitting to {BACKEND_NAME}...")
        raw_syndromes, true_observables = run_and_parse_surface_code(
            backend_name=BACKEND_NAME,
            shots=N_SHOTS,
            rounds=N_ROUNDS,
        )
        np.savez_compressed(FILENAME, syndromes=raw_syndromes, observables=true_observables)
        print(f"[*] Data saved to {FILENAME}")

    n_shots_actual  = raw_syndromes.shape[0]
    base_err_count  = int(np.sum(true_observables))

    print(f"\n[*] Dataset: {n_shots_actual:,} shots, "
          f"{raw_syndromes.shape[1]} syndrome bits per shot")
    print(f"    Uncorrected logical error rate: "
          f"{base_err_count/n_shots_actual:.4%} ({base_err_count}/{n_shots_actual})")

    # ── 6.2 Detector events ───────────────────────────────────────────────────
    print("\n[*] Computing detector events...")
    detector_events = raw_syndromes_to_detectors(raw_syndromes)
    assert detector_events.shape == (n_shots_actual, 20), (
        f"Expected shape ({n_shots_actual}, 20), got {detector_events.shape}"
    )
    print(f"    Detector matrix: {detector_events.shape}  "
          f"(avg firing rate = {detector_events.mean():.4f})")

    # ── 6.3 Build DEM ─────────────────────────────────────────────────────────
    print(f"\n[*] Building Stim DEM (physical_noise={PHYSICAL_NOISE})...")
    dem = build_matching_dem(rounds=N_ROUNDS, physical_noise=PHYSICAL_NOISE)
    print(f"    Detectors: {dem.num_detectors}  |  Error mechanisms: {dem.num_errors}")

    # ── 6.4 Decode ────────────────────────────────────────────────────────────
    print("\n[*] Decoding: Uncorrelated MWPM...")
    m_uncorr      = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    pred_uncorr   = m_uncorr.decode_batch(detector_events)
    count_uncorr  = int(np.sum(np.logical_xor(true_observables, pred_uncorr[:, 0])))
    ler_uncorr    = count_uncorr / n_shots_actual

    print("[*] Decoding: Correlated MWPM...")
    m_corr        = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    pred_corr     = m_corr.decode_batch(detector_events)
    count_corr    = int(np.sum(np.logical_xor(true_observables, pred_corr[:, 0])))
    ler_corr      = count_corr / n_shots_actual

    # ── 6.5 Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  {BACKEND_NAME} | d=3 Z-memory | {N_ROUNDS} rounds | PyMatching DEM")
    print("=" * 58)
    print(f"  Shots              : {n_shots_actual:>10,}")
    print(f"  DEM noise prior p  : {PHYSICAL_NOISE:>10.3f}\n")

    rows = [
        ("Uncorrected",        base_err_count, base_err_count / n_shots_actual),
        ("Uncorrelated MWPM",  count_uncorr,   ler_uncorr),
        ("Correlated MWPM",    count_corr,     ler_corr),
    ]
    print(f"  {'Decoder':<24}  {'Errors':>8}  {'LER':>10}")
    print(f"  {'-'*46}")
    for label, k, ler in rows:
        print(f"  {label:<24}  {k:>8,}  {ler:>10.4%}")

    print()
    if ler_corr < ler_uncorr:
        print(f"  ✓ Correlated MWPM improves by "
              f"{(ler_uncorr - ler_corr) / ler_uncorr * 100:.1f}% over uncorrelated")
    elif ler_corr == ler_uncorr:
        print("  = Correlated and uncorrelated MWPM give identical LER")
    else:
        print("  ✗ Correlated MWPM is worse — check DEM noise calibration")

    print("\n  95% Wilson confidence intervals:")
    for label, k in [("Uncorrelated", count_uncorr), ("Correlated", count_corr)]:
        lo, hi = wilson_ci(k, n_shots_actual)
        print(f"    {label:<14}: [{lo:.4%}, {hi:.4%}]")