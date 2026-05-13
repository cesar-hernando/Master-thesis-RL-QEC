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
# N_ROUNDS=3 matches the canonical Stim surface_code:rotated_memory_z
# generator (1 + (d-1) + 1 effective stabiliser-extraction rounds). One
# round on its own cannot distinguish a data-qubit X error from a Z-ancilla
# readout flip, so it gives no fault tolerance — the canonical 1+d+1
# structure is what makes d=3 actually distance-3.
N_ROUNDS       = 3
BACKEND_NAME   = "Tuna-17"
PHYSICAL_NOISE = 0.03         # ~3% per CX, roughly matched to Tuna-17 specs
DATA_DIR       = "data"
FILENAME       = f"{DATA_DIR}/{BACKEND_NAME}_d3_r{N_ROUNDS}_shots.npz"


# =============================================================================
# 1. HARDWARE CIRCUIT GENERATION
# =============================================================================

# Tuna-17 role assignment derived from the chip's coupling map AND aligned
# to the canonical Stim surface_code:rotated_memory_z generator. The
# generator places the logical-Z observable along the TOP ROW of the data
# grid (X-type top/bottom boundaries, Z-type left/right boundaries), so
# this script swaps roles relative to my earlier hand-rolled schedule:
#
# Stim label  ->  Tuna physical (after the Stim-to-Tuna isomorphism)
#   data Q1, Q3, Q5, Q8, Q10, Q12, Q15, Q17, Q19  ->  Q1, Q2, Q3, Q7, Q8, Q9, Q13, Q14, Q15
#   X-anc Q2  (top boundary)                      ->  Q0
#   X-anc Q11 (top-right interior)                ->  Q5
#   X-anc Q16 (bottom-left interior)              ->  Q11
#   X-anc Q25 (bottom boundary)                   ->  Q16
#   Z-anc Q9  (top-left interior)                 ->  Q4
#   Z-anc Q13 (right boundary)                    ->  Q6
#   Z-anc Q14 (left boundary)                     ->  Q10
#   Z-anc Q18 (bottom-right interior)             ->  Q12
#
# Logical Z = Z(Q1)·Z(Q2)·Z(Q3) (top row in the data 3x3 grid).
DATA_QUBITS = [1, 2, 3, 7, 8, 9, 13, 14, 15]
X_ANCILLAS  = [0, 5, 11, 16]
Z_ANCILLAS  = [4, 6, 10, 12]

# Ancilla measurement order, matching Stim's canonical
#   M 2 9 11 13 14 16 18 25
# under the mapping above. The detector rec[] indices below depend on this
# order, so DO NOT reorder without also fixing the DEM.
ANC_M_ORDER = [0, 4, 5, 6, 10, 11, 12, 16]

# Canonical Tomita-Svore 4-layer CX schedule, taken verbatim from the Stim
# generator's rotated-memory-z circuit and remapped to Tuna-17 indices.
# In each layer X-ancillas drive their data neighbour (CX X_anc -> data)
# while Z-ancillas absorb from theirs (CX data -> Z_anc). All 24 CXs are
# native Tuna-17 edges; each data qubit is touched at most once per layer.
LAYER1_X = [(0, 2), (5, 9), (11, 14)]    # X-anc -> SE-corner data
LAYER1_Z = [(8, 4), (13, 10), (15, 12)]  # SE-corner data -> Z-anc

LAYER2_X = [(0, 1), (5, 8), (11, 13)]    # X-anc -> SW-corner data
LAYER2_Z = [(2, 4), (7, 10), (9, 12)]    # NE-corner data -> Z-anc

LAYER3_X = [(5, 3), (11, 8), (16, 15)]   # X-anc -> NE-corner data
LAYER3_Z = [(7, 4), (9, 6), (14, 12)]    # SW-corner data -> Z-anc

LAYER4_X = [(5, 2), (11, 7), (16, 14)]   # X-anc -> NW-corner data
LAYER4_Z = [(1, 4), (3, 6), (8, 12)]     # NW-corner data -> Z-anc

# Z-stabiliser supports (data-qubit indices into DATA_QUBITS) for each
# Z ancilla. Used by the FINAL detectors and the coupling-map sanity check.
#   Q4  (top-left interior, weight-4): Q1, Q2, Q7, Q8       -> idx 0, 1, 3, 4
#   Q6  (right boundary,   weight-2): Q3, Q9                -> idx 2, 5
#   Q10 (left boundary,    weight-2): Q7, Q13               -> idx 3, 6
#   Q12 (bot-right interior, weight-4): Q8, Q9, Q14, Q15    -> idx 4, 5, 7, 8
Z_STAB_DATA_IDX = {
    4:  [0, 1, 3, 4],
    6:  [2, 5],
    10: [3, 6],
    12: [4, 5, 7, 8],
}


def build_d3_surface_code(rounds: int = 3) -> QuantumCircuit:
    """
    Build the canonical d=3 rotated surface-code Z-memory experiment for
    Tuna-17. Mirrors Stim's surface_code:rotated_memory_z generator
    structure, remapped onto Tuna-17 physical qubits.

    Classical register: single unified register of length 8*rounds + 9.
    Each round writes 8 ancilla measurements in ANC_M_ORDER (X and Z
    interleaved by qubit index). The final 9 bits are the data readouts
    in DATA_QUBITS order, so Q1, Q2, Q3 (the logical-Z top-row support)
    land at indices 0, 1, 2 — i.e. cr[8*rounds], cr[8*rounds+1], cr[8*rounds+2].
    """
    q  = QuantumRegister(17, 'q')
    cr = ClassicalRegister(8 * rounds + 9, 'c')
    qc = QuantumCircuit(q, cr)

    def cx_layer(circuit, x_pairs, z_pairs):
        # X-ancillas drive their data neighbour; Z-ancillas absorb from theirs.
        for ctrl, tgt in x_pairs:
            circuit.cx(q[ctrl], q[tgt])
        for ctrl, tgt in z_pairs:
            circuit.cx(q[ctrl], q[tgt])

    # Reset all 17 qubits at the start: data qubits must be in |0>^9 for the
    # Z-memory experiment, otherwise round-0 Z stabilisers fire on residual
    # thermal excitation and the logical observable is random.
    qc.reset([q[i] for i in range(17)])
    qc.barrier()

    for r in range(rounds):
        # Ancilla MR-style: explicit reset at start of round, measurement
        # at the end. Data qubits are NOT reset between rounds — they
        # carry the encoded state.
        qc.reset([q[a] for a in ANC_M_ORDER])
        qc.h([q[a] for a in X_ANCILLAS])

        cx_layer(qc, LAYER1_X, LAYER1_Z)
        cx_layer(qc, LAYER2_X, LAYER2_Z)
        cx_layer(qc, LAYER3_X, LAYER3_Z)
        cx_layer(qc, LAYER4_X, LAYER4_Z)

        qc.h([q[a] for a in X_ANCILLAS])

        # Measure all 8 ancillas in the canonical Stim M order. The
        # detector rec[] indices in build_matching_dem and
        # raw_syndromes_to_detectors assume this exact order.
        for i, anc in enumerate(ANC_M_ORDER):
            qc.measure(q[anc], cr[r * 8 + i])

        qc.barrier()

    for i, dq in enumerate(DATA_QUBITS):
        qc.measure(q[dq], cr[8 * rounds + i])

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

    # Final 9 data measurements are written in DATA_QUBITS order
    # = [Q1, Q2, Q3, Q7, Q8, Q9, Q13, Q14, Q15]. The canonical Stim
    # surface_code:rotated_memory_z generator uses logical Z = top row,
    # i.e. Q1 ⊗ Q2 ⊗ Q3 — those are at indices 0, 1, 2 in DATA_QUBITS.
    logical_flip = bool((data_bits[0] ^ data_bits[1] ^ data_bits[2]) == 1)

    return syn_bits, data_bits, logical_flip


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

    # ── COUPLING-MAP SANITY CHECK ─────────────────────────────────────────────
    # Verify the 24 CX pairs the circuit needs are actually edges in Tuna-17's
    # coupling map BEFORE transpiling. If they're not, no permutation of the
    # qubit labels can route this circuit without SWAPs and the resulting
    # syndrome data is meaningless under our DEM.
    cmap = backend.coupling_map
    edges = set()
    if cmap is not None:
        for a, b in cmap.get_edges():
            edges.add(frozenset((a, b)))
    print(f"[*] Tuna-17 coupling map: {len(edges)} undirected edges")
    print(f"    edges: {sorted(tuple(sorted(e)) for e in edges)}")

    required = set()
    for pair_list in [LAYER1_X, LAYER1_Z, LAYER2_X, LAYER2_Z,
                      LAYER3_X, LAYER3_Z, LAYER4_X, LAYER4_Z]:
        for a, b in pair_list:
            required.add(frozenset((a, b)))
    missing = required - edges
    if missing and edges:
        print(f"[!] {len(missing)}/{len(required)} required CX pairs are NOT direct edges:")
        for e in sorted(tuple(sorted(x)) for x in missing):
            print(f"      Q{e[0]}-Q{e[1]}")
        print("[!] Identity layout cannot be routed without SWAPs. The data/ancilla")
        print("    role assignment in build_d3_surface_code is incompatible with")
        print("    Tuna-17's physical topology. Fix the qubit indices in Z_NW/X_NW/...")
        print("    to match Tuna-17 before re-submitting.")

    # ── TRANSPILATION ─────────────────────────────────────────────────────────
    # Use optimization_level=1 (NOT 3): level 3 re-synthesises blocks of gates
    # which can silently break the H-CX-H structure that distinguishes X from
    # Z stabiliser readout. SWAP-free routing is MANDATORY: if it fails we
    # abort rather than silently corrupt the error model.
    print("[*] Transpiling for Tuna-17 topology (optimization_level=1, no SWAPs)...")
    qc_t = transpile(
        qc,
        backend=backend,
        optimization_level=1,
        routing_method='none',
        layout_method='trivial',  # honour our hand-chosen qubit indices
    )
    n_swap = qc_t.count_ops().get('swap', 0)
    print(f"    transpiled depth={qc_t.depth()}, gate counts={dict(qc_t.count_ops())}")
    if n_swap > 0:
        raise RuntimeError(
            f"Transpilation inserted {n_swap} SWAP gates — the resulting circuit "
            f"does NOT match the Stim DEM and syndrome data will be invalid. "
            f"Fix the qubit role assignment in build_d3_surface_code() to match "
            f"Tuna-17's coupling map."
        )

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
            if any(k in err_str for k in ("timeout", "network", "connection", "connect to host",
                                            "gaierror", "getaddrinfo", "dns", "temporarily unavailable",
                                            "ssl", "reset by peer")):
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
    raw_data_list = []
    true_observables_list = []

    for shot_str in raw_memory:
        syn_bits, data_bits, logical_flip = parse_memory_shot(shot_str, rounds=rounds)
        raw_syndromes_list.append(syn_bits)
        raw_data_list.append(data_bits)
        true_observables_list.append(logical_flip)

    return (
        np.array(raw_syndromes_list,    dtype=np.int8),
        np.array(raw_data_list,         dtype=np.int8),
        np.array(true_observables_list, dtype=bool),
    )


# =============================================================================
# 3. DETECTOR EVENT CONVERSION
# =============================================================================

def raw_syndromes_to_detectors(
    raw_syndromes: np.ndarray,
    data_bits: np.ndarray,
    rounds: int,
) -> np.ndarray:
    """
    Convert raw ancilla + data measurements into detector events for the
    canonical d=3 rotated-surface-code Z-memory.

    raw_syndromes shape: (n_shots, 8*rounds), each row laid out per round in
        ANC_M_ORDER = [Q0(X), Q4(Z), Q5(X), Q6(Z), Q10(Z), Q11(X), Q12(Z), Q16(X)].
    data_bits shape: (n_shots, 9), each row in DATA_QUBITS order
        [Q1, Q2, Q3, Q7, Q8, Q9, Q13, Q14, Q15].

    Detector layout (4 + 8*(rounds-1) + 4 = 8*rounds total):
        Round 0  : 4 detectors — round-0 Z-anc measurements, absolute.
                   (Z stabs are deterministic in |0>^9; X stabs are
                   random and are NOT promoted to detectors here.)
        Round r >= 1 : 8 detectors — XOR diff of all 8 ancillas vs round r-1.
        Final    : 4 detectors — for each Z ancilla, re-derive the
                   Z stabiliser parity from the FINAL data measurements
                   and XOR against the same Z ancilla's last round
                   measurement. These close the 1+(d-1)+1 detector loop
                   so X errors that occurred between the last syndrome
                   round and the final data readout are still caught.
    """
    expected_cols = 8 * rounds
    assert raw_syndromes.shape[1] == expected_cols, (
        f"Expected {expected_cols} raw syndrome bits ({rounds} rounds × 8 ancillas), "
        f"got {raw_syndromes.shape[1]}"
    )
    assert data_bits.shape[1] == 9, (
        f"Expected 9 data-qubit bits per shot, got {data_bits.shape[1]}"
    )
    n = raw_syndromes.shape[0]
    n_dets = 4 + 8 * (rounds - 1) + 4
    detectors = np.zeros((n, n_dets), dtype=np.int8)

    # Position of each Z ancilla in ANC_M_ORDER. Used to pull Z-anc
    # measurements out of the per-round 8-bit blocks.
    anc_pos = {a: ANC_M_ORDER.index(a) for a in ANC_M_ORDER}
    z_positions = [anc_pos[z] for z in Z_ANCILLAS]   # [1, 3, 4, 6]

    # ── Round 0: 4 absolute Z detectors ──────────────────────────────────────
    for i, zpos in enumerate(z_positions):
        detectors[:, i] = raw_syndromes[:, zpos]

    # ── Rounds r >= 1: 8 XOR detectors per round, all 8 ancillas ─────────────
    for r in range(1, rounds):
        det_offset = 4 + (r - 1) * 8
        prev = (r - 1) * 8
        cur  = r * 8
        detectors[:, det_offset : det_offset + 8] = (
            raw_syndromes[:, cur : cur + 8] ^ raw_syndromes[:, prev : prev + 8]
        )

    # ── Final 4 detectors: re-derive Z stabs from data, XOR last round Z-anc ─
    last_round_offset = (rounds - 1) * 8
    final_offset = 4 + 8 * (rounds - 1)
    for i, z_anc in enumerate(Z_ANCILLAS):
        last_z = raw_syndromes[:, last_round_offset + anc_pos[z_anc]]
        data_parity = np.zeros(n, dtype=np.int8)
        for d_idx in Z_STAB_DATA_IDX[z_anc]:
            data_parity ^= data_bits[:, d_idx]
        detectors[:, final_offset + i] = last_z ^ data_parity

    return detectors


# =============================================================================
# 4. STIM DETECTOR ERROR MODEL
# =============================================================================

def build_matching_dem(
    rounds: int = 3,
    physical_noise: float = 0.05,
) -> stim.DetectorErrorModel:
    """
    Build a Stim DEM mirroring Stim's canonical
    surface_code:rotated_memory_z circuit, remapped to Tuna-17 indices.

    Detector layout matches raw_syndromes_to_detectors:
        4 round-0 Z-anc detectors,
        8 XOR detectors per round 1..rounds-1,
        4 final data-derived Z-stab detectors.

    Observable: logical Z_L = Z(Q1) ⊗ Z(Q2) ⊗ Z(Q3) (top row).
    """
    c = stim.Circuit()
    p = physical_noise

    def cx_layer(circuit, x_pairs, z_pairs):
        for ctrl, tgt in x_pairs:
            circuit.append("CX", [ctrl, tgt])
        for ctrl, tgt in z_pairs:
            circuit.append("CX", [ctrl, tgt])
        flat = [q for pair in (x_pairs + z_pairs) for q in pair]
        if flat:
            circuit.append("DEPOLARIZE2", flat, p)

    for r in range(rounds):
        c.append("R", ANC_M_ORDER)
        if r > 0:
            c.append("X_ERROR", ANC_M_ORDER, p)

        c.append("H", X_ANCILLAS)
        cx_layer(c, LAYER1_X, LAYER1_Z)
        cx_layer(c, LAYER2_X, LAYER2_Z)
        cx_layer(c, LAYER3_X, LAYER3_Z)
        cx_layer(c, LAYER4_X, LAYER4_Z)
        c.append("H", X_ANCILLAS)

        # M with built-in bit-flip noise — modelled inside Stim as a flip
        # of the recorded outcome rather than an X error after the gate.
        c.append("M", ANC_M_ORDER, p)

        # ── Detectors ────────────────────────────────────────────────────
        # ANC_M_ORDER positions (rec offsets within last 8 measurements):
        #   pos 0: Q0  (X) -> rec[-8]
        #   pos 1: Q4  (Z) -> rec[-7]
        #   pos 2: Q5  (X) -> rec[-6]
        #   pos 3: Q6  (Z) -> rec[-5]
        #   pos 4: Q10 (Z) -> rec[-4]
        #   pos 5: Q11 (X) -> rec[-3]
        #   pos 6: Q12 (Z) -> rec[-2]
        #   pos 7: Q16 (X) -> rec[-1]
        z_offsets = [-7, -5, -4, -2]   # Q4, Q6, Q10, Q12

        if r == 0:
            # Round 0: Z-anc only (X-anc round-0 outcomes are random for
            # |0>^9 data and are NOT promoted to detectors).
            for off in z_offsets:
                c.append("DETECTOR", [stim.target_rec(off)])
        else:
            # All 8 ancillas, XOR'd with the same ancilla from round r-1
            # (8 measurements ago in the rec stream).
            for i in range(8):
                cur = -8 + i
                prv = cur - 8
                c.append("DETECTOR",
                         [stim.target_rec(cur), stim.target_rec(prv)])

    # Final data measurement
    c.append("M", DATA_QUBITS, p)

    # ── Final 4 detectors: each Z stabiliser, re-derived from the final
    # data measurements, XOR'd against that Z-anc's last round outcome.
    # DATA_QUBITS rec offsets after final M (last 9 records):
    #   [Q1, Q2, Q3, Q7, Q8, Q9, Q13, Q14, Q15] -> rec[-9..-1]
    # Last-round Z-anc rec offsets (data block of 9 sits between):
    #   Q4  pos 1 -> rec[-9 - (8 - 1)] = rec[-16]
    #   Q6  pos 3 -> rec[-9 - (8 - 3)] = rec[-14]
    #   Q10 pos 4 -> rec[-9 - (8 - 4)] = rec[-13]
    #   Q12 pos 6 -> rec[-9 - (8 - 6)] = rec[-11]
    z_anc_rec = {4: -16, 6: -14, 10: -13, 12: -11}
    for z_anc, idx_list in Z_STAB_DATA_IDX.items():
        # Convert DATA_QUBITS index i -> rec offset (-9 + i)
        data_recs = [stim.target_rec(-9 + i) for i in idx_list]
        c.append("DETECTOR",
                 [stim.target_rec(z_anc_rec[z_anc])] + data_recs)

    # Logical Z = top row, data_bits indices 0, 1, 2 -> rec[-9, -8, -7]
    c.append("OBSERVABLE_INCLUDE", [
        stim.target_rec(-9),   # Q1
        stim.target_rec(-8),   # Q2
        stim.target_rec(-7),   # Q3
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
        raw_data         = data['data_bits'] if 'data_bits' in data.files else None
    else:
        print(f"\n[*] No cached dataset found — submitting to {BACKEND_NAME}...")
        raw_syndromes, raw_data, true_observables = run_and_parse_surface_code(
            backend_name=BACKEND_NAME,
            shots=N_SHOTS,
            rounds=N_ROUNDS,
        )
        np.savez_compressed(
            FILENAME,
            syndromes=raw_syndromes,
            data_bits=raw_data,
            observables=true_observables,
        )
        print(f"[*] Data saved to {FILENAME}")

    n_shots_actual  = raw_syndromes.shape[0]
    base_err_count  = int(np.sum(true_observables))

    print(f"\n[*] Dataset: {n_shots_actual:,} shots, "
          f"{raw_syndromes.shape[1]} syndrome bits per shot")
    print(f"    Uncorrected logical error rate: "
          f"{base_err_count/n_shots_actual:.4%} ({base_err_count}/{n_shots_actual})")

    # ── 6.2 Detector events ───────────────────────────────────────────────────
    print("\n[*] Computing detector events...")
    if raw_data is None:
        raise RuntimeError(
            "Cached .npz lacks 'data_bits'. Delete the cache and re-submit "
            "so build_d3_surface_code / parse_memory_shot can record final "
            "data measurements (needed by the final-round Z-stab detectors)."
        )
    detector_events = raw_syndromes_to_detectors(
        raw_syndromes, raw_data, rounds=N_ROUNDS
    )
    expected_n_dets = 4 + 8 * (N_ROUNDS - 1) + 4
    assert detector_events.shape == (n_shots_actual, expected_n_dets), (
        f"Expected shape ({n_shots_actual}, {expected_n_dets}), got {detector_events.shape}"
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