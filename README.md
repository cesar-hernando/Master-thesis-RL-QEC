# Reinforcement learning for quantum error decoding in the surface code

This codespace is part of my master thesis, part of the master in Quantum Information Science and Technology jointly organized by TU Delft and Leiden University. 

## Project Overview
This is a master thesis implementation of **Reinforcement Learning for Quantum Error Correction (QEC) in the Surface Code**. The project simulates a rotated surface code lattice and implements error correction logic using RL agents.

**Key Domain**: Quantum computing, error correction, topological codes, reinforcement learning

## Architecture Overview

### Core Component: `SurfaceCode` Class (surface_code_env.py)
The single main class implementing a quantum environment simulator for the rotated surface code:

**Lattice Representation**:
- Uses a `(2d+1) × (2d+1)` coordinate grid where `d` is the code distance (must be odd)
- Data qubits: positioned at odd coordinates `(i,j)` where `i,j ∈ {1,3,5,...,2d-1}`
- X-stabilizers: syndrome measurements for bit-flip (X) errors
- Z-stabilizers: syndrome measurements for phase-flip (Z) errors
- Boundary conditions: special handling at lattice edges (i=0, i=2d, j=0, j=2d)

**State Representation**:
- `hidden_state[i,j,0]`: X-component (-1 if bit-flip error, +1 if no error)
- `hidden_state[i,j,1]`: Z-component (-1 if phase-flip error, +1 if no error)
- `syndrome_lattice[i,j,0]`: X-type syndrome measurement
- `syndrome_lattice[i,j,1]`: Z-type syndrome measurement
- `visible_state`: 7-channel observation (x_mask, z_mask, x_syndrome, z_syndrome, data_mask, action_history_X, action_history_Z)

**Error Models**:
1. **'X' (bit-flip)**: Physical errors are X gates only; Z-stabilizers measure errors; X-stabilizers untriggered
2. **'depolarizing'**: Mixed X, Y, Z errors with probabilities `p_phys/3` each; both stabilizer types active

### Key Methods

| Method | Purpose |
|--------|---------|
| `_assign_qubit_coordinates()` | Maps data and stabilizer qubits to lattice positions, handles boundary constraints |
| `_create_masks()` | Generates binary masks indicating qubit/stabilizer positions on grid |
| `_simulate_errors()` | Initializes hidden errors and computes resulting syndrome patterns |
| `_obtain_support_qubits(i,j)` | Returns 2-4 data qubits whose errors trigger stabilizer (i,j) |
| `_is_logically_correct()` | Checks if logical error chain exists (vertical X-chains or horizontal Z-chains) |
| `reset()` | Generates fresh error configuration and returns visible state for RL episode |
| `step(action)` | RL agent applies correction action; returns observation, reward, done flag |
| `render()` | Matplotlib visualization with color-coded plaquettes, syndromes, and errors |

## Critical Design Patterns

### 1. Coordinate System Conversion
Data qubits in the code distance `d` space map to grid as: `(i,j) → (2i+1, 2j+1)`. When extracting from grids, reverse via `i = (coord-1)//2`.

**Pattern Found**: `_obtain_support_qubits()` demonstrates boundary-aware neighbor finding—critical for correct error propagation.

### 2. Syndrome Computation
Syndromes are **products of stabilizer support qubits**:
- If odd number of errors → syndrome = -1
- If even number of errors → syndrome = +1

This leverages numpy's `np.prod()` over error states (±1 representation).

### 3. Logical Error Detection
Logical chains are detected via **row/column parity**:
```python
np.sum(hx[:, col]) == -d  # All d qubits in column have X error → logical Z
np.sum(hz[row, :]) == -d  # All d qubits in row have Z error → logical X
```

## Developer Workflows

### Running the Environment
```bash
# Basic execution with default parameters
python surface_code_env.py

# Creates a d=5 surface code with 20% physical error rate and renders visualization
```

### Key Parameters to `SurfaceCode` Constructor
- `d`: Code distance (odd integer, typically 3-7 for testing)
- `p_phys`: Physical error probability (0.0-1.0)
- `p_meas`: Measurement error probability (currently unused in reset/step)
- `error_model`: 'X' or 'depolarizing'
- `volume_depth`: Temporal extent (currently unused)

### RL Integration Points (In Development)
- **Observation**: `visible_state` (7 channels × (2d+1)² grid)
- **Action Space**: `[i, j, op]` where op ∈ {0:I, 1:X, 2:Z} at coordinates (i,j)
- **Reward**: Currently stubbed; partially implemented in `step()` with penalties for repeated actions
- **Episode Termination**: Triggered by invalid action (already-performed correction at same location)

## Project-Specific Conventions

1. **State Representation**: Errors encoded as ±1 (not binary 0/1) for easy product-based syndrome computation
2. **Boundary Handling**: Special stabilizer placement at lattice edges; methods check `i==0`, `i==2d`, `j==0`, `j==2d` explicitly
3. **Incomplete RL Loop**: `step()` method is partially stubbed—reward system and hidden state updates need completion
4. **Visualization**: Uses matplotlib with specific color scheme (purple=X-syndrome, red=Z-syndrome, gold=errors)

## Integration Points & Dependencies

### External Dependencies
- `numpy`: Array operations, random number generation
- `matplotlib`: 2D visualization of lattice state

### Cross-Component Communication
- `hidden_state` (ground truth) → drives `syndrome_lattice` (observations)
- `syndrome_lattice` → fed to RL agent via `visible_state`
- Agent actions → update `action_history` in `visible_state` (incomplete implementation)

## Common Tasks & Code Locations

| Task | Code Location |
|------|----------------|
| Modify error model | `_simulate_errors()` method |
| Change lattice visualization | `render()` method's color palette and drawing loops |
| Add measurement errors | `_simulate_errors()` (add noise to syndrome readings) |
| Implement RL reward shaping | `step()` method's reward calculation section |
| Test boundary conditions | `_obtain_support_qubits()` for edge cases (i=0, i=2d, j=0, j=2d) |

## Next Steps for Development

1. **Complete `step()` method**: Update hidden state based on corrections; finalize reward logic
2. **Measurement error integration**: Use `p_meas` parameter to add syndrome measurement noise
3. **RL agent wrapper**: Create interface between agent and SurfaceCode environment
4. **Extended visualization**: Add real-time syndrome tracking and correction animation