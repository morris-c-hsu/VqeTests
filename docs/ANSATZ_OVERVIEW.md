# Ansatz Overview

This document describes the ansätze implemented for the SSH-Hubbard VQE.

## Organization

### Main Ansätze (Standard Benchmarks)

The main benchmark suite focuses on **3 core ansätze**:

| Ansatz | Location | Number-Conserving | Purpose |
|--------|----------|-------------------|---------|
| **HEA** | ssh_hubbard_vqe.py | ✗ No | Generic baseline |
| **HVA** | ssh_hubbard_vqe.py | ✓ Yes | Hamiltonian-inspired |
| **NP_HVA** | ssh_hubbard_vqe.py | ✓✓ Strict | Strict number conservation |

### Archived Ansätze (Available for Research)

5 additional ansätze in `src/ansatze/archived_ansatze.py`:

| Ansatz | Number-Conserving | Purpose |
|--------|-------------------|---------|
| **TopoInspired** | ✗ No | SSH topological structure |
| **TopoRN** | ✓ Yes | SSH structure + number-conserving |
| **DQAP** | ✓ Yes | Adiabatic/QAOA-style |
| **TN_MPS** | ✗ No | Tensor network brick-wall |
| **TN_MPS_NP** | ✓ Yes | TN brick-wall + number-conserving |

**Why archived?** To focus testing resources on the most relevant ansätze while preserving implementations for future research.

---

## Main Ansätze (Detailed)

### 1. HEA (Hardware-Efficient Ansatz)
**Command**: `--ansatz hea`
**File**: `src/ssh_hubbard_vqe.py`

**Description**: Standard EfficientSU2 circuit from Qiskit.

**Features**:
- Generic, problem-agnostic structure
- Alternating rotation and entanglement layers
- Does NOT preserve particle number
- Good baseline for comparison

**Structure**:
- Single-qubit rotations: RY and RZ on all qubits
- Entanglement: Circular CX gates
- Depth: Controlled by `--reps`

**Best for**: General-purpose baseline, hardware efficiency

---

### 2. HVA (Hamiltonian-Variational Ansatz)
**Command**: `--ansatz hva`
**File**: `src/ssh_hubbard_vqe.py`

**Description**: Layers mimicking the SSH-Hubbard Hamiltonian structure.

**Features**:
- Hopping layers: XX + YY gates (fermion hopping)
- Interaction layers: ZZ rotations (Hubbard U)
- **Preserves particle number** (using Givens rotations)
- Problem-inspired structure

**Structure**:
```
Layer = Hopping(t1, t2) + Interaction(U) + Single-qubit rotations
```

**Best for**: Hamiltonian-aware optimization, number conservation

---

### 3. NP_HVA (Number-Preserving HVA)
**Command**: `--ansatz np_hva`
**File**: `src/ssh_hubbard_vqe.py`

**Description**: Strict number-conserving HVA using UNP gates.

**Features**:
- **Strictly preserves particle number** (UNP gates)
- Same Hamiltonian-inspired structure as HVA
- Enforces U(1) symmetry at circuit level

**Structure**:
```
Layer = UNP-hopping + Number-preserving interaction + RZ rotations
```

**Best for**: Strict particle-number sectors, physical constraints

---

## Archived Ansätze (Brief Descriptions)

The following ansätze are implemented in `src/ansatze/archived_ansatze.py`. See that file for detailed documentation.

### TopoInspired - Topological/Problem-Inspired
- SSH dimer pattern (strong/weak bonds)
- Edge entanglement
- Does NOT conserve particle number

### TopoRN - RN-Topological
- Same SSH structure as TopoInspired
- **Number-conserving** (RN gates)
- More restricted gate set

### DQAP - Digital-Adiabatic (QAOA-style)
- Adiabatic evolution layers
- **Number-conserving**
- High parameter count

### TN_MPS - Tensor-Network MPS
- Brick-wall MPS-like structure
- Does NOT conserve particle number
- Requires careful initialization

### TN_MPS_NP - Number-Preserving TN-MPS
- Same brick-wall structure
- **Number-conserving** (UNP gates)
- Safe against sector drift

---

## Quick Comparison

| Ansatz | In Benchmarks? | N-Conserving? | SSH-Aware? | Complexity |
|--------|----------------|---------------|------------|------------|
| HEA | ✓ Yes | ✗ No | ✗ No | Low |
| HVA | ✓ Yes | ✓ Yes | ✓ Yes | Medium |
| NP_HVA | ✓ Yes | ✓✓ Strict | ✓ Yes | Medium |
| TopoInspired | Archived | ✗ No | ✓ Yes | Medium |
| TopoRN | Archived | ✓ Yes | ✓ Yes | Medium |
| DQAP | Archived | ✓ Yes | ✓ Yes | High |
| TN_MPS | Archived | ✗ No | ✗ No | Low |
| TN_MPS_NP | Archived | ✓ Yes | ✗ No | Low |

---

## Usage

### Standard Benchmarks (3 Main Ansätze)

```bash
# Compare HEA, HVA, NP_HVA for L=4
python benchmarks/compare_all_ansatze.py --L 4

# Quick benchmark
python benchmarks/quick_benchmark.py

# Individual ansatz
python src/ssh_hubbard_vqe.py --ansatz hea --reps 3
python src/ssh_hubbard_vqe.py --ansatz hva --reps 2
python src/ssh_hubbard_vqe.py --ansatz np_hva --reps 2
```

### Using Archived Ansätze

```python
# Import from archived module
from ansatze.archived_ansatze import (
    build_ansatz_topo_sshh,
    build_ansatz_dqap_sshh,
    # etc.
)

# Build ansatz
circuit = build_ansatz_topo_sshh(L=4, reps=2)

# Use in custom benchmark
# (See archived_ansatze.py for full API)
```

---

## Choosing an Ansatz

**For standard testing** → Use benchmarks with HEA, HVA, NP_HVA

**For accurate physics** → HVA or NP_HVA (number-conserving)

**For baseline comparison** → HEA (generic, no problem structure)

**For custom research** → Archived ansätze in `src/ansatze/archived_ansatze.py`

---

## Performance Notes

**CRITICAL**: This repository uses **single-run optimization** with no ensemble averaging.

Because VQE landscapes are highly non-convex:
- Results reflect a single local minimum
- Performance varies significantly with initialization
- Comparisons are NOT statistically rigorous
- Results are **examples**, not benchmarks

**Do NOT interpret relative performance as ansatz quality**.

For scientifically valid comparisons, you would need:
- Multiple random initializations (10-100 runs)
- Statistical analysis (mean, std, quartiles)
- Convergence diagnostics
- Hardware noise modeling (if relevant)

---

## References

**Ansatz design principles**:
- Hardware-efficient: Kandala et al., Nature 2017
- Hamiltonian-variational: Wecker et al., PRX 2015
- QAOA: Farhi et al., arXiv:1411.4028
- Number-preserving gates: Bravyi et al., arXiv:2108.10182

**SSH-Hubbard model**:
- SSH model: Su, Schrieffer, Heeger, PRL 1979
- Hubbard model: Hubbard, Proc. Royal Soc. 1963
