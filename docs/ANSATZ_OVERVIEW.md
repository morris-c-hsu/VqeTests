# Ansatz Overview

This document describes the 8 ansätze implemented for the SSH-Hubbard VQE.

---

## Available Ansätze

### 1. HEA (Hardware-Efficient Ansatz)
**Command**: `--ansatz hea`

**Description**: Standard EfficientSU2 circuit from Qiskit's circuit library.

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

### 3. TopoInspired (Topological/Problem-Inspired)
**Command**: `--ansatz topoinsp`

**Description**: Dimer pattern reflecting SSH bond structure + edge entanglement.

**Features**:
- Strong bonds (t1): Entangle sites 0↔1, 2↔3, ...
- Weak bonds (t2): Entangle sites 1↔2, 3↔4, ...
- Edge links between dimers
- Does NOT preserve particle number

**Structure**:
```
Layer = Strong-bond-gates + Weak-bond-gates + Edge-links + Rotations
```

**Best for**: Exploiting SSH topological structure

---

### 4. TopoRN (RN-Topological)
**Command**: `--ansatz topo_rn`

**Description**: Number-conserving version of TopoInspired using RN gates.

**Features**:
- Same topological structure as TopoInspired
- **Preserves particle number** (RN gates instead of arbitrary 2-qubit)
- More restricted gate set

**Structure**:
```
Layer = RN-strong-bonds + RN-weak-bonds + RN-edge-links + RZ rotations
```

**Best for**: Number-conserving topological structure

---

### 5. DQAP (Digital-Adiabatic / QAOA-Style)
**Command**: `--ansatz dqap`

**Description**: QAOA/adiabatic-inspired layers with Hamiltonian time evolution.

**Features**:
- Mimics exp(-iHt) decomposition
- Separate hopping and interaction evolution
- **Preserves particle number**
- Many parameters (β, γ angles)

**Structure**:
```
Layer = exp(-iβ*H_hop) + exp(-iγ*H_int) + Mixer
```

**Best for**: Adiabatic-style optimization, deep circuits

---

### 6. NP_HVA (Number-Preserving HVA)
**Command**: `--ansatz np_hva`

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

### 7. TN_MPS (Tensor-Network MPS)
**Command**: `--ansatz tn_mps`

**Description**: Brick-wall MPS-like circuit inspired by tensor networks.

**Features**:
- Brick-wall structure (alternating even/odd bonds)
- Global coverage of all qubits
- Does NOT preserve particle number
- Requires half-filling initialization

**Structure**:
```
Layer = Even-bond gates (0↔1, 2↔3, ...) + Odd-bond gates (1↔2, 3↔4, ...)
```

**Best for**: Tensor network approximation, entanglement structure

**Warning**: Can converge to vacuum sector if initialized wrong!

---

### 8. TN_MPS_NP (Number-Preserving TN-MPS)
**Command**: `--ansatz tn_mps_np`

**Description**: Number-conserving version of TN_MPS.

**Features**:
- Same brick-wall structure as TN_MPS
- **Preserves particle number** (UNP gates)
- Safe against sector drift

**Structure**:
```
Layer = Even UNP gates + Odd UNP gates + RZ rotations
```

**Best for**: Tensor network structure with number conservation

---

## Quick Comparison Table

| Ansatz | Number-Conserving? | SSH-Aware? | Gate Type | Complexity |
|--------|-------------------|------------|-----------|------------|
| HEA | ✗ No | ✗ No | Arbitrary | Low |
| HVA | ✓ Yes | ✓ Yes | Givens | Medium |
| TopoInspired | ✗ No | ✓ Yes | Arbitrary | Medium |
| TopoRN | ✓ Yes | ✓ Yes | RN gates | Medium |
| DQAP | ✓ Yes | ✓ Yes | Hamiltonian | High |
| NP_HVA | ✓✓ Strict | ✓ Yes | UNP | Medium |
| TN_MPS | ✗ No | ✗ No | Arbitrary | Low |
| TN_MPS_NP | ✓ Yes | ✗ No | UNP | Low |

**Legend**:
- ✓ Yes: Feature present
- ✗ No: Feature absent
- ✓✓ Strict: Enforced at circuit level

---

## Choosing an Ansatz

**For accurate physics**:
- Use number-conserving ansätze (HVA, TopoRN, DQAP, NP_HVA, TN_MPS_NP)
- SSH-Hubbard model has strict particle number conservation

**For baseline comparison**:
- HEA: Standard benchmark, no problem structure
- TN_MPS: Tensor network baseline

**For SSH-specific structure**:
- HVA: Hamiltonian structure
- TopoInspired/TopoRN: Dimer pattern
- DQAP: Adiabatic evolution

**For exploration**:
- Try multiple ansätze and compare
- Use `benchmarks/compare_all_ansatze.py` for systematic comparison

---

## Performance Notes

**IMPORTANT**: This repository uses **single-run optimization** with no ensemble averaging.

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

## Usage Examples

### Single ansatz
```bash
# HEA with 3 layers
python src/ssh_hubbard_vqe.py --ansatz hea --reps 3

# HVA with 2 layers
python src/ssh_hubbard_vqe.py --ansatz hva --reps 2

# TopoInspired with 3 layers
python src/ssh_hubbard_vqe.py --ansatz topoinsp --reps 3
```

### Compare all ansätze
```bash
# Compare all 8 ansätze for L=4
python benchmarks/compare_all_ansatze.py --L 4

# For L=6 (larger system)
python benchmarks/compare_all_ansatze.py --L 6
```

### Parameter sweep
```bash
# Sweep dimerization δ from -0.6 to 0.6
python src/ssh_hubbard_vqe.py --ansatz hva --delta-sweep -0.6 0.6 13
```

---

## Circuit Diagrams

Circuit diagrams for L=4 are available in `docs/images/`:
- `hea_L4.png`
- `hva_L4.png`
- `topoinsp_L4.png`
- (etc.)

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
