# SSH–Hubbard Lattice: Exact Diagonalization, VQE, and DMRG (Research Prototype)

This repository implements a set of numerical methods for the spinful Su–Schrieffer–Heeger–Hubbard (SSH–Hubbard) model on one-dimensional chains. It includes:

- **Exact Diagonalization (ED)** for L ≤ 6 sites (validated and internally consistent).
- **Variational Quantum Eigensolver (VQE)** with several ansätze.
- A **TeNPy-based DMRG** implementation, currently known to yield incorrect energies due to a Hamiltonian-construction mismatch.

The purpose of this repository is strictly exploratory. The code is not intended for production-grade many-body calculations, quantitative benchmarking, or claims of algorithmic performance. All results should be interpreted cautiously, given the methodological limitations described below.

---

## 1. Physical Model

The spinful SSH–Hubbard Hamiltonian is

$$H = -\sum_{i=1}^{L-1} t_i \sum_{\sigma=\uparrow,\downarrow} \left(c^\dagger_{i\sigma} c_{i+1\sigma} + \mathrm{h.c.}\right) + U \sum_{i=1}^{L} n_{i\uparrow} n_{i\downarrow}$$

with alternating hopping amplitudes $t_1$ and $t_2$ encoded via a dimerization parameter

$$\delta = \frac{t_1 - t_2}{t_1 + t_2}$$

This model captures the interplay between SSH-like dimerization and local electron–electron repulsion. For small systems (L ≤ 6), the full Hilbert space dimension ($4^L$) is manageable, enabling ED as a ground-truth reference.

---

## 2. Fermion-to-Qubit Mapping

All VQE and tensor-network Hamiltonians use an interleaved Jordan–Wigner mapping:

$$[\text{site}_0\uparrow, \text{site}_0\downarrow, \text{site}_1\uparrow, \text{site}_1\downarrow, \dots]$$

The JW strings, Pauli operators, and qubit indices follow Qiskit's convention (qubit 0 is the rightmost). The Hamiltonian implemented in `ssh_hubbard_vqe.py` has been checked to agree with explicit dense-matrix ED for L ≤ 6.

The tensor-network VQE module (`ssh_hubbard_tn_vqe.py`) originally used inconsistent indexing conventions; these were reconciled so that both modules now construct identical operator matrices for small L.

---

## 3. Implemented Methods

### 3.1 Exact Diagonalization (ED)

Used as the only reliable quantitative method in this repository.

- **Dense diagonalization:** Validated for L = 2, 4, 6 (Hilbert space up to 4,096 dimensions).
- **Sparse Lanczos diagonalization:** Extends exact results to L = 7, 8 and beyond using `scipy.sparse.linalg.eigsh`.
- **Performance:** 227× speedup for L=6, enables previously impossible systems.

All VQE and DMRG results should be compared to ED whenever feasible.

### 3.2 Variational Quantum Eigensolver (VQE)

The VQE implementation:

- uses ideal statevector simulation (no hardware noise, no sampling noise),
- constructs the JW Hamiltonian as Qiskit `SparsePauliOp`s,
- supports multiple optimizers: L-BFGS-B, COBYLA, SLSQP,
- compares energies to ED for L ≤ 6.

**Multi-Start VQE Infrastructure:** The repository now includes multi-start VQE capabilities to address the non-convex optimization landscape:

- 3 optimizers: L-BFGS-B (default), COBYLA, SLSQP
- Multiple random seeds per optimizer for statistical analysis
- Aggregate statistics: mean, std, min, max across all runs
- Convergence visualization with relative error percentage plots
- Automatic COBYLA iteration adjustment (10× base maxiter, minimum 1000) for fair comparison

**Usage modes:**

- **Single-run mode (original):** Quick tests and debugging (`use_multistart=False`)
- **Multi-start mode (recommended):** Robust benchmarking with statistical significance (`use_multistart=True`)

For comprehensive multi-start benchmarking, use:

```bash
python benchmarks/run_multistart_benchmark.py --L 4
```

See `docs/MULTISTART_VQE_GUIDE.md` and `benchmarks/README.md` for details.

### 3.3 Tensor-Network–Inspired VQE

Two brick-wall, MPS-like circuit families are implemented:

- **TN_MPS**
- **TN_MPS_NP** (number-preserving variant)

These circuits do not enforce particle-number conservation unless explicitly designed to do so. The non-number-preserving TN_MPS ansatz can converge to the wrong particle-number sector if initialized in vacuum; half-filling initialization is required to avoid this.

Because particle-number symmetry is a fundamental property of the SSH–Hubbard model, these ansätze are structurally mismatched to the problem unless the number-preserving variant is used.

### 3.4 DMRG

#### TeNPy DMRG (Known Issues)

The TeNPy DMRG implementation currently exhibits a 1–3% systematic energy mismatch relative to ED for L = 4 and L = 6, independent of bond dimension.

This indicates a Hamiltonian-construction inconsistency (e.g., incorrect bond pattern, incorrect interaction mapping, or missing fermionic sign conventions), not a truncation error.

As a result:

- TeNPy DMRG results in this repository should be treated as **qualitative only**,
- TeNPy DMRG cannot be used as a reference or benchmark,
- TeNPy DMRG energies must not be interpreted as approximations to the true ground state.

#### dmrgpy DMRG (Alternative Implementation - NEW)

A new dmrgpy-based DMRG implementation using ITensor backend has been added to test whether ITensor avoids TeNPy's systematic error.

**Status:** Implementation complete, validation pending

**Features:**

- Python wrapper for ITensor (C++ and Julia backends)
- Manual Hamiltonian construction using operator algebra
- Designed to test problematic regime where TeNPy fails (t2/t1 ≥ 0.5)

**Installation:**

```bash
pip install dmrgpy
```

**Usage:**

```python
from ssh_hubbard_dmrgpy import solve_ssh_hubbard_dmrgpy

result = solve_ssh_hubbard_dmrgpy(L=4, t1=1.0, t2=0.6, U=1.0)
print(f"Ground state energy: {result['energy']:.10f}")
```

**Validation:**

```bash
python tests/test_dmrgpy_validation.py
```

See `docs/DMRGPY_IMPLEMENTATION.md` for detailed documentation.

---

## 4. VQE Ansatz Library

Eight ansätze are implemented across two files (`ssh_hubbard_vqe.py` and `ssh_hubbard_tn_vqe.py`). Circuit images for L = 4 are provided in `docs/images/`.

Each ansatz is briefly described below without performance claims.

### Standard / Structure-Aware Ansätze

**HEA (Hardware-Efficient Ansatz)**  
A generic layered circuit with single-qubit rotations and nearest-neighbor entanglers.

**HVA (Hamiltonian-Variational Ansatz)**  
Mimics trotterized time evolution under hopping and interaction terms. Common in fermionic VQE literature and known to preserve some structural features of the Hamiltonian.

**TopoInspired**  
Incorporates the SSH dimer pattern explicitly. The implementation uses alternating strong/weak bond layers; the extent to which it captures edge physics depends on optimization.

### Number-Preserving Ansätze

**Topo_RN**  
Topology-aware design with number-preserving two-qubit blocks.

**DQAP**  
A small-parameter ansatz based on discretized adiabatic evolution.

**NP_HVA**  
A number-preserving variant of the HVA. Enforces U(1) symmetry at every layer.

### Tensor-Network Circuits

**TN_MPS**  
Brick-wall circuit approximating a small-bond-dimension MPS. Lacks number conservation and therefore can converge to unphysical sectors unless initialized correctly.

**TN_MPS_NP**  
Number-preserving variant of TN_MPS.

> No claims are made regarding expressiveness, efficiency, or accuracy. The repository does not contain sufficient statistical or methodological support for such claims.

---

## 5. Benchmarking Status

The repository includes comprehensive multi-start VQE benchmarking infrastructure:

**Multi-Start Benchmarking (Recommended):**

- 3 optimizers: L-BFGS-B, COBYLA, SLSQP
- Multiple random initializations (5 seeds per optimizer by default)
- Ensemble-based statistics: mean, std, min, max across runs
- Convergence visualization with relative error percentage plots
- Statistical significance testing across ansätze
- See `docs/test_results_L4.md` for comprehensive results

**Single-Run Mode (Legacy):** The original single-initialization VQE scripts remain available for quick testing. However:

- only single initializations are used,
- the energy landscape is non-convex,
- results vary significantly across local minima,
- therefore single-run results cannot be considered benchmarks.

For statistically meaningful comparisons, use the multi-start benchmarking tools in `benchmarks/`.

**Tested System Sizes:** L = 4, 6 (multi-start); L = 8 (sparse Lanczos, single-run)

---

## 6. Code Structure

```
src/
  ssh_hubbard_vqe.py                 # Main VQE + 3 main ansätze
  ssh_hubbard_dmrgpy.py              # ⭐ NEW: dmrgpy/ITensor DMRG implementation
  ssh_hubbard_tenpy_dmrg_fixed.py    # TeNPy DMRG (has known issues)
  plot_utils.py                      # Convergence plotting with relative error %
  ansatze/
    archived_ansatze.py              # 5 archived ansätze

benchmarks/
  run_multistart_benchmark.py        # ⭐ Multi-start VQE with 3 optimizers
  compare_all_ansatze.py             # Core comparison infrastructure
  quick_benchmark.py                 # Fast single-run testing
  benchmark_large_systems.py         # L=6,8 system tests (sparse Lanczos)
  run_longer_optimizations.py        # Extended optimization runs
  README.md                          # Benchmark suite documentation

tests/
  test_sparse_lanczos.py             # Sparse Lanczos validation
  test_L7_benchmark.py               # L=7 integration test
  test_dmrgpy_validation.py          # ⭐ NEW: dmrgpy validation suite
  dmrg/                              # TeNPy DMRG error investigation
    01_hamiltonian_mismatch.py       # Main DMRG validation
    02-07_*.py                       # Systematic debugging tests
    DEBUG_REPORT.md                  # Complete investigation report
    README.md                        # DMRG test documentation

docs/
  MULTISTART_VQE_GUIDE.md            # ⭐ Multi-start VQE guide
  DMRGPY_IMPLEMENTATION.md           # ⭐ NEW: dmrgpy/ITensor DMRG docs
  IMPLEMENTATION_SUMMARY.md          # Implementation details & recent updates
  test_results_L4.md                 # L=4 multi-start benchmark results
  ANSATZ_OVERVIEW.md                 # Ansatz documentation
  DMRG_STATUS.md                     # TeNPy DMRG error status
  SPARSE_LANCZOS.md                  # Sparse diagonalization docs
  images/                            # Convergence plots
    convergence_*_L*.png             # Multi-start convergence plots
```

---

## 7. Appropriate Use

This repository is suitable for:

- examining Hamiltonian construction for spinful SSH–Hubbard systems,
- experimenting with different VQE ansätze,
- learning about tensor-network–like quantum circuits,
- comparing exact diagonalization with variational methods on very small systems.

It is **not** suitable for:

- quantitative VQE benchmarking,
- performance comparisons between ansätze,
- large-scale scaling studies,
- accurate DMRG calculations,
- any claim of algorithmic superiority or expressiveness.

All results should be treated as exploratory and non-robust.
