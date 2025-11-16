# SSH–Hubbard Lattice: Exact Diagonalization, VQE, and DMRG (Research Prototype)

This repository implements a set of numerical methods for the spinful Su–Schrieffer–Heeger–Hubbard (SSH–Hubbard) model on one-dimensional chains. It includes:

1. **Exact Diagonalization (ED)** for L ≤ 6 sites (validated and internally consistent).
2. **Variational Quantum Eigensolver (VQE)** with several ansätze.
3. **A TeNPy-based DMRG implementation**, currently known to yield incorrect energies due to a Hamiltonian-construction mismatch.

The purpose of this repository is **strictly exploratory**. The code is not intended for production-grade many-body calculations, quantitative benchmarking, or claims of algorithmic performance. All results should be interpreted cautiously, given the methodological limitations described below.

---

## 1. Physical Model

The spinful SSH–Hubbard Hamiltonian is

\[
H = -\sum_{i=1}^{L-1} t_i
\sum_{\sigma=\uparrow,\downarrow}
(c^\dagger_{i\sigma} c_{i+1\sigma} + \mathrm{h.c.})
+ U \sum_{i=1}^{L} n_{i\uparrow} n_{i\downarrow},
\]

with alternating hopping amplitudes \(t_1\) and \(t_2\) encoded via a dimerization parameter

\[
\delta = \frac{t_1 - t_2}{t_1 + t_2}.
\]

This model captures the interplay between SSH-like dimerization and local electron–electron repulsion. For small systems (L ≤ 6), the full Hilbert space dimension \(4^L\) is manageable, enabling ED as a ground-truth reference.

---

## 2. Fermion-to-Qubit Mapping

All VQE and tensor-network Hamiltonians use an **interleaved Jordan–Wigner mapping**:

\[
[\text{site}_0\uparrow,\; \text{site}_0\downarrow,\;
 \text{site}_1\uparrow,\; \text{site}_1\downarrow,\; \dots ].
\]

The JW strings, Pauli operators, and qubit indices follow Qiskit's convention (qubit 0 is the rightmost).
The Hamiltonian implemented in `ssh_hubbard_vqe.py` has been checked to agree with explicit dense-matrix ED for L ≤ 6.

The tensor-network VQE module (`ssh_hubbard_tn_vqe.py`) originally used inconsistent indexing conventions; these were reconciled so that both modules now construct identical operator matrices for small L.

---

## 3. Implemented Methods

### 3.1 Exact Diagonalization (ED)

- Used as the only reliable quantitative method in this repository.
- Validated for L = 2, 4, 6.
- All VQE and DMRG results should be compared only to ED for these sizes.

### 3.2 Variational Quantum Eigensolver (VQE)

The VQE implementation:

- uses ideal statevector simulation (no hardware noise, no sampling noise),
- constructs the JW Hamiltonian as Qiskit SparsePauliOps,
- optimizes circuit parameters using L-BFGS-B,
- compares energies to ED for L ≤ 6.

**Important methodological limitation:**
Only **a single optimization run** is performed per ansatz and parameter set (no multi-start or ensemble averaging).
Because VQE loss landscapes are highly non-convex, this means that:

- the reported energies reflect a *single* local minimum,
- the results cannot be interpreted as representative of an ansatz's general performance,
- comparisons between ansätze are not statistically meaningful.

Thus, all VQE results in the repository should be read as examples of *particular optimization trajectories*, not as evidence of relative ansatz quality.

### 3.3 Tensor-Network–Inspired VQE

Two brick-wall, MPS-like circuit families are implemented:

- **TN_MPS**
- **TN_MPS_NP** (number-preserving variant)

These circuits do **not** enforce particle-number conservation unless explicitly designed to do so. The non-number-preserving TN_MPS ansatz can converge to the wrong particle-number sector if initialized in vacuum; half-filling initialization is required to avoid this.

Because particle-number symmetry is a fundamental property of the SSH–Hubbard model, these ansätze are structurally mismatched to the problem unless the number-preserving variant is used.

### 3.4 DMRG (TeNPy)

The DMRG implementation currently exhibits a **1–3% systematic energy mismatch** relative to ED for L = 4 and L = 6, **independent of bond dimension**.

This indicates a Hamiltonian-construction inconsistency (e.g., incorrect bond pattern, incorrect interaction mapping, or missing fermionic sign conventions), not a truncation error.

As a result:

- DMRG results in this repository should be treated as *qualitative only*,
- DMRG cannot be used as a reference or benchmark,
- DMRG energies must not be interpreted as approximations to the true ground state.

---

## 4. VQE Ansatz Library

Eight ansätze are implemented across two files (`ssh_hubbard_vqe.py` and `ssh_hubbard_tn_vqe.py`).
Circuit images for L = 4 are provided in `docs/images/`.

Each ansatz is briefly described below **without performance claims**.

### Standard / Structure-Aware Ansätze

1. **HEA (Hardware-Efficient Ansatz)**
   A generic layered circuit with single-qubit rotations and nearest-neighbor entanglers.

2. **HVA (Hamiltonian-Variational Ansatz)**
   Mimics trotterized time evolution under hopping and interaction terms. Common in fermionic VQE literature and known to preserve some structural features of the Hamiltonian.

3. **TopoInspired**
   Incorporates the SSH dimer pattern explicitly. The implementation uses alternating strong/weak bond layers; the extent to which it captures edge physics depends on optimization.

### Number-Preserving Ansätze

4. **Topo_RN**
   Topology-aware design with number-preserving two-qubit blocks.

5. **DQAP**
   A small-parameter ansatz based on discretized adiabatic evolution.

6. **NP_HVA**
   A number-preserving variant of the HVA. Enforces U(1) symmetry at every layer.

### Tensor-Network Circuits

7. **TN_MPS**
   Brick-wall circuit approximating a small-bond-dimension MPS.
   Lacks number conservation and therefore can converge to unphysical sectors unless initialized correctly.

8. **TN_MPS_NP**
   Number-preserving variant of TN_MPS.

No claims are made regarding expressiveness, efficiency, or accuracy.
The repository does not contain sufficient statistical or methodological support for such claims.

---

## 5. Benchmarking Status (L ≤ 6 Only)

The repository includes scripts to run VQE on L = 6 for selected values of \(U\) and \(\delta\).
However:

- only single initializations were used,
- the energy landscape is non-convex,
- results vary significantly across local minima,
- no ensemble-based statistics are provided,
- therefore **the numerical results cannot be considered benchmarks**.

They should be interpreted strictly as examples of running VQE with these circuits.

---

## 6. Code Structure

```

src/
ssh_hubbard_vqe.py          # Main VQE + 6 ansätze
ssh_hubbard_tn_vqe.py       # TN-based ansätze
ssh_hubbard_tenpy_dmrg_fixed.py  # DMRG (inconsistent results)
docs/
IMPLEMENTATION_SUMMARY.md
images/
tests/
test_hamiltonian_consistency.py

```

---

## 7. Appropriate Use

This repository is suitable for:

- examining Hamiltonian construction for spinful SSH–Hubbard systems,
- experimenting with different VQE ansätze,
- learning about tensor-network–like quantum circuits,
- comparing exact diagonalization with variational methods on *very small* systems.

It is **not** suitable for:

- quantitative VQE benchmarking,
- performance comparisons between ansätze,
- large-scale scaling studies,
- accurate DMRG calculations,
- any claim of algorithmic superiority or expressiveness.

All results should be treated as exploratory and non-robust.

---
