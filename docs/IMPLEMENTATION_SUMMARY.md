# Implementation Summary for the SSH–Hubbard Simulation Framework

This document describes the implementation details, assumptions, and limitations underlying the SSH–Hubbard codebase. All numerical results should be interpreted cautiously due to the methodological constraints noted below.

---

# 1. Hamiltonian Construction

All Hamiltonians are constructed using the interleaved Jordan–Wigner transformation:

\[
[\text{site}_0\uparrow,\; \text{site}_0\downarrow,\;\dots,\; \text{site}_{L-1}\uparrow,\;\text{site}_{L-1}\downarrow].
\]

For L ≤ 6, the SparsePauliOp Hamiltonians have been checked against explicit dense-matrix constructions and agree exactly.

Both VQE modules (`ssh_hubbard_vqe.py` and `ssh_hubbard_tn_vqe.py`) now use consistent conventions after prior discrepancies were corrected.

---

# 2. Exact Diagonalization (ED)

- Implemented via SciPy dense eigensolvers.
- Serves as the only quantitatively reliable benchmark in the repository.
- Used exclusively for L ≤ 6.

---

# 3. VQE Implementation

The VQE driver performs:

1. JW Hamiltonian generation
2. Initial half-filling state preparation
3. Construction of an ansatz with variational parameters
4. Energy evaluation via Qiskit statevector
5. Optimization via L-BFGS-B

**Methodological constraints:**

- Only **one optimization run** is used per ansatz and per parameter set.
- No ensemble averaging or multi-start strategy is implemented.
- No noise models, measurement sampling, or hardware backends are used.

Therefore, all results reflect the behavior of a **single optimization trajectory** and are not robust to local minima or parameter initialization.

---

# 4. Ansatz Implementations

Eight ansätze are provided. Their implementation details are summarized below, without interpreting their numerical performance.

### 4.1 HEA
Generic hardware-efficient layered circuit.

### 4.2 HVA
Hopping and interaction unitaries applied in alternating layers.
Uses XX + YY for hopping and ZZ-type rotations for on-site repulsion.

### 4.3 TopoInspired
Applies strong-bond and weak-bond entanglers in alternating layers.
Designed to reflect the SSH structure; behavior depends on optimization.

### 4.4 Topo_RN
Topology-aware, but replaces two-qubit entanglers with number-preserving blocks.

### 4.5 DQAP
Small-parameter discretized adiabatic evolution.

### 4.6 NP_HVA
Number-preserving version of the HVA.
Maintains U(1) particle-number symmetry.

### 4.7 TN_MPS
Tensor-network–inspired brick-wall circuit approximating a small-bond-dimension MPS.
Does not preserve particle number; requires careful initialization to avoid unphysical sectors.

### 4.8 TN_MPS_NP
Number-preserving variant of TN_MPS.

Circuit images (L = 4) are provided under `docs/images/`.

---

# 5. Tensor-Network VQE Considerations

The TN_MPS and TN_MPS_NP circuits are structurally inspired by matrix product states but do not guarantee symmetry preservation unless explicitly designed to do so.

The non-number-preserving variant can converge to the vacuum sector unless the initial state is explicitly set to half-filling.

This behavior should be kept in mind when interpreting any numerical results.

---

# 6. DMRG Implementation (TeNPy)

The DMRG implementation currently does not quantitatively match ED results:

- Errors of 1–3% persist for L = 4 and L = 6,
- Increasing the bond dimension does not remove this error.

This indicates an inconsistency in Hamiltonian construction for the DMRG module.
Until this is corrected, DMRG results should be regarded as **qualitative only**, and **not as approximate ground-state energies**.

---

# 7. Numerical Results (Interpretation Warning)

The repository contains numerical outputs for L = 6 and selected parameters.
However:

- each data point is from a single VQE optimization run,
- the results may reflect local minima,
- no statistical characterization is available,
- ansatz comparisons are therefore not meaningful.

These results should be interpreted strictly as **examples of possible VQE outcomes**, not as benchmarks.

---

# 8. Intended Use and Scientific Scope

This repository is intended for:

- exploring Hamiltonian construction for interacting fermion models,
- learning how VQE ansätze are implemented,
- experimenting with tensor-network–inspired circuits,
- contrasting variational methods with ED on very small systems.

It is **not intended** for:

- quantitative benchmarking of VQE algorithms,
- claims of ansatz optimality,
- scaling studies beyond L = 6,
- accurate DMRG calculations,
- hardware-based validation.

---

# 9. Summary

The repository provides a transparent, exploratory implementation of ED, VQE, and DMRG for the SSH–Hubbard model. The ED routines are reliable for L ≤ 6. The VQE implementation is suitable for demonstration and experimentation but does not generate statistically meaningful benchmarks due to single-run optimization. The DMRG implementation is not currently consistent with ED and should not be used for quantitative purposes.

All numerical results should be interpreted with these limitations in mind.
