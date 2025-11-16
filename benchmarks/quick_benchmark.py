#!/usr/bin/env python3
"""
Quick benchmark of all ansätze for a single small system.

This provides a snapshot of performance, but remember:
- Single optimization run per ansatz (no ensemble averaging)
- Results depend on random initialization
- Not statistically rigorous
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import (
    ssh_hubbard_hamiltonian,
    build_ansatz_hea,
    build_ansatz_hva_sshh,
    build_ansatz_topo_sshh,
    build_ansatz_topo_rn_sshh,
    build_ansatz_dqap_sshh,
    build_ansatz_np_hva_sshh,
    build_ansatz_tn_mps_sshh,
    build_ansatz_tn_mps_np_sshh
)

from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms import VQE
try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator
from qiskit import QuantumCircuit

print("=" * 80)
print("QUICK ANSATZ BENCHMARK")
print("=" * 80)

# Parameters
L = 4
t1 = 1.0
t2 = 0.6
U = 2.0
reps = 2
maxiter = 200

print(f"\nSystem: L={L}, t1={t1}, t2={t2}, U={U}")
print(f"VQE: reps={reps}, maxiter={maxiter}")
print(f"Note: Single run per ansatz (not statistically rigorous)")

# Build Hamiltonian
H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
N = 2 * L

# Exact energy
from scipy.sparse.linalg import eigsh
H_sparse = H.to_matrix(sparse=True)
E_exact = eigsh(H_sparse, k=1, which='SA', return_eigenvectors=False)[0]

print(f"\nExact ground energy: {E_exact:.8f}")

# Ansätze to test
ansatze = {
    'HEA': lambda: build_ansatz_hea(N, reps),
    'HVA': lambda: build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True),
    'TopoInspired': lambda: build_ansatz_topo_sshh(L, reps, use_edge_link=True),
    'TopoRN': lambda: build_ansatz_topo_rn_sshh(L, reps, use_edge_link=True),
    'DQAP': lambda: build_ansatz_dqap_sshh(L, reps, include_U=True),
    'NP_HVA': lambda: build_ansatz_np_hva_sshh(L, reps),
    'TN_MPS': lambda: build_ansatz_tn_mps_sshh(L, reps),
    'TN_MPS_NP': lambda: build_ansatz_tn_mps_np_sshh(L, reps),
}

# Number-conserving ansätze need half-filling initialization
number_conserving = {'HVA', 'DQAP', 'NP_HVA', 'TN_MPS_NP'}

results = []

print("\n" + "=" * 80)
print("RUNNING VQE...")
print("=" * 80)

for name, builder in ansatze.items():
    print(f"\n{name}:")

    # Build ansatz
    ansatz = builder()
    n_params = ansatz.num_parameters

    # Initial state if needed
    if name in number_conserving:
        # Half-filling: L electrons in 2L orbitals
        # Simple: fill first L/2 up and L/2 down
        init_state = QuantumCircuit(N)
        for i in range(0, L, 2):  # Every other site
            init_state.x(2*i)      # Spin up
            init_state.x(2*i + 1)  # Spin down
        ansatz = init_state.compose(ansatz)

    # VQE
    estimator = Estimator()
    optimizer = L_BFGS_B(maxiter=maxiter)

    np.random.seed(42)  # Fixed seed for reproducibility
    initial_point = 0.01 * np.random.randn(n_params)

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(H)

    energy = result.eigenvalue.real
    error = energy - E_exact
    rel_error = abs(error / E_exact) * 100
    evals = result.cost_function_evals

    print(f"  Parameters: {n_params}")
    print(f"  Energy:     {energy:.8f}")
    print(f"  Error:      {error:+.2e} ({rel_error:.3f}%)")
    print(f"  Evaluations: {evals}")

    results.append({
        'name': name,
        'energy': energy,
        'error': error,
        'rel_error': rel_error,
        'params': n_params,
        'evals': evals,
        'conserves_N': name in number_conserving
    })

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Ansatz':<15} {'N_params':<10} {'Energy':<14} {'Error':<12} {'%Error':<10} {'Evals':<8} {'N-cons':<6}")
print("-" * 85)

for r in sorted(results, key=lambda x: abs(x['error'])):
    n_mark = "✓" if r['conserves_N'] else "✗"
    print(f"{r['name']:<15} {r['params']:<10} {r['energy']:<14.8f} {r['error']:+12.2e} "
          f"{r['rel_error']:>9.3f}% {r['evals']:<8} {n_mark:<6}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

best = min(results, key=lambda x: abs(x['error']))
print(f"\nBest result (this run): {best['name']}")
print(f"  Energy: {best['energy']:.8f}")
print(f"  Error:  {best['error']:+.2e} ({best['rel_error']:.3f}%)")

print("\n⚠ IMPORTANT CAVEAT:")
print("This is a SINGLE optimization run per ansatz.")
print("VQE landscapes are highly non-convex with many local minima.")
print("Rankings will change with different:")
print("  - Random seeds")
print("  - Initial parameters")
print("  - Optimization settings")
print("\nFor rigorous comparison, need:")
print("  - Multiple runs (10-100) per ansatz")
print("  - Statistical analysis (mean ± std)")
print("  - Ensemble averaging")
print("\nThese results are EXAMPLES, not benchmarks!")

print("\n" + "=" * 80)
