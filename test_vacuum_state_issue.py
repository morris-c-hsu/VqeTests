#!/usr/bin/env python3
"""
Test: Does TN_MPS benefit from initial state preparation?

Compares TN_MPS performance with and without half-filling initial state.
"""

import numpy as np
import time
import warnings

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import QuantumCircuit

try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator

try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import L_BFGS_B
except ImportError:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import L_BFGS_B

from ssh_hubbard_vqe import (
    ssh_hubbard_hamiltonian,
    prepare_half_filling_state,
)

from ssh_hubbard_tn_vqe import (
    build_ansatz_tn_mps_sshh,
    ssh_hubbard_hamiltonian as tn_hamiltonian,
)

warnings.filterwarnings('ignore', category=DeprecationWarning)


def exact_diagonalization(H: SparsePauliOp):
    """Compute exact ground state energy."""
    H_matrix = H.to_matrix()
    eigenvalues = np.linalg.eigh(H_matrix)[0]
    return eigenvalues[0]


def run_vqe_test(ansatz, H, seed=42, maxiter=300):
    """Run VQE with given ansatz."""
    estimator = Estimator()
    optimizer = L_BFGS_B(maxiter=maxiter)

    np.random.seed(seed)
    initial_point = 0.01 * np.random.randn(ansatz.num_parameters)

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    start_time = time.time()
    result = vqe.compute_minimum_eigenvalue(H)
    runtime = time.time() - start_time

    return {
        'energy': result.eigenvalue.real,
        'evaluations': result.cost_function_evals,
        'runtime': runtime,
    }


def main():
    """Test vacuum state issue for TN_MPS."""
    print("=" * 80)
    print("VACUUM STATE TEST: TN_MPS with/without Initial State Preparation")
    print("=" * 80)

    # Test parameters - use a case where we saw mediocre TN_MPS performance
    L = 6
    t1 = 1.0
    t2 = 0.5
    U = 2.0
    reps = 2
    maxiter = 300  # Longer optimization

    N = 2 * L
    delta = (t1 - t2) / (t1 + t2)

    print(f"\nSystem Parameters:")
    print(f"  L = {L} sites ({N} qubits)")
    print(f"  δ = {delta:.3f} (dimerization)")
    print(f"  U = {U:.2f} (interaction)")
    print(f"  VQE maxiter = {maxiter}")

    # Build Hamiltonian
    print(f"\nBuilding Hamiltonian...")
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    print(f"  Pauli terms: {len(H)}")

    # Exact ground state
    print(f"\nComputing exact ground state...")
    E_exact = exact_diagonalization(H)
    print(f"  E_exact = {E_exact:.10f}")

    # Test 1: TN_MPS from vacuum (current approach)
    print(f"\n" + "=" * 80)
    print("TEST 1: TN_MPS from VACUUM state |00...0⟩")
    print("=" * 80)

    ansatz_vacuum = build_ansatz_tn_mps_sshh(L, reps)
    print(f"  Parameters: {ansatz_vacuum.num_parameters}")
    print(f"  Depth: {ansatz_vacuum.depth()}")
    print(f"  Running VQE...")

    result_vacuum = run_vqe_test(ansatz_vacuum, H, seed=42, maxiter=maxiter)
    error_vacuum = abs(result_vacuum['energy'] - E_exact)
    rel_error_vacuum = 100 * error_vacuum / abs(E_exact)

    print(f"  ✓ Energy:      {result_vacuum['energy']:.10f}")
    print(f"  ✓ Error:       {error_vacuum:.6f} ({rel_error_vacuum:.2f}%)")
    print(f"  ✓ Evaluations: {result_vacuum['evaluations']}")
    print(f"  ✓ Runtime:     {result_vacuum['runtime']:.2f}s")

    # Test 2: TN_MPS with half-filling initial state
    print(f"\n" + "=" * 80)
    print("TEST 2: TN_MPS with HALF-FILLING initial state preparation")
    print("=" * 80)

    ansatz_base = build_ansatz_tn_mps_sshh(L, reps)
    initial_state = prepare_half_filling_state(L)

    # Compose initial state + ansatz
    ansatz_prepared = QuantumCircuit(N)
    ansatz_prepared.compose(initial_state, inplace=True)
    ansatz_prepared.compose(ansatz_base, inplace=True)

    print(f"  Parameters: {ansatz_prepared.num_parameters}")
    print(f"  Depth: {ansatz_prepared.depth()}")
    print(f"  Running VQE...")

    result_prepared = run_vqe_test(ansatz_prepared, H, seed=42, maxiter=maxiter)
    error_prepared = abs(result_prepared['energy'] - E_exact)
    rel_error_prepared = 100 * error_prepared / abs(E_exact)

    print(f"  ✓ Energy:      {result_prepared['energy']:.10f}")
    print(f"  ✓ Error:       {error_prepared:.6f} ({rel_error_prepared:.2f}%)")
    print(f"  ✓ Evaluations: {result_prepared['evaluations']}")
    print(f"  ✓ Runtime:     {result_prepared['runtime']:.2f}s")

    # Comparison
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    improvement = error_vacuum - error_prepared
    improvement_pct = 100 * improvement / error_vacuum if error_vacuum > 0 else 0

    print(f"\nEnergy Errors:")
    print(f"  From vacuum:        {rel_error_vacuum:.2f}%")
    print(f"  With preparation:   {rel_error_prepared:.2f}%")
    print(f"  Improvement:        {improvement:.6f} ({improvement_pct:+.1f}%)")

    print(f"\nRuntime:")
    print(f"  From vacuum:        {result_vacuum['runtime']:.2f}s")
    print(f"  With preparation:   {result_prepared['runtime']:.2f}s")

    print(f"\nEvaluations:")
    print(f"  From vacuum:        {result_vacuum['evaluations']}")
    print(f"  With preparation:   {result_prepared['evaluations']}")

    if improvement > 0:
        print(f"\n✓ CONCLUSION: Initial state preparation IMPROVES TN_MPS accuracy!")
        print(f"  Error reduced by {improvement_pct:.1f}%")
    elif improvement < -0.01:
        print(f"\n✗ CONCLUSION: Initial state preparation DEGRADES TN_MPS accuracy")
        print(f"  Error increased by {-improvement_pct:.1f}%")
    else:
        print(f"\n≈ CONCLUSION: Initial state preparation has MINIMAL effect")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
