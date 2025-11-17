#!/usr/bin/env python3
"""
SSH-Hubbard VQE Quick Test for L=6 (12 qubits)

Designed for Google Colab to verify implementations work before full sweep.

Estimated runtime: 5-10 minutes on Colab CPU
Total VQE runs: 12 (3 ansätze × 2 optimizers × 2 seeds)
"""

import sys
import time
from datetime import datetime

print("=" * 70)
print("SSH-HUBBARD VQE TEST: L=6 (12 qubits)")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# Installation (Colab only)
# ============================================================================

if 'google.colab' in sys.modules:
    print("Running on Google Colab - installing dependencies...")
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'qiskit', 'qiskit-algorithms',
                    'matplotlib', 'numpy', 'scipy'], check=True)
    print("✓ Dependencies installed\n")
else:
    print("Running locally\n")

# ============================================================================
# Imports
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse.linalg import eigsh
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA

try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator

import qiskit
print(f"Qiskit version: {qiskit.__version__}")
print(f"NumPy version: {np.__version__}\n")

# ============================================================================
# Hamiltonian Construction
# ============================================================================

def ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False):
    """
    Build SSH-Hubbard Hamiltonian with Jordan-Wigner transformation.

    Qubit convention: [site0↑, site0↓, site1↑, site1↓, ..., site(L-1)↑, site(L-1)↓]
    """
    N = 2 * L
    pauli_list = []

    def q_index(site, spin):
        return 2 * site + (0 if spin == 'up' else 1)

    def add_hopping(site_i, site_j, t, spin):
        qi = q_index(site_i, spin)
        qj = q_index(site_j, spin)
        a = min(qi, qj)
        b = max(qi, qj)

        pauli_str_xx = ['I'] * N
        pauli_str_xx[N-1-a] = 'X'
        pauli_str_xx[N-1-b] = 'X'
        for k in range(a + 1, b):
            pauli_str_xx[N-1-k] = 'Z'

        pauli_str_yy = ['I'] * N
        pauli_str_yy[N-1-a] = 'Y'
        pauli_str_yy[N-1-b] = 'Y'
        for k in range(a + 1, b):
            pauli_str_yy[N-1-k] = 'Z'

        pauli_list.append((''.join(pauli_str_xx), -t/2))
        pauli_list.append((''.join(pauli_str_yy), -t/2))

    for spin in ['up', 'down']:
        for i in range(L - 1):
            t = t1 if i % 2 == 0 else t2
            add_hopping(i, i+1, t, spin)

    for i in range(L):
        qi_up = q_index(i, 'up')
        qi_dn = q_index(i, 'down')
        pauli_list.append(('I'*N, U/4))
        z_up_str = ['I'] * N
        z_up_str[N-1-qi_up] = 'Z'
        pauli_list.append((''.join(z_up_str), -U/4))
        z_dn_str = ['I'] * N
        z_dn_str[N-1-qi_dn] = 'Z'
        pauli_list.append((''.join(z_dn_str), -U/4))
        zz_str = ['I'] * N
        zz_str[N-1-qi_up] = 'Z'
        zz_str[N-1-qi_dn] = 'Z'
        pauli_list.append((''.join(zz_str), U/4))

    return SparsePauliOp.from_list(pauli_list).simplify()

# ============================================================================
# Ansatz Construction
# ============================================================================

def q_index(site, spin, L):
    return 2 * site + (0 if spin == 'up' else 1)

def prepare_half_filling_state(L):
    N = 2 * L
    qc = QuantumCircuit(N)
    for site in range(L):
        if site % 2 == 0:
            qc.x(q_index(site, 'up', L))
        else:
            qc.x(q_index(site, 'down', L))
    return qc

def apply_unp_gate(qc, theta, phi, q0, q1):
    qc.crz(phi, q0, q1)
    qc.h(q1)
    qc.cx(q1, q0)
    qc.ry(theta, q0)
    qc.cx(q1, q0)
    qc.h(q1)

def build_ansatz_hea(N, depth):
    return RealAmplitudes(N, reps=depth, entanglement='full')

def build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True):
    N = 2 * L
    qc = prepare_half_filling_state(L)

    for rep in range(reps):
        for i in range(0, L-1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i+1, spin, L)
                theta = Parameter(f'θ_t1_{rep}_{i}_{spin}')
                qc.rxx(theta, qi, qj)
                qc.ryy(theta, qi, qj)

        for i in range(1, L-1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i+1, spin, L)
                theta = Parameter(f'θ_t2_{rep}_{i}_{spin}')
                qc.rxx(theta, qi, qj)
                qc.ryy(theta, qi, qj)

        if include_U:
            for i in range(L):
                qi_up = q_index(i, 'up', L)
                qi_dn = q_index(i, 'down', L)
                phi = Parameter(f'φ_U_{rep}_{i}')
                qc.rzz(phi, qi_up, qi_dn)

    return qc

def build_ansatz_np_hva_sshh(L, reps):
    N = 2 * L
    qc = prepare_half_filling_state(L)

    for rep in range(reps):
        for i in range(0, L-1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i+1, spin, L)
                theta_t1 = Parameter(f'θ_t1_np_{rep}_{i}_{spin}')
                phi_t1 = Parameter(f'φ_t1_np_{rep}_{i}_{spin}')
                apply_unp_gate(qc, theta_t1, phi_t1, qi, qj)

        for i in range(1, L-1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i+1, spin, L)
                theta_t2 = Parameter(f'θ_t2_np_{rep}_{i}_{spin}')
                phi_t2 = Parameter(f'φ_t2_np_{rep}_{i}_{spin}')
                apply_unp_gate(qc, theta_t2, phi_t2, qi, qj)

        for i in range(L):
            qi_up = q_index(i, 'up', L)
            qi_dn = q_index(i, 'down', L)
            gamma = Parameter(f'γ_np_{rep}_{i}')
            qc.rzz(gamma, qi_up, qi_dn)

    return qc

# ============================================================================
# VQE Runner
# ============================================================================

class VQERunner:
    def __init__(self, maxiter=50, optimizer_name='L_BFGS_B'):
        self.maxiter = maxiter
        self.optimizer_name = optimizer_name
        self.energy_history = []
        self.nfev = 0

    def callback(self, nfev, params, value, meta):
        self.energy_history.append(value)
        self.nfev = nfev

    def run(self, ansatz, hamiltonian, initial_point=None, seed=None):
        self.energy_history = []
        self.nfev = 0

        if self.optimizer_name == 'L_BFGS_B':
            optimizer = L_BFGS_B(maxiter=self.maxiter)
        elif self.optimizer_name == 'COBYLA':
            cobyla_maxiter = max(1000, self.maxiter * 10)
            optimizer = COBYLA(maxiter=cobyla_maxiter)

        if initial_point is None and seed is not None:
            rng = np.random.default_rng(seed)
            initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

        estimator = Estimator()
        vqe = VQE(estimator, ansatz, optimizer, callback=self.callback,
                  initial_point=initial_point)

        start_time = time.time()
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        runtime = time.time() - start_time

        return {
            'energy': result.eigenvalue,
            'runtime': runtime,
            'nfev': self.nfev,
            'seed': seed
        }

def run_multistart_vqe(runner, ansatz, hamiltonian, seeds):
    per_seed_results = []
    for seed in seeds:
        result = runner.run(ansatz, hamiltonian, seed=seed)
        per_seed_results.append(result)

    energies = np.array([r['energy'] for r in per_seed_results])
    best_idx = int(np.argmin(energies))

    return {
        'per_seed': per_seed_results,
        'best_energy': float(energies[best_idx]),
        'mean_energy': float(energies.mean()),
        'std_energy': float(energies.std())
    }

# ============================================================================
# Test Configuration
# ============================================================================

print("=" * 70)
print("TEST CONFIGURATION")
print("=" * 70)

L = 6  # 6 sites = 12 qubits
delta = 0.0
t1 = 1.0
t2 = t1 * (1 - delta) / (1 + delta)
U = 0.0

ansatz_reps = 2
maxiter = 50
seeds = [0, 1]
optimizers = ['L_BFGS_B', 'COBYLA']
ansatze = ['HEA', 'HVA', 'NP_HVA']

print(f"System size: L = {L} ({2*L} qubits)")
print(f"Parameters: δ = {delta:.2f}, U = {U:.2f}")
print(f"  → t1 = {t1:.2f}, t2 = {t2:.2f}")
print(f"\nVQE configuration:")
print(f"  Ansätze: {ansatze}")
print(f"  Optimizers: {optimizers}")
print(f"  Seeds: {seeds}")
print(f"  Max iterations: {maxiter}")
print(f"\nTotal VQE runs: {len(ansatze) * len(optimizers) * len(seeds)}")
print("=" * 70 + "\n")

# ============================================================================
# Build Hamiltonian and Get Exact Solution
# ============================================================================

print("Building Hamiltonian and computing exact solution...")
H = ssh_hubbard_hamiltonian(L, t1, t2, U)
print(f"  Hamiltonian: {H.num_qubits} qubits, {len(H.paulis)} terms")

H_matrix = H.to_matrix(sparse=True)
print(f"  Matrix dimension: {H_matrix.shape[0]} × {H_matrix.shape[1]}")
print(f"  Computing exact ground state...")

exact_start = time.time()
eigenvalues, _ = eigsh(H_matrix, k=1, which='SA')
E_exact = eigenvalues[0]
exact_time = time.time() - exact_start

print(f"  ✓ Exact energy: {E_exact:.6f} (computed in {exact_time:.2f}s)\n")

# ============================================================================
# Run VQE Tests
# ============================================================================

print("=" * 70)
print("RUNNING VQE TESTS")
print("=" * 70)

results = {}
test_start = time.time()

for ansatz_name in ansatze:
    print(f"\n{'=' * 70}")
    print(f"{ansatz_name} ANSATZ")
    print('=' * 70)

    # Build ansatz
    N = 2 * L
    if ansatz_name == 'HEA':
        ansatz = build_ansatz_hea(N, ansatz_reps)
    elif ansatz_name == 'HVA':
        ansatz = build_ansatz_hva_sshh(L, ansatz_reps, t1, t2, include_U=True)
    elif ansatz_name == 'NP_HVA':
        ansatz = build_ansatz_np_hva_sshh(L, ansatz_reps)

    print(f"Circuit: {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters")
    print(f"Depth: {ansatz.depth()}\n")

    results[ansatz_name] = {}

    for opt_name in optimizers:
        print(f"  [{opt_name}]")

        # Run multi-start VQE
        runner = VQERunner(maxiter=maxiter, optimizer_name=opt_name)
        multistart_result = run_multistart_vqe(runner, ansatz, H, seeds)

        # Calculate error
        best_energy = multistart_result['best_energy']
        rel_error = 100 * abs(best_energy - E_exact) / abs(E_exact)

        results[ansatz_name][opt_name] = {
            'energy': best_energy,
            'error': rel_error,
            'multistart': multistart_result
        }

        print(f"    Best energy: {best_energy:.6f}")
        print(f"    Rel. error:  {rel_error:.2f}%")
        print(f"    Mean ± std:  {multistart_result['mean_energy']:.6f} ± {multistart_result['std_energy']:.6f}")

        # Check for success
        if abs(best_energy) < 1e-6:
            print(f"    ⚠️  WARNING: Near-zero energy - likely broken!")
        elif rel_error < 10:
            print(f"    ✅ EXCELLENT performance")
        elif rel_error < 30:
            print(f"    ✅ GOOD performance")
        else:
            print(f"    ⚠️  High error - may need more iterations")

test_time = time.time() - test_start

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Exact ground state energy: {E_exact:.6f}\n")

print(f"{'Ansatz':<10} {'Optimizer':<12} {'Best Energy':<14} {'Error %':<10} {'Status'}")
print("-" * 70)

for ansatz_name in ansatze:
    for opt_name in optimizers:
        energy = results[ansatz_name][opt_name]['energy']
        error = results[ansatz_name][opt_name]['error']

        if abs(energy) < 1e-6:
            status = '❌ BROKEN'
        elif error < 10:
            status = '✅ EXCELLENT'
        elif error < 30:
            status = '✅ GOOD'
        else:
            status = '⚠️  NEEDS WORK'

        print(f"{ansatz_name:<10} {opt_name:<12} {energy:<14.6f} {error:<10.2f} {status}")

# ============================================================================
# Pass/Fail Check
# ============================================================================

print("\n" + "=" * 70)
print("PASS/FAIL ANALYSIS")
print("=" * 70)

hva_best = min(results['HVA'][opt]['error'] for opt in optimizers)
np_hva_best = min(results['NP_HVA'][opt]['error'] for opt in optimizers)
hea_best = min(results['HEA'][opt]['error'] for opt in optimizers)

print(f"\nBest errors:")
print(f"  HVA:    {hva_best:.2f}% ", end='')
if hva_best < 15:
    print("✅ PASSED")
else:
    print("❌ FAILED (expected <15%)")

print(f"  NP_HVA: {np_hva_best:.2f}% ", end='')
if np_hva_best < 20:
    print("✅ PASSED")
else:
    print("❌ FAILED (expected <20%)")

print(f"  HEA:    {hea_best:.2f}% (baseline)")

overall_pass = (hva_best < 15 and np_hva_best < 20)

print("\n" + "=" * 70)
if overall_pass:
    print("✅ OVERALL: TEST PASSED")
    print("   L=6 implementations working correctly!")
    print("   Ready for full parameter sweep on Colab.")
else:
    print("❌ OVERALL: TEST FAILED")
    print("   Some ansätze not achieving expected accuracy.")
    print("   May need more iterations or debugging.")
print("=" * 70)

print(f"\nTotal runtime: {test_time:.1f}s ({test_time/60:.2f} min)")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
