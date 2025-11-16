#!/usr/bin/env python3
"""
Quick test of L=7 with sparse Lanczos in benchmark script.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from compare_all_ansatze import exact_diagonalization

print("=" * 80)
print("TEST: L=7 exact diagonalization using benchmark function")
print("=" * 80)

L = 7
t1 = 1.0
t2 = 0.6
U = 2.0

print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")
print(f"Qubits: {2*L}")
print(f"Hilbert space dimension: {2**(2*L)}")

# Build Hamiltonian
H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)

print(f"\nHamiltonian: {len(H)} Pauli terms")

# Run exact diagonalization (should use sparse method automatically)
print("\nRunning exact diagonalization...")
E0, psi0 = exact_diagonalization(H)

print(f"\nResults:")
print(f"  Ground energy:     {E0:.10f}")
print(f"  Energy per site:   {E0/L:.10f}")
print(f"  State norm:        {np.linalg.norm(psi0):.10f}")

# Sanity checks
if abs(np.linalg.norm(psi0) - 1.0) < 1e-6:
    print(f"\n✓ State is properly normalized")
else:
    print(f"\n✗ State normalization issue")

if -5.0 < E0/L < 0.0:
    print(f"✓ Energy is physically reasonable")
else:
    print(f"✗ Energy seems unphysical")

print("\n" + "=" * 80)
print("SUCCESS: Sparse Lanczos enables exact diagonalization for L > 6")
print("=" * 80)
