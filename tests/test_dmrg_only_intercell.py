#!/usr/bin/env python3
"""
Test L=4 with t1=0, t2≠0 (only inter-cell hopping).

This will test ONLY the problematic inter-cell coupling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

print("=" * 80)
print("TEST: L=4 WITH ONLY INTER-CELL HOPPING (t1=0, t2≠0)")
print("=" * 80)

L = 4
t1 = 0.0  # NO intra-dimer hopping
t2 = 1.0  # ONLY inter-dimer hopping
U = 2.0

print(f"\nParameters: L={L}, t1={t1:.3f}, t2={t2:.3f}, U={U:.3f}")
print()
print("With t1=0, only bond 1↔2 exists (inter-dimer bond)")
print("Bonds 0↔1 and 2↔3 are ABSENT")
print()

# Get exact result
print("=" * 80)
print("EXACT DIAGONALIZATION (VQE)")
print("=" * 80)

H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
H_vqe_matrix = H_vqe.to_matrix()
eig_vqe = np.linalg.eigvalsh(H_vqe_matrix)
E0_exact = eig_vqe[0]

print(f"Ground state energy: {E0_exact:.10f}")

# Get DMRG result
print("\n" + "=" * 80)
print("DMRG RESULT")
print("=" * 80)

result = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=100, verbose=False)
E0_dmrg = result['energy']

print(f"Ground state energy: {E0_dmrg:.10f}")
print(f"Bond dimensions:     {result['chi']}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = E0_dmrg - E0_exact
rel_err = abs(diff / E0_exact) * 100

print(f"\nExact:  {E0_exact:.10f}")
print(f"DMRG:   {E0_dmrg:.10f}")
print(f"Error:  {diff:+.10f}")
print(f"Rel %:  {rel_err:.6f}%")

if rel_err < 0.01:
    print("\n✓ Inter-cell coupling works correctly!")
    print("  → Error must be from INTERACTION between t1 and t2?")
    print("  → Or specific to the SSH pattern?")
elif rel_err > 1.0:
    print(f"\n✗ ERROR FOUND in inter-cell coupling! ({rel_err:.2f}%)")
    print("  → This is the source of the bug")
else:
    print(f"\n⚠ Small error ({rel_err:.3f}%) - unclear")

print("\n" + "=" * 80)
