#!/usr/bin/env python3
"""
Test L=4 system with t2=0 (isolated dimers).

If the error is in the inter-cell coupling, this should give perfect agreement.
If the error persists, it's something else about having multiple unit cells.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

print("=" * 80)
print("TEST: L=4 WITH ISOLATED DIMERS (t2=0)")
print("=" * 80)

L = 4
t1 = 1.0
t2 = 0.0  # NO inter-dimer hopping
U = 2.0

print(f"\nParameters: L={L}, t1={t1:.3f}, t2={t2:.3f}, U={U:.3f}")
print()
print("With t2=0, we have two ISOLATED dimers:")
print("  Dimer 0: sites (0,1) with hopping t1")
print("  Dimer 1: sites (2,3) with hopping t1")
print("  NO coupling between dimers")
print()
print("Expected: E_total = 2 × E_single_dimer")
print("         where E_single_dimer = -1.236068 (from L=2 test)")
print(f"         So E_total ≈ {2 * (-1.236068):.6f}")

# Get exact result
print("\n" + "=" * 80)
print("EXACT DIAGONALIZATION (VQE)")
print("=" * 80)

H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
H_vqe_matrix = H_vqe.to_matrix()
eig_vqe = np.linalg.eigvalsh(H_vqe_matrix)
E0_exact = eig_vqe[0]

print(f"Ground state energy: {E0_exact:.10f}")
print(f"Energy per dimer:    {E0_exact/2:.10f}")
print(f"Expected per dimer:  -1.2360679775")

# Get DMRG result
print("\n" + "=" * 80)
print("DMRG RESULT")
print("=" * 80)

result = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=100, verbose=False)
E0_dmrg = result['energy']

print(f"Ground state energy: {E0_dmrg:.10f}")
print(f"Energy per dimer:    {E0_dmrg/2:.10f}")
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
    print("\n✓✓✓ PERFECT! Error is in the t2 (inter-cell) coupling!")
    print("    → The issue is with how TeNPy adds the dx=[1] coupling")
    print("    → Or possibly in how the lattice boundary is handled")
elif rel_err > 1.0:
    print("\n✗✗ ERROR PERSISTS even with t2=0")
    print("   → The issue is NOT in the inter-cell coupling")
    print("   → It's something about having multiple unit cells")
    print("   → Possibly wrong fermion signs or unit cell structure")
else:
    print(f"\n⚠ Small error ({rel_err:.3f}%) - inconclusive")

print("\n" + "=" * 80)
