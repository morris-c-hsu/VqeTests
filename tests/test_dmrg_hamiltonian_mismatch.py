"""
Test to investigate DMRG Hamiltonian mismatch

ISSUE: DMRG energies show 1-3% systematic offset compared to exact diagonalization.
This offset does NOT decrease with increased bond dimension (χ_max), indicating
a Hamiltonian construction mismatch rather than a convergence issue.

This test compares the VQE Hamiltonian with DMRG to identify the source of the discrepancy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.sparse.linalg import eigsh

def test_dmrg_vs_exact_energy():
    """
    Compare DMRG ground state energy with exact diagonalization.

    Expected: DMRG should match ED within numerical precision for small L with large χ.
    Observed: DMRG shows 1-3% offset that persists even at χ=500.
    """
    try:
        from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
        from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard
    except ImportError as e:
        print(f"Import error: {e}")
        print("Cannot run test - required modules not found")
        return

    # Test parameters
    L = 4  # Small system for exact comparison
    t1, t2, U = 1.0, 0.6, 2.0

    print("=" * 70)
    print("DMRG HAMILTONIAN MISMATCH INVESTIGATION")
    print("=" * 70)
    print(f"\nSystem: L={L}, t1={t1}, t2={t2}, U={U}")

    # 1. Get exact energy from VQE Hamiltonian
    print("\n1. Computing EXACT energy via VQE Hamiltonian...")
    H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    H_vqe_sparse = H_vqe.to_matrix(sparse=True)
    eigenvalues_exact, _ = eigsh(H_vqe_sparse, k=1, which='SA')
    E_exact = eigenvalues_exact[0]
    print(f"   Exact ground state energy: {E_exact:.10f}")

    # 2. Get DMRG energy
    print("\n2. Computing DMRG energy...")
    try:
        dmrg_result = run_dmrg_ssh_hubbard(
            L=L, t1=t1, t2=t2, U=U,
            chi_max=200,
            verbose=False
        )
        E_dmrg = dmrg_result['energy']
        print(f"   DMRG ground state energy: {E_dmrg:.10f}")
    except Exception as e:
        print(f"   DMRG failed: {e}")
        return

    # 3. Calculate discrepancy
    abs_error = abs(E_dmrg - E_exact)
    rel_error = abs(abs_error / E_exact) * 100

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Exact Energy (VQE):     {E_exact:.10f}")
    print(f"DMRG Energy:            {E_dmrg:.10f}")
    print(f"Absolute Error:         {abs_error:.10f}")
    print(f"Relative Error:         {rel_error:.4f}%")
    print("=" * 70)

    # 4. Diagnosis
    print("\nDIAGNOSIS:")
    if rel_error < 0.01:
        print("✓ DMRG matches exact energy within 0.01% - No mismatch!")
    elif rel_error < 0.1:
        print("⚠ Small discrepancy (<0.1%) - Likely numerical precision")
    elif rel_error < 2.0:
        print("⚠ MODERATE MISMATCH (1-2%) - Confirms reported systematic offset")
        print("  → This is NOT a convergence issue (χ=200 is sufficient for L=4)")
        print("  → Likely causes:")
        print("    1. SSH bond pattern ordering difference between VQE and TeNPy")
        print("    2. Unit-cell interpretation mismatch")
        print("    3. Jordan-Wigner or fermion sign conventions")
        print("    4. Factor of 2 in hopping or interaction terms")
    else:
        print("✗ LARGE MISMATCH (>2%) - Major Hamiltonian construction error")

    print("\nNEXT STEPS:")
    print("1. Verify SSH bond pattern in TeNPy model:")
    print("   - Strong bonds (t1): (0,1), (2,3)")
    print("   - Weak bonds (t2): (1,2), (3,4) for open BC")
    print("2. Check if TeNPy uses different site indexing")
    print("3. Add explicit Hamiltonian matrix comparison (if TeNPy allows)")
    print("4. Test single hopping term: H = -t1 * (c†_0 c_1 + h.c.)")
    print("=" * 70)


def test_ssh_bond_pattern_verification():
    """
    Verify that the SSH bond pattern is correctly implemented.

    SSH Model: Dimerized chain with alternating strong/weak bonds
    - Strong bonds (t1): connect (0,1), (2,3), (4,5), ...
    - Weak bonds (t2): connect (1,2), (3,4), (5,6), ...
    """
    print("\n" + "=" * 70)
    print("SSH BOND PATTERN VERIFICATION")
    print("=" * 70)

    L = 6
    print(f"\nFor L={L} sites with open boundary conditions:")
    print("\nExpected SSH bond pattern:")
    print("  Strong bonds (t1):", [(2*i, 2*i+1) for i in range(L//2)])
    print("  Weak bonds (t2):  ", [(2*i+1, 2*i+2) for i in range(L//2-1)])

    print("\n⚠️ TODO: Add code to extract actual bond pattern from TeNPy model")
    print("   and compare with expected pattern")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DMRG HAMILTONIAN MISMATCH TEST SUITE")
    print("=" * 70)
    print("\nPurpose: Investigate the 1-3% systematic energy offset in DMRG results")
    print("that does not improve with increasing bond dimension.\n")

    test_dmrg_vs_exact_energy()
    test_ssh_bond_pattern_verification()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("This systematic offset indicates a Hamiltonian construction mismatch")
    print("between the VQE and TeNPy DMRG implementations.")
    print("\nUntil resolved, DMRG results should be treated as approximate and")
    print("cannot serve as exact benchmarks for VQE validation.")
    print("=" * 70 + "\n")
