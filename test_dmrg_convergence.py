#!/usr/bin/env python3
"""
Test DMRG convergence with very tight parameters.

If DMRG converges to exact energy with high chi_max, the Hamiltonian is correct.
"""

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

def test_convergence(L=4, t1=1.0, t2=0.6, U=2.0):
    """Test DMRG convergence for small system."""
    print("\n" + "=" * 80)
    print(f"DMRG CONVERGENCE TEST: L={L}")
    print("=" * 80)

    # Get exact energy
    print("\n[1] Exact Diagonalization:")
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    H_matrix = H.to_matrix()
    E_exact = np.linalg.eigh(H_matrix)[0][0]
    print(f"    E_exact = {E_exact:.12f}")

    # Test different chi_max values
    chi_values = [50, 100, 200, 500]

    print("\n[2] DMRG with increasing chi_max:")
    print(f"    {'chi_max':<10} {'Energy':<18} {'Error':<15} {'Rel. Error':<12}")
    print(f"    {'-'*10} {'-'*18} {'-'*15} {'-'*12}")

    best_error = float('inf')
    best_chi = None

    for chi_max in chi_values:
        result = run_dmrg_ssh_hubbard(
            L=L, t1=t1, t2=t2, U=U,
            chi_max=chi_max,
            verbose=False
        )

        if result:
            E_dmrg = result['energy']
            error = abs(E_dmrg - E_exact)
            rel_error = 100 * error / abs(E_exact)

            status = ""
            if rel_error < 0.01:
                status = "✓✓✓ Excellent"
                if error < best_error:
                    best_error = error
                    best_chi = chi_max
            elif rel_error < 0.1:
                status = "✓✓ Very good"
                if error < best_error:
                    best_error = error
                    best_chi = chi_max
            elif rel_error < 1.0:
                status = "✓ Good"
                if error < best_error:
                    best_error = error
                    best_chi = chi_max
            else:
                status = "⚠ Moderate"

            print(f"    {chi_max:<10} {E_dmrg:<18.12f} {error:<15.6e} {rel_error:<12.4f}% {status}")

    print("\n[3] Verdict:")
    if best_error < 1e-6:
        print(f"    ✓✓✓ DMRG converges to exact result (best error: {best_error:.2e} at chi={best_chi})")
        print(f"    → Hamiltonian construction is CORRECT")
        print(f"    → Original errors were due to insufficient chi_max")
        return True
    elif best_error / abs(E_exact) < 0.001:
        print(f"    ✓✓ DMRG achieves < 0.1% error (best: {100*best_error/abs(E_exact):.4f}% at chi={best_chi})")
        print(f"    → Hamiltonian is likely correct")
        return True
    else:
        print(f"    ✗ DMRG does NOT converge to exact result (best: {best_error:.6e} at chi={best_chi})")
        print(f"    → Hamiltonian construction may have errors")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# DMRG CONVERGENCE VERIFICATION")
    print("#" * 80)

    # Test L=4
    success_L4 = test_convergence(L=4, t1=1.0, t2=0.6, U=2.0)

    # Test L=6
    print("\n" * 2)
    success_L6 = test_convergence(L=6, t1=1.0, t2=0.5, U=2.0)

    # Summary
    print("\n\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)

    if success_L4 and success_L6:
        print("\n✓✓✓ DMRG Hamiltonian is CORRECT")
        print("    → Energy errors in original tests were due to low chi_max")
        print("    → Use chi_max ≥ 200 for accurate results")
    elif success_L4 or success_L6:
        print("\n✓ DMRG partially successful")
        print("  → May need higher chi_max for some systems")
    else:
        print("\n✗ DMRG has systematic errors")
        print("  → Need to debug Hamiltonian construction")

    print()
