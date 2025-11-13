#!/usr/bin/env python3
"""
Verify DMRG Hamiltonian matches VQE Hamiltonian.

Extracts the Hamiltonian matrix from TeNPy DMRG model and compares
with the Qiskit VQE Hamiltonian.
"""

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import SpinfulSSHHubbard

def test_hamiltonian_equivalence(L=4, t1=1.0, t2=0.6, U=2.0):
    """
    Compare DMRG and VQE Hamiltonians for small L.

    Parameters
    ----------
    L : int
        Number of physical sites (must be even)
    t1, t2 : float
        SSH hopping parameters
    U : float
        Hubbard interaction
    """
    print("\n" + "=" * 80)
    print(f"HAMILTONIAN VERIFICATION: L={L}")
    print("=" * 80)

    # 1. Get VQE Hamiltonian (Qiskit)
    print("\n[1] Building VQE Hamiltonian (Qiskit)...")
    H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    H_vqe_matrix = H_vqe.to_matrix()

    # Get ground state from exact diagonalization
    eigenvalues = np.linalg.eigh(H_vqe_matrix)[0]
    E_exact = eigenvalues[0]

    print(f"    Hilbert space dim: {H_vqe_matrix.shape[0]}")
    print(f"    Ground energy:     {E_exact:.10f}")

    # 2. Get DMRG Hamiltonian (TeNPy)
    print("\n[2] Building DMRG Hamiltonian (TeNPy)...")
    model_params = {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        'bc_MPS': 'finite',
    }

    model = SpinfulSSHHubbard(model_params)

    # Extract Hamiltonian as MPO and convert to matrix
    H_mpo = model.calc_H_MPO()
    H_dmrg_matrix = H_mpo.to_matrix()

    print(f"    Hilbert space dim: {H_dmrg_matrix.shape[0]}")

    # 3. Compare matrices
    print("\n[3] Comparing Hamiltonians...")

    # Check if dimensions match
    if H_vqe_matrix.shape != H_dmrg_matrix.shape:
        print(f"    ✗ ERROR: Dimension mismatch!")
        print(f"      VQE:  {H_vqe_matrix.shape}")
        print(f"      DMRG: {H_dmrg_matrix.shape}")
        return False

    # Compute difference
    diff = H_vqe_matrix - H_dmrg_matrix
    diff_norm = np.linalg.norm(diff, 'fro')
    max_diff = np.max(np.abs(diff))

    print(f"    Matrix shape:      {H_vqe_matrix.shape}")
    print(f"    Frobenius norm:    {np.linalg.norm(H_vqe_matrix, 'fro'):.6f}")
    print(f"    Difference (Frob): {diff_norm:.6e}")
    print(f"    Max element diff:  {max_diff:.6e}")

    # Check ground state energies
    E_dmrg_diag = np.linalg.eigh(H_dmrg_matrix)[0][0]
    energy_diff = abs(E_exact - E_dmrg_diag)

    print(f"\n    Ground energies:")
    print(f"      VQE (exact):     {E_exact:.10f}")
    print(f"      DMRG (from H):   {E_dmrg_diag:.10f}")
    print(f"      Difference:      {energy_diff:.6e}")

    # Verdict
    print("\n[4] Verdict:")
    if diff_norm < 1e-10:
        print("    ✓✓✓ Hamiltonians are IDENTICAL!")
        return True
    elif diff_norm < 1e-6:
        print("    ✓✓ Hamiltonians are very close (numerical precision)")
        return True
    elif diff_norm / np.linalg.norm(H_vqe_matrix, 'fro') < 1e-4:
        print(f"    ✓ Hamiltonians are similar (rel. diff = {diff_norm / np.linalg.norm(H_vqe_matrix, 'fro'):.2e})")
        return True
    else:
        print(f"    ✗ Hamiltonians DIFFER significantly!")
        print(f"      This explains the DMRG energy errors.")

        # Print a sample of differences
        print(f"\n    Sample of non-zero differences:")
        non_zero_idx = np.where(np.abs(diff) > 1e-10)
        n_samples = min(10, len(non_zero_idx[0]))
        for k in range(n_samples):
            i, j = non_zero_idx[0][k], non_zero_idx[1][k]
            print(f"      H[{i:3d},{j:3d}]: VQE={H_vqe_matrix[i,j].real:+.6f}, "
                  f"DMRG={H_dmrg_matrix[i,j].real:+.6f}, "
                  f"diff={diff[i,j].real:+.6f}")

        return False


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# DMRG HAMILTONIAN VERIFICATION")
    print("#" * 80)

    # Test L=4 (small enough to compare matrices)
    success_L4 = test_hamiltonian_equivalence(L=4, t1=1.0, t2=0.6, U=2.0)

    print("\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)

    if success_L4:
        print("\n✓ DMRG Hamiltonian construction is CORRECT")
        print("  → DMRG energy errors are due to convergence, not wrong Hamiltonian")
        print("  → Increase chi_max or max_sweeps to improve accuracy")
    else:
        print("\n✗ DMRG Hamiltonian construction has ERRORS")
        print("  → Need to fix the unit cell structure or term definitions")

    print()
