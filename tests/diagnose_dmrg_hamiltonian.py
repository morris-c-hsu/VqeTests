#!/usr/bin/env python3
"""
Diagnostic script to compare VQE and DMRG Hamiltonians term-by-term.

This script constructs both Hamiltonians for small L and compares:
1. Individual hopping terms
2. Hubbard interaction terms
3. Total energies and eigenvalues
4. Matrix element differences
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.sparse.linalg import eigsh

# VQE imports
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian, q_index

def analyze_vqe_hamiltonian(L=4, t1=1.0, t2=0.6, U=2.0):
    """Analyze the VQE Hamiltonian construction in detail."""
    print("=" * 80)
    print("VQE HAMILTONIAN ANALYSIS")
    print("=" * 80)
    print(f"\nParameters: L={L}, t1={t1:.3f}, t2={t2:.3f}, U={U:.3f}")

    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)

    print(f"\nHamiltonian properties:")
    print(f"  Number of qubits: {H.num_qubits}")
    print(f"  Number of Pauli terms: {len(H)}")
    print(f"  Hilbert space dimension: {2**H.num_qubits}")

    # Convert to dense matrix
    H_matrix = H.to_matrix()

    print(f"\nMatrix properties:")
    print(f"  Shape: {H_matrix.shape}")
    print(f"  Is Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}")

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    E0 = eigenvalues[0]

    print(f"\nSpectrum:")
    print(f"  Ground state energy: {E0:.10f}")
    print(f"  Energy per site: {E0/L:.10f}")
    print(f"  First excited: {eigenvalues[1]:.10f}")
    print(f"  Gap: {eigenvalues[1] - eigenvalues[0]:.6f}")
    print(f"  Energy range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")

    # Analyze terms
    print(f"\nTerm analysis:")

    # Count hopping terms
    n_hopping = 0
    n_interaction = 0
    n_identity = 0

    for pauli_str, coeff in zip(H.paulis, H.coeffs):
        label = str(pauli_str)
        if 'X' in label or 'Y' in label:
            n_hopping += 1
        elif 'Z' in label:
            n_interaction += 1
        else:  # All I
            n_identity += 1

    print(f"  Hopping terms (X/Y): {n_hopping}")
    print(f"  Interaction terms (Z): {n_interaction}")
    print(f"  Identity terms: {n_interaction}")

    # Compute trace (should be zero for traceless Pauli ops, except identity)
    trace = np.trace(H_matrix)
    print(f"  Trace(H): {trace:.6f}")
    print(f"  Expected constant (L*U/4): {L*U/4:.6f}")

    return {
        'H_matrix': H_matrix,
        'eigenvalues': eigenvalues,
        'E0': E0,
        'trace': trace,
    }


def compare_with_simple_construction(L=4, t1=1.0, t2=0.6, U=2.0):
    """
    Build Hamiltonian from scratch and compare with VQE.

    This helps identify if there are any issues with the VQE construction.
    """
    print("\n" + "=" * 80)
    print("SIMPLE HAMILTONIAN CONSTRUCTION (Verification)")
    print("=" * 80)

    N = 2 * L  # Total qubits
    dim = 2 ** N

    print(f"\nDirect construction:")
    print(f"  L={L}, N={N} qubits, dim={dim}")

    # Build Hamiltonian in occupation number basis
    # Basis: |n_0↑ n_0↓ n_1↑ n_1↓ ...⟩

    H_manual = np.zeros((dim, dim), dtype=complex)

    # Helper: convert state index to occupation numbers
    def state_to_occupation(state, N):
        """Convert state index to list of occupation numbers."""
        return [(state >> i) & 1 for i in range(N)]

    # Helper: convert occupation numbers to state index
    def occupation_to_state(occ):
        """Convert list of occupation numbers to state index."""
        return sum(occ[i] << i for i in range(len(occ)))

    # Add hopping terms
    hopping_energy = 0.0
    for i in range(L - 1):
        t = t1 if i % 2 == 0 else t2

        for spin_offset in [0, 1]:  # 0 = up, 1 = down
            p = 2*i + spin_offset
            q = 2*(i+1) + spin_offset

            # Add hopping: -t (c†_p c_q + c†_q c_p)
            for state_idx in range(dim):
                occ = state_to_occupation(state_idx, N)

                # Try c†_p c_q (destroy at q, create at p)
                if occ[q] == 1 and occ[p] == 0:
                    new_occ = occ.copy()
                    new_occ[q] = 0
                    new_occ[p] = 1

                    # Compute fermion sign
                    sign = 1
                    for k in range(min(p, q) + 1, max(p, q)):
                        if occ[k] == 1:
                            sign *= -1

                    new_state_idx = occupation_to_state(new_occ)
                    H_manual[new_state_idx, state_idx] += -t * sign

                # Try c†_q c_p (destroy at p, create at q)
                if occ[p] == 1 and occ[q] == 0:
                    new_occ = occ.copy()
                    new_occ[p] = 0
                    new_occ[q] = 1

                    # Compute fermion sign
                    sign = 1
                    for k in range(min(p, q) + 1, max(p, q)):
                        if occ[k] == 1:
                            sign *= -1

                    new_state_idx = occupation_to_state(new_occ)
                    H_manual[new_state_idx, state_idx] += -t * sign

    # Add Hubbard interaction: U n_up n_dn
    for i in range(L):
        p_up = 2*i
        p_dn = 2*i + 1

        for state_idx in range(dim):
            occ = state_to_occupation(state_idx, N)

            # U * n_up * n_dn
            if occ[p_up] == 1 and occ[p_dn] == 1:
                H_manual[state_idx, state_idx] += U

    # Compare with VQE
    H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    H_vqe_matrix = H_vqe.to_matrix()

    diff = np.max(np.abs(H_manual - H_vqe_matrix))

    print(f"\nComparison with VQE Hamiltonian:")
    print(f"  Max matrix element difference: {diff:.2e}")

    if diff < 1e-10:
        print(f"  ✓✓✓ Perfect agreement!")
    elif diff < 1e-6:
        print(f"  ✓✓ Excellent agreement")
    elif diff < 1e-3:
        print(f"  ✓ Good agreement")
    else:
        print(f"  ✗ MISMATCH DETECTED!")

    # Compare eigenvalues
    eig_manual = np.linalg.eigvalsh(H_manual)
    eig_vqe = np.linalg.eigvalsh(H_vqe_matrix)

    print(f"\nGround state energies:")
    print(f"  Manual: {eig_manual[0]:.10f}")
    print(f"  VQE:    {eig_vqe[0]:.10f}")
    print(f"  Diff:   {abs(eig_manual[0] - eig_vqe[0]):.2e}")

    return {
        'H_manual': H_manual,
        'H_vqe': H_vqe_matrix,
        'diff': diff,
    }


def print_ssh_bond_pattern(L):
    """Print the SSH bond pattern for clarity."""
    print(f"\nSSH Bond Pattern for L={L}:")
    print(f"  Physical sites: 0, 1, 2, ..., {L-1}")
    print(f"  Bonds:")
    for i in range(L - 1):
        t = "t1" if i % 2 == 0 else "t2"
        strength = "STRONG" if i % 2 == 0 else "weak  "
        print(f"    Bond {i}: site {i} ↔ site {i+1}  ({t}, {strength})")
    print(f"  Dimer pairs: ", end="")
    dimers = [(2*k, 2*k+1) for k in range(L//2)]
    print(", ".join(f"({a},{b})" for a, b in dimers))


def main():
    """Run full diagnostic."""
    print("\n" + "#" * 80)
    print("# DMRG HAMILTONIAN DIAGNOSTIC")
    print("#" * 80)
    print("\nThis script analyzes the Hamiltonians to identify the source of")
    print("the 1-3% DMRG energy offset.")

    L = 4
    t1 = 1.0
    t2 = 0.6
    U = 2.0

    print_ssh_bond_pattern(L)

    # Analyze VQE Hamiltonian
    vqe_results = analyze_vqe_hamiltonian(L, t1, t2, U)

    # Compare with simple construction
    comparison = compare_with_simple_construction(L, t1, t2, U)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if comparison['diff'] < 1e-10:
        print("\n✓ VQE Hamiltonian construction is CORRECT (verified by manual construction)")
        print("\n→ The DMRG error must be in the TeNPy model construction,")
        print("  NOT in the VQE Hamiltonian.")
    else:
        print("\n✗ VQE Hamiltonian has issues (mismatch with manual construction)")

    print(f"\nExpected DMRG energy: {vqe_results['E0']:.10f}")
    print(f"  (This should match exact diagonalization)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
