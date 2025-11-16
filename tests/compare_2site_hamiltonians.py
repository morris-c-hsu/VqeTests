#!/usr/bin/env python3
"""
Compare VQE and DMRG Hamiltonians for L=2 (smallest system).

This will allow us to examine the full Hamiltonian matrix and identify
exactly where the discrepancy comes from.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian

# For TeNPy
from ssh_hubbard_tenpy_dmrg_fixed import SpinfulSSHHubbard
from tenpy.linalg.np_conserved import tensordot
from tenpy.networks.mps import MPS

print("=" * 80)
print("2-SITE HAMILTONIAN COMPARISON")
print("=" * 80)

# Simplest parameters
L = 2
t1 = 1.0
t2 = 0.0  # No inter-dimer hopping for L=2
U = 2.0

print(f"\nParameters: L={L}, t1={t1:.3f}, t2={t2:.3f}, U={U:.3f}")
print(f"For L=2: Only one bond (0-1) with strength t1={t1}")
print()

# Get VQE Hamiltonian
print("=" * 80)
print("VQE HAMILTONIAN")
print("=" * 80)

H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
H_vqe_matrix = H_vqe.to_matrix()

print(f"VQE Hamiltonian properties:")
print(f"  Qubits: {H_vqe.num_qubits}")
print(f"  Dimension: {H_vqe_matrix.shape[0]}")
print(f"  Is Hermitian: {np.allclose(H_vqe_matrix, H_vqe_matrix.conj().T)}")

# Get eigenvalues
eig_vqe = np.linalg.eigvalsh(H_vqe_matrix)
E0_vqe = eig_vqe[0]

print(f"\nVQE spectrum:")
print(f"  Ground state: {E0_vqe:.10f}")
print(f"  First 5 eigenvalues:")
for i in range(min(5, len(eig_vqe))):
    print(f"    E[{i}] = {eig_vqe[i]:.6f}")

# Get DMRG result
print("\n" + "=" * 80)
print("DMRG RESULT")
print("=" * 80)

from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

result_dmrg = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=50, verbose=False)
E0_dmrg = result_dmrg['energy']

print(f"\nDMRG ground state: {E0_dmrg:.10f}")
print(f"Bond dimensions: {result_dmrg.get('chi', 'N/A')}")

# Try to extract full Hamiltonian from TeNPy model
print("\n" + "=" * 80)
print("TENPY HAMILTONIAN MATRIX")
print("=" * 80)

# Build the model
model_params = {'L': L, 't1': t1, 't2': t2, 'U': U}
model = SpinfulSSHHubbard(model_params)

# Get the MPO (matrix product operator) representation
H_mpo = model.H_MPO

print(f"TeNPy MPO properties:")
print(f"  Number of sites: {H_mpo.L}")
print(f"  Bond dimensions: {H_mpo.chi}")

# Try to convert MPO to dense matrix
# For small systems, we can do this
print("\nConverting MPO to dense matrix...")

try:
    from tenpy.linalg import np_conserved as npc

    # Start with identity on the left
    W = H_mpo.get_W(0)  # Get first MPO tensor
    # Contract in the left virtual index
    matrix = W.take_slice([0], ['wL'])  # Take left boundary

    # Contract with remaining MPO tensors
    for i in range(1, H_mpo.L):
        W_i = H_mpo.get_W(i)
        matrix = npc.tensordot(matrix, W_i, axes=(['wR'], ['wL']))
        # Contract physical indices if needed

    # Take right boundary
    matrix = matrix.take_slice([0], ['wR'])

    # Convert to dense numpy array
    # This is complex because of charge conservation...
    print("  (MPO to matrix conversion is complex with charge conservation)")
    print("  Skipping direct matrix comparison...")

except Exception as e:
    print(f"  Could not convert MPO to matrix: {e}")

# Compare energies
print("\n" + "=" * 80)
print("ENERGY COMPARISON")
print("=" * 80)

diff = E0_dmrg - E0_vqe
rel_err = abs(diff / E0_vqe) * 100

print(f"\nVQE ground energy:  {E0_vqe:.10f}")
print(f"DMRG ground energy: {E0_dmrg:.10f}")
print(f"Absolute difference: {diff:+.10f}")
print(f"Relative error:      {rel_err:.6f}%")

if rel_err < 0.01:
    print("\n✓✓✓ EXCELLENT AGREEMENT!")
elif rel_err < 0.1:
    print("\n✓✓ Very good agreement")
elif rel_err < 1.0:
    print("\n✓ Good agreement")
elif rel_err < 2.0:
    print("\n⚠ Moderate disagreement (similar to L=4 error)")
else:
    print("\n✗✗✗ LARGE DISAGREEMENT")

# Detailed term-by-term analysis
print("\n" + "=" * 80)
print("DETAILED ANALYSIS: What should the Hamiltonian be?")
print("=" * 80)

print(f"\nFor L=2 SSH-Hubbard with t1={t1}, t2={t2}, U={U}:")
print()
print(f"Physical sites: 0, 1")
print(f"Qubits (interleaved): 0↑, 0↓, 1↑, 1↓")
print()
print("Hopping terms (only one bond 0↔1):")
print(f"  H_hop = -t1 ∑_σ (c†_0σ c_1σ + c†_1σ c_0σ)")
print(f"        = -1.0 (c†_0↑ c_1↑ + h.c.) - 1.0 (c†_0↓ c_1↓ + h.c.)")
print()
print("Hubbard interaction:")
print(f"  H_U = U n_0↑ n_0↓ + U n_1↑ n_1↓")
print(f"      = 2.0 n_0↑ n_0↓ + 2.0 n_1↑ n_1↓")

# Let's manually build the Hamiltonian and compare
print("\n" + "=" * 80)
print("MANUAL HAMILTONIAN CONSTRUCTION")
print("=" * 80)

from itertools import product

# Basis states: |n_0↑ n_0↓ n_1↑ n_1↓⟩
# dim = 2^4 = 16

dim = 16
H_manual = np.zeros((dim, dim), dtype=complex)

def state_to_occ(s):
    """Convert state index to occupation numbers."""
    return [(s >> i) & 1 for i in range(4)]

def occ_to_state(occ):
    """Convert occupation numbers to state index."""
    return sum(occ[i] << i for i in range(4))

# Add hopping: -t1 (c†_p c_q + c†_q c_p) for p=0, q=2 (site 0↑ to site 1↑)
# and p=1, q=3 (site 0↓ to site 1↓)

print("Adding hopping terms...")

for spin in [0, 1]:  # 0=up, 1=down
    p = spin  # qubit index for site 0, spin σ
    q = 2 + spin  # qubit index for site 1, spin σ

    for s in range(dim):
        occ = state_to_occ(s)

        # c†_p c_q: destroy at q, create at p
        if occ[q] == 1 and occ[p] == 0:
            new_occ = occ.copy()
            new_occ[q] = 0
            new_occ[p] = 1

            # Fermion sign
            sign = 1
            for k in range(min(p,q)+1, max(p,q)):
                if occ[k] == 1:
                    sign *= -1

            new_s = occ_to_state(new_occ)
            H_manual[new_s, s] += -t1 * sign

        # c†_q c_p: destroy at p, create at q
        if occ[p] == 1 and occ[q] == 0:
            new_occ = occ.copy()
            new_occ[p] = 0
            new_occ[q] = 1

            # Fermion sign
            sign = 1
            for k in range(min(p,q)+1, max(p,q)):
                if occ[k] == 1:
                    sign *= -1

            new_s = occ_to_state(new_occ)
            H_manual[new_s, s] += -t1 * sign

# Add Hubbard interaction
print("Adding Hubbard interaction...")

for site in [0, 1]:
    p_up = 2 * site
    p_dn = 2 * site + 1

    for s in range(dim):
        occ = state_to_occ(s)
        if occ[p_up] == 1 and occ[p_dn] == 1:
            H_manual[s, s] += U

# Compare
diff_manual = np.max(np.abs(H_manual - H_vqe_matrix))

print(f"\nManual vs VQE Hamiltonian:")
print(f"  Max difference: {diff_manual:.2e}")

if diff_manual < 1e-10:
    print(f"  ✓✓✓ Perfect match! VQE Hamiltonian is correct.")
else:
    print(f"  ✗ Mismatch detected!")

# Compare eigenvalues
eig_manual = np.linalg.eigvalsh(H_manual)

print(f"\nEigenvalue comparison:")
print(f"  Manual: {eig_manual[0]:.10f}")
print(f"  VQE:    {eig_vqe[0]:.10f}")
print(f"  DMRG:   {E0_dmrg:.10f}")

print("\n" + "=" * 80)
print("CONCLUSION FOR L=2")
print("=" * 80)

if diff_manual < 1e-10:
    print("\n✓ VQE Hamiltonian construction is verified correct")

    if rel_err < 0.01:
        print("✓ DMRG also gives correct result for L=2!")
        print("→ The error only appears for L≥4?")
    else:
        print(f"⚠ DMRG has {rel_err:.2f}% error even for L=2")
        print("→ This suggests a fundamental issue with TeNPy model")
else:
    print("✗ VQE Hamiltonian has issues")

print("\n" + "=" * 80)
