#!/usr/bin/env python3
"""
Verification test: Ensure both implementations produce identical Hamiltonians.
"""

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian as H_main
from ssh_hubbard_tn_vqe import ssh_hubbard_hamiltonian as H_tn

# Test parameters
L = 4
t1 = 1.0
t2 = 0.5
U = 2.0

print("=" * 70)
print("HAMILTONIAN CONSISTENCY TEST")
print("=" * 70)

# Build Hamiltonians from both implementations
print("\nBuilding Hamiltonian from ssh_hubbard_vqe.py...")
H1 = H_main(L, t1, t2, U, periodic=False)
print(f"  Terms: {len(H1)}, Qubits: {H1.num_qubits}")

print("\nBuilding Hamiltonian from ssh_hubbard_tn_vqe.py...")
H2 = H_tn(L, t1, t2, U)
print(f"  Terms: {len(H2)}, Qubits: {H2.num_qubits}")

# Convert to matrices
print("\nConverting to matrices...")
M1 = H1.to_matrix()
M2 = H2.to_matrix()

# Check if they're identical
diff = np.linalg.norm(M1 - M2)
print(f"\nMatrix difference (Frobenius norm): {diff:.2e}")

if diff < 1e-10:
    print("✅ PASS: Hamiltonians are identical!")
else:
    print("❌ FAIL: Hamiltonians differ!")
    print(f"\nMax element difference: {np.max(np.abs(M1 - M2)):.2e}")

# Check eigenvalues
print("\nChecking ground state energies...")
E1 = np.linalg.eigvalsh(M1)[0]
E2 = np.linalg.eigvalsh(M2)[0]
print(f"  ssh_hubbard_vqe.py:    {E1:.10f}")
print(f"  ssh_hubbard_tn_vqe.py: {E2:.10f}")
print(f"  Difference:            {abs(E1 - E2):.2e}")

if abs(E1 - E2) < 1e-10:
    print("✅ PASS: Ground state energies match!")
else:
    print("❌ FAIL: Ground state energies differ!")

print("\n" + "=" * 70)
