#!/usr/bin/env python3
"""
Find the exact threshold of t2 where the error appears.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

print("=" * 80)
print("FINDING ERROR THRESHOLD")
print("=" * 80)

L = 4
t1 = 1.0
U = 2.0

# Scan t2 values around the threshold
t2_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"\nL={L}, t1={t1}, U={U}")
print(f"\nScanning t2 from 0.20 to 0.60 to find error threshold...")
print()
print(f"{'t2':>6s} {'E_exact':>14s} {'E_DMRG':>14s} {'Error %':>10s}")
print("-" * 60)

for t2 in t2_values:
    # Exact
    H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    E_exact = np.linalg.eigvalsh(H_vqe.to_matrix())[0]

    # DMRG
    result = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=100, verbose=False)
    E_dmrg = result['energy']

    # Error
    rel_err = abs((E_dmrg - E_exact) / E_exact) * 100
    status = "✓" if rel_err < 0.01 else "✗"

    print(f"{t2:6.2f} {E_exact:14.8f} {E_dmrg:14.8f} {rel_err:10.6f}% {status}")

print("\n" + "=" * 80)
