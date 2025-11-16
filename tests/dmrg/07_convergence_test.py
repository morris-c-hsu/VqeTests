#!/usr/bin/env python3
"""
Test if the error for large t2 is a convergence issue.

If increasing chi_max fixes it, it's just DMRG convergence.
If not, it's a real bug in the Hamiltonian.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

print("=" * 80)
print("CONVERGENCE TEST: Does higher χ fix the t2=0.6 error?")
print("=" * 80)

L = 4
t1 = 1.0
t2 = 0.6
U = 2.0

# Get exact
H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
E_exact = np.linalg.eigvalsh(H_vqe.to_matrix())[0]

print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")
print(f"Exact energy: {E_exact:.10f}")
print()

chi_values = [10, 20, 50, 100, 200, 500, 1000]

print(f"{'χ_max':>6s} {'E_DMRG':>14s} {'Error':>12s} {'%':>8s} {'χ_actual':>10s}")
print("-" * 80)

for chi_max in chi_values:
    result = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=chi_max, verbose=False)
    E_dmrg = result['energy']
    error = E_dmrg - E_exact
    rel_err = abs(error / E_exact) * 100
    chi_actual = max(result['chi']) if 'chi' in result and result['chi'] else 'N/A'

    print(f"{chi_max:6d} {E_dmrg:14.10f} {error:+12.8f} {rel_err:8.4f}% {str(chi_actual):>10s}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
If error decreases with increasing χ:
  → It's a DMRG convergence issue (need higher bond dimension)
  → Not a bug in the Hamiltonian

If error is constant regardless of χ:
  → It's a systematic error in the Hamiltonian construction
  → THIS is what we observed before (error doesn't change with χ)
""")

print("=" * 80)
