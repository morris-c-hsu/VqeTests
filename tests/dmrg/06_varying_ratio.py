#!/usr/bin/env python3
"""
Test DMRG with varying t1/t2 ratios to understand the error pattern.

If the error is proportional to t2, that suggests something about the inter-cell coupling.
If it's proportional to both, it suggests an interaction effect.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

print("=" * 80)
print("TEST: VARYING t1/t2 RATIO")
print("=" * 80)

L = 4
U = 2.0

test_cases = [
    # (t1, t2, description)
    (1.0, 0.0, "Only t1 (isolated dimers)"),
    (0.0, 1.0, "Only t2 (inter-cell only)"),
    (1.0, 0.1, "Weak t2"),
    (1.0, 0.3, "Moderate t2"),
    (1.0, 0.6, "Original parameters"),
    (1.0, 1.0, "Equal hopping"),
    (0.6, 1.0, "Reversed SSH"),
]

print(f"\nL={L}, U={U}")
print()
print(f"{'t1':>6s} {'t2':>6s} {'E_exact':>14s} {'E_DMRG':>14s} {'Error':>12s} {'%':>8s}")
print("-" * 80)

results = []

for t1, t2, desc in test_cases:
    # Exact
    H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    H_matrix = H_vqe.to_matrix()
    E_exact = np.linalg.eigvalsh(H_matrix)[0]

    # DMRG
    result = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=100, verbose=False)
    E_dmrg = result['energy']

    # Error
    error = E_dmrg - E_exact
    rel_err = abs(error / E_exact) * 100

    results.append({
        't1': t1,
        't2': t2,
        'E_exact': E_exact,
        'E_dmrg': E_dmrg,
        'error': error,
        'rel_err': rel_err,
        'desc': desc
    })

    print(f"{t1:6.2f} {t2:6.2f} {E_exact:14.8f} {E_dmrg:14.8f} {error:+12.8f} {rel_err:8.4f}%")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nError vs parameters:")
for r in results:
    status = "✓" if r['rel_err'] < 0.01 else "✗"
    print(f"  {status} t1={r['t1']:.1f}, t2={r['t2']:.1f}: {r['rel_err']:6.3f}% - {r['desc']}")

# Look for patterns
print("\nPattern analysis:")

# Check if error is proportional to t2
has_t2 = [r for r in results if r['t2'] > 0]
no_t2 = [r for r in results if r['t2'] == 0]

if all(r['rel_err'] < 0.01 for r in no_t2) and any(r['rel_err'] > 0.1 for r in has_t2):
    print("  → Error appears when t2 > 0 (inter-cell coupling present)")

    # But we know t2-only is fine, so...
    has_both = [r for r in results if r['t1'] > 0 and r['t2'] > 0]
    if all(r['rel_err'] > 0.1 for r in has_both):
        print("  → Error ONLY when BOTH t1 and t2 are nonzero!")
        print("  → This suggests interference between intra-cell and inter-cell couplings")
else:
    print("  → Pattern unclear from these tests")

# Check if error magnitude correlates with t2
print("\nError vs t2 (for fixed t1=1.0):")
fixed_t1 = [r for r in results if r['t1'] == 1.0 and r['t2'] > 0]
for r in sorted(fixed_t1, key=lambda x: x['t2']):
    print(f"  t2={r['t2']:.2f}: error = {r['rel_err']:.3f}%")

print("\n" + "=" * 80)
