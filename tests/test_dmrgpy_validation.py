#!/usr/bin/env python3
"""
Validation Tests for dmrgpy SSH-Hubbard Implementation

This script tests the dmrgpy/ITensor implementation of SSH-Hubbard DMRG
against exact diagonalization, with specific focus on the regime where
TeNPy exhibits systematic errors (t2/t1 >= 0.5).

Purpose:
    1. Verify dmrgpy correctly implements SSH-Hubbard Hamiltonian
    2. Test whether ITensor avoids TeNPy's systematic error
    3. Characterize accuracy across parameter regimes

Test Cases:
    - L=2: Baseline (trivial, all methods should agree)
    - L=4, t2/t1 < 0.5: Safe regime (TeNPy works)
    - L=4, t2/t1 = 0.5: Threshold (TeNPy transition point)
    - L=4, t2/t1 >= 0.5: Critical regime (TeNPy fails with 1-6% error)

Expected Outcomes:
    - If dmrgpy gives <0.1% error: EXCELLENT (ITensor superior to TeNPy)
    - If dmrgpy gives <1.0% error: GOOD (better than TeNPy)
    - If dmrgpy gives >1.0% error: Needs investigation

Usage:
    python test_dmrgpy_validation.py

Requirements:
    - dmrgpy
    - ssh_hubbard_vqe module (for exact diagonalization)
    - ssh_hubbard_dmrgpy module
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check imports
try:
    from ssh_hubbard_dmrgpy import solve_ssh_hubbard_dmrgpy, compare_with_exact, HAS_DMRGPY
    has_dmrgpy_module = True
except ImportError as e:
    print(f"ERROR: Cannot import ssh_hubbard_dmrgpy: {e}")
    has_dmrgpy_module = False

try:
    from ssh_hubbard_vqe import exact_diagonalization_ssh_hubbard
    has_ed = True
except ImportError as e:
    print(f"ERROR: Cannot import exact_diagonalization_ssh_hubbard: {e}")
    has_ed = False


def test_dmrgpy_availability():
    """Test 1: Check if dmrgpy is available."""
    print("\n" + "="*70)
    print("TEST 1: dmrgpy Availability")
    print("="*70)

    if not has_dmrgpy_module:
        print("✗ FAIL: ssh_hubbard_dmrgpy module not found")
        return False

    if not HAS_DMRGPY:
        print("✗ FAIL: dmrgpy library not installed")
        print("\nTo install:")
        print("  pip install dmrgpy")
        return False

    print("✓ PASS: dmrgpy is available")
    return True


def test_small_system_l2():
    """Test 2: L=2 system (baseline test)."""
    print("\n" + "="*70)
    print("TEST 2: L=2 Baseline (All methods should agree)")
    print("="*70)

    if not HAS_DMRGPY or not has_ed:
        print("SKIP: Missing dependencies")
        return None

    L, t1, t2, U = 2, 1.0, 0.6, 1.0

    # Exact energy
    exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U, verbose=False)
    print(f"\nExact energy: {exact_energy:.10f}")

    # DMRG energy
    result = solve_ssh_hubbard_dmrgpy(L, t1, t2, U, maxm=100, nsweeps=5, verbose=False)

    if result['energy'] is None:
        print("✗ FAIL: DMRG did not converge")
        return False

    print(f"DMRG energy:  {result['energy']:.10f}")

    # Compare
    comp = compare_with_exact(L, t1, t2, U, exact_energy, result['energy'], verbose=False)

    error_pct = abs(comp['error_rel_pct'])
    print(f"\nRelative error: {comp['error_rel_pct']:+.6f}%")

    if error_pct < 0.01:
        print("✓ PASS: Error < 0.01% (excellent agreement)")
        return True
    elif error_pct < 0.1:
        print("✓ PASS: Error < 0.1% (good agreement)")
        return True
    else:
        print(f"✗ FAIL: Error {error_pct:.6f}% too large for L=2")
        return False


def test_safe_regime():
    """Test 3: L=4, t2/t1 < 0.5 (TeNPy works here)."""
    print("\n" + "="*70)
    print("TEST 3: Safe Regime (t2/t1 < 0.5)")
    print("="*70)

    if not HAS_DMRGPY or not has_ed:
        print("SKIP: Missing dependencies")
        return None

    L, t1, t2, U = 4, 1.0, 0.4, 1.0
    print(f"Parameters: L={L}, t1={t1}, t2={t2}, U={U}")
    print(f"t2/t1 ratio: {t2/t1:.4f} (< 0.5, safe regime)")

    # Exact energy
    exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U, verbose=False)
    print(f"\nExact energy: {exact_energy:.10f}")

    # DMRG energy
    result = solve_ssh_hubbard_dmrgpy(L, t1, t2, U, maxm=200, nsweeps=10, verbose=False)

    if result['energy'] is None:
        print("✗ FAIL: DMRG did not converge")
        return False

    print(f"DMRG energy:  {result['energy']:.10f}")

    # Compare
    comp = compare_with_exact(L, t1, t2, U, exact_energy, result['energy'], verbose=False)

    error_pct = abs(comp['error_rel_pct'])
    print(f"\nRelative error: {comp['error_rel_pct']:+.6f}%")

    if error_pct < 0.1:
        print("✓ PASS: Error < 0.1% in safe regime")
        return True
    elif error_pct < 1.0:
        print("✓ PASS: Error < 1.0% (acceptable)")
        return True
    else:
        print(f"✗ FAIL: Error {error_pct:.6f}% too large")
        return False


def test_threshold_regime():
    """Test 4: L=4, t2/t1 = 0.5 (TeNPy transition point)."""
    print("\n" + "="*70)
    print("TEST 4: Threshold Regime (t2/t1 = 0.5)")
    print("="*70)

    if not HAS_DMRGPY or not has_ed:
        print("SKIP: Missing dependencies")
        return None

    L, t1, t2, U = 4, 1.0, 0.5, 1.0
    print(f"Parameters: L={L}, t1={t1}, t2={t2}, U={U}")
    print(f"t2/t1 ratio: {t2/t1:.4f} (= 0.5, threshold)")
    print("TeNPy: Tiny error (~0.01%) at this point")

    # Exact energy
    exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U, verbose=False)
    print(f"\nExact energy: {exact_energy:.10f}")

    # DMRG energy
    result = solve_ssh_hubbard_dmrgpy(L, t1, t2, U, maxm=200, nsweeps=10, verbose=False)

    if result['energy'] is None:
        print("✗ FAIL: DMRG did not converge")
        return False

    print(f"DMRG energy:  {result['energy']:.10f}")

    # Compare
    comp = compare_with_exact(L, t1, t2, U, exact_energy, result['energy'], verbose=False)

    error_pct = abs(comp['error_rel_pct'])
    print(f"\nRelative error: {comp['error_rel_pct']:+.6f}%")

    if error_pct < 0.1:
        print("✓ PASS: Error < 0.1% at threshold")
        return True
    elif error_pct < 1.0:
        print("✓ PASS: Error < 1.0% (acceptable)")
        return True
    else:
        print(f"✗ FAIL: Error {error_pct:.6f}% too large")
        return False


def test_critical_regime():
    """Test 5: L=4, t2/t1 = 0.6 (TeNPy FAILS with 1.68% error)."""
    print("\n" + "="*70)
    print("TEST 5: CRITICAL REGIME (t2/t1 = 0.6)")
    print("="*70)
    print("⚠️  This is where TeNPy has systematic 1.68% error!")

    if not HAS_DMRGPY or not has_ed:
        print("SKIP: Missing dependencies")
        return None

    L, t1, t2, U = 4, 1.0, 0.6, 1.0
    print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")
    print(f"t2/t1 ratio: {t2/t1:.4f} (> 0.5, critical regime)")
    print("TeNPy gives: -2.6139 (exact: -2.6585, error: 1.68%)")

    # Exact energy
    exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U, verbose=False)
    print(f"\nExact energy: {exact_energy:.10f}")

    # DMRG energy
    result = solve_ssh_hubbard_dmrgpy(L, t1, t2, U, maxm=200, nsweeps=10, verbose=False)

    if result['energy'] is None:
        print("✗ FAIL: DMRG did not converge")
        return False

    print(f"DMRG energy:  {result['energy']:.10f}")

    # Compare
    comp = compare_with_exact(L, t1, t2, U, exact_energy, result['energy'], verbose=False)

    error_pct = abs(comp['error_rel_pct'])
    print(f"\nRelative error: {comp['error_rel_pct']:+.6f}%")

    print(f"\n{'─'*70}")
    print("CRITICAL TEST EVALUATION:")
    if error_pct < 0.1:
        print(f"✓✓ EXCELLENT: dmrgpy/ITensor achieves {error_pct:.6f}% error")
        print(f"   This is 17× better than TeNPy's 1.68% error!")
        print(f"   → ITensor does NOT have TeNPy's systematic bug")
        result_status = "excellent"
    elif error_pct < 1.0:
        print(f"✓ GOOD: dmrgpy/ITensor achieves {error_pct:.6f}% error")
        print(f"   This is better than TeNPy's 1.68% error")
        print(f"   → ITensor performs better than TeNPy in this regime")
        result_status = "good"
    elif error_pct < 1.5:
        print(f"△ SIMILAR: dmrgpy/ITensor has {error_pct:.6f}% error")
        print(f"   This is comparable to TeNPy's 1.68% error")
        print(f"   → ITensor may have similar issues to TeNPy")
        result_status = "similar"
    else:
        print(f"✗ WORSE: dmrgpy/ITensor has {error_pct:.6f}% error")
        print(f"   This is worse than TeNPy's 1.68% error")
        print(f"   → Needs investigation")
        result_status = "worse"
    print(f"{'─'*70}")

    # Pass if better than TeNPy
    if result_status in ["excellent", "good"]:
        print("✓ PASS: dmrgpy/ITensor outperforms TeNPy")
        return True
    else:
        print("✗ FAIL: dmrgpy/ITensor does not solve TeNPy's problem")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("SSH-HUBBARD DMRGPY VALIDATION TEST SUITE")
    print("="*70)
    print("\nTesting dmrgpy/ITensor implementation against exact diagonalization")
    print("Focus: Regime where TeNPy has systematic errors (t2/t1 >= 0.5)")
    print("="*70)

    if not has_dmrgpy_module:
        print("\nERROR: ssh_hubbard_dmrgpy module not found")
        print("Make sure you're running from tests/ directory")
        return

    if not has_ed:
        print("\nERROR: Cannot import exact diagonalization from ssh_hubbard_vqe")
        return

    results = {}

    # Run tests
    results['availability'] = test_dmrgpy_availability()

    if results['availability']:
        results['l2_baseline'] = test_small_system_l2()
        results['safe_regime'] = test_safe_regime()
        results['threshold'] = test_threshold_regime()
        results['critical'] = test_critical_regime()
    else:
        print("\nSKIPPING remaining tests (dmrgpy not available)")
        print("\nTo install dmrgpy:")
        print("  pip install dmrgpy")
        return

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    test_names = {
        'availability': 'dmrgpy Availability',
        'l2_baseline': 'L=2 Baseline',
        'safe_regime': 'Safe Regime (t2/t1 < 0.5)',
        'threshold': 'Threshold (t2/t1 = 0.5)',
        'critical': 'CRITICAL (t2/t1 = 0.6)'
    }

    for key, name in test_names.items():
        if key in results:
            status = results[key]
            if status is True:
                print(f"✓ PASS: {name}")
            elif status is False:
                print(f"✗ FAIL: {name}")
            else:
                print(f"⊘ SKIP: {name}")

    print("="*70)

    # Overall verdict
    if results.get('availability') and results.get('critical'):
        print("\n🎉 SUCCESS: dmrgpy/ITensor successfully solves SSH-Hubbard!")
        print("   It performs better than TeNPy in the problematic regime.")
        print("\nRecommendation: Use dmrgpy for SSH-Hubbard DMRG calculations")
    elif results.get('availability') and results.get('critical') is False:
        print("\n⚠️  WARNING: dmrgpy/ITensor does not solve TeNPy's issue")
        print("   Further investigation needed.")
    else:
        print("\n⊘ INCOMPLETE: Not all tests could run")

    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
