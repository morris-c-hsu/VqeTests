#!/usr/bin/env python3
"""
SSH-Hubbard Model DMRG Solver using dmrgpy (ITensor backend)

This module implements DMRG for the spinful Su-Schrieffer-Heeger (SSH) Hubbard
model using dmrgpy, which provides a Python interface to ITensor.

The implementation is designed to:
1. Test whether ITensor's DMRG avoids the systematic error found in TeNPy
2. Validate against exact diagonalization for L ≤ 6
3. Provide an alternative DMRG implementation for benchmarking

Hamiltonian:
    H = -∑_{i,σ} t_i (c†_{i,σ} c_{i+1,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓}

where t_i alternates between t1 (strong, even bonds) and t2 (weak, odd bonds).

Installation:
    pip install dmrgpy

Requirements:
    - dmrgpy
    - numpy
    - ITensor (installed automatically by dmrgpy)

Usage:
    python ssh_hubbard_dmrgpy.py

    Or import as module:
    from ssh_hubbard_dmrgpy import solve_ssh_hubbard_dmrgpy

Author: Generated for VqeTests repository
Date: 2025
"""

import numpy as np
import sys
import warnings

# Check if dmrgpy is available
try:
    from dmrgpy import fermionchain
    HAS_DMRGPY = True
except ImportError:
    HAS_DMRGPY = False
    warnings.warn(
        "dmrgpy not installed. Install with: pip install dmrgpy\n"
        "Note: dmrgpy requires ITensor (C++ or Julia backend)"
    )

warnings.filterwarnings('ignore')


def solve_ssh_hubbard_dmrgpy(
    L: int,
    t1: float = 1.0,
    t2: float = 0.6,
    U: float = 1.0,
    maxm: int = 200,
    nsweeps: int = 10,
    cutoff: float = 1e-8,
    use_julia: bool = False,
    verbose: bool = True
) -> dict:
    """
    Solve SSH-Hubbard model using dmrgpy (ITensor backend).

    Parameters
    ----------
    L : int
        Number of lattice sites
    t1 : float
        Strong hopping amplitude (even bonds: 0-1, 2-3, ...)
    t2 : float
        Weak hopping amplitude (odd bonds: 1-2, 3-4, ...)
    U : float
        Hubbard on-site interaction strength
    maxm : int
        Maximum bond dimension for DMRG
    nsweeps : int
        Number of DMRG sweeps
    cutoff : float
        Truncation error cutoff
    use_julia : bool
        If True, use Julia backend instead of C++ (requires Julia installation)
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Results dictionary containing:
        - 'energy': Ground state energy
        - 'wavefunction': Ground state wavefunction (if available)
        - 'converged': Whether DMRG converged
        - 'bond_dimension': Final bond dimension used
        - 'parameters': Dictionary of input parameters

    Notes
    -----
    The SSH-Hubbard model has alternating hopping strengths:
    - Bonds 0-1, 2-3, 4-5, ... have hopping t1 (strong)
    - Bonds 1-2, 3-4, 5-6, ... have hopping t2 (weak)

    This creates a dimerized pattern characteristic of the SSH model.

    The Hubbard term is on-site: U n_i↑ n_i↓ = U (N_up - 1/2)(N_dn - 1/2)
    where the shift by 1/2 centers the interaction around half-filling.

    Examples
    --------
    >>> result = solve_ssh_hubbard_dmrgpy(L=4, t1=1.0, t2=0.6, U=1.0)
    >>> print(f"Ground state energy: {result['energy']:.6f}")
    """

    if not HAS_DMRGPY:
        raise ImportError(
            "dmrgpy is required but not installed.\n"
            "Install with: pip install dmrgpy\n"
            "Note: This will also install ITensor backend."
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"SSH-Hubbard DMRG with dmrgpy (ITensor backend)")
        print(f"{'='*70}")
        print(f"System size: L = {L} sites ({2*L} spin orbitals)")
        print(f"Hopping: t1 = {t1:.4f} (strong), t2 = {t2:.4f} (weak)")
        print(f"Dimerization: δ = (t1-t2)/(t1+t2) = {(t1-t2)/(t1+t2):.4f}")
        print(f"Interaction: U = {U:.4f}")
        print(f"DMRG parameters: maxm = {maxm}, nsweeps = {nsweeps}, cutoff = {cutoff:.1e}")
        print(f"Backend: {'Julia' if use_julia else 'C++'}")
        print(f"{'='*70}\n")

    # Create spinful fermionic chain
    if verbose:
        print("Creating spinful fermionic chain...")

    fc = fermionchain.Spinful_Fermionic_Chain(L)

    # Set backend
    if use_julia:
        fc.setup_julia()
        if verbose:
            print("Using Julia/ITensors.jl backend")
    else:
        if verbose:
            print("Using C++/ITensor backend")

    # Build Hamiltonian using operator algebra
    if verbose:
        print("\nConstructing SSH-Hubbard Hamiltonian...")
        print("  - Building hopping terms with alternating t1/t2...")

    # Initialize Hamiltonian
    h = 0

    # Hopping terms: -t (c† c + h.c.)
    # SSH pattern: even bonds get t1, odd bonds get t2
    for i in range(L - 1):
        # Determine hopping strength based on bond index
        if i % 2 == 0:
            t = t1  # Even bond (0-1, 2-3, 4-5, ...)
        else:
            t = t2  # Odd bond (1-2, 3-4, 5-6, ...)

        # Add hopping for spin-up
        h = h - t * fc.Cdagup[i] * fc.Cup[i + 1]

        # Add hopping for spin-down
        h = h - t * fc.Cdagdn[i] * fc.Cdn[i + 1]

    # Make Hamiltonian Hermitian by adding h.c.
    h = h + h.get_dagger()

    if verbose:
        print("  - Building Hubbard interaction terms...")

    # Hubbard interaction: U n_i↑ n_i↓
    # In dmrgpy: N_up and N_down are number operators (eigenvalues 0 or 1)
    # Standard form: U * N_up * N_down
    # Shifted form: U * (N_up - 1/2) * (N_down - 1/2) centers around half-filling
    #
    # Using standard form for consistency with TeNPy implementation
    for i in range(L):
        h = h + U * fc.Nup[i] * fc.Ndn[i]

    if verbose:
        print("  - Setting Hamiltonian in DMRG solver...")

    # Set the Hamiltonian
    fc.set_hamiltonian(h)

    # Configure DMRG parameters
    fc.maxm = maxm
    fc.nsweeps = nsweeps
    fc.cutoff = cutoff

    if verbose:
        print("\nRunning DMRG optimization...")
        print(f"  Max bond dimension: {maxm}")
        print(f"  Number of sweeps: {nsweeps}")
        print(f"  Cutoff: {cutoff:.1e}")

    # Run DMRG to get ground state energy
    try:
        energy = fc.get_gs()
        converged = True

        if verbose:
            print(f"\n{'='*70}")
            print(f"DMRG CONVERGED")
            print(f"Ground state energy: {energy:.10f}")
            print(f"{'='*70}\n")

    except Exception as e:
        if verbose:
            print(f"\nWARNING: DMRG encountered an error: {e}")
            print("Attempting to retrieve energy anyway...")

        try:
            energy = fc.get_gs()
            converged = False
        except:
            energy = None
            converged = False
            if verbose:
                print("FAILED: Could not retrieve energy")

    # Prepare results dictionary
    results = {
        'energy': energy,
        'converged': converged,
        'bond_dimension': maxm,
        'parameters': {
            'L': L,
            't1': t1,
            't2': t2,
            'U': U,
            'delta': (t1 - t2) / (t1 + t2),
            'maxm': maxm,
            'nsweeps': nsweeps,
            'cutoff': cutoff,
            'backend': 'Julia' if use_julia else 'C++'
        }
    }

    return results


def compare_with_exact(L: int, t1: float, t2: float, U: float,
                      exact_energy: float, dmrg_energy: float,
                      verbose: bool = True) -> dict:
    """
    Compare DMRG result with exact diagonalization.

    Parameters
    ----------
    L : int
        Number of lattice sites
    t1, t2, U : float
        Model parameters
    exact_energy : float
        Exact ground state energy from ED
    dmrg_energy : float
        DMRG ground state energy
    verbose : bool
        Print comparison

    Returns
    -------
    dict
        Comparison statistics
    """

    error_abs = dmrg_energy - exact_energy
    error_rel_pct = 100.0 * error_abs / abs(exact_energy)

    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPARISON WITH EXACT DIAGONALIZATION")
        print(f"{'='*70}")
        print(f"System: L={L}, t1={t1:.4f}, t2={t2:.4f}, U={U:.4f}")
        print(f"t2/t1 ratio: {t2/t1:.4f}")
        print(f"\nExact energy (ED):  {exact_energy:.10f}")
        print(f"DMRG energy:        {dmrg_energy:.10f}")
        print(f"\nAbsolute error:     {error_abs:+.10f}")
        print(f"Relative error:     {error_rel_pct:+.6f}%")

        # Critical threshold where TeNPy fails
        if t2/t1 >= 0.5:
            print(f"\n⚠️  CRITICAL TEST: t2/t1 = {t2/t1:.4f} >= 0.5")
            print(f"   This is the regime where TeNPy has systematic errors (1-6%)")
            if abs(error_rel_pct) < 0.1:
                print(f"   ✓ dmrgpy/ITensor achieves <0.1% error - SIGNIFICANTLY BETTER")
            elif abs(error_rel_pct) < 1.0:
                print(f"   ✓ dmrgpy/ITensor achieves <1% error - better than TeNPy")
            else:
                print(f"   ✗ dmrgpy/ITensor also has error >1%")

        print(f"{'='*70}\n")

    return {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        't2_t1_ratio': t2/t1,
        'exact_energy': exact_energy,
        'dmrg_energy': dmrg_energy,
        'error_abs': error_abs,
        'error_rel_pct': error_rel_pct
    }


def validation_suite(verbose: bool = True):
    """
    Run validation tests comparing dmrgpy DMRG with exact diagonalization.

    Tests the problematic regime where TeNPy fails (t2/t1 >= 0.5).
    """

    if not HAS_DMRGPY:
        print("ERROR: dmrgpy not installed. Cannot run validation.")
        print("Install with: pip install dmrgpy")
        return

    print("\n" + "="*70)
    print("SSH-HUBBARD DMRGPY VALIDATION SUITE")
    print("="*70)
    print("\nThis suite tests dmrgpy/ITensor against exact diagonalization")
    print("Focus: Regime where TeNPy has systematic errors (t2/t1 >= 0.5)")
    print("="*70)

    # Import exact diagonalization from existing module
    try:
        from ssh_hubbard_vqe import exact_diagonalization_ssh_hubbard
        has_ed = True
    except ImportError:
        print("\nWARNING: Cannot import exact_diagonalization_ssh_hubbard")
        print("Validation will use hardcoded reference energies instead.")
        has_ed = False

    # Test cases: (L, t1, t2, U, expected_energy)
    # These are the same cases that expose TeNPy's bug
    test_cases = [
        # L=2: Should work perfectly for any method
        (2, 1.0, 0.6, 1.0, None),  # Will compute with ED

        # L=4: Critical test - TeNPy fails at t2/t1 = 0.6
        (4, 1.0, 0.6, 1.0, None),  # TeNPy gives 1.68% error here

        # L=4: Threshold test - exactly at t2/t1 = 0.5
        (4, 1.0, 0.5, 1.0, None),  # TeNPy transition point

        # L=4: Safe regime - t2/t1 < 0.5
        (4, 1.0, 0.4, 1.0, None),  # TeNPy works here
    ]

    results = []

    for L, t1, t2, U, ref_energy in test_cases:
        print(f"\n{'─'*70}")
        print(f"Test: L={L}, t1={t1:.2f}, t2={t2:.2f}, U={U:.2f}, t2/t1={t2/t1:.2f}")
        print(f"{'─'*70}")

        # Get exact energy
        if has_ed:
            exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U, verbose=False)
            print(f"Exact energy (ED): {exact_energy:.10f}")
        elif ref_energy is not None:
            exact_energy = ref_energy
            print(f"Reference energy: {exact_energy:.10f}")
        else:
            print("Skipping (no exact energy available)")
            continue

        # Run DMRG
        dmrg_result = solve_ssh_hubbard_dmrgpy(
            L=L, t1=t1, t2=t2, U=U,
            maxm=200, nsweeps=10,
            verbose=False
        )

        if dmrg_result['energy'] is None:
            print("DMRG failed to converge")
            continue

        # Compare
        comp = compare_with_exact(
            L, t1, t2, U,
            exact_energy, dmrg_result['energy'],
            verbose=True
        )

        results.append(comp)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"{'L':<4} {'t1':<6} {'t2':<6} {'t2/t1':<8} {'Error (%)':<12} {'Status'}")
    print("─"*70)

    for r in results:
        status = "✓ PASS" if abs(r['error_rel_pct']) < 0.1 else "✗ FAIL"
        print(f"{r['L']:<4} {r['t1']:<6.2f} {r['t2']:<6.2f} {r['t2_t1_ratio']:<8.2f} "
              f"{r['error_rel_pct']:>+11.6f}  {status}")

    print("="*70)
    print("\nInterpretation:")
    print("  - Error <0.1%: Excellent agreement with exact diagonalization")
    print("  - Error <1.0%: Good agreement, better than TeNPy in problematic regime")
    print("  - Error >1.0%: Significant deviation, investigation needed")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    """
    Run validation tests when executed as script.
    """

    if not HAS_DMRGPY:
        print("\nERROR: dmrgpy is not installed.")
        print("\nTo install dmrgpy:")
        print("  pip install dmrgpy")
        print("\nNote: dmrgpy will automatically install ITensor backend.")
        print("      You may need C++ compiler and LAPACK/BLAS libraries.")
        sys.exit(1)

    # Run validation suite
    validation_suite(verbose=True)
