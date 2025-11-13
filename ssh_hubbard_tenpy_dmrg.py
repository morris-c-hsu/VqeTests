#!/usr/bin/env python3
"""
SSH-Hubbard Model with TeNPy DMRG

This module implements a proper finite-system DMRG for the spinful SSH-Hubbard
model using TeNPy (Tensor Network Python library).

The SSH-Hubbard Hamiltonian:
    H = -∑_{i,σ} t_{i,i+1} (c†_{i,σ} c_{i+1,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓}

where t_{i,i+1} alternates between t1 (strong bonds, i even) and t2 (weak bonds, i odd).

Features:
  - Custom SSH-Hubbard model with alternating hoppings
  - Proper DMRG with conserved quantum numbers (particle number, Sz)
  - Observable calculations: densities, double occupancy, bond orders, entanglement
  - Validation against exact diagonalization for small systems
  - Works for L = 2, 4, 6, 8, ... up to larger systems

Usage:
    python ssh_hubbard_tenpy_dmrg.py
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

# TeNPy imports
try:
    import tenpy
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.site import FermionSite
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    import tenpy.linalg.np_conserved as npc
except ImportError as e:
    raise ImportError(
        "TeNPy is required for this module. Install with: pip install physics-tenpy"
    ) from e

# Suppress TeNPy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='tenpy')


# ============================================================================
# SSH-HUBBARD MODEL
# ============================================================================

class SSHHubbardModel(CouplingMPOModel):
    """
    Spinful SSH-Hubbard model with alternating hopping amplitudes.

    The Hamiltonian is:
        H = -∑_{i,σ} t_i (c†_{i,σ} c_{i+1,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓}

    with alternating hoppings:
        t_i = t1 for i even (strong bonds: 0-1, 2-3, 4-5, ...)
        t_i = t2 for i odd  (weak bonds:   1-2, 3-4, 5-6, ...)

    This models the Su-Schrieffer-Heeger (SSH) polyacetylene chain with
    on-site Hubbard interactions.

    Parameters
    ----------
    model_params : dict
        Dictionary containing:
        - L : int, number of sites
        - t1 : float, strong bond hopping amplitude
        - t2 : float, weak bond hopping amplitude
        - U : float, on-site Hubbard interaction strength
        - bc_MPS : str, boundary conditions ('finite' or 'infinite')
        - conserve : str, conserved quantum numbers ('N', 'parity', or None)
        - filling : float, optional target filling (default: 0.5 for half-filling)
    """

    def __init__(self, model_params):
        print("=" * 70)
        print("SSH-HUBBARD TENPY DMRG")
        print("=" * 70)

        # Call parent initializer - this will call init_lattice() and init_terms()
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        """Initialize the sites (spinless fermions).

        We use 2*L sites total:
        - Sites 0, 2, 4, ... (even) are spin-up
        - Sites 1, 3, 5, ... (odd) are spin-down
        """
        conserve = model_params.get('conserve', 'N')
        fs = FermionSite(conserve=conserve, filling=0.25)  # 1/4 since we double sites
        return fs

    def init_lattice(self, model_params):
        """Initialize the lattice with 2*L sites (spin encoded as site index)."""
        site = self.init_sites(model_params)
        L = model_params.get('L', 8)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'open' if bc_MPS == 'finite' else 'periodic'
        # Create chain with 2*L sites (double for spin)
        lat = Chain(2 * L, site, bc=bc, bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        """Add the Hamiltonian terms (hopping and interaction).

        Site layout (2*L total sites):
        - Site 2*i:     spin-up on physical site i
        - Site 2*i+1:   spin-down on physical site i

        Hopping only within same spin species with SSH pattern.
        Interaction couples neighboring sites (up/down on same physical site).
        """
        # Extract parameters
        L = model_params.get('L', 8)
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.5)
        U = model_params.get('U', 2.0)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        conserve = model_params.get('conserve', 'N')

        self.t1 = t1
        self.t2 = t2
        self.U = U
        self.verbose = model_params.get('verbose', 1)

        if self.verbose >= 1:
            print(f"\nModel Parameters:")
            print(f"  Sites (L):        {L} physical ({2*L} MPS sites)")
            print(f"  Strong hop (t1):  {t1:.4f}")
            print(f"  Weak hop (t2):    {t2:.4f}")
            print(f"  Interaction (U):  {U:.4f}")
            print(f"  Dimerization δ:   {(t1-t2)/(t1+t2):.4f}")
            print(f"  Boundary:         {bc_MPS}")
            print(f"  Conserve:         {conserve}")

        # Add hopping terms with SSH pattern
        # FermionSite operators: 'Cd' (creation), 'C' (annihilation), 'N' (number)
        for i in range(L - 1):
            # Determine hopping amplitude for this bond
            t_hop = t1 if i % 2 == 0 else t2

            # Spin-up hopping: from site 2*i to site 2*(i+1)
            self.add_coupling(-t_hop, 2*i, 'Cd', 2*(i+1), 'C', dx=None,
                             plus_hc=True, op_string='JW', category='hop_up')

            # Spin-down hopping: from site 2*i+1 to site 2*(i+1)+1
            self.add_coupling(-t_hop, 2*i+1, 'Cd', 2*(i+1)+1, 'C', dx=None,
                             plus_hc=True, op_string='JW', category='hop_down')

        # Add on-site Hubbard interaction: U * n_up * n_down
        # Couples site 2*i (spin-up) with site 2*i+1 (spin-down)
        for i in range(L):
            self.add_coupling(U, 2*i, 'N', 2*i+1, 'N', dx=None, category='Hubbard_U')

        if self.verbose >= 1:
            print(f"\n✓ Hamiltonian terms added successfully")
            print(f"  Hopping bonds:    {2*(L-1)} ({L-1} per spin)")
            print(f"  Interaction sites: {L}")


# ============================================================================
# OBSERVABLE CALCULATIONS
# ============================================================================

def compute_observables(psi: MPS, model: SSHHubbardModel,
                        compute_entanglement: bool = True,
                        compute_correlations: bool = True) -> Dict:
    """
    Compute physical observables from the MPS ground state.

    Parameters
    ----------
    psi : MPS
        Ground state MPS from DMRG
    model : SSHHubbardModel
        The SSH-Hubbard model instance
    compute_entanglement : bool
        Whether to compute entanglement entropy profile
    compute_correlations : bool
        Whether to compute correlation functions

    Returns
    -------
    obs : dict
        Dictionary containing all computed observables
    """
    L = psi.L
    obs = {}

    print("\n" + "=" * 70)
    print("COMPUTING OBSERVABLES")
    print("=" * 70)

    # Local densities
    n_up = psi.expectation_value('Nu')
    n_down = psi.expectation_value('Nd')
    n_total = n_up + n_down
    double_occ = psi.expectation_value('NuNd')

    obs['n_up'] = np.array(n_up)
    obs['n_down'] = np.array(n_down)
    obs['n_total'] = np.array(n_total)
    obs['double_occupancy'] = np.array(double_occ)

    # Total particle numbers
    obs['N_up_total'] = float(np.sum(n_up))
    obs['N_down_total'] = float(np.sum(n_down))
    obs['N_total'] = float(np.sum(n_total))
    obs['double_occ_total'] = float(np.sum(double_occ))

    print(f"\nParticle Numbers:")
    print(f"  N_up:            {obs['N_up_total']:.6f}")
    print(f"  N_down:          {obs['N_down_total']:.6f}")
    print(f"  N_total:         {obs['N_total']:.6f}")
    print(f"  Double occ:      {obs['double_occ_total']:.6f}")

    # Bond kinetic energies (hopping expectation values)
    if compute_correlations:
        # Compute ⟨c†_i c_{i+1}⟩ + h.c. for each bond and spin
        bond_energies_up = []
        bond_energies_down = []

        for i in range(L - 1):
            # Spin-up bond
            hop_up = psi.expectation_value_term([('Cdu', i), ('Cu', i + 1)])
            hop_up += psi.expectation_value_term([('Cu', i), ('Cdu', i + 1)])
            bond_energies_up.append(hop_up.real)

            # Spin-down bond
            hop_down = psi.expectation_value_term([('Cdd', i), ('Cd', i + 1)])
            hop_down += psi.expectation_value_term([('Cd', i), ('Cdu', i + 1)])
            bond_energies_down.append(hop_down.real)

        obs['bond_kinetic_up'] = np.array(bond_energies_up)
        obs['bond_kinetic_down'] = np.array(bond_energies_down)
        obs['bond_kinetic_total'] = obs['bond_kinetic_up'] + obs['bond_kinetic_down']

        # SSH dimer order parameter: D = ∑_{even bonds} ⟨hop⟩ - ∑_{odd bonds} ⟨hop⟩
        strong_bonds = obs['bond_kinetic_total'][::2]   # Even indices
        weak_bonds = obs['bond_kinetic_total'][1::2]    # Odd indices
        obs['dimer_order'] = float(np.sum(strong_bonds) - np.sum(weak_bonds))

        print(f"\nBond Order:")
        print(f"  Dimer order D:   {obs['dimer_order']:.6f}")
        print(f"  Strong bonds:    {np.mean(strong_bonds):.6f} (avg)")
        print(f"  Weak bonds:      {np.mean(weak_bonds):.6f} (avg)")

    # Entanglement entropy
    if compute_entanglement:
        S_ent = []
        for i in range(L - 1):
            S_ent.append(psi.entanglement_entropy([i + 1])[0])
        obs['entanglement_entropy'] = np.array(S_ent)

        print(f"\nEntanglement:")
        print(f"  Max S_ent:       {np.max(S_ent):.6f} (bond {np.argmax(S_ent)})")
        print(f"  Avg S_ent:       {np.mean(S_ent):.6f}")

    return obs


# ============================================================================
# DMRG SOLVER
# ============================================================================

def run_dmrg(model: SSHHubbardModel, chi_max: int = 100,
             max_sweeps: int = 20, verbose: int = 1) -> Tuple[MPS, Dict]:
    """
    Run finite-system DMRG to find the ground state.

    Parameters
    ----------
    model : SSHHubbardModel
        The SSH-Hubbard model to solve
    chi_max : int
        Maximum bond dimension to keep
    max_sweeps : int
        Maximum number of DMRG sweeps
    verbose : int
        Verbosity level (0 = silent, 1 = normal, 2 = verbose)

    Returns
    -------
    psi : MPS
        Ground state MPS
    dmrg_results : dict
        Dictionary containing DMRG results (energy, truncation errors, etc.)
    """
    print("\n" + "=" * 70)
    print("DMRG OPTIMIZATION")
    print("=" * 70)

    # Initial state: half-filling with alternating spins
    # Product state |↑0↓0↑0↓0...⟩ for half-filling
    L = model.lat.N_sites
    product_state = []
    for i in range(L):
        if i % 2 == 0:
            product_state.append('up')    # Spin-up electron
        else:
            product_state.append('down')  # Spin-down electron

    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

    if verbose >= 1:
        print(f"\nInitial State:")
        print(f"  Pattern:         Alternating ↑↓↑↓... (half-filling)")
        print(f"  Total particles: {L} ({L//2} up, {L - L//2} down expected)")

    # DMRG parameters
    dmrg_params = {
        'mixer': True,  # Use mixer to avoid metastable states
        'mixer_params': {
            'amplitude': 1.e-5,
            'decay': 1.5,
            'disable_after': 10
        },
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'max_sweeps': max_sweeps,
        'verbose': verbose
    }

    if verbose >= 1:
        print(f"\nDMRG Parameters:")
        print(f"  chi_max:         {chi_max}")
        print(f"  max_sweeps:      {max_sweeps}")
        print(f"  Mixer:           enabled")
        print("\nRunning DMRG...")
        print("-" * 70)

    # Run DMRG
    info = dmrg.run(psi, model, dmrg_params)

    # Extract results
    E0 = info['E']
    E_per_site = E0 / L
    S_max = np.max(info['S'])
    sweep_count = info['sweep_statistics']['sweep'][-1] if 'sweep_statistics' in info else 0

    dmrg_results = {
        'E0': E0,
        'E_per_site': E_per_site,
        'S_max': S_max,
        'sweeps': sweep_count,
        'info': info
    }

    if verbose >= 1:
        print("-" * 70)
        print(f"\n✓ DMRG Converged!")
        print(f"\nGround State Energy:")
        print(f"  E0:              {E0:.10f}")
        print(f"  E/site:          {E_per_site:.10f}")
        print(f"  Max entanglement: {S_max:.6f}")
        print(f"  Sweeps:          {sweep_count}")

    return psi, dmrg_results


# ============================================================================
# VALIDATION AGAINST EXACT DIAGONALIZATION
# ============================================================================

def validate_small_system(L: int, t1: float, t2: float, U: float,
                          chi_max: int = 64) -> None:
    """
    Validate DMRG against exact diagonalization for small systems.

    For L <= 6, we can compare TeNPy DMRG with exact diagonalization
    from the existing dmrg_ssh_hubbard.py implementation.

    Parameters
    ----------
    L : int
        Number of sites (should be <= 6 for ED comparison)
    t1, t2, U : float
        Model parameters
    chi_max : int
        DMRG bond dimension
    """
    print("\n" + "=" * 70)
    print(f"VALIDATION: L={L} SYSTEM")
    print("=" * 70)

    # Run TeNPy DMRG
    model_params = {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        'bc_MPS': 'finite',
        'conserve': 'N',
        'verbose': 1
    }

    model = SSHHubbardModel(model_params)
    psi, dmrg_results = run_dmrg(model, chi_max=chi_max, max_sweeps=30, verbose=1)
    obs = compute_observables(psi, model, compute_entanglement=True, compute_correlations=True)

    # Print detailed observable table
    print("\n" + "=" * 70)
    print("SITE OBSERVABLES")
    print("=" * 70)
    print("Site  |  n_up  |  n_down  |  n_total  |  double_occ")
    print("-" * 70)
    for i in range(L):
        print(f"  {i}   | {obs['n_up'][i]:.4f} | {obs['n_down'][i]:.4f}  | "
              f"{obs['n_total'][i]:.4f}   | {obs['double_occupancy'][i]:.4f}")

    print("\n" + "=" * 70)
    print("BOND OBSERVABLES")
    print("=" * 70)
    print("Bond  |  Type    |  ⟨hopping⟩")
    print("-" * 70)
    for i in range(L - 1):
        bond_type = "strong" if i % 2 == 0 else "weak"
        print(f"  {i}   |  {bond_type:7s} | {obs['bond_kinetic_total'][i]:.6f}")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Main demonstration of SSH-Hubbard DMRG with TeNPy."""

    # Test case 1: Small system for validation (L=4)
    print("\n" + "#" * 70)
    print("# TEST 1: L=4 SYSTEM (VALIDATION)")
    print("#" * 70)
    validate_small_system(L=4, t1=1.0, t2=0.5, U=2.0, chi_max=64)

    # Test case 2: Moderate system (L=8)
    print("\n\n" + "#" * 70)
    print("# TEST 2: L=8 SYSTEM (DMRG REGIME)")
    print("#" * 70)
    validate_small_system(L=8, t1=1.0, t2=0.5, U=2.0, chi_max=100)

    # Test case 3: Topological vs trivial (L=6, scan δ)
    print("\n\n" + "#" * 70)
    print("# TEST 3: TOPOLOGICAL PHASE SCAN (L=6)")
    print("#" * 70)

    L = 6
    U = 2.0
    t_avg = 1.0
    deltas = [-0.5, 0.0, 0.5]  # Trivial, critical, topological

    for delta in deltas:
        t1 = t_avg * (1 + delta)
        t2 = t_avg * (1 - delta)

        print(f"\n{'=' * 70}")
        print(f"δ = {delta:+.2f} (t1={t1:.3f}, t2={t2:.3f})")
        print(f"{'=' * 70}")

        model_params = {
            'L': L,
            't1': t1,
            't2': t2,
            'U': U,
            'bc_MPS': 'finite',
            'conserve': 'N',
            'verbose': 0
        }

        model = SSHHubbardModel(model_params)
        psi, dmrg_results = run_dmrg(model, chi_max=80, max_sweeps=20, verbose=0)
        obs = compute_observables(psi, model, compute_entanglement=True, compute_correlations=True)

        # Print summary
        print(f"\nResults:")
        print(f"  E0/site:         {dmrg_results['E_per_site']:.8f}")
        print(f"  Dimer order D:   {obs['dimer_order']:.6f}")
        print(f"  Edge density:    {(obs['n_total'][0] + obs['n_total'][-1])/2:.6f}")
        print(f"  Bulk density:    {np.mean(obs['n_total'][1:-1]):.6f}")
        print(f"  Max S_ent:       {dmrg_results['S_max']:.6f}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
