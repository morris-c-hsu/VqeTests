#!/usr/bin/env python3
"""
Simplified TeNPy DMRG for SSH-Hubbard using single-site unit cells.

Uses the simplest possible lattice structure and adds terms explicitly.
"""

import numpy as np
import warnings

try:
    import tenpy
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import FermionSite
    from tenpy.models.lattice import Chain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    HAS_TENPY = True
except ImportError:
    HAS_TENPY = False

warnings.filterwarnings('ignore')


class SimpleSSHHubbard(CouplingMPOModel):
    """
    Simplified spinful SSH-Hubbard model using single-site unit cells.

    Each MPS site is a spinless fermion. Sites are ordered:
    [0↑, 0↓, 1↑, 1↓, 2↑, 2↓, ...]

    Uses simple Chain lattice with explicit term addition.
    """

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        site = FermionSite(conserve='N')
        return site

    def init_lattice(self, model_params):
        L_phys = model_params.get('L', 4)
        N_mps = 2 * L_phys
        bc_MPS = model_params.get('bc_MPS', 'finite')

        site = self.init_sites(model_params)

        # Simple 1D chain with single-site unit cells
        lat = Chain(L=N_mps, site=site, bc_MPS=bc_MPS, bc='open')

        return lat

    def init_terms(self, model_params):
        L_phys = model_params.get('L', 4)
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.6)
        U = model_params.get('U', 2.0)

        print(f"  Building simplified SSH-Hubbard Hamiltonian:")
        print(f"    Physical sites:      {L_phys}")
        print(f"    MPS sites:           {2*L_phys}")
        print(f"    Strong hopping (t1): {t1:.3f}")
        print(f"    Weak hopping (t2):   {t2:.3f}")
        print(f"    Interaction (U):     {U:.3f}")

        # Add hopping terms explicitly for each bond
        # With single-site unit cells, u1=u2=0 and dx is the cell displacement
        for i_phys in range(L_phys - 1):
            # Determine hopping for this physical bond
            t_hop = t1 if i_phys % 2 == 0 else t2

            # MPS site indices (interleaved spins)
            up_i = 2 * i_phys
            dn_i = 2 * i_phys + 1
            up_j = 2 * (i_phys + 1)
            dn_j = 2 * (i_phys + 1) + 1

            # Add hopping: -t * (c† c + h.c.)
            # Spin-up hopping: from cell up_i to cell up_j (dx = up_j - up_i)
            self.add_coupling(-t_hop, 0, 'Cd', 0, 'C', dx=up_j - up_i, plus_hc=True)

            # Spin-down hopping: from cell dn_i to cell dn_j (dx = dn_j - dn_i)
            self.add_coupling(-t_hop, 0, 'Cd', 0, 'C', dx=dn_j - dn_i, plus_hc=True)

        # Add Hubbard interaction at each physical site
        for i_phys in range(L_phys):
            up_site = 2 * i_phys
            dn_site = 2 * i_phys + 1

            # U * n_up * n_down
            # Couples cells up_site and dn_site (dx = dn_site - up_site = 1)
            self.add_coupling(U, 0, 'N', 0, 'N', dx=dn_site - up_site)


def run_simple_dmrg(L=4, t1=1.0, t2=0.6, U=2.0, chi_max=100):
    """Run DMRG with simplified model."""
    if not HAS_TENPY:
        print("ERROR: TeNPy not available")
        return None

    print(f"\nSystem: L={L} physical sites ({2*L} MPS sites)")

    model_params = {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        'bc_MPS': 'finite',
    }

    # Build model
    model = SimpleSSHHubbard(model_params)

    # Initial state: half-filling
    N_mps = 2 * L
    product_state = ['full' if i < L else 'empty' for i in range(N_mps)]

    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc='finite')

    # DMRG parameters
    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10},
        'max_E_err': 1.e-10,
        'max_sweeps': 30,
        'mixer': True,
        'verbose': 0,
    }

    # Run DMRG
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E_dmrg, psi = eng.run()

    print(f"  DMRG Energy: {E_dmrg:.12f}")

    return {'energy': E_dmrg, 'psi': psi}


if __name__ == "__main__":
    from ssh_hubbard_vqe import ssh_hubbard_hamiltonian

    print("\n" + "#" * 80)
    print("# SIMPLIFIED DMRG TEST")
    print("#" * 80)

    # Test L=4
    print("\n" + "=" * 80)
    print("TEST: L=4, t1=1.0, t2=0.6, U=2.0")
    print("=" * 80)

    # Get exact energy
    H = ssh_hubbard_hamiltonian(4, 1.0, 0.6, 2.0, periodic=False)
    E_exact = np.linalg.eigh(H.to_matrix())[0][0]
    print(f"\nExact energy: {E_exact:.12f}")

    # Run simplified DMRG
    result = run_simple_dmrg(L=4, t1=1.0, t2=0.6, U=2.0, chi_max=200)

    if result:
        E_dmrg = result['energy']
        error = abs(E_dmrg - E_exact)
        rel_error = 100 * error / abs(E_exact)

        print(f"\nComparison:")
        print(f"  Exact:       {E_exact:.12f}")
        print(f"  DMRG:        {E_dmrg:.12f}")
        print(f"  Error:       {error:.6e} ({rel_error:.4f}%)")

        if rel_error < 0.1:
            print(f"  ✓✓✓ Excellent agreement!")
        elif rel_error < 1.0:
            print(f"  ✓✓ Very good!")
        else:
            print(f"  ⚠ Needs investigation")

    print()
