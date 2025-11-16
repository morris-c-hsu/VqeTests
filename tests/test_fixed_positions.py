#!/usr/bin/env python3
"""
Test DMRG with CORRECTED position parameters.

The unit cell has 4 sites: [A↑, A↓, B↑, B↓]
But A↑ and A↓ are at the SAME spatial location (sublattice A)
And B↑ and B↓ are at the SAME spatial location (sublattice B)

So positions should be [[0.], [0.], [d], [d]] where d is the dimer bond length.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Lattice
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian

print("=" * 80)
print("TEST: CORRECTED POSITION PARAMETERS")
print("=" * 80)

class FixedSSHHubbard(CouplingMPOModel):
    """SSH-Hubbard with CORRECT positions."""

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        return FermionSite(conserve='N')

    def init_lattice(self, model_params):
        L_phys = model_params.get('L', 4)
        N_cells = L_phys // 2
        site = FermionSite(conserve='N')
        unit_cell = [site] * 4

        # CORRECTED: A↑ and A↓ at same position, B↑ and B↓ at same position
        # Using d=1.0 as the intra-dimer distance
        positions_corrected = [[0.0], [0.0], [1.0], [1.0]]

        lat = Lattice([N_cells], unit_cell,
                     bc_MPS='finite',
                     bc='open',
                     basis=[[2.0]],  # Unit cell length (includes both A and B, plus gap to next cell)
                     positions=positions_corrected)
        return lat

    def init_terms(self, model_params):
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.6)
        U = model_params.get('U', 2.0)

        # Hopping terms (same as before)
        self.add_coupling(-t1, 0, 'Cd', 2, 'C', dx=[0], plus_hc=True)  # A↑ → B↑
        self.add_coupling(-t1, 1, 'Cd', 3, 'C', dx=[0], plus_hc=True)  # A↓ → B↓

        self.add_coupling(-t2, 2, 'Cd', 0, 'C', dx=[1], plus_hc=True)  # B↑ → A↑
        self.add_coupling(-t2, 3, 'Cd', 1, 'C', dx=[1], plus_hc=True)  # B↓ → A↓

        # Hubbard U
        self.add_coupling(U, 0, 'N', 1, 'N', dx=[0])  # n_A↑ n_A↓
        self.add_coupling(U, 2, 'N', 3, 'N', dx=[0])  # n_B↑ n_B↓


def run_dmrg_fixed(L=4, t1=1.0, t2=0.6, U=2.0, chi_max=100):
    """Run DMRG with corrected positions."""
    params = {'L': L, 't1': t1, 't2': t2, 'U': U}
    model = FixedSSHHubbard(params)

    # Initial state
    product_state = ['empty'] * model.lat.N_sites
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc='finite')

    # DMRG
    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10},
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'verbose': 0,
    }

    info = dmrg.run(psi, model, dmrg_params)
    E0 = info['E']
    return E0

# Test
L = 4
t1 = 1.0
t2 = 0.6
U = 2.0

print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")

# Exact
H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
E_exact = np.linalg.eigvalsh(H_vqe.to_matrix())[0]

print(f"\nExact energy (VQE): {E_exact:.10f}")

# Original (wrong positions)
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard
E_orig = run_dmrg_ssh_hubbard(L=L, t1=t1, t2=t2, U=U, chi_max=100, verbose=False)['energy']
err_orig = abs((E_orig - E_exact) / E_exact) * 100

print(f"\nDMRG (original positions):  {E_orig:.10f}")
print(f"  Error: {err_orig:.4f}%")

# Fixed positions
print("\nTesting with CORRECTED positions [[0,0,1,1]]...")
E_fixed = run_dmrg_fixed(L=L, t1=t1, t2=t2, U=U, chi_max=100)
err_fixed = abs((E_fixed - E_exact) / E_exact) * 100

print(f"\nDMRG (corrected positions): {E_fixed:.10f}")
print(f"  Error: {err_fixed:.4f}%")

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if err_fixed < 0.01:
    print("\n✓✓✓ FIX WORKS! Corrected positions give exact result!")
    print("    → The bug was in the position parameter")
    print("    → TeNPy was misinterpreting the lattice geometry")
elif err_fixed < err_orig / 2:
    print(f"\n✓ IMPROVEMENT! Error reduced from {err_orig:.3f}% to {err_fixed:.3f}%")
    print("  → Positions matter, but may not be the only issue")
else:
    print(f"\n✗ NO IMPROVEMENT. Error still {err_fixed:.3f}%")
    print("  → Positions are not the issue")

print("\n" + "=" * 80)
