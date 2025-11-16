#!/usr/bin/env python3
"""
Test if the problem is with plus_hc=True.

Instead of using plus_hc=True, add both directions of hopping manually.
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
print("TEST: Manual hermitian conjugate (no plus_hc=True)")
print("=" * 80)

class ManualHCModel(CouplingMPOModel):
    """SSH-Hubbard WITHOUT plus_hc, adding h.c. manually."""

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        return FermionSite(conserve='N')

    def init_lattice(self, model_params):
        L_phys = model_params.get('L', 4)
        N_cells = L_phys // 2
        site = FermionSite(conserve='N')
        unit_cell = [site] * 4
        lat = Lattice([N_cells], unit_cell,
                     bc_MPS='finite',
                     bc='open',
                     basis=[[1.]],
                     positions=[[0.], [0.25], [0.5], [0.75]])
        return lat

    def init_terms(self, model_params):
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.6)
        U = model_params.get('U', 2.0)

        print("\nAdding hopping terms WITHOUT plus_hc:")

        # Intra-cell: A → B and B → A SEPARATELY
        print(f"  Intra-cell t1={t1}:")
        print(f"    add_coupling(-{t1}, 0, 'Cd', 2, 'C', plus_hc=False)  # A↑ → B↑")
        self.add_coupling(-t1, 0, 'Cd', 2, 'C', dx=[0], plus_hc=False)
        print(f"    add_coupling(-{t1}, 2, 'Cd', 0, 'C', plus_hc=False)  # B↑ → A↑ (h.c.)")
        self.add_coupling(-t1, 2, 'Cd', 0, 'C', dx=[0], plus_hc=False)

        print(f"    add_coupling(-{t1}, 1, 'Cd', 3, 'C', plus_hc=False)  # A↓ → B↓")
        self.add_coupling(-t1, 1, 'Cd', 3, 'C', dx=[0], plus_hc=False)
        print(f"    add_coupling(-{t1}, 3, 'Cd', 1, 'C', plus_hc=False)  # B↓ → A↓ (h.c.)")
        self.add_coupling(-t1, 3, 'Cd', 1, 'C', dx=[0], plus_hc=False)

        # Inter-cell: B → A and A → B SEPARATELY
        print(f"\n  Inter-cell t2={t2}:")
        print(f"    add_coupling(-{t2}, 2, 'Cd', 0, 'C', dx=[1], plus_hc=False)  # B↑ → A↑")
        self.add_coupling(-t2, 2, 'Cd', 0, 'C', dx=[1], plus_hc=False)
        print(f"    add_coupling(-{t2}, 0, 'Cd', 2, 'C', dx=[-1], plus_hc=False)  # A↑ → B↑ (h.c.)")
        self.add_coupling(-t2, 0, 'Cd', 2, 'C', dx=[-1], plus_hc=False)

        print(f"    add_coupling(-{t2}, 3, 'Cd', 1, 'C', dx=[1], plus_hc=False)  # B↓ → A↓")
        self.add_coupling(-t2, 3, 'Cd', 1, 'C', dx=[1], plus_hc=False)
        print(f"    add_coupling(-{t2}, 1, 'Cd', 3, 'C', dx=[-1], plus_hc=False)  # A↓ → B↓ (h.c.)")
        self.add_coupling(-t2, 1, 'Cd', 3, 'C', dx=[-1], plus_hc=False)

        # Hubbard U (no plus_hc needed for NN)
        print(f"\n  Hubbard U={U}:")
        self.add_coupling(U, 0, 'N', 1, 'N', dx=[0])
        self.add_coupling(U, 2, 'N', 3, 'N', dx=[0])


def run_manual_hc_dmrg(L=4, t1=1.0, t2=0.6, U=2.0, chi_max=100):
    params = {'L': L, 't1': t1, 't2': t2, 'U': U}
    model = ManualHCModel(params)

    # Initial state
    L_cells = L // 2
    product_state = []
    for cell in range(L_cells):
        product_state.extend(['full', 'full', 'empty', 'empty'])

    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc='finite')

    # DMRG
    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10},
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        'verbose': 0,
    }

    info = dmrg.run(psi, model, dmrg_params)
    return info['E']

# Test
L = 4
t1 = 1.0
t2 = 0.6
U = 2.0

print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")

# Exact
H_vqe = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
E_exact = np.linalg.eigvalsh(H_vqe.to_matrix())[0]

print(f"\nExact energy: {E_exact:.10f}")

# Test with manual h.c.
E_manual = run_manual_hc_dmrg(L, t1, t2, U, chi_max=100)
err_manual = abs((E_manual - E_exact) / E_exact) * 100

print(f"\nDMRG (manual h.c.):  {E_manual:.10f}")
print(f"  Error: {err_manual:.4f}%")

# Compare with original
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard
E_orig = run_dmrg_ssh_hubbard(L, t1, t2, U, chi_max=100, verbose=False)['energy']
err_orig = abs((E_orig - E_exact) / E_exact) * 100

print(f"\nDMRG (plus_hc=True): {E_orig:.10f}")
print(f"  Error: {err_orig:.4f}%")

print("\n" + "=" * 80)
if err_manual < 0.01:
    print("✓✓✓ FIXED! Manual h.c. gives correct result!")
    print("    → The bug is in TeNPy's plus_hc=True implementation")
elif err_manual < err_orig / 2:
    print(f"✓ IMPROVEMENT from {err_orig:.2f}% to {err_manual:.2f}%")
else:
    print(f"✗ No improvement. Error still {err_manual:.2f}%")
print("=" * 80)
