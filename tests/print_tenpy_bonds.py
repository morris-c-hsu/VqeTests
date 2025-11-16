#!/usr/bin/env python3
"""
Print all bond terms that TeNPy is actually adding to the Hamiltonian.

This will help us see if there's any double-counting or unexpected terms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Lattice

print("=" * 80)
print("TENPY BOND TERMS INSPECTION")
print("=" * 80)

L = 4
t1 = 1.0
t2 = 0.6
U = 2.0

print(f"\nParameters: L={L}, t1={t1}, t2={t2}, U={U}")

class InspectedModel(CouplingMPOModel):
    """Model that prints what it's doing."""

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
        L_phys = model_params.get('L', 4)
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.6)
        U = model_params.get('U', 2.0)

        print("\n" + "=" * 80)
        print("ADDING COUPLINGS")
        print("=" * 80)

        # Intra-cell
        print(f"\n1. Intra-cell hopping (t1={t1}):")
        print(f"   add_coupling(-{t1}, u0=0, 'Cd', u1=2, 'C', dx=[0], plus_hc=True)")
        self.add_coupling(-t1, 0, 'Cd', 2, 'C', dx=[0], plus_hc=True)

        print(f"   add_coupling(-{t1}, u0=1, 'Cd', u1=3, 'C', dx=[0], plus_hc=True)")
        self.add_coupling(-t1, 1, 'Cd', 3, 'C', dx=[0], plus_hc=True)

        # Inter-cell
        print(f"\n2. Inter-cell hopping (t2={t2}):")
        print(f"   add_coupling(-{t2}, u0=2, 'Cd', u1=0, 'C', dx=[1], plus_hc=True)")
        self.add_coupling(-t2, 2, 'Cd', 0, 'C', dx=[1], plus_hc=True)

        print(f"   add_coupling(-{t2}, u0=3, 'Cd', u1=1, 'C', dx=[1], plus_hc=True)")
        self.add_coupling(-t2, 3, 'Cd', 1, 'C', dx=[1], plus_hc=True)

        # Hubbard
        print(f"\n3. Hubbard interaction (U={U}):")
        print(f"   add_coupling({U}, u0=0, 'N', u1=1, 'N', dx=[0])")
        self.add_coupling(U, 0, 'N', 1, 'N', dx=[0])

        print(f"   add_coupling({U}, u0=2, 'N', u1=3, 'N', dx=[0])")
        self.add_coupling(U, 2, 'N', 3, 'N', dx=[0])

# Build model
params = {'L': L, 't1': t1, 't2': t2, 'U': U}
model = InspectedModel(params)

# Access the internal coupling terms
print("\n" + "=" * 80)
print("ACTUAL TERMS IN MODEL (from model.coupling_terms)")
print("=" * 80)

print(f"\nModel has {model.lat.N_sites} MPS sites")
print(f"Unit cells: {model.lat.N_cells}")
print(f"Sites per unit cell: {len(model.lat.unit_cell)}")

# Print all coupling terms
print("\nAll coupling terms:")
for key, terms in model.coupling_terms.items():
    print(f"\n  Coupling type: {key}")
    for i_bond, term_list in enumerate(terms):
        if term_list:  # Only print if not empty
            print(f"    Bond {i_bond}: {len(term_list)} terms")
            for term in term_list:
                print(f"      {term}")

# Check for on-site terms
print("\nOn-site terms:")
for i, terms in enumerate(model.onsite_terms):
    if terms:
        print(f"  Site {i}: {len(terms)} terms")
        for term in terms:
            print(f"    {term}")

# Try to access the H_bond
print("\n" + "=" * 80)
print("BOND HAMILTONIANS")
print("=" * 80)

if hasattr(model, 'H_bond'):
    print(f"\nNumber of bond Hamiltonians: {len(model.H_bond)}")
    for i, H_b in enumerate(model.H_bond):
        if H_b is not None:
            print(f"  Bond {i}: shape {H_b.shape}, norm {H_b.norm():.6f}")
else:
    print("  (H_bond not available in CouplingMPOModel)")

print("\n" + "=" * 80)
print("MPS SITE MAPPING")
print("=" * 80)

N_cells = L // 2
print(f"\n{N_cells} unit cells with 4 sites each:")
for cell_idx in range(N_cells):
    print(f"\n  Cell {cell_idx}:")
    for u in range(4):
        mps_idx = cell_idx * 4 + u
        sublat = 'A' if u in [0, 1] else 'B'
        spin = '↑' if u in [0, 2] else '↓'
        phys_site = cell_idx * 2 + (0 if u in [0, 1] else 1)
        print(f"    MPS[{mps_idx}] = {sublat}{cell_idx}{spin} = physical site {phys_site}, spin {spin}")

print("\n" + "=" * 80)
print("EXPECTED BONDS FOR SSH")
print("=" * 80)

print(f"\nFor L={L}:")
print("  Expected physical bonds:")
for i in range(L-1):
    t = t1 if i % 2 == 0 else t2
    print(f"    Bond {i}: site {i} ↔ site {i+1} (t={t})")

print("\n  Mapping to MPS bonds:")
print("    Intra-cell (t1):")
print("      Cell 0: MPS 0↔2, MPS 1↔3  (site 0↔1)")
print("      Cell 1: MPS 4↔6, MPS 5↔7  (site 2↔3)")
print("    Inter-cell (t2):")
print("      Between cells 0&1: MPS 2↔4, MPS 3↔5  (site 1↔2)")

print("\n" + "=" * 80)
