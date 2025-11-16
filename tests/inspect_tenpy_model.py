#!/usr/bin/env python3
"""
Inspect the TeNPy model construction in detail.

This script creates the TeNPy model and prints:
1. Unit cell structure and site ordering
2. All coupling terms being added
3. Lattice connectivity
4. Expected vs actual bond pattern
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Lattice

print("=" * 80)
print("TENPY MODEL INSPECTION")
print("=" * 80)

# Create a simple SSH-Hubbard model
L = 4
t1 = 1.0
t2 = 0.6
U = 2.0

print(f"\nParameters: L={L}, t1={t1:.3f}, t2={t2:.3f}, U={U:.3f}")
print(f"Expected SSH pattern: (0)--t1--(1)--t2--(2)--t1--(3)")
print(f"  Strong bonds (t1=1.0): 0-1, 2-3")
print(f"  Weak bonds   (t2=0.6): 1-2")

# Build unit cell
print("\n" + "=" * 80)
print("UNIT CELL CONSTRUCTION")
print("=" * 80)

fermion_site = FermionSite(conserve=None)
unit_cell = [fermion_site] * 4  # [A_up, A_dn, B_up, B_dn]
N_cells = L // 2

print(f"\nUnit cell structure:")
print(f"  Sites per cell: 4 [A↑, A↓, B↑, B↓]")
print(f"  Number of cells: {N_cells}")
print(f"  Total MPS sites: {4 * N_cells}")

# Create lattice
bc = 'open'
lat = Lattice([N_cells], unit_cell, bc=bc)

print(f"\nLattice properties:")
print(f"  Shape: {lat.shape}")
print(f"  N_sites: {lat.N_sites}")
print(f"  Boundary conditions: {bc}")

# Print MPS site to physical site mapping
print(f"\nMPS site to physical site mapping:")
print(f"  (Assuming A and B within each cell form a dimer)")
print()
print(f"  MPS site | Cell | In-cell pos | Spin | Physical meaning")
print(f"  ---------|------|-------------|------|------------------")
for mps_idx in range(lat.N_sites):
    cell_idx = mps_idx // 4
    in_cell_pos = mps_idx % 4

    if in_cell_pos == 0:
        sublattice = 'A'
        spin = '↑'
    elif in_cell_pos == 1:
        sublattice = 'A'
        spin = '↓'
    elif in_cell_pos == 2:
        sublattice = 'B'
        spin = '↑'
    else:
        sublattice = 'B'
        spin = '↓'

    phys_site = cell_idx * 2 + (0 if sublattice == 'A' else 1)
    print(f"  {mps_idx:8d} | {cell_idx:4d} | {in_cell_pos:11d} | {spin:4s} | site {phys_site}, {sublattice} sublattice")

# Now examine what couplings are added
print("\n" + "=" * 80)
print("EXPECTED SSH BOND PATTERN")
print("=" * 80)

print(f"\nFor L={L} SSH chain:")
print(f"  Physical sites: 0, 1, 2, 3")
print(f"  Expected bonds:")
for i in range(L - 1):
    t = t1 if i % 2 == 0 else t2
    bond_type = "STRONG" if i % 2 == 0 else "weak  "
    print(f"    Bond {i}: {i} ↔ {i+1}  (t={t:.1f}, {bond_type})")

print(f"\n  In terms of dimers:")
print(f"    Dimer 0: sites (0, 1)  [A0, B0]")
print(f"    Dimer 1: sites (2, 3)  [A1, B1]")
print(f"  Strong bonds (t1): within dimers  (0-1, 2-3)")
print(f"  Weak bonds   (t2): between dimers (1-2)")

# Now let's trace what the model actually does
print("\n" + "=" * 80)
print("TRACING MODEL CONSTRUCTION")
print("=" * 80)

# I'll create a minimal version of the model to trace what's happening
from tenpy.models.model import CouplingMPOModel

class TracedSSHHubbard(CouplingMPOModel):
    def __init__(self, model_params):
        print("\nInitializing model...")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        """Initialize sites."""
        return FermionSite(conserve='N')

    def init_lattice(self, model_params):
        """Initialize lattice."""
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
        """Add coupling terms with detailed tracing."""
        L_phys = model_params.get('L', 4)
        t1 = model_params.get('t1', 1.0)
        t2 = model_params.get('t2', 0.5)
        U = model_params.get('U', 2.0)

        print(f"\nAdding hopping terms:")
        print(f"  (u0, u1) are unit cell indices")
        print(f"  dx is offset in unit cells")
        print()

        # Intra-cell hopping (A to B within same dimer)
        print(f"  Intra-cell hopping (t1={t1:.3f}, strong):")
        print(f"    add_coupling(-{t1}, u0=0, op1='Cd', u1=2, op2='C', dx=[0], plus_hc=True)")
        print(f"    → This couples: (cell i, sublattice A, spin σ) to (cell i, sublattice B, spin σ)")
        print(f"    → For each spin: within-dimer hopping")
        self.add_coupling(-t1, 0, 'Cd', 2, 'C', dx=[0], plus_hc=True)  # up
        self.add_coupling(-t1, 1, 'Cd', 3, 'C', dx=[0], plus_hc=True)  # down

        print()

        # Inter-cell hopping (B of cell i to A of cell i+1)
        print(f"  Inter-cell hopping (t2={t2:.3f}, weak):")
        print(f"    add_coupling(-{t2}, u0=2, op1='Cd', u1=0, op2='C', dx=[1], plus_hc=True)")
        print(f"    → This couples: (cell i, sublattice B, spin σ) to (cell i+1, sublattice A, spin σ)")
        print(f"    → For each spin: between-dimer hopping")
        self.add_coupling(-t2, 2, 'Cd', 0, 'C', dx=[1], plus_hc=True)  # up
        self.add_coupling(-t2, 3, 'Cd', 1, 'C', dx=[1], plus_hc=True)  # down

        print()

        # Hubbard U
        print(f"  Hubbard interaction (U={U:.3f}):")
        print(f"    add_coupling({U}, u0=0, op1='N', u1=1, op2='N', dx=[0])")
        print(f"    → (cell i, A↑) × (cell i, A↓) for each cell")
        print(f"    add_coupling({U}, u0=2, op1='N', u1=3, op2='N', dx=[0])")
        print(f"    → (cell i, B↑) × (cell i, B↓) for each cell")
        self.add_coupling(U, 0, 'N', 1, 'N', dx=[0])  # n_A_up * n_A_dn
        self.add_coupling(U, 2, 'N', 3, 'N', dx=[0])  # n_B_up * n_B_dn

# Create the model
params = {
    'L': L,
    't1': t1,
    't2': t2,
    'U': U,
}

model = TracedSSHHubbard(params)

print("\n" + "=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)

print(f"\nMapping to physical sites for L={L}:")
print()
print(f"  Cell 0:")
print(f"    MPS sites [0,1,2,3] = [A0↑, A0↓, B0↑, B0↓]")
print(f"    → Physical sites: A0=0, B0=1")
print()
print(f"  Cell 1:")
print(f"    MPS sites [4,5,6,7] = [A1↑, A1↓, B1↑, B1↓]")
print(f"    → Physical sites: A1=2, B1=3")
print()

print(f"  Intra-cell couplings (t1={t1}):")
print(f"    Cell 0: A0 ↔ B0  (MPS: 0↔2 and 1↔3)")
print(f"            Physical: 0 ↔ 1  ✓ CORRECT (strong bond)")
print(f"    Cell 1: A1 ↔ B1  (MPS: 4↔6 and 5↔7)")
print(f"            Physical: 2 ↔ 3  ✓ CORRECT (strong bond)")
print()
print(f"  Inter-cell couplings (t2={t2}):")
print(f"    Cell 0 to Cell 1: B0 ↔ A1  (MPS: 2↔4 and 3↔5)")
print(f"                      Physical: 1 ↔ 2  ✓ CORRECT (weak bond)")

print("\n" + "=" * 80)
print("POTENTIAL ISSUES TO INVESTIGATE")
print("=" * 80)

print("""
1. TeNPy's plus_hc behavior:
   - Does plus_hc=True add: -t (c†c + cc†) ?
   - Or does it add: -t c†c - t* (c†c)† = -t (c†c + h.c.) ?
   - Test already showed it does NOT add 1/2 factor

2. Jordan-Wigner string ordering:
   - VQE uses interleaved [site0↑, site0↓, site1↑, site1↓, ...]
   - TeNPy uses [A0↑, A0↓, B0↑, B0↓, A1↑, A1↓, B1↑, B1↓, ...]
   - Physical sites: A0=0, B0=1, A1=2, B1=3
   - TeNPy order: 0↑,0↓,1↑,1↓, 2↑,2↓,3↑,3↓
   - VQE order:   0↑,0↓,1↑,1↓, 2↑,2↓,3↑,3↓
   - SAME! So J-W strings should match

3. Fermion anticommutation:
   - Check if TeNPy respects fermionic statistics correctly
   - Verify sign conventions match

4. Coupling strength interpretation:
   - Does add_coupling(-t, ..., plus_hc=True) give H = -t(c†c + cc†)?
   - Or H = -t(c†c + h.c.) = -t c†c - t* c c† ?
   - Need to check TeNPy source code or documentation
""")

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. Check TeNPy documentation for add_coupling with plus_hc=True
2. Build a minimal 2-site model and compare matrix elements
3. Test whether the issue is in coupling strength vs operator ordering
4. Consider that the 1.68% error might be from something else entirely:
   - Numerical precision in DMRG?
   - Different energy zero-point?
   - Missing constant shift in Hamiltonian?
""")
print("=" * 80)
