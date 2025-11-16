#!/usr/bin/env python3
"""
Check what TeNPy considers as nearest neighbors in our lattice.

This might reveal if TeNPy is automatically adding extra couplings
based on spatial positions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Lattice

print("=" * 80)
print("LATTICE NEIGHBOR DETECTION")
print("=" * 80)

L = 4
N_cells = L // 2

# Build lattice as in the DMRG code
site = FermionSite(conserve='N')
unit_cell_sites = [site] * 4

lat = Lattice([N_cells], unit_cell_sites,
             bc_MPS='finite',
             bc='open',
             basis=[[1.]],
             positions=[[0.], [0.25], [0.5], [0.75]])

print(f"\nLattice properties:")
print(f"  N_sites: {lat.N_sites}")
print(f"  N_cells: {lat.N_cells}")
print(f"  Ls: {lat.Ls}")
print(f"  Unit cell size: {len(lat.unit_cell)}")

# Check nearest neighbors
print("\n" + "=" * 80)
print("NEAREST NEIGHBOR PAIRS")
print("=" * 80)

# Get pairs for different distances
for dist_idx in range(min(5, lat.N_sites)):
    try:
        pairs = lat.pairs['nearest_neighbors']
        print(f"\nNearest neighbors (automatic detection):")
        for pair in pairs:
            print(f"  {pair}")
        break
    except (KeyError, AttributeError):
        pass

# Try to get coupling pairs for specific unit cell positions
print("\n" + "=" * 80)
print("COUPLING PAIRS (what add_coupling uses)")
print("=" * 80)

# The coupling (u0, u1, dx) maps to MPS indices
print("\nMapping couplings to MPS bonds:")

def coupling_to_mps_pairs(u0, u1, dx, N_cells):
    """Convert coupling (u0, u1, dx) to list of MPS site pairs."""
    pairs = []
    for cell_i in range(N_cells):
        if dx == [0]:  # Same cell
            mps_i = cell_i * 4 + u0
            mps_j = cell_i * 4 + u1
            pairs.append((mps_i, mps_j))
        elif dx == [1]:  # Adjacent cells
            if cell_i < N_cells - 1:
                mps_i = cell_i * 4 + u0
                mps_j = (cell_i + 1) * 4 + u1
                pairs.append((mps_i, mps_j))
    return pairs

# Intra-cell t1 couplings
print("\nIntra-cell hopping (t1):")
print("  (u0=0, u1=2, dx=[0])  # A↑ → B↑ within cell")
pairs = coupling_to_mps_pairs(0, 2, [0], N_cells)
print(f"    MPS pairs: {pairs}")

print("  (u0=1, u1=3, dx=[0])  # A↓ → B↓ within cell")
pairs = coupling_to_mps_pairs(1, 3, [0], N_cells)
print(f"    MPS pairs: {pairs}")

# Inter-cell t2 couplings
print("\nInter-cell hopping (t2):")
print("  (u0=2, u1=0, dx=[1])  # B↑ → A↑ between cells")
pairs = coupling_to_mps_pairs(2, 0, [1], N_cells)
print(f"    MPS pairs: {pairs}")

print("  (u0=3, u1=1, dx=[1])  # B↓ → A↓ between cells")
pairs = coupling_to_mps_pairs(3, 1, [1], N_cells)
print(f"    MPS pairs: {pairs}")

# Check what the position-based distances are
print("\n" + "=" * 80)
print("POSITION-BASED DISTANCES")
print("=" * 80)

print("\nPhysical positions of MPS sites:")
for i in range(lat.N_sites):
    pos = lat.position(lat.mps2lat_idx(i))
    cell = i // 4
    u = i % 4
    print(f"  MPS {i} (cell {cell}, u={u}): position {pos}")

print("\nDistances between consecutive MPS sites:")
for i in range(lat.N_sites - 1):
    pos_i = lat.position(lat.mps2lat_idx(i))
    pos_j = lat.position(lat.mps2lat_idx(i+1))
    dist = np.linalg.norm(pos_j - pos_i)
    print(f"  MPS {i} ↔ {i+1}: distance = {dist:.4f}")

print("\nKey observation:")
print("  Distance within cell (A↔B): should correspond to intra-cell hopping")
print("  Distance between cells: should correspond to inter-cell hopping")
print()
print("  But TeNPy might be using these positions for automatic neighbor detection!")
print("  If so, it could be adding EXTRA couplings we didn't ask for.")

print("\n" + "=" * 80)
