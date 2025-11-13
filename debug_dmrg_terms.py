#!/usr/bin/env python3
"""
Debug DMRG model to understand what terms are actually being added.

Print all coupling terms to verify the Hamiltonian construction.
"""

from ssh_hubbard_tenpy_dmrg_fixed import SpinfulSSHHubbard
import numpy as np

def debug_model_terms(L=4, t1=1.0, t2=0.6, U=2.0):
    """Print all terms in the DMRG model."""
    print("\n" + "=" * 80)
    print(f"DEBUG: DMRG Model Terms (L={L})")
    print("=" * 80)

    model_params = {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        'bc_MPS': 'finite',
    }

    model = SpinfulSSHHubbard(model_params)

    print("\n[1] Lattice Structure:")
    print(f"    N_sites (MPS): {model.lat.N_sites}")
    print(f"    N_cells:       {model.lat.N_cells}")
    print(f"    Unit cell size: {model.lat.N_sites_per_ring}")

    # Print all coupling terms
    print("\n[2] Coupling Terms:")

    # TeNPy stores terms in self.all_coupling_terms()
    terms = model.all_coupling_terms()

    print(f"\n    Total number of terms: {len(terms.to_TermList())}")

    # Let's manually print what we expect
    print("\n[3] Expected SSH-Hubbard Terms (L=4):")
    print("\n    Hopping terms:")
    print("      Physical bond 0→1 (MPS 0→2 for up, 1→3 for down): t1 = 1.0")
    print("      Physical bond 1→2 (MPS 2→4 for up, 3→5 for down): t2 = 0.6")
    print("      Physical bond 2→3 (MPS 4→6 for up, 5→7 for down): t1 = 1.0")

    print("\n    Interaction terms:")
    print("      Physical site 0 (MPS 0,1): U = 2.0")
    print("      Physical site 1 (MPS 2,3): U = 2.0")
    print("      Physical site 2 (MPS 4,5): U = 2.0")
    print("      Physical site 3 (MPS 6,7): U = 2.0")

    # Try to extract bond energies
    print("\n[4] Analyzing MPS structure...")
    print(f"    MPS sites: {model.lat.mps_sites()}")

    print("\n[5] Unit cell details:")
    for u in range(len(model.lat.unit_cell)):
        site = model.lat.unit_cell[u]
        print(f"    Position {u}: {site}")

    # Check if we can access H_bond
    if hasattr(model, 'H_bond'):
        print("\n[6] Bond Hamiltonians:")
        for i, H in enumerate(model.H_bond):
            if H is not None:
                evals = np.linalg.eigvalsh(H)
                print(f"    Bond {i}: shape={H.shape}, min_eval={evals[0]:.6f}, max_eval={evals[-1]:.6f}")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# DEBUG: DMRG MODEL STRUCTURE")
    print("#" * 80)

    debug_model_terms(L=4, t1=1.0, t2=0.6, U=2.0)

    print("\n" + "#" * 80)

    # Additional test: Try to understand the unit cell mapping
    print("\n[DIAGNOSTIC] Unit cell interpretation:")
    print("\n  In unit cell [A↑, A↓, B↑, B↓]:")
    print("  - Sites 0,1 = Physical site A (one dimer half)")
    print("  - Sites 2,3 = Physical site B (other dimer half)")
    print("\n  For L=4 with 2 unit cells:")
    print("  - Cell 0: MPS [0,1,2,3] = Physical sites [0,1]")
    print("  - Cell 1: MPS [4,5,6,7] = Physical sites [2,3]")
    print("\n  Intra-cell hopping (dx=0):")
    print("  - Connects A→B within cell")
    print("  - Cell 0: 0→2 (phys 0→1), 1→3 (phys 0→1)")
    print("  - Cell 1: 4→6 (phys 2→3), 5→7 (phys 2→3)")
    print("\n  Inter-cell hopping (dx=1):")
    print("  - Connects B of cell i → A of cell i+1")
    print("  - Cell 0→1: 2→4 (phys 1→2), 3→5 (phys 1→2)")

    print("\n  This matches SSH pattern: 0→1 (t1), 1→2 (t2), 2→3 (t1) ✓")
    print()
