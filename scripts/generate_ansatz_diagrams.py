#!/usr/bin/env python3
"""
Generate circuit diagrams for all ansätze in the repository.

Creates PNG images of each ansatz for documentation purposes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from ssh_hubbard_vqe import (
    build_ansatz_hea,
    build_ansatz_hva_sshh,
    build_ansatz_topo_sshh,
    build_ansatz_topo_rn_sshh,
    build_ansatz_dqap_sshh,
    build_ansatz_np_hva_sshh,
    prepare_half_filling_state,
)

from ssh_hubbard_tn_vqe import (
    build_ansatz_tn_mps_sshh,
    build_ansatz_tn_mps_np_sshh,
)

from qiskit.circuit import QuantumCircuit


def save_ansatz_diagram(ansatz, name, description, output_dir='../docs/images'):
    """Save ansatz circuit diagram as PNG using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating diagram for {name}...")
    print(f"  Circuit: {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters, depth {ansatz.depth()}")

    try:
        # Use matplotlib drawer with simplified approach
        from qiskit.visualization import circuit_drawer

        fig = circuit_drawer(ansatz, output='mpl', style={'backgroundcolor': '#EEEEEE'}, fold=-1)
        filename = f"{output_dir}/{name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ Saved to {filename}")

    except Exception as e:
        print(f"  ✗ Error with matplotlib: {e}")
        print(f"  Trying text mode fallback...")
        try:
            # Fallback to text mode
            text_output = circuit_drawer(ansatz, output='text', fold=80)
            filename_txt = f"{output_dir}/{name}.txt"
            with open(filename_txt, 'w') as f:
                f.write(str(text_output))  # Convert TextDrawing to string
            print(f"  ✓ Saved text version to {filename_txt}")
        except Exception as e2:
            print(f"  ✗ Text fallback also failed: {e2}")


def main():
    """Generate all ansatz diagrams."""
    L = 4  # Small system for clear diagrams
    reps = 1  # Single repetition layer for clarity
    t1, t2 = 1.0, 0.5  # SSH parameters for HVA

    print("=" * 70)
    print("GENERATING ANSATZ CIRCUIT DIAGRAMS")
    print("=" * 70)
    print(f"System size: L={L} ({2*L} qubits)")
    print(f"Repetitions: {reps}")
    print()

    # 1. HEA - Hardware-Efficient Ansatz
    print("\n1. HEA (Hardware-Efficient Ansatz)")
    print("-" * 70)
    ansatz_hea = build_ansatz_hea(L, reps)
    save_ansatz_diagram(
        ansatz_hea,
        "ansatz_hea",
        "Hardware-Efficient Ansatz (EfficientSU2)"
    )

    # 2. HVA - Hamiltonian-Variational Ansatz
    print("\n2. HVA (Hamiltonian-Variational Ansatz)")
    print("-" * 70)
    ansatz_hva = build_ansatz_hva_sshh(L, reps, t1, t2)
    save_ansatz_diagram(
        ansatz_hva,
        "ansatz_hva",
        "Hamiltonian-Variational Ansatz"
    )

    # 3. TopoInspired - Topological Ansatz
    print("\n3. TopoInspired (Topology-Inspired Ansatz)")
    print("-" * 70)
    ansatz_topo = build_ansatz_topo_sshh(L, reps)
    save_ansatz_diagram(
        ansatz_topo,
        "ansatz_topoinsp",
        "Topology-Inspired Ansatz"
    )

    # 4. Topo_RN - RN-Topological Ansatz
    print("\n4. Topo_RN (RN-Topological Ansatz)")
    print("-" * 70)
    ansatz_topo_rn = build_ansatz_topo_rn_sshh(L, reps)
    save_ansatz_diagram(
        ansatz_topo_rn,
        "ansatz_topo_rn",
        "RN-Topological Ansatz"
    )

    # 5. DQAP - Discretized Quantum Adiabatic Process
    print("\n5. DQAP (Discretized QAP)")
    print("-" * 70)
    ansatz_dqap = build_ansatz_dqap_sshh(L, layers=reps, include_U=True)
    save_ansatz_diagram(
        ansatz_dqap,
        "ansatz_dqap",
        "Discretized Quantum Adiabatic Process"
    )

    # 6. NP_HVA - Number-Preserving HVA
    print("\n6. NP_HVA (Number-Preserving HVA)")
    print("-" * 70)
    ansatz_np_hva = build_ansatz_np_hva_sshh(L, reps)
    save_ansatz_diagram(
        ansatz_np_hva,
        "ansatz_np_hva",
        "Number-Preserving HVA"
    )

    # 7. TN_MPS - Tensor Network MPS
    print("\n7. TN_MPS (Tensor Network MPS)")
    print("-" * 70)
    ansatz_tn_mps = build_ansatz_tn_mps_sshh(L, reps)
    save_ansatz_diagram(
        ansatz_tn_mps,
        "ansatz_tn_mps",
        "Tensor Network MPS (brick-wall)"
    )

    # 8. TN_MPS_NP - Number-Preserving TN_MPS
    print("\n8. TN_MPS_NP (Number-Preserving TN_MPS)")
    print("-" * 70)
    ansatz_tn_mps_np = build_ansatz_tn_mps_np_sshh(L, reps)
    save_ansatz_diagram(
        ansatz_tn_mps_np,
        "ansatz_tn_mps_np",
        "Number-Preserving Tensor Network MPS"
    )

    # Also generate initial state diagram
    print("\n9. Initial State (Half-Filling)")
    print("-" * 70)
    initial_state = prepare_half_filling_state(L)
    save_ansatz_diagram(
        initial_state,
        "initial_state_half_filling",
        "Initial State: Half-Filling (1 fermion per site)"
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"All diagrams saved to docs/images/")
    print("\nGenerated files:")
    print("  - ansatz_hea.png")
    print("  - ansatz_hva.png")
    print("  - ansatz_topoinsp.png")
    print("  - ansatz_topo_rn.png")
    print("  - ansatz_dqap.png")
    print("  - ansatz_np_hva.png")
    print("  - ansatz_tn_mps.png")
    print("  - ansatz_tn_mps_np.png")
    print("  - initial_state_half_filling.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
