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
    """Save ansatz circuit diagram as PNG using matplotlib with manual gate drawing."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating diagram for {name}...")
    print(f"  Circuit: {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters, depth {ansatz.depth()}")

    try:
        # Try matplotlib with latex-free approach
        from qiskit.visualization import circuit_drawer

        # Use 'mpl' output with latex=False in style
        style = {
            'displaytext': {},  # Don't use latex for gate labels
            'backgroundcolor': '#FFFFFF',
            'textcolor': '#000000',
        }

        fig = circuit_drawer(
            ansatz,
            output='mpl',
            style=style,
            plot_barriers=False,
            fold=100  # Wider fold for better readability
        )

        filename = f"{output_dir}/{name}.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved to {filename}")
        return True

    except ImportError as e:
        if 'pylatexenc' in str(e):
            print(f"  ⚠ pylatexenc not available, trying alternative method...")
            # Try using latex source instead
            try:
                from qiskit.visualization import circuit_drawer
                # Generate LaTeX source
                latex_source = circuit_drawer(ansatz, output='latex_source', fold=100)

                # Save LaTeX source for manual compilation
                filename_tex = f"{output_dir}/{name}.tex"
                with open(filename_tex, 'w') as f:
                    f.write(latex_source)
                print(f"  ✓ Saved LaTeX source to {filename_tex}")
                print(f"    (requires pdflatex to compile to PDF)")
                return False
            except Exception as e3:
                print(f"  ✗ LaTeX source generation failed: {e3}")
                return False
        else:
            print(f"  ✗ Import error: {e}")
            return False

    except Exception as e:
        print(f"  ✗ Error with matplotlib: {e}")
        return False


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
