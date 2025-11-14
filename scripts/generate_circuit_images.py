#!/usr/bin/env python3
"""
Generate circuit images using matplotlib directly, bypassing pylatexenc requirement.

Uses a custom circuit drawer that creates simple box-and-line diagrams.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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


def draw_circuit_custom(circuit, filename, title=""):
    """
    Custom circuit drawer using matplotlib without latex.
    Creates a simple visualization with boxes for gates and lines for wires.
    """
    num_qubits = circuit.num_qubits

    # Get circuit data
    data = circuit.data

    # Create figure
    fig_width = min(20, max(10, len(data) * 0.5))
    fig_height = max(4, num_qubits * 0.6)

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, len(data) + 0.5)
    ax.set_ylim(-0.5, num_qubits - 0.5)
    ax.set_aspect('equal')

    # Draw qubit wires
    for q in range(num_qubits):
        y = num_qubits - 1 - q
        ax.plot([-0.3, len(data) + 0.3], [y, y], 'k-', linewidth=1, zorder=1)
        ax.text(-0.4, y, f'q{q}', ha='right', va='center', fontsize=8)

    # Draw gates
    x_pos = 0
    for instruction in data:
        gate = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        gate_name = gate.name

        # Simplify parameter names
        if hasattr(gate, 'params') and len(gate.params) > 0:
            # Just show gate type, not full parameters
            display_name = gate_name.upper()
        else:
            display_name = gate_name.upper()

        if len(qubits) == 1:
            # Single qubit gate
            q = qubits[0]
            y = num_qubits - 1 - q

            # Draw box
            rect = patches.Rectangle(
                (x_pos - 0.15, y - 0.15),
                0.3, 0.3,
                linewidth=1,
                edgecolor='blue',
                facecolor='lightblue',
                zorder=2
            )
            ax.add_patch(rect)

            # Add label (first 3 chars to save space)
            label = display_name[:3] if len(display_name) > 3 else display_name
            ax.text(x_pos, y, label, ha='center', va='center',
                   fontsize=6, weight='bold', zorder=3)

        elif len(qubits) == 2:
            # Two qubit gate
            q0, q1 = qubits
            y0 = num_qubits - 1 - q0
            y1 = num_qubits - 1 - q1

            # Draw connecting line
            ax.plot([x_pos, x_pos], [y0, y1], 'b-', linewidth=2, zorder=2)

            # Draw control/target circles
            if gate_name.lower() in ['cx', 'cnot']:
                # Control dot
                circle1 = patches.Circle((x_pos, y0), 0.08,
                                        facecolor='blue', edgecolor='blue', zorder=3)
                ax.add_patch(circle1)

                # Target circle with X
                circle2 = patches.Circle((x_pos, y1), 0.15,
                                        facecolor='white', edgecolor='blue',
                                        linewidth=2, zorder=3)
                ax.add_patch(circle2)
                ax.plot([x_pos - 0.1, x_pos + 0.1], [y1, y1], 'b-', linewidth=2, zorder=4)
                ax.plot([x_pos, x_pos], [y1 - 0.1, y1 + 0.1], 'b-', linewidth=2, zorder=4)
            else:
                # Generic two-qubit gate - draw boxes
                for q in [q0, q1]:
                    y = num_qubits - 1 - q
                    rect = patches.Rectangle(
                        (x_pos - 0.15, y - 0.15),
                        0.3, 0.3,
                        linewidth=1,
                        edgecolor='green',
                        facecolor='lightgreen',
                        zorder=3
                    )
                    ax.add_patch(rect)

                # Label
                label = display_name[:2] if len(display_name) > 2 else display_name
                ax.text(x_pos, (y0 + y1) / 2, label,
                       ha='center', va='center', fontsize=5, zorder=4)

        x_pos += 1

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add title
    if title:
        ax.set_title(title, fontsize=10, pad=10)

    # Add info text
    info_text = f"{num_qubits} qubits, {circuit.num_parameters} params, depth {circuit.depth()}"
    ax.text(len(data) / 2, -0.8, info_text, ha='center', fontsize=8, style='italic')

    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return True


def main():
    """Generate all ansatz diagrams as PNG images."""
    L = 4
    reps = 1
    t1, t2 = 1.0, 0.5

    output_dir = '../docs/images'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING ANSATZ CIRCUIT IMAGES (PNG)")
    print("=" * 70)
    print(f"System: L={L} ({2*L} qubits), reps={reps}")
    print(f"Output: {output_dir}/")
    print()

    ansatze = [
        (build_ansatz_hea(L, reps), "ansatz_hea", "HEA: Hardware-Efficient Ansatz"),
        (build_ansatz_hva_sshh(L, reps, t1, t2), "ansatz_hva", "HVA: Hamiltonian-Variational"),
        (build_ansatz_topo_sshh(L, reps), "ansatz_topoinsp", "TopoInspired: Topology-Inspired"),
        (build_ansatz_topo_rn_sshh(L, reps), "ansatz_topo_rn", "Topo_RN: RN-Topological"),
        (build_ansatz_dqap_sshh(L, layers=reps, include_U=True), "ansatz_dqap", "DQAP: Discretized QAP"),
        (build_ansatz_np_hva_sshh(L, reps), "ansatz_np_hva", "NP_HVA: Number-Preserving HVA"),
        (build_ansatz_tn_mps_sshh(L, reps), "ansatz_tn_mps", "TN_MPS: Tensor Network MPS"),
        (build_ansatz_tn_mps_np_sshh(L, reps), "ansatz_tn_mps_np", "TN_MPS_NP: Number-Preserving TN"),
        (prepare_half_filling_state(L), "initial_state_half_filling", "Initial State: Half-Filling"),
    ]

    success_count = 0
    for i, (ansatz, name, title) in enumerate(ansatze, 1):
        print(f"{i}. {title}")
        print("-" * 70)

        filename = f"{output_dir}/{name}.png"

        try:
            success = draw_circuit_custom(ansatz, filename, title)
            if success:
                print(f"  ✓ Saved to {filename}")
                success_count += 1
            else:
                print(f"  ✗ Failed to save {filename}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()

    print("=" * 70)
    print(f"COMPLETE: {success_count}/{len(ansatze)} diagrams generated")
    print("=" * 70)


if __name__ == "__main__":
    main()
