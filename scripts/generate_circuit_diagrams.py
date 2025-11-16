#!/usr/bin/env python3
"""
Generate circuit diagram images for all ansätze using Qiskit's native circuit drawer.

Creates PNG images of quantum circuits using Qiskit's built-in draw() method
with matplotlib backend and default style.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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

from qiskit.circuit import QuantumCircuit, Parameter


def strip_params_for_plot(circ):
    """
    For 'structure-only' diagrams: assign all params to 0 so the drawer
    just prints gate types (Ry, Rz, RXX, RYY, RZZ, CX, …).
    """
    if not circ.parameters:
        return circ
    return circ.assign_parameters({p: 0.0 for p in circ.parameters}, inplace=False)


def shorten_params_for_plot(circ):
    """
    For 'symbolic' diagrams: keep parameters but rename them θ0, θ1, …
    to avoid huge labels.
    """
    if not circ.parameters:
        return circ

    mapping = {}
    for i, p in enumerate(sorted(circ.parameters, key=lambda x: x.name)):
        mapping[p] = Parameter(f"θ{i}")
    return circ.assign_parameters(mapping, inplace=False)


def save_circuit_diagram(
    circuit,
    filename,
    description,
    output_dir='../docs/images',
    mode='structure',   # 'structure' or 'symbols'
):
    """Save circuit diagram as PNG with specified rendering mode."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating: {filename}")
    print(f"  {description}")
    print(f"  Circuit: {circuit.num_qubits} qubits, {circuit.num_parameters} params, depth {circuit.depth()}")

    try:
        # Choose how much text to show
        if mode == 'symbols':
            plot_circ = shorten_params_for_plot(circuit)
        else:
            plot_circ = strip_params_for_plot(circuit)

        fig = plot_circ.draw(
            output="mpl",
            fold=-1,
            idle_wires=False,
            scale=0.8,       # moderate; not huge
        )
        fig.set_size_inches(14, 4)

        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  ✓ Saved to {filepath}")
        return True

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main():
    """Generate all ansatz circuit diagrams."""
    L = 4  # Small system for clear diagrams
    reps = 1  # Single repetition layer for clarity
    t1, t2 = 1.0, 0.5  # SSH parameters

    print("=" * 70)
    print("GENERATING ANSATZ CIRCUIT DIAGRAMS")
    print("=" * 70)
    print(f"System size: L={L} ({2*L} qubits)")
    print(f"Repetitions: {reps}")
    print()

    success_count = 0
    total_count = 0

    # Define all circuits to generate (circuit, filename, description, mode)
    circuits = [
        (prepare_half_filling_state(L),
         "initial_state_half_filling.png",
         "Initial State: Half-Filling (1 fermion per site)",
         "structure"),

        (build_ansatz_hea(L, reps),
         "ansatz_hea.png",
         "HEA: Hardware-Efficient Ansatz",
         "structure"),

        (build_ansatz_hva_sshh(L, reps, t1, t2),
         "ansatz_hva.png",
         "HVA: Hamiltonian-Variational Ansatz",
         "symbols"),

        (build_ansatz_topo_sshh(L, reps),
         "ansatz_topoinsp.png",
         "TopoInspired: Topology-Inspired Ansatz",
         "symbols"),

        (build_ansatz_topo_rn_sshh(L, reps),
         "ansatz_topo_rn.png",
         "Topo_RN: RN-Topological Ansatz",
         "symbols"),

        (build_ansatz_dqap_sshh(L, layers=reps, include_U=True),
         "ansatz_dqap.png",
         "DQAP: Discretized Quantum Adiabatic Process",
         "symbols"),

        (build_ansatz_np_hva_sshh(L, reps),
         "ansatz_np_hva.png",
         "NP_HVA: Number-Preserving HVA",
         "symbols"),

        (build_ansatz_tn_mps_sshh(L, reps),
         "ansatz_tn_mps.png",
         "TN_MPS: Tensor Network MPS (brick-wall)",
         "structure"),

        (build_ansatz_tn_mps_np_sshh(L, reps),
         "ansatz_tn_mps_np.png",
         "TN_MPS_NP: Number-Preserving Tensor Network MPS",
         "structure"),
    ]

    # Generate all diagrams
    for i, (circuit, filename, description, mode) in enumerate(circuits, 1):
        print(f"\n{i}. {description}")
        print("-" * 70)

        if save_circuit_diagram(circuit, filename, description, mode=mode):
            success_count += 1
        total_count += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"COMPLETE: {success_count}/{total_count} diagrams generated successfully")
    print("=" * 70)

    if success_count < total_count:
        print("\nWARNING: Some diagrams failed to generate")
        sys.exit(1)
    else:
        print("\nAll circuit diagrams generated successfully!")
        print(f"Output directory: docs/images/")
        sys.exit(0)


if __name__ == '__main__':
    main()
