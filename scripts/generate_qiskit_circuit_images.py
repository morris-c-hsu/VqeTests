#!/usr/bin/env python3
"""
Generate circuit diagram images using native Qiskit matplotlib drawer.
Uses pylatexenc for proper mathematical symbols and professional appearance.
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
    build_ansatz_tn_mps_sshh,
    build_ansatz_tn_mps_np_sshh,
    prepare_half_filling_state,
)


def save_circuit_image(circuit, filename, title="", output_dir='../docs/images'):
    """Save circuit diagram as PNG using Qiskit's native matplotlib drawer."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating: {filename}")
    print(f"  Circuit: {circuit.num_qubits} qubits, {circuit.num_parameters} params, depth {circuit.depth()}")

    try:
        # Check pylatexenc availability
        try:
            import pylatexenc
            print("  Using pylatexenc for high-quality rendering")
        except ImportError:
            print("  WARNING: pylatexenc not installed - may have rendering issues")
            print("  Install with: pip install pylatexenc")

        # Use Qiskit's native matplotlib drawer
        fig = circuit.draw(
            output='mpl',
            style={'backgroundcolor': '#FFFFFF'},
            fold=100,  # Fold long circuits
            plot_barriers=False,
            scale=0.8
        )

        # Save with high quality
        filepath = f"{output_dir}/{filename}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  SUCCESS: Saved to {filepath}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Generate all ansatz circuit diagrams using Qiskit's native drawer."""
    L = 4  # Small system for clear diagrams
    reps = 1  # Single repetition layer for clarity
    t1, t2 = 1.0, 0.5  # SSH parameters

    print("=" * 70)
    print("GENERATING CIRCUIT DIAGRAMS WITH QISKIT NATIVE DRAWER")
    print("=" * 70)
    print(f"System size: L={L} ({2*L} qubits)")
    print(f"Repetitions: {reps}")
    print()

    success_count = 0
    total_count = 0

    # 0. Initial State - Half-Filling
    print("\n0. Initial State Preparation (Half-Filling)")
    print("-" * 70)
    initial_state = prepare_half_filling_state(L)
    if save_circuit_image(initial_state, "ansatz_initial_state.png", "Initial State: Half-Filling"):
        success_count += 1
    total_count += 1

    # 1. HEA - Hardware-Efficient Ansatz
    print("\n1. HEA (Hardware-Efficient Ansatz)")
    print("-" * 70)
    ansatz_hea = build_ansatz_hea(2*L, reps)  # N = 2*L qubits
    if save_circuit_image(ansatz_hea, "ansatz_hea.png", "HEA: Hardware-Efficient Ansatz"):
        success_count += 1
    total_count += 1

    # 2. HVA - Hamiltonian-Variational Ansatz
    print("\n2. HVA (Hamiltonian-Variational Ansatz)")
    print("-" * 70)
    ansatz_hva = build_ansatz_hva_sshh(L, reps, t1, t2)
    if save_circuit_image(ansatz_hva, "ansatz_hva.png", "HVA: Hamiltonian-Variational Ansatz"):
        success_count += 1
    total_count += 1

    # 3. TopoInspired - Topology-Inspired Ansatz
    print("\n3. TopoInspired (Topology-Inspired Ansatz)")
    print("-" * 70)
    ansatz_topo = build_ansatz_topo_sshh(L, reps)
    if save_circuit_image(ansatz_topo, "ansatz_topoinsp.png", "TopoInspired: Topology-Inspired Ansatz"):
        success_count += 1
    total_count += 1

    # 4. Topo_RN - RN-Topological Ansatz
    print("\n4. Topo_RN (RN-Topological Ansatz)")
    print("-" * 70)
    ansatz_topo_rn = build_ansatz_topo_rn_sshh(L, reps)
    if save_circuit_image(ansatz_topo_rn, "ansatz_topo_rn.png", "Topo_RN: RN-Topological Ansatz"):
        success_count += 1
    total_count += 1

    # 5. DQAP - Discretized QAP
    print("\n5. DQAP (Discretized QAP)")
    print("-" * 70)
    ansatz_dqap = build_ansatz_dqap_sshh(L, layers=reps, include_U=True)
    if save_circuit_image(ansatz_dqap, "ansatz_dqap.png", "DQAP: Discretized QAP"):
        success_count += 1
    total_count += 1

    # 6. NP_HVA - Number-Preserving HVA
    print("\n6. NP_HVA (Number-Preserving HVA)")
    print("-" * 70)
    ansatz_np_hva = build_ansatz_np_hva_sshh(L, reps)
    if save_circuit_image(ansatz_np_hva, "ansatz_np_hva.png", "NP_HVA: Number-Preserving HVA"):
        success_count += 1
    total_count += 1

    # 7. TN_MPS - Tensor Network MPS
    print("\n7. TN_MPS (Tensor Network MPS)")
    print("-" * 70)
    ansatz_tn_mps = build_ansatz_tn_mps_sshh(L, reps)
    if save_circuit_image(ansatz_tn_mps, "ansatz_tn_mps.png", "TN_MPS: Tensor Network MPS"):
        success_count += 1
    total_count += 1

    # 8. TN_MPS_NP - Number-Preserving TN_MPS
    print("\n8. TN_MPS_NP (Number-Preserving TN_MPS)")
    print("-" * 70)
    ansatz_tn_mps_np = build_ansatz_tn_mps_np_sshh(L, reps)
    if save_circuit_image(ansatz_tn_mps_np, "ansatz_tn_mps_np.png", "TN_MPS_NP: Number-Preserving TN_MPS"):
        success_count += 1
    total_count += 1

    print("\n" + "=" * 70)
    print(f"COMPLETE: {success_count}/{total_count} diagrams generated successfully")
    print("=" * 70)

    if success_count < total_count:
        print("\nWARNING: Some diagrams failed to generate")
        sys.exit(1)
    else:
        print("\nAll circuit diagrams generated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
