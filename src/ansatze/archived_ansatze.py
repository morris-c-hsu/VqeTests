#!/usr/bin/env python3
"""
Archived Ansätze for SSH-Hubbard VQE

This module contains additional ansatz implementations that are preserved for
future use but not included in the main benchmark suite.

Main ansätze (in ssh_hubbard_vqe.py):
- HEA: Hardware-Efficient Ansatz (baseline)
- HVA: Hamiltonian-Variational Ansatz (Hamiltonian-inspired, number-conserving)
- NP_HVA: Number-Preserving HVA (strict number conservation with UNP gates)

Archived ansätze (in this file):
- TopoInspired: Topological/Problem-Inspired (SSH dimer pattern)
- TopoRN: RN-Topological (number-conserving, SSH structure)
- DQAP: Digital-Adiabatic/QAOA-style (adiabatic evolution)
- TN_MPS: Tensor-Network MPS (brick-wall structure)
- TN_MPS_NP: Number-Preserving TN-MPS (brick-wall with number conservation)

Usage:
    from ansatze.archived_ansatze import build_ansatz_topo_sshh

    # Build TopoInspired ansatz for L=4, 2 repetitions
    circuit = build_ansatz_topo_sshh(L=4, reps=2)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Import helper functions from parent module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ssh_hubbard_vqe import q_index, unp_gate, rn_gate


# =============================================================================
# TOPOLOGICAL / PROBLEM-INSPIRED ANSÄTZE
# =============================================================================

def build_ansatz_topo_sshh(L: int, reps: int, use_edge_link: bool = True) -> QuantumCircuit:
    """
    Problem-Inspired / Topological Ansatz for SSH-Hubbard.

    ARCHIVED: Use HEA, HVA, or NP_HVA for main tests.

    Structure per repetition:
      1. Local Ry rotations on all qubits
      2. Strong bond layer (even bonds with XX+YY)
      3. Weak bond layer (odd bonds with XX+YY)
      4. Edge link (connects sites 0 and L-1) if enabled

    Parameters:
        L: Number of lattice sites
        reps: Number of repetitions
        use_edge_link: Whether to include edge-to-edge entangler

    Returns:
        Parameterized quantum circuit
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    for rep in range(reps):
        # Local rotations
        for q in range(N):
            theta = Parameter(f'ry_{rep}_{q}')
            qc.ry(theta, q)

        # Strong bonds (even indices)
        for i in range(0, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                gamma = Parameter(f'γ_str_{rep}_{i}_{spin}')

                try:
                    qc.rxx(gamma, qi, qj)
                    qc.ryy(gamma, qi, qj)
                except:
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(gamma, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Weak bonds (odd indices)
        for i in range(1, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                gamma = Parameter(f'γ_weak_{rep}_{i}_{spin}')

                try:
                    qc.rxx(gamma, qi, qj)
                    qc.ryy(gamma, qi, qj)
                except:
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(gamma, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Edge link (topological feature)
        if use_edge_link and L > 2:
            for spin in ['up', 'down']:
                q_left = q_index(0, spin, L)
                q_right = q_index(L - 1, spin, L)
                gamma_edge = Parameter(f'γ_edge_{rep}_{spin}')

                try:
                    qc.rxx(gamma_edge, q_left, q_right)
                    qc.ryy(gamma_edge, q_left, q_right)
                except:
                    qc.h(q_left)
                    qc.h(q_right)
                    qc.cx(q_left, q_right)
                    qc.rz(gamma_edge, q_right)
                    qc.cx(q_left, q_right)
                    qc.h(q_left)
                    qc.h(q_right)

    return qc


def build_ansatz_topo_rn_sshh(L: int, reps: int, use_edge_link: bool = True) -> QuantumCircuit:
    """
    RN-Topological Ansatz for SSH-Hubbard (Ciaramelletti-style).

    ARCHIVED: Use HEA, HVA, or NP_HVA for main tests.

    Structure per repetition:
      1. Local single-qubit Ry (and optionally Rz) rotations
      2. Strong bonds (even) with RN gates
      3. Weak bonds (odd) with RN gates
      4. Topological edge link with RN gate (if enabled)

    The RN gate preserves excitation number and provides number-conserving
    entanglement suitable for topological systems.

    Parameters:
        L: Number of lattice sites
        reps: Number of repetitions
        use_edge_link: Whether to include edge-to-edge RN gate

    Returns:
        Parameterized quantum circuit
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    for rep in range(reps):
        # Local Ry rotations
        for q in range(N):
            theta = Parameter(f'ry_{rep}_{q}')
            qc.ry(theta, q)

        # Strong bonds (even) with RN gates
        for i in range(0, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                theta_rn = Parameter(f'θ_str_{rep}_{i}_{spin}')
                rn_gate(qc, qi, qj, theta_rn)

        # Weak bonds (odd) with RN gates
        for i in range(1, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                theta_rn = Parameter(f'θ_weak_{rep}_{i}_{spin}')
                rn_gate(qc, qi, qj, theta_rn)

        # Edge link with RN gate
        if use_edge_link and L > 2:
            for spin in ['up', 'down']:
                q_left = q_index(0, spin, L)
                q_right = q_index(L - 1, spin, L)
                theta_edge = Parameter(f'θ_edge_{rep}_{spin}')
                rn_gate(qc, q_left, q_right, theta_edge)

        # Additional Rz layer for expressiveness
        for q in range(N):
            phi = Parameter(f'rz_{rep}_{q}')
            qc.rz(phi, q)

    return qc


# =============================================================================
# DIGITAL-ADIABATIC / QAOA-STYLE ANSATZ
# =============================================================================

def build_ansatz_dqap_sshh(L: int, layers: int, include_U: bool = True) -> QuantumCircuit:
    """
    Digital-Adiabatic / QAOA-Style Ansatz for SSH-Hubbard.

    ARCHIVED: Use HEA, HVA, or NP_HVA for main tests.

    Mimics adiabatic evolution with alternating problem and mixer Hamiltonians.
    Each layer applies:
      - Hopping evolution exp(-i β H_hop)
      - Interaction evolution exp(-i γ H_U) (if include_U=True)
      - Mixer (single-qubit rotations)

    Parameters:
        L: Number of lattice sites
        layers: Number of QAOA-like layers
        include_U: Whether to include interaction term evolution

    Returns:
        Parameterized quantum circuit
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    for layer in range(layers):
        # Hopping evolution (approximate exp(-i β H_hop))
        # Using Trotter-like decomposition

        # Strong bonds (t1)
        for i in range(0, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                beta = Parameter(f'β_t1_{layer}_{i}_{spin}')

                # XX + YY ~ hopping
                try:
                    qc.rxx(beta, qi, qj)
                    qc.ryy(beta, qi, qj)
                except:
                    # Fallback decomposition
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(beta, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Weak bonds (t2)
        for i in range(1, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                beta = Parameter(f'β_t2_{layer}_{i}_{spin}')

                try:
                    qc.rxx(beta, qi, qj)
                    qc.ryy(beta, qi, qj)
                except:
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(beta, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Interaction evolution (approximate exp(-i γ H_U))
        if include_U:
            for i in range(L):
                qi_up = q_index(i, 'up', L)
                qi_dn = q_index(i, 'down', L)
                gamma = Parameter(f'γ_U_{layer}_{i}')
                qc.rzz(gamma, qi_up, qi_dn)

        # Mixer layer (single-qubit rotations)
        for q in range(N):
            theta_y = Parameter(f'mix_y_{layer}_{q}')
            theta_z = Parameter(f'mix_z_{layer}_{q}')
            qc.ry(theta_y, q)
            qc.rz(theta_z, q)

    return qc


# =============================================================================
# TENSOR-NETWORK-INSPIRED MPS ANSÄTZE
# =============================================================================

def build_ansatz_tn_mps_sshh(L: int, reps: int) -> QuantumCircuit:
    """
    Tensor-network-inspired quantum-circuit MPS (qMPS) ansatz for SSH-Hubbard.

    ARCHIVED: Use HEA, HVA, or NP_HVA for main tests.

    Brick-wall structure alternating even/odd bond entanglers.
    Does NOT preserve particle number - requires careful initialization.

    Structure:
      Layer = Even-bond gates (0-1, 2-3, ...) + Odd-bond gates (1-2, 3-4, ...) + Rotations

    Parameters:
        L: Number of lattice sites
        reps: Number of brick-wall repetitions

    Returns:
        QuantumCircuit: The TN-MPS ansatz circuit with parameters
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    for rep in range(reps):
        # Even bonds: (0,1), (2,3), (4,5), ...
        for i in range(0, N - 1, 2):
            # General two-qubit unitary (15 parameters)
            # Simplified to 4 parameters for efficiency
            theta1 = Parameter(f'tn_θ1_e_{rep}_{i}')
            theta2 = Parameter(f'tn_θ2_e_{rep}_{i}')
            phi = Parameter(f'tn_φ_e_{rep}_{i}')
            lam = Parameter(f'tn_λ_e_{rep}_{i}')

            qc.ry(theta1, i)
            qc.ry(theta2, i + 1)
            qc.cx(i, i + 1)
            qc.rz(phi, i + 1)
            qc.ry(lam, i + 1)
            qc.cx(i, i + 1)

        # Odd bonds: (1,2), (3,4), (5,6), ...
        for i in range(1, N - 1, 2):
            theta1 = Parameter(f'tn_θ1_o_{rep}_{i}')
            theta2 = Parameter(f'tn_θ2_o_{rep}_{i}')
            phi = Parameter(f'tn_φ_o_{rep}_{i}')
            lam = Parameter(f'tn_λ_o_{rep}_{i}')

            qc.ry(theta1, i)
            qc.ry(theta2, i + 1)
            qc.cx(i, i + 1)
            qc.rz(phi, i + 1)
            qc.ry(lam, i + 1)
            qc.cx(i, i + 1)

        # Single-qubit rotations
        for q in range(N):
            ry_angle = Parameter(f'tn_ry_{rep}_{q}')
            rz_angle = Parameter(f'tn_rz_{rep}_{q}')
            qc.ry(ry_angle, q)
            qc.rz(rz_angle, q)

    return qc


def build_ansatz_tn_mps_np_sshh(L: int, reps: int) -> QuantumCircuit:
    """
    Number-preserving brick-wall TN ansatz for SSH-Hubbard.

    ARCHIVED: Use HEA, HVA, or NP_HVA for main tests.

    Same layout as build_ansatz_tn_mps_sshh, but each 2-qubit block
    uses UNP (number-preserving) gates instead of arbitrary unitaries.
    This ensures the ansatz stays within the correct particle-number sector.

    Structure:
      Layer = Even UNP gates + Odd UNP gates + Single-qubit rotations

    Parameters:
        L: Number of lattice sites
        reps: Number of brick-wall repetitions

    Returns:
        QuantumCircuit: The number-preserving TN-MPS ansatz circuit
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    for rep in range(reps):
        # Even bonds with UNP gates
        for i in range(0, N - 1, 2):
            theta = Parameter(f'tnp_θ_e_{rep}_{i}')
            phi1 = Parameter(f'tnp_φ1_e_{rep}_{i}')
            phi2 = Parameter(f'tnp_φ2_e_{rep}_{i}')
            unp_gate(qc, i, i + 1, theta, phi1, phi2)

        # Odd bonds with UNP gates
        for i in range(1, N - 1, 2):
            theta = Parameter(f'tnp_θ_o_{rep}_{i}')
            phi1 = Parameter(f'tnp_φ1_o_{rep}_{i}')
            phi2 = Parameter(f'tnp_φ2_o_{rep}_{i}')
            unp_gate(qc, i, i + 1, theta, phi1, phi2)

        # Single-qubit Rz rotations (preserve number)
        for q in range(N):
            rz_angle = Parameter(f'tnp_rz_{rep}_{q}')
            qc.rz(rz_angle, q)

    return qc


# =============================================================================
# SUMMARY
# =============================================================================

ARCHIVED_ANSATZE = {
    'topoinsp': build_ansatz_topo_sshh,
    'topo_rn': build_ansatz_topo_rn_sshh,
    'dqap': build_ansatz_dqap_sshh,
    'tn_mps': build_ansatz_tn_mps_sshh,
    'tn_mps_np': build_ansatz_tn_mps_np_sshh,
}

__all__ = [
    'build_ansatz_topo_sshh',
    'build_ansatz_topo_rn_sshh',
    'build_ansatz_dqap_sshh',
    'build_ansatz_tn_mps_sshh',
    'build_ansatz_tn_mps_np_sshh',
    'ARCHIVED_ANSATZE',
]
