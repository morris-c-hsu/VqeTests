#!/usr/bin/env python3
"""
SSH-Hubbard VQE with Topology-Aware Ansätze

This script implements Variational Quantum Eigensolver (VQE) for the spinful
Su-Schrieffer-Heeger (SSH) Hubbard model with three ansatz options:

1. HEA (Hardware-Efficient Ansatz): Standard EfficientSU2 circuit
2. HVA (Hamiltonian-Variational Ansatz): Layers of hopping and interaction terms
3. TopoInspired (Topological/Problem-Inspired): Dimer pattern + edge links

Usage examples:
  python ssh_hubbard_vqe.py --ansatz hea
  python ssh_hubbard_vqe.py --ansatz hva --reps 2
  python ssh_hubbard_vqe.py --ansatz topoinsp --reps 3
  python ssh_hubbard_vqe.py --ansatz topoinsp --delta-sweep -0.6 0.6 13

Features:
  - Exact diagonalization benchmarking
  - Warm-start parameter sweep in dimerization δ
  - Topology diagnostics: edge concurrence, bond purity
  - Extensive observable calculations (energy, correlations, bond orders, etc.)
  - Convergence tracking and plotting
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Optional

# Qiskit imports with version compatibility
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, DensityMatrix

# Ansatz imports
try:
    from qiskit.circuit.library import efficient_su2, EfficientSU2
    HAS_EFFICIENT_SU2_FUNC = True
except ImportError:
    from qiskit.circuit.library import EfficientSU2
    HAS_EFFICIENT_SU2_FUNC = False

# Circuit building
from qiskit.circuit import QuantumCircuit, Parameter

# Primitives (Qiskit 1.x compatibility)
try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    try:
        from qiskit.primitives import Estimator
    except ImportError:
        from qiskit_aer.primitives import Estimator

# VQE and optimizer
try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import L_BFGS_B
except ImportError:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import L_BFGS_B


# ============================================================================
# HELPER FUNCTIONS: QUBIT MAPPING AND PAULI STRING CONSTRUCTION
# ============================================================================

def q_index(site: int, spin: str, L: int) -> int:
    """
    Map lattice site and spin to qubit index.

    Convention: [site0↑, site0↓, site1↑, site1↓, ..., site(L-1)↑, site(L-1)↓]

    Parameters:
        site: Lattice site index (0 to L-1)
        spin: 'up' or 'down'
        L: Number of lattice sites

    Returns:
        Qubit index in [0, 2*L-1]
    """
    return 2 * site + (0 if spin == 'up' else 1)


def label_from_ops(N: int, ops: Dict[int, str]) -> str:
    """
    Build Pauli string with specified operators at given qubits.

    Qiskit convention: rightmost character = qubit 0.

    Parameters:
        N: Total number of qubits
        ops: Dict mapping qubit index -> Pauli operator ('X', 'Y', 'Z')

    Returns:
        Pauli string of length N
    """
    pauli_list = ['I'] * N
    for q, op in ops.items():
        pauli_list[N - 1 - q] = op
    return ''.join(pauli_list)


def add_term(H: SparsePauliOp, label: str, coeff: complex) -> SparsePauliOp:
    """Add a Pauli term to Hamiltonian."""
    term = SparsePauliOp(label, coeff)
    return (H + term).simplify()


# ============================================================================
# JORDAN-WIGNER MAPPING UTILITIES
# ============================================================================

def jw_number_op(q: int, N: int) -> SparsePauliOp:
    """
    Number operator for mode q: n_q = (I - Z_q)/2

    Parameters:
        q: Qubit (mode) index
        N: Total number of qubits

    Returns:
        SparsePauliOp representing n_q
    """
    I_label = 'I' * N
    Z_ops = {q: 'Z'}
    Z_label = label_from_ops(N, Z_ops)

    return SparsePauliOp.from_list([
        (I_label, 0.5),
        (Z_label, -0.5)
    ]).simplify()


def jw_sz_op(site: int, L: int) -> SparsePauliOp:
    """
    Local spin-z operator: S^z_i = (n_{i↑} - n_{i↓})/2

    Parameters:
        site: Lattice site index
        L: Number of lattice sites

    Returns:
        SparsePauliOp representing S^z at site
    """
    N = 2 * L
    up_q = q_index(site, 'up', L)
    dn_q = q_index(site, 'down', L)

    n_up = jw_number_op(up_q, N)
    n_dn = jw_number_op(dn_q, N)

    return ((n_up - n_dn) * 0.5).simplify()


def jw_hop_two_fermion(p: int, q: int, N: int, coeff: float) -> SparsePauliOp:
    """
    Jordan-Wigner hopping term: coeff * (c†_p c_q + c†_q c_p)

    Under JW this becomes:
        (coeff/2) * (X_p Z_{p+1...q-1} X_q + Y_p Z_{p+1...q-1} Y_q)

    Parameters:
        p, q: Mode indices (assumes p < q without loss of generality)
        N: Total number of qubits
        coeff: Hopping amplitude

    Returns:
        SparsePauliOp representing Hermitian hopping term
    """
    a = min(p, q)
    b = max(p, q)

    # Build XX term: X_a Z_{a+1...b-1} X_b
    ops_x = {a: 'X', b: 'X'}
    for k in range(a + 1, b):
        ops_x[k] = 'Z'
    label_x = label_from_ops(N, ops_x)

    # Build YY term: Y_a Z_{a+1...b-1} Y_b
    ops_y = {a: 'Y', b: 'Y'}
    for k in range(a + 1, b):
        ops_y[k] = 'Z'
    label_y = label_from_ops(N, ops_y)

    return SparsePauliOp.from_list([
        (label_x, coeff * 0.5),
        (label_y, coeff * 0.5)
    ]).simplify()


# ============================================================================
# SSH-HUBBARD HAMILTONIAN BUILDER
# ============================================================================

def ssh_hubbard_hamiltonian(L: int = 8, t1: float = 1.0, t2: float = 0.6,
                           U: float = 2.0, periodic: bool = False) -> SparsePauliOp:
    """
    Construct SSH-Hubbard Hamiltonian:
        H = -Σ_{<i,j>,σ} t_{ij} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}

    SSH pattern: t1 on even bonds (0-1, 2-3, ...), t2 on odd bonds (1-2, 3-4, ...)

    Parameters:
        L: Number of lattice sites
        t1: Strong hopping amplitude
        t2: Weak hopping amplitude
        U: Hubbard on-site interaction
        periodic: Use periodic boundary conditions

    Returns:
        SparsePauliOp representing the full Hamiltonian
    """
    N = 2 * L  # Total qubits
    H = SparsePauliOp("I" * N, 0.0)

    # Hopping terms for both spins
    for spin in ['up', 'down']:
        # Nearest-neighbor bonds
        for i in range(L - 1):
            t = t1 if i % 2 == 0 else t2
            p = q_index(i, spin, L)
            q = q_index(i + 1, spin, L)
            H += jw_hop_two_fermion(p, q, N, -t)

        # Periodic boundary condition
        if periodic:
            t = t2 if (L - 1) % 2 == 1 else t1
            p = q_index(L - 1, spin, L)
            q = q_index(0, spin, L)
            H += jw_hop_two_fermion(p, q, N, -t)

    # Hubbard interaction: U n_{i↑} n_{i↓}
    # Expand: U n_up n_dn = U/4 (I - Z_up)(I - Z_dn)
    #                     = U/4 (I - Z_up - Z_dn + Z_up Z_dn)
    for i in range(L):
        up_q = q_index(i, 'up', L)
        dn_q = q_index(i, 'down', L)

        I_label = 'I' * N
        Z_up_label = label_from_ops(N, {up_q: 'Z'})
        Z_dn_label = label_from_ops(N, {dn_q: 'Z'})
        ZZ_label = label_from_ops(N, {up_q: 'Z', dn_q: 'Z'})

        H = add_term(H, I_label, U / 4.0)
        H = add_term(H, Z_up_label, -U / 4.0)
        H = add_term(H, Z_dn_label, -U / 4.0)
        H = add_term(H, ZZ_label, U / 4.0)

    return H.simplify()


# ============================================================================
# OBSERVABLE OPERATORS
# ============================================================================

def double_occupancy_operator(L: int) -> SparsePauliOp:
    """
    Average double occupancy: (1/L) Σ_i n_{i↑} n_{i↓}
    """
    N = 2 * L
    op = SparsePauliOp("I" * N, 0.0)

    for i in range(L):
        up_q = q_index(i, 'up', L)
        dn_q = q_index(i, 'down', L)

        # n_up * n_dn = 1/4 (I - Z_up)(I - Z_dn)
        I_label = 'I' * N
        Z_up = label_from_ops(N, {up_q: 'Z'})
        Z_dn = label_from_ops(N, {dn_q: 'Z'})
        ZZ = label_from_ops(N, {up_q: 'Z', dn_q: 'Z'})

        op = add_term(op, I_label, 0.25)
        op = add_term(op, Z_up, -0.25)
        op = add_term(op, Z_dn, -0.25)
        op = add_term(op, ZZ, 0.25)

    return (op * (1.0 / L)).simplify()


def spin_correlation_nn_operator(L: int, periodic: bool = False) -> SparsePauliOp:
    """
    Nearest-neighbor spin correlation: (1/N_bonds) Σ_{<i,j>} S^z_i S^z_j
    """
    bonds = []
    for i in range(L - 1):
        bonds.append((i, i + 1))
    if periodic:
        bonds.append((L - 1, 0))

    N_bonds = len(bonds)
    op = SparsePauliOp("I" * (2 * L), 0.0)

    for i, j in bonds:
        Sz_i = jw_sz_op(i, L)
        Sz_j = jw_sz_op(j, L)
        op += (Sz_i @ Sz_j)

    return (op * (1.0 / N_bonds)).simplify()


def bond_order_operator(i: int, j: int, L: int, spin: str) -> SparsePauliOp:
    """
    Bond order: c†_{i,σ} c_{j,σ} + h.c.
    """
    N = 2 * L
    p = q_index(i, spin, L)
    q = q_index(j, spin, L)
    return jw_hop_two_fermion(p, q, N, 1.0)


def dimer_order_operator(L: int, periodic: bool = False) -> SparsePauliOp:
    """
    Dimer order parameter: (1/L) Σ_i (-1)^i B_i
    where B_i = (B_{i,up} + B_{i,down})/2 is spin-averaged bond order
    """
    bonds = []
    for i in range(L - 1):
        bonds.append((i, i + 1))
    if periodic:
        bonds.append((L - 1, 0))

    op = SparsePauliOp("I" * (2 * L), 0.0)

    for idx, (i, j) in enumerate(bonds):
        B_up = bond_order_operator(i, j, L, 'up')
        B_dn = bond_order_operator(i, j, L, 'down')
        B_avg = ((B_up + B_dn) * 0.5).simplify()

        sign = (-1) ** idx
        op += (B_avg * sign)

    return (op * (1.0 / L)).simplify()


def edge_density_operator(L: int) -> Tuple[SparsePauliOp, SparsePauliOp]:
    """
    Edge densities: (n_left, n_right)
    n_left = n_{0,up} + n_{0,down}
    n_right = n_{L-1,up} + n_{L-1,down}
    """
    N = 2 * L

    n_left = (jw_number_op(q_index(0, 'up', L), N) +
              jw_number_op(q_index(0, 'down', L), N)).simplify()

    n_right = (jw_number_op(q_index(L - 1, 'up', L), N) +
               jw_number_op(q_index(L - 1, 'down', L), N)).simplify()

    return n_left, n_right


def edge_spin_correlation_operator(L: int) -> SparsePauliOp:
    """
    Edge spin correlation: S^z_0 S^z_{L-1}
    """
    Sz_left = jw_sz_op(0, L)
    Sz_right = jw_sz_op(L - 1, L)
    return (Sz_left @ Sz_right).simplify()


def structure_factor_szz_operator(L: int, q: float) -> SparsePauliOp:
    """
    Structure factor: S^{zz}(q) = (1/L) Σ_{i,j} exp(iq(i-j)) S^z_i S^z_j
    (keeping only real part of phase)
    """
    op = SparsePauliOp("I" * (2 * L), 0.0)

    for i in range(L):
        for j in range(L):
            phase = np.exp(1j * q * (i - j))
            Sz_i = jw_sz_op(i, L)
            Sz_j = jw_sz_op(j, L)
            op += ((Sz_i @ Sz_j) * phase.real)

    return (op * (1.0 / L)).simplify()


# ============================================================================
# TOPOLOGICAL DIAGNOSTICS
# ============================================================================

def compute_concurrence_2qubit(rho: np.ndarray) -> float:
    """
    Wootters concurrence for a 2-qubit density matrix.

    C = max(0, λ1 - λ2 - λ3 - λ4)
    where λi are eigenvalues (descending) of R = sqrt(sqrt(ρ) ρ_tilde sqrt(ρ))
    and ρ_tilde = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
    """
    if rho.shape != (4, 4):
        return 0.0

    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y_tensor = np.kron(sigma_y, sigma_y)

    # Spin-flipped density matrix
    rho_tilde = Y_tensor @ np.conj(rho) @ Y_tensor

    # Compute R matrix
    sqrt_rho = np.linalg.matrix_power(rho, 1)  # Actually need proper sqrt
    # For proper implementation:
    evals, evecs = np.linalg.eigh(rho)
    evals = np.maximum(evals, 0)
    sqrt_rho = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    R = sqrt_rho @ rho_tilde @ sqrt_rho

    # Eigenvalues of R
    lambdas = np.linalg.eigvalsh(R)
    lambdas = np.sqrt(np.maximum(lambdas, 0))
    lambdas = np.sort(lambdas)[::-1]

    C = max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
    return float(C)


def compute_edge_concurrence(state: Statevector, L: int) -> float:
    """
    Average edge concurrence for up and down spins.

    Computes concurrence between qubits at sites 0 and L-1 for each spin,
    then averages.
    """
    N = 2 * L
    qL_up = q_index(0, 'up', L)
    qR_up = q_index(L - 1, 'up', L)
    qL_dn = q_index(0, 'down', L)
    qR_dn = q_index(L - 1, 'down', L)

    # Get full density matrix
    rho_full = DensityMatrix(state).data

    # Trace out all qubits except the two edge qubits for spin-up
    keep_qubits_up = [qL_up, qR_up]
    other_qubits_up = [q for q in range(N) if q not in keep_qubits_up]
    rho_edge_up = partial_trace(DensityMatrix(rho_full), other_qubits_up).data
    C_up = compute_concurrence_2qubit(rho_edge_up)

    # Same for spin-down
    keep_qubits_dn = [qL_dn, qR_dn]
    other_qubits_dn = [q for q in range(N) if q not in keep_qubits_dn]
    rho_edge_dn = partial_trace(DensityMatrix(rho_full), other_qubits_dn).data
    C_dn = compute_concurrence_2qubit(rho_edge_dn)

    return (C_up + C_dn) / 2.0


def compute_bond_purity(state: Statevector, site_i: int, site_j: int,
                       L: int, spin: str) -> float:
    """
    Purity Tr(ρ²) of 2-qubit reduced density matrix for a bond.
    """
    N = 2 * L
    qi = q_index(site_i, spin, L)
    qj = q_index(site_j, spin, L)

    rho_full = DensityMatrix(state).data
    keep_qubits = [qi, qj]
    other_qubits = [q for q in range(N) if q not in keep_qubits]
    rho_bond = partial_trace(DensityMatrix(rho_full), other_qubits).data

    purity = np.trace(rho_bond @ rho_bond).real
    return float(purity)


# ============================================================================
# OBSERVABLE MEASUREMENT AND COMPARISON
# ============================================================================

def measure_observable(state: Statevector, operator: SparsePauliOp) -> float:
    """Compute expectation value of operator in given state."""
    return float(state.expectation_value(operator).real)


def compute_vqe_observables(state: Statevector, L: int, t1: float, t2: float,
                           U: float, periodic: bool = False,
                           measure_structure_factor: bool = False,
                           compute_topology: bool = False) -> Dict:
    """
    Compute all observables from a state.

    Parameters:
        state: Quantum state (Statevector)
        L, t1, t2, U, periodic: Model parameters
        measure_structure_factor: Whether to compute S^zz(π)
        compute_topology: Whether to compute edge concurrence and bond purity

    Returns:
        Dictionary of observables
    """
    obs = {}

    # Rebuild Hamiltonian
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic)

    # Energy per site
    energy_total = measure_observable(state, H)
    obs['energy'] = energy_total / L

    # Energy variance
    H_matrix = H.to_matrix()
    state_vec = state.data
    E = np.vdot(state_vec, H_matrix @ state_vec).real
    E2 = np.vdot(state_vec, H_matrix @ H_matrix @ state_vec).real
    obs['varH'] = E2 - E ** 2

    # Z-only observables
    obs['double_occupancy'] = measure_observable(state, double_occupancy_operator(L))
    obs['SzSz_nn'] = measure_observable(state, spin_correlation_nn_operator(L, periodic))

    if measure_structure_factor:
        obs['Szz_pi'] = measure_observable(state, structure_factor_szz_operator(L, np.pi))
    else:
        obs['Szz_pi'] = None

    # XY-type: bond orders
    bonds = [(i, i + 1) for i in range(L - 1)]
    if periodic:
        bonds.append((L - 1, 0))

    bond_orders = []
    for i, j in bonds:
        B_up = bond_order_operator(i, j, L, 'up')
        B_dn = bond_order_operator(i, j, L, 'down')
        b_up = measure_observable(state, B_up)
        b_dn = measure_observable(state, B_dn)
        bond_orders.append((b_up + b_dn) / 2.0)

    obs['bond_orders'] = bond_orders
    obs['D_dimer'] = measure_observable(state, dimer_order_operator(L, periodic))

    # Edge diagnostics (only for open boundaries)
    if not periodic:
        n_left_op, n_right_op = edge_density_operator(L)
        n_left = measure_observable(state, n_left_op)
        n_right = measure_observable(state, n_right_op)
        obs['edge_n'] = (n_left + n_right) / 2.0
        obs['edge_SzSz'] = measure_observable(state, edge_spin_correlation_operator(L))
    else:
        obs['edge_n'] = None
        obs['edge_SzSz'] = None

    # Topological diagnostics (statevector only)
    if compute_topology:
        obs['C_edge'] = compute_edge_concurrence(state, L)

        # Bond purity for first strong bond (0-1) and first weak bond (1-2)
        if L >= 3:
            obs['P_strong'] = compute_bond_purity(state, 0, 1, L, 'up')
            obs['P_weak'] = compute_bond_purity(state, 1, 2, L, 'up')
        else:
            obs['P_strong'] = 0.0
            obs['P_weak'] = 0.0
    else:
        obs['C_edge'] = None
        obs['P_strong'] = 0.0
        obs['P_weak'] = 0.0

    return obs


def print_observable_comparison(vqe_obs: Dict, ed_obs: Optional[Dict] = None):
    """Pretty-print observable values and comparison with ED."""
    print("\n" + "=" * 70)
    print("OBSERVABLE RESULTS")
    print("=" * 70)

    # Z-only group
    print("\n--- Z-basis Observables ---")
    print(f"  Energy/site:        {vqe_obs['energy']:.8f}")
    print(f"  Energy variance:    {vqe_obs['varH']:.8e}")
    print(f"  Double occupancy:   {vqe_obs['double_occupancy']:.6f}")
    print(f"  <S^z S^z>_NN:       {vqe_obs['SzSz_nn']:.6f}")
    if vqe_obs['Szz_pi'] is not None:
        print(f"  S^zz(π):            {vqe_obs['Szz_pi']:.6f}")

    # XY group
    print("\n--- XY-basis Observables ---")
    print(f"  Dimer order D:      {vqe_obs['D_dimer']:.6f}")
    print("  Bond orders:")
    for idx, b in enumerate(vqe_obs['bond_orders']):
        bond_type = "strong" if idx % 2 == 0 else "weak"
        print(f"    Bond {idx} ({bond_type}):  {b:.6f}")

    # Edge diagnostics
    if vqe_obs['edge_n'] is not None:
        print("\n--- Edge Diagnostics (Open BC) ---")
        print(f"  Edge density:       {vqe_obs['edge_n']:.6f}")
        print(f"  Edge S^z corr:      {vqe_obs['edge_SzSz']:.6f}")

    # Topological diagnostics
    if vqe_obs['C_edge'] is not None:
        print("\n--- Topological Diagnostics ---")
        print(f"  Edge concurrence:   {vqe_obs['C_edge']:.6f}")
        print(f"  Strong bond purity: {vqe_obs['P_strong']:.6f}")
        print(f"  Weak bond purity:   {vqe_obs['P_weak']:.6f}")

    # Comparison table with ED
    if ed_obs is not None:
        print("\n" + "=" * 70)
        print("VQE vs ED COMPARISON")
        print("=" * 70)
        print(f"{'Observable':<20} {'VQE':>12} {'ED':>12} {'Error':>12} {'Rel%':>10}")
        print("-" * 70)

        metrics = [
            ('Energy/site', 'energy'),
            ('Double occ', 'double_occupancy'),
            ('<SzSz>_NN', 'SzSz_nn'),
            ('D_dimer', 'D_dimer'),
        ]

        if vqe_obs['Szz_pi'] is not None and ed_obs['Szz_pi'] is not None:
            metrics.append(('S^zz(π)', 'Szz_pi'))

        for name, key in metrics:
            vqe_val = vqe_obs[key]
            ed_val = ed_obs[key]
            err = abs(vqe_val - ed_val)
            rel = 100 * err / abs(ed_val) if abs(ed_val) > 1e-10 else 0.0
            print(f"{name:<20} {vqe_val:>12.6f} {ed_val:>12.6f} {err:>12.2e} {rel:>9.2f}%")


# ============================================================================
# ANSATZ BUILDERS
# ============================================================================

def build_ansatz_hea(N: int, depth: int) -> QuantumCircuit:
    """
    Hardware-Efficient Ansatz using EfficientSU2.

    Parameters:
        N: Number of qubits
        depth: Number of repetitions

    Returns:
        Parameterized quantum circuit
    """
    if HAS_EFFICIENT_SU2_FUNC:
        ansatz = efficient_su2(
            num_qubits=N,
            reps=depth,
            entanglement='linear',
            su2_gates=['ry', 'rz'],
            insert_barriers=False
        )
    else:
        ansatz = EfficientSU2(
            num_qubits=N,
            reps=depth,
            entanglement='linear',
            su2_gates=['ry', 'rz'],
            insert_barriers=False
        )

    return ansatz


def build_ansatz_hva_sshh(L: int, reps: int, t1: float, t2: float,
                         include_U: bool = True) -> QuantumCircuit:
    """
    Hamiltonian-Variational Ansatz for SSH-Hubbard.

    Each block consists of:
      1. Hopping layer for t1 bonds (even)
      2. Hopping layer for t2 bonds (odd)
      3. On-site U interaction layer

    Uses number-conserving XX+YY gates for hopping and ZZ for interaction.

    Parameters:
        L: Number of lattice sites
        reps: Number of repetitions
        t1, t2: SSH hopping parameters (for structure, not values)
        include_U: Whether to include Hubbard U layer

    Returns:
        Parameterized quantum circuit
    """
    N = 2 * L
    qc = QuantumCircuit(N)

    param_idx = 0

    for rep in range(reps):
        # Layer 1: Even bonds (strong, t1)
        for i in range(0, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                theta = Parameter(f'θ_t1_{rep}_{i}_{spin}')

                # XX+YY gate: exp(-i θ/2 (XX+YY))
                # Fallback: RXX + RYY with same angle
                try:
                    qc.rxx(theta, qi, qj)
                    qc.ryy(theta, qi, qj)
                except:
                    # Alternative implementation
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(theta, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Layer 2: Odd bonds (weak, t2)
        for i in range(1, L - 1, 2):
            for spin in ['up', 'down']:
                qi = q_index(i, spin, L)
                qj = q_index(i + 1, spin, L)
                theta = Parameter(f'θ_t2_{rep}_{i}_{spin}')

                try:
                    qc.rxx(theta, qi, qj)
                    qc.ryy(theta, qi, qj)
                except:
                    qc.h(qi)
                    qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(theta, qj)
                    qc.cx(qi, qj)
                    qc.h(qi)
                    qc.h(qj)

        # Layer 3: On-site Hubbard U (ZZ between up and down at each site)
        if include_U:
            for i in range(L):
                qi_up = q_index(i, 'up', L)
                qi_dn = q_index(i, 'down', L)
                phi = Parameter(f'φ_U_{rep}_{i}')
                qc.rzz(phi, qi_up, qi_dn)

    return qc


def build_ansatz_topo_sshh(L: int, reps: int, use_edge_link: bool = True) -> QuantumCircuit:
    """
    Problem-Inspired / Topological Ansatz for SSH-Hubbard.

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


# ============================================================================
# VQE HISTORY TRACKING
# ============================================================================

class VQEHistory:
    """Track VQE optimization progress."""

    def __init__(self):
        self.it = []
        self.energy = []

    def callback(self, eval_count, params, mean, std):
        """Callback function for VQE optimizer."""
        self.it.append(eval_count)
        self.energy.append(float(mean))


# ============================================================================
# WARM-START DELTA SWEEP
# ============================================================================

def warmstart_delta_sweep(L: int, U: float, periodic: bool,
                          ansatz_kind: str, reps: int,
                          deltas: np.ndarray, optimizer, estimator) -> List[Dict]:
    """
    Warm-start parameter sweep over dimerization δ.

    For each δ, compute t1 and t2, build Hamiltonian, run VQE starting from
    previous optimal parameters (warm-start).

    Parameters:
        L: Number of sites
        U: Hubbard interaction
        periodic: Boundary conditions
        ansatz_kind: 'hea', 'hva', or 'topoinsp'
        reps: Ansatz depth
        deltas: Array of dimerization values
        optimizer: Qiskit optimizer instance
        estimator: Qiskit estimator instance

    Returns:
        List of result dictionaries for each δ
    """
    results = []
    theta_prev = None
    t_avg = 1.0

    print("\n" + "=" * 70)
    print(f"WARM-START DELTA SWEEP ({len(deltas)} points)")
    print("=" * 70)
    print(f"Ansatz: {ansatz_kind}, Reps: {reps}, L: {L}, U: {U}")
    print("-" * 70)

    for idx, delta in enumerate(deltas):
        print(f"\n[{idx + 1}/{len(deltas)}] δ = {delta:+.4f}")

        # Compute hopping parameters
        t1 = t_avg * (1 + delta)
        t2 = t_avg * (1 - delta)
        print(f"  t1 = {t1:.4f}, t2 = {t2:.4f}")

        # Build Hamiltonian
        H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic)
        N = 2 * L

        # Build ansatz
        if ansatz_kind == 'hea':
            ansatz = build_ansatz_hea(N, reps)
        elif ansatz_kind == 'hva':
            ansatz = build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True)
        elif ansatz_kind == 'topoinsp':
            ansatz = build_ansatz_topo_sshh(L, reps, use_edge_link=True)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz_kind}")

        # Initial point: warm-start or random
        if theta_prev is None or len(theta_prev) != ansatz.num_parameters:
            np.random.seed(42 + idx)
            theta0 = 0.01 * np.random.randn(ansatz.num_parameters)
            print(f"  Initial: random ({ansatz.num_parameters} params)")
        else:
            theta0 = theta_prev
            print(f"  Initial: warm-start from previous δ")

        # Run VQE
        history = VQEHistory()
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=estimator,
            initial_point=theta0,
            callback=history.callback
        )

        result = vqe.compute_minimum_eigenvalue(operator=H)
        vqe_energy = float(result.eigenvalue.real)

        # Get optimal parameters for next iteration
        if hasattr(result, 'optimal_point'):
            theta_prev = result.optimal_point
        elif hasattr(result, 'optimal_parameters'):
            theta_prev = np.array(list(result.optimal_parameters.values()))
        else:
            theta_prev = theta0

        # Exact diagonalization for reference
        H_matrix = H.to_matrix()
        if H_matrix.shape[0] < 2000:
            eigenvalues = np.linalg.eigvalsh(H_matrix)
        else:
            from scipy.sparse.linalg import eigsh as sparse_eigsh
            eigenvalues, _ = sparse_eigsh(H, k=1, which='SA')
        ed_energy = eigenvalues[0]

        abs_err = abs(vqe_energy - ed_energy)
        rel_err = abs_err / abs(ed_energy) if abs(ed_energy) > 1e-10 else 0.0

        print(f"  VQE energy:  {vqe_energy:.8f}")
        print(f"  ED energy:   {ed_energy:.8f}")
        print(f"  Abs error:   {abs_err:.2e}")
        print(f"  Evaluations: {len(history.energy)}")

        results.append({
            'delta': delta,
            't1': t1,
            't2': t2,
            'vqe_energy': vqe_energy,
            'ed_energy': ed_energy,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'evals': len(history.energy),
            'history': history
        })

    return results


# ============================================================================
# MAIN VQE ROUTINE
# ============================================================================

def main():
    """Main VQE simulation."""

    # Command-line interface
    parser = argparse.ArgumentParser(
        description='SSH-Hubbard VQE with topology-aware ansätze',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ssh_hubbard_vqe.py --ansatz hea
  python ssh_hubbard_vqe.py --ansatz hva --reps 2
  python ssh_hubbard_vqe.py --ansatz topoinsp --reps 3
  python ssh_hubbard_vqe.py --ansatz topoinsp --delta-sweep -0.6 0.6 13
        """
    )

    parser.add_argument('--ansatz', type=str, default='hea',
                       choices=['hea', 'hva', 'topoinsp'],
                       help='Ansatz type (default: hea)')
    parser.add_argument('--reps', type=int, default=3,
                       help='Ansatz depth/repetitions (default: 3)')
    parser.add_argument('--delta-sweep', nargs=3, type=float, metavar=('START', 'END', 'NUM'),
                       help='Run delta sweep: start end num_points')
    parser.add_argument('--L', type=int, default=6,
                       help='Number of lattice sites (default: 6)')
    parser.add_argument('--U', type=float, default=2.0,
                       help='Hubbard interaction strength (default: 2.0)')
    parser.add_argument('--t1', type=float, default=1.0,
                       help='Strong hopping (default: 1.0)')
    parser.add_argument('--t2', type=float, default=0.8,
                       help='Weak hopping (default: 0.8)')
    parser.add_argument('--periodic', action='store_true',
                       help='Use periodic boundary conditions')
    parser.add_argument('--maxiter', type=int, default=300,
                       help='Max optimizer iterations (default: 300)')

    args = parser.parse_args()

    # Model parameters
    L = args.L
    U = args.U
    periodic = args.periodic

    # Create results directory
    os.makedirs('../results', exist_ok=True)

    # ========================================================================
    # DELTA SWEEP MODE
    # ========================================================================
    if args.delta_sweep:
        delta_start, delta_end, delta_num = args.delta_sweep
        deltas = np.linspace(delta_start, delta_end, int(delta_num))

        optimizer = L_BFGS_B(maxiter=args.maxiter)
        estimator = Estimator()

        results = warmstart_delta_sweep(
            L=L,
            U=U,
            periodic=periodic,
            ansatz_kind=args.ansatz,
            reps=args.reps,
            deltas=deltas,
            optimizer=optimizer,
            estimator=estimator
        )

        # Save results to CSV
        csv_path = f'../results/L{L}_{args.ansatz}_delta_sweep.csv'
        with open(csv_path, 'w') as f:
            f.write('delta,t1,t2,vqe_energy,ed_energy,abs_err,rel_err,evals\n')
            for r in results:
                f.write(f"{r['delta']:.6f},{r['t1']:.6f},{r['t2']:.6f},"
                       f"{r['vqe_energy']:.10f},{r['ed_energy']:.10f},"
                       f"{r['abs_err']:.3e},{r['rel_err']:.6f},{r['evals']}\n")
        print(f"\n✓ Results saved to {csv_path}")

        # Plot error vs delta
        fig, ax = plt.subplots(figsize=(8, 5))
        deltas_plot = [r['delta'] for r in results]
        errors = [r['abs_err'] for r in results]

        ax.semilogy(deltas_plot, errors, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Dimerization δ', fontsize=12)
        ax.set_ylabel('|E_VQE - E_ED|', fontsize=12)
        ax.set_title(f'VQE Error vs Dimerization (L={L}, {args.ansatz.upper()})', fontsize=13)
        ax.grid(True, alpha=0.3)

        plot_path = f'../results/L{L}_{args.ansatz}_delta_error.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Plot saved to {plot_path}")

        return

    # ========================================================================
    # SINGLE-POINT VQE MODE
    # ========================================================================

    t1 = args.t1
    t2 = args.t2
    delta_t = (t1 - t2) / (t1 + t2)

    print("\n" + "=" * 70)
    print("SSH-HUBBARD VQE SIMULATION")
    print("=" * 70)
    print(f"System: L={L} sites, N={2*L} qubits")
    print(f"Ansatz: {args.ansatz.upper()}, Reps={args.reps}")
    print(f"Parameters: t1={t1}, t2={t2}, U={U}, δ={delta_t:.4f}")
    print(f"Boundary: {'Periodic' if periodic else 'Open'}")
    print("=" * 70)

    # Build Hamiltonian
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic)
    N = 2 * L
    print(f"\nHamiltonian: {len(H)} Pauli terms on {N} qubits")

    # Exact diagonalization
    print("\n--- Exact Diagonalization ---")
    H_matrix = H.to_matrix()
    print(f"Hilbert space dimension: {H_matrix.shape[0]}")

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    exact_energy = eigenvalues[0]
    exact_state = Statevector(np.ascontiguousarray(eigenvectors[:, 0]))

    print(f"Ground state energy: {exact_energy:.10f}")
    print(f"Energy per site:     {exact_energy / L:.10f}")

    # ED observables
    measure_sf = (L <= 8)
    ed_obs = compute_vqe_observables(
        exact_state, L, t1, t2, U, periodic,
        measure_structure_factor=measure_sf,
        compute_topology=True
    )

    # Build ansatz
    print(f"\n--- Building {args.ansatz.upper()} Ansatz ---")
    if args.ansatz == 'hea':
        ansatz = build_ansatz_hea(N, args.reps)
    elif args.ansatz == 'hva':
        ansatz = build_ansatz_hva_sshh(L, args.reps, t1, t2, include_U=True)
    elif args.ansatz == 'topoinsp':
        ansatz = build_ansatz_topo_sshh(L, args.reps, use_edge_link=True)
    else:
        raise ValueError(f"Unknown ansatz: {args.ansatz}")

    print(f"Circuit depth:  {ansatz.depth()}")
    print(f"Parameters:     {ansatz.num_parameters}")
    print(f"Qubits:         {ansatz.num_qubits}")

    # Try to save circuit diagram
    try:
        fig = ansatz.decompose().draw('mpl', style='iqp', fold=-1)
        fig_path = f'../results/L{L}_{args.ansatz}_circuit.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Circuit saved to {fig_path}")
    except Exception as e:
        print(f"  (Could not save circuit diagram: {e})")

    # Initial parameters
    np.random.seed(42)
    theta0 = 0.01 * np.random.randn(ansatz.num_parameters)

    # Run VQE
    print("\n--- Running VQE Optimization ---")
    estimator = Estimator()
    optimizer = L_BFGS_B(maxiter=args.maxiter)
    history = VQEHistory()

    vqe = VQE(
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        initial_point=theta0,
        callback=history.callback
    )

    result = vqe.compute_minimum_eigenvalue(operator=H)
    vqe_energy = float(result.eigenvalue.real)

    # Build optimized state
    if hasattr(result, 'optimal_point'):
        optimal_params = result.optimal_point
    elif hasattr(result, 'optimal_parameters'):
        optimal_params = np.array(list(result.optimal_parameters.values()))
    else:
        optimal_params = theta0

    vqe_circuit = ansatz.assign_parameters(optimal_params)
    vqe_state = Statevector.from_instruction(vqe_circuit)

    # Compute VQE observables
    vqe_obs = compute_vqe_observables(
        vqe_state, L, t1, t2, U, periodic,
        measure_structure_factor=measure_sf,
        compute_topology=True
    )

    # Print results
    print("\n" + "=" * 70)
    print("ENERGY COMPARISON")
    print("=" * 70)
    abs_err = abs(vqe_energy - exact_energy)
    rel_err = 100 * abs_err / abs(exact_energy)
    print(f"ED energy:      {exact_energy:.10f}")
    print(f"VQE energy:     {vqe_energy:.10f}")
    print(f"Absolute error: {abs_err:.3e}")
    print(f"Relative error: {rel_err:.4f}%")
    print(f"Parameters:     {ansatz.num_parameters}")
    print(f"Evaluations:    {len(history.energy)}")

    # Print observables
    print_observable_comparison(vqe_obs, ed_obs)

    # Measurement grouping summary
    print("\n" + "=" * 70)
    print("MEASUREMENT STRATEGY")
    print("=" * 70)
    print("Z-only group (1 circuit):")
    print("  - Energy, variance, double occupancy, spin correlations, edge densities")
    print("\nXY group (~L circuits):")
    print("  - Bond orders, dimer order")
    print("\nTopology group (statevector only):")
    print("  - Edge concurrence, bond purity")
    print(f"\nApproximate measurement count: {1 + L}")

    # Convergence plots
    if len(history.energy) > 0:
        try:
            print("\n--- Generating Convergence Plots ---")

            it = np.arange(1, len(history.energy) + 1)
            Ek = np.array(history.energy)
            abs_err_hist = np.abs(Ek - exact_energy)

            # Plot 1: Energy convergence
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(it, Ek, 'b-', linewidth=2, label='VQE Energy')
            ax.axhline(exact_energy, color='r', linestyle='--', linewidth=2, label='ED Energy')
            ax.set_xlabel('Evaluation', fontsize=12)
            ax.set_ylabel('Energy', fontsize=12)
            ax.set_title(f'VQE Convergence (L={L}, {args.ansatz.upper()})', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            conv_path = f'../results/L{L}_{args.ansatz}_energy_convergence.png'
            plt.savefig(conv_path, dpi=150)
            plt.close()
            print(f"✓ Energy convergence: {conv_path}")

            # Plot 2: Error convergence (log scale)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(it, abs_err_hist, 'b-', linewidth=2)
            ax.set_xlabel('Evaluation', fontsize=12)
            ax.set_ylabel('|E_VQE - E_ED|', fontsize=12)
            ax.set_title(f'VQE Error (log scale) - L={L}, {args.ansatz.upper()}', fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            err_path = f'../results/L{L}_{args.ansatz}_error_log.png'
            plt.savefig(err_path, dpi=150)
            plt.close()
            print(f"✓ Error convergence:  {err_path}")

            # Summary statistics
            initial_err = abs_err_hist[0]
            final_err = abs_err_hist[-1]
            reduction = initial_err / final_err if final_err > 0 else np.inf

            print(f"\nConvergence summary:")
            print(f"  Initial error: {initial_err:.3e}")
            print(f"  Final error:   {final_err:.3e}")
            print(f"  Reduction:     {reduction:.1f}x")

        except Exception as e:
            print(f"Could not generate plots: {e}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
