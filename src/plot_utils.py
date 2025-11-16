#!/usr/bin/env python3
"""
Plotting utilities for VQE convergence analysis.

Shared functions used across main VQE script and benchmark scripts.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import List, Optional


def plot_vqe_convergence(
    energy_history: List[float],
    exact_energy: float,
    ansatz_name: str,
    L: int,
    output_dir: str = '../results',
    prefix: str = '',
    show_stats: bool = True
) -> tuple:
    """
    Generate convergence plots for VQE optimization.

    Creates two plots:
    1. Energy convergence: VQE energy vs evaluation count
    2. Error convergence: Absolute error vs evaluation count (log scale)

    Parameters
    ----------
    energy_history : List[float]
        VQE energy at each evaluation
    exact_energy : float
        Exact ground state energy (from ED or DMRG)
    ansatz_name : str
        Name of ansatz (for plot titles and filenames)
    L : int
        Number of lattice sites
    output_dir : str, optional
        Directory to save plots (default: '../results')
    prefix : str, optional
        Prefix for output filenames (default: '')
    show_stats : bool, optional
        Print convergence statistics (default: True)

    Returns
    -------
    (energy_path, error_path) : tuple
        Paths to saved plot files

    Examples
    --------
    >>> history = [−4.5, −4.7, −4.8, −4.82]
    >>> plot_vqe_convergence(history, −4.823, 'hea', 6)
    ('results/L6_hea_energy_convergence.png', 'results/L6_hea_error_log.png')
    """
    if len(energy_history) == 0:
        if show_stats:
            print("  Warning: No energy history to plot")
        return None, None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    it = np.arange(1, len(energy_history) + 1)
    Ek = np.array(energy_history)
    rel_err_hist = 100 * np.abs(Ek - exact_energy) / abs(exact_energy)

    # Generate filename prefix
    if prefix:
        file_prefix = f"{prefix}_L{L}_{ansatz_name}"
    else:
        file_prefix = f"L{L}_{ansatz_name}"

    # Plot 1: Energy convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(it, Ek, 'b-', linewidth=2, label='VQE Energy')
    ax.axhline(exact_energy, color='r', linestyle='--', linewidth=2, label='Exact Energy')
    ax.set_xlabel('Evaluation', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'VQE Convergence (L={L}, {ansatz_name.upper()})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    energy_path = os.path.join(output_dir, f'{file_prefix}_energy_convergence.png')
    plt.savefig(energy_path, dpi=150)
    plt.close()

    # Plot 2: Error convergence (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(it, rel_err_hist, 'b-', linewidth=2)
    ax.set_xlabel('Evaluation', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title(f'VQE Relative Error (log scale) - L={L}, {ansatz_name.upper()}', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    error_path = os.path.join(output_dir, f'{file_prefix}_error_log.png')
    plt.savefig(error_path, dpi=150)
    plt.close()

    # Print statistics
    if show_stats:
        initial_err = rel_err_hist[0]
        final_err = rel_err_hist[-1]
        reduction = initial_err / final_err if final_err > 1e-15 else np.inf

        print(f"  ✓ Convergence plots saved:")
        print(f"      Energy: {os.path.basename(energy_path)}")
        print(f"      Error:  {os.path.basename(error_path)}")
        print(f"  Convergence: {initial_err:.2f}% → {final_err:.2f}% ({reduction:.1f}x reduction)")

    return energy_path, error_path


def plot_multi_ansatz_comparison(
    results_dict: dict,
    L: int,
    exact_energy: float,
    output_dir: str = '../results',
    filename: str = 'ansatz_comparison.png'
) -> str:
    """
    Generate comparison plot of convergence for multiple ansätze.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping ansatz_name -> energy_history list
    L : int
        Number of lattice sites
    exact_energy : float
        Exact ground state energy
    output_dir : str, optional
        Directory to save plot (default: '../results')
    filename : str, optional
        Output filename (default: 'ansatz_comparison.png')

    Returns
    -------
    output_path : str
        Path to saved plot file

    Examples
    --------
    >>> results = {'hea': [−4.5, −4.8], 'hva': [−4.6, −4.82]}
    >>> plot_multi_ansatz_comparison(results, 6, −4.823)
    'results/ansatz_comparison.png'
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for idx, (ansatz_name, energy_history) in enumerate(results_dict.items()):
        if len(energy_history) == 0:
            continue

        it = np.arange(1, len(energy_history) + 1)
        Ek = np.array(energy_history)
        rel_err = 100 * np.abs(Ek - exact_energy) / abs(exact_energy)

        # Energy plot
        ax1.plot(it, Ek, '-', linewidth=2, label=ansatz_name.upper(),
                 color=colors[idx])

        # Error plot
        ax2.semilogy(it, rel_err, '-', linewidth=2, label=ansatz_name.upper(),
                     color=colors[idx])

    # Exact energy reference
    ax1.axhline(exact_energy, color='k', linestyle='--', linewidth=1.5,
                label='Exact', alpha=0.7)

    # Configure energy plot
    ax1.set_xlabel('Evaluation', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Energy Convergence Comparison (L={L})', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Configure error plot
    ax2.set_xlabel('Evaluation', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title(f'Relative Error Convergence (log scale, L={L})', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Multi-ansatz comparison saved: {os.path.basename(output_path)}")

    return output_path


def plot_multistart_convergence(
    per_seed_results: List[dict],
    exact_energy: float,
    ansatz_name: str,
    optimizer_name: str,
    L: int,
    output_dir: str = '../docs/images',
    show_stats: bool = True
) -> str:
    """
    Plot convergence curves for multi-start VQE runs.

    Shows all seed runs with different transparency, highlights best seed,
    and displays mean ± std bands.

    Parameters
    ----------
    per_seed_results : List[dict]
        List of individual VQE run results, each with 'energy_history' and 'seed'
    exact_energy : float
        Exact ground state energy
    ansatz_name : str
        Name of ansatz (for plot titles and filenames)
    optimizer_name : str
        Name of optimizer (L_BFGS_B, COBYLA, SLSQP)
    L : int
        Number of lattice sites
    output_dir : str, optional
        Directory to save plots (default: '../docs/images')
    show_stats : bool, optional
        Print convergence statistics (default: True)

    Returns
    -------
    output_path : str
        Path to saved plot file

    Examples
    --------
    >>> results = [{'energy_history': [...], 'seed': 0, 'energy': -4.82}, ...]
    >>> plot_multistart_convergence(results, -4.823, 'hea', 'L_BFGS_B', 4)
    '../docs/images/convergence_hea_L_BFGS_B_L4.png'
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Find best seed (lowest final energy)
    energies = [r['energy'] for r in per_seed_results]
    best_idx = int(np.argmin(energies))

    # Track max iteration count for alignment
    max_iters = max(len(r['energy_history']) for r in per_seed_results)

    # Prepare data for mean/std calculation
    # Pad shorter histories with their final value
    all_histories = []
    for result in per_seed_results:
        hist = result['energy_history']
        if len(hist) < max_iters:
            # Pad with final value
            padded = hist + [hist[-1]] * (max_iters - len(hist))
        else:
            padded = hist
        all_histories.append(padded)

    all_histories = np.array(all_histories)
    mean_energy = np.mean(all_histories, axis=0)
    std_energy = np.std(all_histories, axis=0)

    # Plot individual seeds
    for idx, result in enumerate(per_seed_results):
        history = result['energy_history']
        seed = result['seed']
        it = np.arange(1, len(history) + 1)

        if idx == best_idx:
            # Highlight best seed
            ax1.plot(it, history, '-', linewidth=2, alpha=0.9,
                    color='blue', label=f'Seed {seed} (best)')

            rel_err = 100 * np.abs(np.array(history) - exact_energy) / abs(exact_energy)
            ax2.semilogy(it, rel_err, '-', linewidth=2, alpha=0.9,
                        color='blue', label=f'Seed {seed} (best)')
        else:
            # Other seeds with transparency
            ax1.plot(it, history, '-', linewidth=1, alpha=0.3,
                    color='gray', label=f'Seed {seed}' if idx == 0 else '')

            rel_err = 100 * np.abs(np.array(history) - exact_energy) / abs(exact_energy)
            ax2.semilogy(it, rel_err, '-', linewidth=1, alpha=0.3,
                        color='gray', label=f'Other seeds' if idx == 0 else '')

    # Plot mean ± std band
    it_full = np.arange(1, max_iters + 1)
    ax1.plot(it_full, mean_energy, 'r--', linewidth=2, alpha=0.7, label='Mean')
    ax1.fill_between(it_full, mean_energy - std_energy, mean_energy + std_energy,
                     alpha=0.2, color='red', label='±1 std')

    # Exact energy reference
    ax1.axhline(exact_energy, color='green', linestyle='--', linewidth=1.5,
               label='Exact', alpha=0.7)

    # Configure energy plot
    ax1.set_xlabel('Evaluation', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Multi-Start Convergence: {ansatz_name.upper()} ({optimizer_name})\nL={L}',
                 fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Configure error plot
    ax2.set_xlabel('Evaluation', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title(f'Relative Error Convergence (log scale)\n{ansatz_name.upper()} ({optimizer_name}), L={L}',
                 fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    filename = f'convergence_{ansatz_name}_{optimizer_name}_L{L}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()

    if show_stats:
        best_energy = energies[best_idx]
        mean_final = mean_energy[-1]
        std_final = std_energy[-1]

        print(f"  ✓ Multi-start convergence plot saved: {filename}")
        print(f"    Best energy:  {best_energy:.10f} (seed {per_seed_results[best_idx]['seed']})")
        print(f"    Mean ± std:   {mean_final:.10f} ± {std_final:.3e}")

    return output_path
