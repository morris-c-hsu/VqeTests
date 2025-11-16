#!/usr/bin/env python3
"""
SSH-Hubbard Parameter Sweep for L=4

Comprehensive sweep over dimerization (δ) and interaction (U) parameters
using multi-start VQE with 3 optimizers and 5 seeds.

Parameter ranges:
- δ (dimerization) = (t1-t2)/(t1+t2): Controls SSH topology
- U (Hubbard interaction): Controls electron-electron interaction strength
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_all_ansatze import compare_ansatze


def delta_to_t2(delta, t1=1.0):
    """Convert dimerization parameter δ to hopping t2."""
    # δ = (t1-t2)/(t1+t2) => t2 = t1*(1-δ)/(1+δ)
    return t1 * (1 - delta) / (1 + delta)


def run_parameter_sweep(
    delta_values,
    U_values,
    L=4,
    t1=1.0,
    reps=2,
    maxiter=100,
    output_file=None
):
    """
    Run comprehensive parameter sweep over δ and U.

    Parameters:
    -----------
    delta_values : list
        List of dimerization parameters δ = (t1-t2)/(t1+t2)
    U_values : list
        List of Hubbard interaction strengths
    L : int
        Lattice size (default: 4)
    t1 : float
        Strong hopping parameter (default: 1.0)
    reps : int
        Ansatz repetitions (default: 2)
    maxiter : int
        Maximum optimizer iterations (default: 100)
    output_file : str
        Output markdown file path
    """

    print("=" * 80)
    print("SSH-HUBBARD PARAMETER SWEEP")
    print("=" * 80)
    print(f"System size: L = {L}")
    print(f"Fixed parameters: t1 = {t1}, reps = {reps}, maxiter = {maxiter}")
    print(f"Dimerization sweep (δ): {delta_values}")
    print(f"Interaction sweep (U): {U_values}")
    print(f"Total parameter points: {len(delta_values)} × {len(U_values)} = {len(delta_values) * len(U_values)}")
    print(f"VQE runs per point: 3 ansätze × 3 optimizers × 5 seeds = 45")
    print(f"Total VQE runs: {len(delta_values) * len(U_values) * 45}")
    print("=" * 80)
    print()

    # Storage for all results
    all_results = []
    start_time = time.time()

    # Parameter sweep
    total_points = len(delta_values) * len(U_values)
    current_point = 0

    for delta in delta_values:
        t2 = delta_to_t2(delta, t1)

        for U in U_values:
            current_point += 1

            print("\n" + "=" * 80)
            print(f"PARAMETER POINT {current_point}/{total_points}")
            print(f"δ = {delta:.3f}, U = {U:.2f} (t1 = {t1:.2f}, t2 = {t2:.3f})")
            print("=" * 80)

            point_start = time.time()

            # Run multi-start VQE comparison
            result = compare_ansatze(
                L=L,
                t1=t1,
                t2=t2,
                U=U,
                reps=reps,
                maxiter=maxiter,
                verbose=True,
                use_multistart=True
            )

            point_time = time.time() - point_start

            # Store result with metadata
            all_results.append({
                'delta': delta,
                'U': U,
                't1': t1,
                't2': t2,
                'result': result,
                'runtime': point_time
            })

            print(f"\n✓ Parameter point completed in {point_time:.1f}s")
            elapsed = time.time() - start_time
            avg_time = elapsed / current_point
            remaining_points = total_points - current_point
            eta = avg_time * remaining_points

            print(f"Progress: {current_point}/{total_points} ({100*current_point/total_points:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time

    # Generate comprehensive report
    if output_file:
        generate_sweep_report(all_results, output_file, L, total_time)

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Parameter points: {total_points}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return all_results


def generate_sweep_report(all_results, output_file, L, total_time):
    """Generate comprehensive markdown report for parameter sweep."""

    with open(output_file, 'w') as f:
        # Header
        f.write(f"# SSH-Hubbard Parameter Sweep Results (L={L})\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Runtime**: {total_time/60:.1f} minutes ({total_time:.1f} seconds)\n\n")
        f.write(f"**Parameter points**: {len(all_results)}\n\n")
        f.write(f"**VQE runs**: {len(all_results) * 45}\n\n")
        f.write("---\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write("- **System size**: L = 4 (8 qubits)\n")
        f.write("- **Optimizers**: L-BFGS-B, COBYLA, SLSQP\n")
        f.write("- **Seeds per optimizer**: 5 [0, 1, 2, 3, 4]\n")
        f.write("- **Ansätze**: HEA, HVA, NP_HVA\n")
        f.write("- **Method**: Multi-start VQE (45 runs per parameter point)\n\n")
        f.write("---\n\n")

        # Best performers summary
        f.write("## Best Performers Across All Parameters\n\n")

        # Find global best for each ansatz+optimizer combo
        best_combos = {}
        for res in all_results:
            delta = res['delta']
            U = res['U']
            result = res['result']

            for ansatz_name, ansatz_data in result['ansatze'].items():
                for opt_name, opt_data in ansatz_data.items():
                    key = f"{ansatz_name}+{opt_name}"
                    best_energy = opt_data['best_energy']
                    exact_energy = result['exact']['energy']
                    rel_error = 100 * abs(best_energy - exact_energy) / abs(exact_energy)

                    if key not in best_combos or rel_error < best_combos[key]['rel_error']:
                        best_combos[key] = {
                            'ansatz': ansatz_name,
                            'optimizer': opt_name,
                            'delta': delta,
                            'U': U,
                            'energy': best_energy,
                            'exact': exact_energy,
                            'rel_error': rel_error
                        }

        # Sort by rel_error
        sorted_combos = sorted(best_combos.values(), key=lambda x: x['rel_error'])

        f.write("| Rank | Ansatz | Optimizer | δ | U | Best Energy | Exact | Rel Error % |\n")
        f.write("|------|--------|-----------|---|---|-------------|-------|-------------|\n")

        for i, combo in enumerate(sorted_combos[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            f.write(f"| {medal} {i} | {combo['ansatz']} | {combo['optimizer']} | "
                   f"{combo['delta']:.3f} | {combo['U']:.1f} | {combo['energy']:.6f} | "
                   f"{combo['exact']:.6f} | {combo['rel_error']:.2f}% |\n")

        f.write("\n---\n\n")

        # Heat maps data section
        f.write("## Performance Heat Map Data\n\n")
        f.write("Best relative error (%) for each (δ, U) parameter point:\n\n")

        # Organize by ansatz
        for ansatz_name in ['HEA', 'HVA', 'NP_HVA']:
            f.write(f"### {ansatz_name} - Best Across All Optimizers\n\n")

            # Create grid
            delta_vals = sorted(set(res['delta'] for res in all_results))
            U_vals = sorted(set(res['U'] for res in all_results))

            # Header
            f.write("| δ \\ U |")
            for U in U_vals:
                f.write(f" {U:.1f} |")
            f.write("\n")

            f.write("|-------|")
            for _ in U_vals:
                f.write("------|")
            f.write("\n")

            # Data rows
            for delta in delta_vals:
                f.write(f"| {delta:.3f} |")
                for U in U_vals:
                    # Find this parameter point
                    point = next((r for r in all_results
                                if abs(r['delta']-delta)<1e-6 and abs(r['U']-U)<1e-6), None)

                    if point and ansatz_name.lower() in point['result']['ansatze']:
                        ansatz_data = point['result']['ansatze'][ansatz_name.lower()]
                        exact = point['result']['exact']['energy']

                        # Best across optimizers
                        best_err = float('inf')
                        for opt_data in ansatz_data.values():
                            err = 100 * abs(opt_data['best_energy'] - exact) / abs(exact)
                            if err < best_err:
                                best_err = err

                        f.write(f" {best_err:.2f}% |")
                    else:
                        f.write(" N/A |")
                f.write("\n")

            f.write("\n")

        f.write("---\n\n")

        # Detailed results per parameter point
        f.write("## Detailed Results by Parameter Point\n\n")

        for res in sorted(all_results, key=lambda x: (x['delta'], x['U'])):
            delta = res['delta']
            U = res['U']
            t1 = res['t1']
            t2 = res['t2']
            result = res['result']
            runtime = res['runtime']

            exact_energy = result['exact']['energy']

            f.write(f"### δ = {delta:.3f}, U = {U:.2f} (t1={t1:.2f}, t2={t2:.3f})\n\n")
            f.write(f"**Exact energy**: {exact_energy:.6f}\n\n")
            f.write(f"**Runtime**: {runtime:.1f}s\n\n")

            f.write("| Ansatz | Optimizer | Best Energy | Mean ± Std | Rel Error % |\n")
            f.write("|--------|-----------|-------------|------------|-------------|\n")

            for ansatz_name, ansatz_data in sorted(result['ansatze'].items()):
                for opt_name, opt_data in sorted(ansatz_data.items()):
                    best_e = opt_data['best_energy']
                    mean_e = opt_data['mean_energy']
                    std_e = opt_data['std_energy']
                    rel_err = 100 * abs(best_e - exact_energy) / abs(exact_energy)

                    f.write(f"| {ansatz_name.upper()} | {opt_name} | {best_e:.6f} | "
                           f"{mean_e:.6f} ± {std_e:.2e} | {rel_err:.2f}% |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SSH-Hubbard parameter sweep for L=4"
    )
    parser.add_argument(
        '--delta-values',
        type=float,
        nargs='+',
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
        help='Dimerization values (default: 0.0 0.2 0.4 0.6 0.8)'
    )
    parser.add_argument(
        '--U-values',
        type=float,
        nargs='+',
        default=[0.0, 1.0, 2.0, 4.0],
        help='Interaction strengths (default: 0.0 1.0 2.0 4.0)'
    )
    parser.add_argument(
        '--L',
        type=int,
        default=4,
        help='Lattice size (default: 4)'
    )
    parser.add_argument(
        '--maxiter',
        type=int,
        default=100,
        help='Max optimizer iterations (default: 100)'
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=2,
        help='Ansatz repetitions (default: 2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output markdown file (default: auto-generated)'
    )

    args = parser.parse_args()

    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'../docs/parameter_sweep_L{args.L}_{timestamp}.md'

    # Run sweep
    results = run_parameter_sweep(
        delta_values=args.delta_values,
        U_values=args.U_values,
        L=args.L,
        t1=1.0,
        reps=args.reps,
        maxiter=args.maxiter,
        output_file=args.output
    )

    print(f"\n✓ All results saved to: {args.output}")
