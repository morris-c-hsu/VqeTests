#!/usr/bin/env python3
"""
Multi-Start VQE Benchmark with Multiple Optimizers

Runs comprehensive VQE benchmarks with:
- 3 optimizers: L-BFGS-B, COBYLA, SLSQP
- 5 random seeds per optimizer: [0, 1, 2, 3, 4]
- All 3 main ansätze: HEA, HVA, NP_HVA
- Convergence tracking and plotting
- Comprehensive results documentation

Usage:
    # Run benchmark for L=4
    python run_multistart_benchmark.py --L 4

    # Run with custom parameters
    python run_multistart_benchmark.py --L 4 --t1 1.0 --t2 0.5 --U 2.0 --reps 2 --maxiter 200

    # Run multiple system sizes
    python run_multistart_benchmark.py --L 4 6 --output-doc results_L4_L6.md
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compare_all_ansatze import compare_ansatze


def format_results_table(results):
    """
    Format multi-start results as a markdown table.

    Parameters
    ----------
    results : dict
        Results from compare_ansatze with use_multistart=True

    Returns
    -------
    table : str
        Markdown-formatted table
    """
    lines = []

    # System parameters
    sys_params = results['system']
    lines.append(f"\n### System: L={sys_params['L']}, δ={sys_params['delta']:.3f}, U={sys_params['U']:.2f}\n")

    # Exact energy
    E_exact = results['exact']['energy']
    lines.append(f"**Exact Energy**: `{E_exact:.10f}`\n")

    # Results table header
    lines.append("| Ansatz | Optimizer | Best Energy | Mean ± Std | Min | Max | Error (Best) | Rel % |")
    lines.append("|--------|-----------|-------------|------------|-----|-----|--------------|-------|")

    # Results rows
    for ansatz_name, opt_results in results['ansatze'].items():
        if 'error' in opt_results:
            lines.append(f"| {ansatz_name.upper()} | ALL | ERROR | {opt_results['error']} | - | - | - | - |")
            continue

        for opt_name, metrics in opt_results.items():
            best_e = metrics['best_energy']
            mean_e = metrics['mean_energy']
            std_e = metrics['std_energy']
            min_e = metrics['min_energy']
            max_e = metrics['max_energy']
            abs_err = metrics['abs_error_best']
            rel_err = metrics['rel_error_best_percent']

            lines.append(
                f"| {ansatz_name.upper()} | {opt_name} | "
                f"{best_e:.8f} | {mean_e:.8f} ± {std_e:.2e} | "
                f"{min_e:.8f} | {max_e:.8f} | "
                f"{abs_err:.2e} | {rel_err:.2f}% |"
            )

    return "\n".join(lines)


def format_per_seed_details(results):
    """
    Format per-seed details for each ansatz/optimizer combination.

    Parameters
    ----------
    results : dict
        Results from compare_ansatze with use_multistart=True

    Returns
    -------
    details : str
        Markdown-formatted per-seed details
    """
    lines = []
    lines.append("\n## Per-Seed Details\n")

    for ansatz_name, opt_results in results['ansatze'].items():
        if 'error' in opt_results:
            continue

        for opt_name, metrics in opt_results.items():
            lines.append(f"\n### {ansatz_name.upper()} - {opt_name}\n")

            lines.append("| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |")
            lines.append("|------|--------|-------------|-------------|-------------------|")

            for seed_result in metrics['per_seed']:
                seed = seed_result['seed']
                energy = seed_result['energy']
                evals = seed_result['evaluations']
                runtime = seed_result['runtime']
                conv_steps = len(seed_result['energy_history'])

                lines.append(
                    f"| {seed} | {energy:.10f} | {evals} | {runtime:.2f} | {conv_steps} |"
                )

    return "\n".join(lines)


def generate_markdown_report(all_results, output_file):
    """
    Generate comprehensive markdown report of multi-start benchmark results.

    Parameters
    ----------
    all_results : list of (description, results) tuples
        List of benchmark results for different configurations
    output_file : str
        Path to output markdown file
    """
    lines = []

    # Header
    lines.append("# Multi-Start VQE Benchmark Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append("- **Optimizers**: L-BFGS-B, COBYLA, SLSQP")
    lines.append("- **Seeds**: [0, 1, 2, 3, 4]")
    lines.append("- **Ansätze**: HEA, HVA, NP_HVA")
    lines.append("- **Method**: Multi-start VQE (5 random initializations per optimizer)\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Test | Ansatz | Optimizer | Best Energy | Mean ± Std | Error (Best) | Rel % |")
    lines.append("|------|--------|-----------|-------------|------------|--------------|-------|")

    for description, result in all_results:
        test_name = description.split('(')[0].strip()
        for ansatz_name, opt_results in result['ansatze'].items():
            if 'error' in opt_results:
                continue

            for opt_name, metrics in opt_results.items():
                best_e = metrics['best_energy']
                mean_e = metrics['mean_energy']
                std_e = metrics['std_energy']
                abs_err = metrics['abs_error_best']
                rel_err = metrics['rel_error_best_percent']

                lines.append(
                    f"| {test_name} | {ansatz_name.upper()} | {opt_name} | "
                    f"{best_e:.8f} | {mean_e:.8f} ± {std_e:.2e} | "
                    f"{abs_err:.2e} | {rel_err:.2f}% |"
                )

    # Detailed results for each configuration
    lines.append("\n---\n")
    lines.append("# Detailed Results\n")

    for description, result in all_results:
        lines.append(f"\n## {description}\n")
        lines.append(format_results_table(result))
        lines.append(format_per_seed_details(result))

        # Convergence plots reference
        lines.append("\n### Convergence Plots\n")
        L = result['system']['L']
        for ansatz_name in result['ansatze'].keys():
            if 'error' in result['ansatze'][ansatz_name]:
                continue
            for opt_name in ["L_BFGS_B", "COBYLA", "SLSQP"]:
                if opt_name in result['ansatze'][ansatz_name]:
                    plot_path = f"../docs/images/convergence_{ansatz_name}_{opt_name}_L{L}.png"
                    lines.append(f"- [{ansatz_name.upper()} - {opt_name}]({plot_path})")

    # Best performers analysis
    lines.append("\n---\n")
    lines.append("# Best Performers\n")

    for description, result in all_results:
        lines.append(f"\n## {description}\n")

        # Collect all ansatz/optimizer combinations
        combos = []
        for ansatz_name, opt_results in result['ansatze'].items():
            if 'error' in opt_results:
                continue
            for opt_name, metrics in opt_results.items():
                combos.append((ansatz_name, opt_name, metrics))

        if not combos:
            continue

        # Best accuracy
        best_acc = min(combos, key=lambda x: x[2]['abs_error_best'])
        lines.append(
            f"**Best Accuracy**: {best_acc[0].upper()} + {best_acc[1]} "
            f"({best_acc[2]['rel_error_best_percent']:.2f}% error)"
        )

        # Most consistent (lowest std)
        best_std = min(combos, key=lambda x: x[2]['std_energy'])
        lines.append(
            f"**Most Consistent**: {best_std[0].upper()} + {best_std[1]} "
            f"(std = {best_std[2]['std_energy']:.3e})"
        )

    # Footer
    lines.append("\n---\n")
    lines.append("## Notes\n")
    lines.append("- **Best Energy**: Lowest energy found across all 5 seeds")
    lines.append("- **Mean ± Std**: Statistics across 5 random initializations")
    lines.append("- **Error**: Absolute and relative error compared to exact diagonalization")
    lines.append("- **Convergence plots**: Show all seed trajectories (gray), best seed (blue), and mean ± std (red band)")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n{'=' * 70}")
    print(f"✓ Comprehensive report saved: {output_file}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-start VQE benchmark with multiple optimizers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--L', type=int, nargs='+', default=[4],
                       help='System sizes to test (default: 4)')
    parser.add_argument('--t1', type=float, default=1.0,
                       help='Strong hopping amplitude (default: 1.0)')
    parser.add_argument('--t2', type=float, default=0.5,
                       help='Weak hopping amplitude (default: 0.5)')
    parser.add_argument('--U', type=float, default=2.0,
                       help='Hubbard interaction strength (default: 2.0)')
    parser.add_argument('--reps', type=int, default=2,
                       help='Ansatz repetitions (default: 2)')
    parser.add_argument('--maxiter', type=int, default=200,
                       help='Maximum optimizer iterations (default: 200)')
    parser.add_argument('--output-doc', type=str, default=None,
                       help='Output markdown file (default: auto-generated)')

    args = parser.parse_args()

    # Generate output filename
    if args.output_doc is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        L_str = '_'.join(map(str, args.L))
        args.output_doc = f'../docs/multistart_benchmark_L{L_str}_{timestamp}.md'

    print("#" * 70)
    print("# MULTI-START VQE BENCHMARK")
    print("#" * 70)
    print(f"Optimizers: L-BFGS-B, COBYLA, SLSQP")
    print(f"Seeds: [0, 1, 2, 3, 4]")
    print(f"System sizes: {args.L}")
    print(f"Parameters: t1={args.t1}, t2={args.t2}, U={args.U}")
    print(f"Ansatz reps: {args.reps}, Max iterations: {args.maxiter}")
    print("#" * 70)

    all_results = []
    total_start = time.time()

    # Run benchmarks for each system size
    for L in args.L:
        delta = (args.t1 - args.t2) / (args.t1 + args.t2)
        description = f"L={L}, δ={delta:.3f}, U={args.U:.2f}"

        print(f"\n{'=' * 70}")
        print(f"Running: {description}")
        print(f"{'=' * 70}\n")

        result = compare_ansatze(
            L=L,
            t1=args.t1,
            t2=args.t2,
            U=args.U,
            reps=args.reps,
            maxiter=args.maxiter,
            verbose=True,
            use_multistart=True
        )

        all_results.append((description, result))

    total_time = time.time() - total_start

    # Generate markdown report
    generate_markdown_report(all_results, args.output_doc)

    print(f"\nTotal benchmark time: {total_time:.1f}s")
    print(f"Results saved to: {args.output_doc}")


if __name__ == "__main__":
    main()
