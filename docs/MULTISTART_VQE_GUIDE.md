# Multi-Start VQE Benchmark Guide

## Overview

This guide describes the multi-start VQE benchmarking infrastructure implemented for robust ansatz comparison.

### Key Features

- **3 Optimizers**: L-BFGS-B, COBYLA, SLSQP
- **5 Random Seeds**: [0, 1, 2, 3, 4] for each optimizer
- **Convergence Tracking**: Full energy history for each run
- **Statistical Analysis**: Mean, std, min, max across all seeds
- **Visualization**: Multi-seed convergence plots with best seed highlighting
- **Backward Compatible**: Single-run mode still available

### Why Multi-Start?

VQE optimization landscapes are highly non-convex with many local minima. Single-run benchmarks are unreliable because:

1. Results depend heavily on random initialization
2. Different optimizers may find different local minima
3. No statistical confidence without multiple runs

Multi-start VQE addresses this by:
- Running multiple times with different initializations
- Computing aggregate statistics across all runs
- Identifying best, mean, and worst-case performance
- Providing statistical confidence (mean ± std)

---

## Architecture

### Core Components

**1. VQERunner Class** (`benchmarks/compare_all_ansatze.py`)

Enhanced to support:
- Multiple optimizers: L_BFGS_B, COBYLA, SLSQP
- Per-run random seeds with `np.random.default_rng(seed)`
- Convergence history tracking via callback

```python
runner = VQERunner(maxiter=200, optimizer_name='L_BFGS_B')
result = runner.run(ansatz, hamiltonian, seed=42)
# Returns: energy, optimal_params, evaluations, runtime, energy_history, seed
```

**2. run_multistart_vqe Function**

Orchestrates multiple VQE runs:

```python
multistart_result = run_multistart_vqe(
    runner=runner,
    ansatz=ansatz,
    hamiltonian=H,
    seeds=[0, 1, 2, 3, 4]
)
# Returns:
# {
#   'per_seed': [...],           # Individual run results
#   'best': {...},               # Best run (lowest energy)
#   'mean_energy': float,        # Mean across seeds
#   'std_energy': float,         # Std across seeds
#   'min_energy': float,         # Minimum energy found
#   'max_energy': float          # Maximum energy found
# }
```

**3. compare_ansatze Function**

Updated with `use_multistart` parameter:

```python
# Multi-start mode (default)
results = compare_ansatze(
    L=4, t1=1.0, t2=0.5, U=2.0,
    reps=2, maxiter=200,
    use_multistart=True  # 3 optimizers × 5 seeds
)

# Single-run mode (backward compatible)
results = compare_ansatze(
    L=4, t1=1.0, t2=0.5, U=2.0,
    reps=2, maxiter=200,
    use_multistart=False  # Just L-BFGS-B
)
```

**Result Structure (Multi-Start)**:

```python
{
  'system': {'L': 4, 't1': 1.0, 't2': 0.5, 'U': 2.0, 'delta': 0.333},
  'exact': {'energy': -8.123456, 'energy_per_site': -2.030864},
  'multistart': True,
  'ansatze': {
    'hea': {
      'L_BFGS_B': {
        'best_energy': -8.120000,
        'mean_energy': -8.115000,
        'std_energy': 0.003,
        'min_energy': -8.120000,
        'max_energy': -8.110000,
        'abs_error_best': 0.003456,
        'rel_error_best_percent': 0.04,
        'per_seed': [
          {'energy': -8.120, 'seed': 0, 'runtime': 5.2, ...},
          {'energy': -8.118, 'seed': 1, 'runtime': 4.8, ...},
          ...
        ],
        'num_params': 56,
        'depth': 15
      },
      'COBYLA': {...},
      'SLSQP': {...}
    },
    'hva': {...},
    'np_hva': {...}
  }
}
```

---

## Visualization

### plot_multistart_convergence Function

Located in `src/plot_utils.py`, generates dual plots:

**Left Panel**: Energy Convergence
- Gray lines (α=0.3): All seed trajectories
- Blue line (bold): Best seed (lowest energy)
- Red dashed: Mean trajectory
- Red band: ±1 std deviation
- Green dashed: Exact energy

**Right Panel**: Error Convergence (log scale)
- Same color scheme as left panel
- Y-axis: |E_VQE - E_exact|

**Example**:

```python
from plot_utils import plot_multistart_convergence

plot_multistart_convergence(
    per_seed_results=multistart_result['per_seed'],
    exact_energy=E_exact,
    ansatz_name='hea',
    optimizer_name='L_BFGS_B',
    L=4,
    output_dir='../docs/images'
)
# Saves: ../docs/images/convergence_hea_L_BFGS_B_L4.png
```

---

## Usage

### CLI Tool: run_multistart_benchmark.py

Comprehensive multi-start benchmarking with automatic report generation.

**Basic Usage**:

```bash
# Run for L=4 with default parameters
python benchmarks/run_multistart_benchmark.py --L 4

# Custom parameters
python benchmarks/run_multistart_benchmark.py \
  --L 4 \
  --t1 1.0 --t2 0.5 --U 2.0 \
  --reps 2 --maxiter 200 \
  --output-doc results_L4.md

# Multiple system sizes
python benchmarks/run_multistart_benchmark.py --L 4 6 --output-doc results_L4_L6.md
```

**Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--L` | int list | [4] | System sizes to test |
| `--t1` | float | 1.0 | Strong hopping amplitude |
| `--t2` | float | 0.5 | Weak hopping amplitude |
| `--U` | float | 2.0 | Hubbard interaction |
| `--reps` | int | 2 | Ansatz repetitions |
| `--maxiter` | int | 200 | Max optimizer iterations |
| `--output-doc` | str | auto | Output markdown file |

**What It Does**:

1. Runs VQE for all 3 ansätze (HEA, HVA, NP_HVA)
2. Uses all 3 optimizers (L-BFGS-B, COBYLA, SLSQP)
3. Each optimizer runs 5 times with seeds [0, 1, 2, 3, 4]
4. **Total runs**: 3 ansätze × 3 optimizers × 5 seeds = **45 VQE runs** per system size
5. Generates convergence plots for each ansatz/optimizer pair
6. Creates comprehensive markdown report with:
   - Summary tables
   - Per-seed details
   - Best performer analysis
   - References to convergence plots

---

## Python API Usage

### Example: Single Ansatz, Single Optimizer

```python
from compare_all_ansatze import VQERunner, run_multistart_vqe
from ssh_hubbard_vqe import ssh_hubbard_hamiltonian, build_ansatz_hea

# Build Hamiltonian
H = ssh_hubbard_hamiltonian(L=4, t1=1.0, t2=0.5, U=2.0, periodic=False)

# Build ansatz
ansatz = build_ansatz_hea(N=8, reps=2)

# Run multi-start VQE
runner = VQERunner(maxiter=200, optimizer_name='L_BFGS_B')
results = run_multistart_vqe(
    runner=runner,
    ansatz=ansatz,
    hamiltonian=H,
    seeds=[0, 1, 2, 3, 4]
)

print(f"Best energy: {results['best']['energy']:.8f}")
print(f"Mean ± std:  {results['mean_energy']:.8f} ± {results['std_energy']:.3e}")
print(f"Min/Max:     {results['min_energy']:.8f} / {results['max_energy']:.8f}")
```

### Example: Full Comparison

```python
from compare_all_ansatze import compare_ansatze

# Run full multi-start comparison
results = compare_ansatze(
    L=4,
    t1=1.0, t2=0.5, U=2.0,
    reps=2,
    maxiter=200,
    verbose=True,
    use_multistart=True
)

# Access results
for ansatz_name, opt_results in results['ansatze'].items():
    print(f"\n{ansatz_name.upper()}:")
    for opt_name, metrics in opt_results.items():
        print(f"  {opt_name}: {metrics['best_energy']:.8f} ± {metrics['std_energy']:.3e}")
```

---

## Results Interpretation

### Understanding Multi-Start Statistics

**Best Energy**: Lowest energy found across all 5 seeds
- Represents "best case" performance
- What you'd hope to achieve with good initialization

**Mean ± Std**: Average and spread across 5 seeds
- Mean: Expected performance with random initialization
- Std: Variability/consistency of optimizer
- Low std = robust/consistent optimizer
- High std = sensitive to initialization

**Min/Max**: Range of energies found
- Max - Min = spread of local minima
- Large range indicates rugged landscape

### Comparing Optimizers

**L-BFGS-B**:
- Quasi-Newton, gradient-based
- Fast convergence when landscape is smooth
- Can get stuck in local minima
- Best for: Well-behaved problems

**COBYLA**:
- Gradient-free, derivative-free
- Robust to noise and non-smooth landscapes
- Slower convergence than L-BFGS-B
- Best for: Noisy or non-smooth problems

**SLSQP**:
- Sequential Least Squares Programming
- Gradient-based with constraint handling
- Middle ground between L-BFGS-B and COBYLA
- Best for: Constrained optimization

### Statistical Significance

With only 5 seeds, statistics are indicative but not rigorous:
- Std is noisy estimate (need 10-100 runs for confidence)
- Results show trends, not definitive rankings
- Use for qualitative comparison, not publication-quality claims

For rigorous benchmarking:
- Use 50-100 seeds
- Compute confidence intervals
- Perform statistical tests (t-test, Mann-Whitney)
- Consider multiple parameter regimes

---

## Example Workflow

### 1. Quick Test (L=4, Single Configuration)

```bash
python benchmarks/run_multistart_benchmark.py --L 4
```

**Expected Output**:
- 45 VQE runs (3 ansätze × 3 optimizers × 5 seeds)
- Runtime: ~5-10 minutes (depends on system)
- Generates: `docs/multistart_benchmark_L4_<timestamp>.md`
- Plots: 9 convergence plots in `docs/images/`

### 2. Multi-Size Comparison

```bash
python benchmarks/run_multistart_benchmark.py --L 4 6 --output-doc comparison_L4_L6.md
```

**Expected Output**:
- 90 VQE runs (45 per system size)
- Runtime: ~15-30 minutes
- Allows comparison of ansatz scaling

### 3. Custom Parameter Regime

```bash
# Strong interaction regime
python benchmarks/run_multistart_benchmark.py --L 4 --U 4.0 --output-doc strong_U.md

# Topological regime (large dimerization)
python benchmarks/run_multistart_benchmark.py --L 4 --t1 1.0 --t2 0.2 --output-doc strong_SSH.md
```

### 4. Backward Compatible Single-Run

```python
# For quick testing without multi-start overhead
from compare_all_ansatze import compare_ansatze

results = compare_ansatze(
    L=4, t1=1.0, t2=0.5, U=2.0,
    use_multistart=False  # Single L-BFGS-B run
)
```

---

## Performance Considerations

### Runtime Scaling

Single VQE run: ~5-20 seconds (L=4, 200 iterations)

**Multi-start (5 seeds)**:
- Single optimizer: ~25-100 seconds
- All 3 optimizers: ~75-300 seconds

**Full benchmark (3 ansätze × 3 optimizers × 5 seeds)**:
- L=4: ~5-10 minutes
- L=6: ~15-30 minutes (larger Hilbert space)

### Memory Usage

- L=4 (8 qubits): 2^8 = 256 states → minimal memory
- L=6 (12 qubits): 2^12 = 4096 states → still manageable
- L=8 (16 qubits): 2^16 = 65536 states → uses sparse Lanczos

### Parallelization

Current implementation is **sequential** (runs one seed at a time).

For parallel execution:
```python
# Future enhancement (not yet implemented)
from multiprocessing import Pool

def run_seed(seed):
    return runner.run(ansatz, H, seed=seed)

with Pool(5) as p:
    results = p.map(run_seed, [0, 1, 2, 3, 4])
```

---

## Troubleshooting

### Import Errors

```
ImportError: cannot import name 'plot_multistart_convergence'
```

**Fix**: Make sure you're using the updated `plot_utils.py`:
```bash
git pull origin claude/read-this-r-01DLdEcvW8hustGKsyPjZzLM
```

### Missing Plots

```
Warning: Could not generate convergence plots: No module named 'matplotlib'
```

**Fix**: Install matplotlib:
```bash
pip install matplotlib
```

### SLSQP Not Found

```
ImportError: cannot import name 'SLSQP'
```

**Fix**: Update Qiskit:
```bash
pip install --upgrade qiskit qiskit-algorithms
```

Or remove SLSQP from optimizer list (edit `compare_all_ansatze.py` line 348).

### Slow Performance

If benchmarks are taking too long:

1. Reduce `--maxiter`:
   ```bash
   python benchmarks/run_multistart_benchmark.py --L 4 --maxiter 100
   ```

2. Use fewer seeds (edit `compare_all_ansatze.py` line 349):
   ```python
   seeds = [0, 1, 2] if use_multistart else [None]  # 3 instead of 5
   ```

3. Test single ansatz:
   ```python
   # Comment out unwanted ansätze in compare_all_ansatze.py lines 334-337
   ansatz_configs = [
       ('hea', lambda: build_ansatz_hea(N, reps), False),
       # ('hva', ...),  # Commented out
       # ('np_hva', ...),  # Commented out
   ]
   ```

---

## Future Enhancements

### Planned Features

1. **Parallel Execution**: Use multiprocessing to run seeds in parallel
2. **More Optimizers**: Add SPSA, Powell, Nelder-Mead
3. **Adaptive Seeds**: Automatically determine number of seeds based on convergence
4. **Confidence Intervals**: Bootstrap or t-distribution confidence intervals
5. **Interactive Plots**: Plotly or Bokeh for interactive convergence exploration
6. **Database Storage**: SQLite for storing all run data
7. **Web Dashboard**: Flask/Dash app for visualizing results

### Contributing

To add new optimizers:

1. Import optimizer in `compare_all_ansatze.py`:
   ```python
   from qiskit_algorithms.optimizers import YourOptimizer
   ```

2. Add to VQERunner.run():
   ```python
   elif self.optimizer_name == 'YOUR_OPT':
       optimizer = YourOptimizer(maxiter=self.maxiter)
   ```

3. Add to supported list:
   ```python
   supported_optimizers = ['L_BFGS_B', 'COBYLA', 'SLSQP', 'YOUR_OPT']
   ```

4. Update optimizer list in compare_ansatze:
   ```python
   optimizers = ["L_BFGS_B", "COBYLA", "SLSQP", "YOUR_OPT"] if use_multistart else ["L_BFGS_B"]
   ```

---

## References

### Optimizer Documentation

- **L-BFGS-B**: [Qiskit L_BFGS_B](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.L_BFGS_B.html)
- **COBYLA**: [Qiskit COBYLA](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.COBYLA.html)
- **SLSQP**: [Qiskit SLSQP](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.SLSQP.html)

### VQE and Multi-Start Methods

- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor", Nature Communications (2014)
- Kandala et al., "Hardware-efficient variational quantum eigensolver for small molecules", Nature (2017)
- Cervera-Lierta et al., "Meta-Variational Quantum Eigensolver", arXiv:2009.13545
- Grimsley et al., "An adaptive variational algorithm for exact molecular simulations", Nature Communications (2019)

### SSH-Hubbard Model

- Su, Schrieffer, Heeger, "Solitons in Polyacetylene", PRL (1979)
- Hubbard, "Electron Correlations in Narrow Energy Bands", Proc. Royal Soc. (1963)

---

## Summary

This multi-start VQE infrastructure provides:

✓ **Robustness**: 5 random initializations per optimizer
✓ **Completeness**: 3 optimizers for comprehensive comparison
✓ **Statistics**: Mean, std, min, max for confidence
✓ **Visualization**: Convergence plots with all seed trajectories
✓ **Automation**: CLI tool for batch benchmarking
✓ **Documentation**: Markdown reports with tables and plots
✓ **Backward Compatible**: Single-run mode still available

Use this for reliable ansatz benchmarking that goes beyond single-point comparisons.
