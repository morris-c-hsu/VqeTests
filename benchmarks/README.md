# VQE Benchmarks

This directory contains benchmark scripts for comparing VQE ansätze on the SSH-Hubbard model.

## Available Scripts

### 1. run_multistart_benchmark.py ⭐ NEW

**Multi-start VQE with 3 optimizers and 5 random seeds**

Comprehensive benchmarking infrastructure for robust ansatz comparison:
- 3 optimizers: L-BFGS-B, COBYLA, SLSQP
- 5 random seeds per optimizer: [0, 1, 2, 3, 4]
- Statistical analysis: mean, std, min, max
- Convergence visualization with multi-seed plots
- Automatic markdown report generation
- **Enhanced Features**:
  - Relative error percentage plots for intuitive accuracy assessment
  - COBYLA gets 10× iterations (min 1000) for fair comparison with gradient-based optimizers
  - Professional plot formatting with dense tick marks and plain number display

**Usage**:
```bash
# Quick test (L=4)
python run_multistart_benchmark.py --L 4

# Multiple system sizes
python run_multistart_benchmark.py --L 4 6 --output-doc results.md

# Custom parameters
python run_multistart_benchmark.py --L 4 --t1 1.0 --t2 0.5 --U 2.0 --reps 2 --maxiter 200
```

**Output**:
- Markdown report with summary tables and per-seed details
- Convergence plots in `docs/images/`
- Best performer analysis

**Documentation**: See [`docs/MULTISTART_VQE_GUIDE.md`](../docs/MULTISTART_VQE_GUIDE.md)

**Total runs**: 3 ansätze × 3 optimizers × 5 seeds = **45 VQE runs** per system size

---

### 2. quick_benchmark.py

**Fast single-run comparison for testing**

Quick sanity check of all 3 main ansätze (HEA, HVA, NP_HVA):
- Single L-BFGS-B run per ansatz
- L=4, standard parameters (δ=0.33, U=2.0)
- Convergence plots generated

**Usage**:
```bash
python quick_benchmark.py
```

**Runtime**: ~30-60 seconds

**When to use**:
- Quick sanity check after code changes
- Testing new ansätze implementations
- Debugging VQE setup

---

### 3. benchmark_large_systems.py

**Benchmarks for L=6, 8 with sparse Lanczos**

Tests ansätze on larger systems where exact diagonalization requires sparse methods:
- L=6 (12 qubits, dim=4096): Dense ED
- L=8 (16 qubits, dim=65536): Sparse Lanczos

**Usage**:
```bash
python benchmark_large_systems.py
```

**Features**:
- Automatic sparse/dense method selection
- Multiple parameter regimes
- Performance comparison across system sizes

**Runtime**: ~10-30 minutes (L=8 is slow!)

---

### 4. run_longer_optimizations.py

**Extended optimization runs (500-1000 iterations)**

For ansätze that need more iterations to converge:
- maxiter=500 or maxiter=1000
- L=4 and L=6
- Tracks convergence to assess if more iterations help

**Usage**:
```bash
python run_longer_optimizations.py
```

**Runtime**: ~30-60 minutes

**When to use**:
- Ansatz not converging in 200 iterations
- Investigating optimizer performance limits
- Research on convergence behavior

---

## Ansätze Tested

All benchmarks test the **3 main ansätze**:

| Ansatz | Number-Conserving | Purpose |
|--------|-------------------|---------|
| **HEA** | ✗ No | Generic baseline (EfficientSU2) |
| **HVA** | ✓ Yes | Hamiltonian-inspired (Givens rotations) |
| **NP_HVA** | ✓✓ Strict | Strict number conservation (UNP gates) |

**Archived ansätze** (5 additional) available in `src/ansatze/archived_ansatze.py`:
- TopoInspired, TopoRN, DQAP, TN_MPS, TN_MPS_NP

See [`docs/ANSATZ_OVERVIEW.md`](../docs/ANSATZ_OVERVIEW.md) for details.

---

## Quick Start

### 1. First-time setup

```bash
# Install dependencies
pip install qiskit qiskit-algorithms numpy scipy matplotlib

# Verify installation
python quick_benchmark.py
```

### 2. Run multi-start benchmark

```bash
# Standard benchmark (recommended)
python run_multistart_benchmark.py --L 4

# Check results
cat ../docs/multistart_benchmark_L4_*.md
```

### 3. View convergence plots

```bash
ls -lh ../docs/images/convergence_*.png
```

---

## Benchmark Comparison

| Script | Ansätze | Optimizers | Seeds | L values | Runtime | Output |
|--------|---------|------------|-------|----------|---------|--------|
| `run_multistart_benchmark.py` ⭐ | 3 | 3 | 5 | Custom | ~5-10 min | MD report + plots |
| `quick_benchmark.py` | 3 | 1 | 1 | 4 | ~30-60 sec | Terminal + plots |
| `benchmark_large_systems.py` | 3 | 1 | 1 | 6, 8 | ~10-30 min | Terminal output |
| `run_longer_optimizations.py` | 3 | 1 | 1 | 4, 6 | ~30-60 min | Terminal output |

---

## Common Workflows

### Testing a New Ansatz

1. Add ansatz to `ssh_hubbard_vqe.py` or `ansatze/archived_ansatze.py`
2. Quick test: `python quick_benchmark.py`
3. Full test: `python run_multistart_benchmark.py --L 4`
4. Large system: `python benchmark_large_systems.py`

### Comparing Parameter Regimes

```bash
# Weak interaction
python run_multistart_benchmark.py --L 4 --U 0.5 --output-doc weak_U.md

# Strong interaction
python run_multistart_benchmark.py --L 4 --U 4.0 --output-doc strong_U.md

# Topological regime
python run_multistart_benchmark.py --L 4 --t1 1.0 --t2 0.2 --output-doc topo.md
```

### Investigating Convergence

```bash
# Short run to check if VQE is working
python quick_benchmark.py

# Longer run to see if more iterations help
python run_longer_optimizations.py

# Multi-start to assess initialization sensitivity
python run_multistart_benchmark.py --L 4
```

---

## Output Files

### Plots (`../docs/images/`)

**Multi-start convergence plots**:
- `convergence_{ansatz}_{optimizer}_L{L}.png`
- Shows all 5 seed trajectories + mean ± std
- Example: `convergence_hea_L_BFGS_B_L4.png`
- **Features**:
  - Left panel: Energy convergence vs iteration
  - Right panel: Relative error percentage (log scale) = 100 × |E_VQE - E_exact| / |E_exact|
  - Dense tick marks (15 major + 100 minor) for better readability
  - Plain number formatting (e.g., "10" instead of "1e1")
  - Gray lines: all seed runs; Blue: best seed; Red: mean ± std

**Single-run plots** (from quick_benchmark.py):
- `L{L}_{ansatz}_energy_convergence.png`: Energy vs iteration
- `L{L}_{ansatz}_error_log.png`: Log-scale relative error percentage convergence

### Reports (`../docs/`)

**Multi-start benchmark reports**:
- `multistart_benchmark_L{L}_{timestamp}.md`
- Contains:
  - Summary tables (all ansätze × optimizers)
  - Per-seed details
  - Best performer analysis
  - References to convergence plots

---

## Troubleshooting

### "No module named 'qiskit_algorithms'"

```bash
pip install --upgrade qiskit qiskit-algorithms
```

### "ImportError: cannot import name 'plot_multistart_convergence'"

Update to latest code:
```bash
git pull origin claude/read-this-r-01DLdEcvW8hustGKsyPjZzLM
```

### Benchmarks too slow

Reduce iterations or system size:
```bash
python run_multistart_benchmark.py --L 4 --maxiter 100
```

### Out of memory (L=8+)

The sparse Lanczos method is automatically used for L > 6. If still running out of memory:
- Close other applications
- Reduce `reps` parameter
- Test on smaller L first

---

## Performance Tips

1. **Start small**: Test with L=4 before running L=6 or L=8
2. **Use quick_benchmark.py first**: Verify setup before multi-start
3. **Reduce maxiter for testing**: Use 50-100 for quick tests, 200+ for final runs
4. **Monitor progress**: All scripts print progress to terminal

---

## Further Reading

- **Multi-start VQE Guide**: [`docs/MULTISTART_VQE_GUIDE.md`](../docs/MULTISTART_VQE_GUIDE.md)
- **Ansatz Overview**: [`docs/ANSATZ_OVERVIEW.md`](../docs/ANSATZ_OVERVIEW.md)
- **Sparse Lanczos**: [`docs/SPARSE_LANCZOS.md`](../docs/SPARSE_LANCZOS.md)
- **Main README**: [`README.md`](../README.md)

---

## Archived Scripts

Legacy benchmark scripts have been moved to `../docs/archived_benchmarks/` for historical reference:

- **compare_all_ansatze.py** - Older comprehensive comparison tool superseded by `run_multistart_benchmark.py`

These scripts are preserved for reference but are not recommended for new work. See `../docs/archived_benchmarks/README.md` for details.

Additionally, several archived ansätze (TopoInspired, Topo_RN, DQAP, TN_MPS, TN_MPS_NP) are available in `src/ansatze/archived_ansatze.py` but are not part of the standard benchmark suite.

---

## Contributing

When adding new benchmarks:

1. Follow naming convention: `{action}_{description}.py`
2. Add argparse for command-line options
3. Include docstring with usage examples
4. Update this README
5. Add output examples to `docs/`

---

## Summary

For most use cases, use:

**Quick test** → `quick_benchmark.py`
**Comprehensive benchmark** → `run_multistart_benchmark.py`
**Large systems** → `benchmark_large_systems.py`

The multi-start infrastructure (`run_multistart_benchmark.py`) provides the most robust and statistically meaningful results.
