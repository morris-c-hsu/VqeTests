# dmrgpy Implementation for SSH-Hubbard DMRG

## Overview

This document describes the **dmrgpy-based DMRG implementation** for the SSH-Hubbard model, providing an alternative to the TeNPy implementation that exhibits systematic errors in certain parameter regimes.

### Why dmrgpy?

**TeNPy Issue**: The TeNPy DMRG implementation has a systematic 1-3% energy error for `t2/t1 ≥ 0.5` that is **independent of bond dimension**, indicating a Hamiltonian construction bug rather than convergence issues.

**dmrgpy Solution**: dmrgpy provides Python bindings to **ITensor** (C++ and Julia), a well-tested tensor network library that may avoid TeNPy's systematic error.

---

## Installation

### Prerequisites

dmrgpy requires:
- Python 3.7+
- C++ compiler (g++ ≥ 6) OR Julia runtime
- LAPACK and BLAS libraries

### Install dmrgpy

```bash
pip install dmrgpy
```

This will automatically install ITensor backend dependencies.

### Verify Installation

```python
from dmrgpy import fermionchain
print("dmrgpy installed successfully!")
```

---

## Implementation Details

### File Structure

```
src/
  ssh_hubbard_dmrgpy.py        # Main dmrgpy DMRG implementation

tests/
  test_dmrgpy_validation.py    # Validation against exact diagonalization

docs/
  DMRGPY_IMPLEMENTATION.md     # This file
```

### Hamiltonian Construction

dmrgpy uses **operator algebra** to build Hamiltonians:

```python
from dmrgpy import fermionchain

# Create spinful fermionic chain
fc = fermionchain.Spinful_Fermionic_Chain(L)

# Build hopping terms with alternating t1/t2
h = 0
for i in range(L - 1):
    t = t1 if i % 2 == 0 else t2  # SSH alternation

    # Spin-up hopping
    h = h - t * fc.Cdagup[i] * fc.Cup[i + 1]

    # Spin-down hopping
    h = h - t * fc.Cdagdn[i] * fc.Cdn[i + 1]

# Add Hermitian conjugate
h = h + h.get_dagger()

# Hubbard interaction: U * n_up * n_down
for i in range(L):
    h = h + U * fc.Nup[i] * fc.Ndn[i]

# Set Hamiltonian and run DMRG
fc.set_hamiltonian(h)
energy = fc.get_gs()
```

### Key Differences from TeNPy

| Feature | TeNPy | dmrgpy |
|---------|-------|--------|
| **Backend** | Pure Python + C extensions | ITensor (C++ or Julia) |
| **API Style** | High-level (`add_coupling`) | Low-level (operator algebra) |
| **Fermions** | Built-in `FermiSite` | Manual Jordan-Wigner via operators |
| **Hamiltonian** | Automatic MPO construction | Manual operator building |
| **Error at t2/t1≥0.5** | 1-3% systematic error | Unknown (to be tested) |

---

## Usage

### Basic Usage

```python
from ssh_hubbard_dmrgpy import solve_ssh_hubbard_dmrgpy

# Solve SSH-Hubbard model
result = solve_ssh_hubbard_dmrgpy(
    L=4,           # Number of sites
    t1=1.0,        # Strong hopping
    t2=0.6,        # Weak hopping (t2/t1 = 0.6, critical regime!)
    U=1.0,         # Hubbard interaction
    maxm=200,      # Bond dimension
    nsweeps=10,    # DMRG sweeps
    cutoff=1e-8    # Truncation cutoff
)

print(f"Ground state energy: {result['energy']:.10f}")
print(f"Converged: {result['converged']}")
```

### Validation Against Exact Diagonalization

```python
from ssh_hubbard_dmrgpy import solve_ssh_hubbard_dmrgpy, compare_with_exact
from ssh_hubbard_vqe import exact_diagonalization_ssh_hubbard

# Parameters in TeNPy's problematic regime
L, t1, t2, U = 4, 1.0, 0.6, 1.0

# Exact energy
exact_energy = exact_diagonalization_ssh_hubbard(L, t1, t2, U)

# DMRG energy
dmrg_result = solve_ssh_hubbard_dmrgpy(L, t1, t2, U)

# Compare
comparison = compare_with_exact(
    L, t1, t2, U,
    exact_energy, dmrg_result['energy']
)

print(f"Relative error: {comparison['error_rel_pct']:.6f}%")
```

### Run Validation Suite

```bash
python tests/test_dmrgpy_validation.py
```

This runs systematic tests comparing dmrgpy against exact diagonalization in:
- Safe regime (t2/t1 < 0.5)
- Threshold (t2/t1 = 0.5)
- **Critical regime (t2/t1 ≥ 0.5) where TeNPy fails**

---

## Expected Outcomes

### Test Cases

| System | t2/t1 | TeNPy Error | dmrgpy Expected |
|--------|-------|-------------|-----------------|
| L=2    | 0.6   | Perfect (<0.01%) | Perfect |
| L=4    | 0.4   | Perfect | Perfect |
| L=4    | 0.5   | Threshold (~0.01%) | Should be perfect |
| L=4    | 0.6   | **1.68% ERROR** | **Test if ITensor avoids this** |
| L=4    | 0.7   | 2-3% error | Test |
| L=6    | 0.6   | 2.61% error | Test |

### Success Criteria

**Excellent** (dmrgpy error < 0.1%):
- ITensor does NOT have TeNPy's bug
- dmrgpy can be used for SSH-Hubbard DMRG
- Recommendation: Switch from TeNPy to dmrgpy

**Good** (dmrgpy error < 1.0%):
- ITensor performs better than TeNPy
- Acceptable for qualitative studies
- Still better alternative to TeNPy

**Problematic** (dmrgpy error > 1.0%):
- ITensor may have similar issues
- Requires investigation
- May need to stick with exact diagonalization

---

## Scientific Validation

### Hamiltonian Verification

The SSH-Hubbard Hamiltonian is:

```
H = -∑_{i,σ} t_i (c†_{i,σ} c_{i+1,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓}
```

where:
- t_i alternates: t₁ for even bonds (0-1, 2-3, ...), t₂ for odd bonds (1-2, 3-4, ...)
- U is the on-site Hubbard repulsion

### Operator Mapping

dmrgpy operators for spinful fermions:

| Physics | dmrgpy Operator |
|---------|----------------|
| c†_{i,↑} | `fc.Cdagup[i]` |
| c_{i,↑} | `fc.Cup[i]` |
| c†_{i,↓} | `fc.Cdagdn[i]` |
| c_{i,↓} | `fc.Cdn[i]` |
| n_{i,↑} | `fc.Nup[i]` |
| n_{i,↓} | `fc.Ndn[i]` |

### Jordan-Wigner Signs

dmrgpy automatically handles Jordan-Wigner strings for fermionic anticommutation. The operator products like `fc.Cdagup[i] * fc.Cup[j]` correctly include the necessary phase factors.

---

## Backend Options

### C++ Backend (Default)

```python
fc = fermionchain.Spinful_Fermionic_Chain(L)
# Uses ITensor C++ (faster, requires compilation)
```

**Pros**: Faster, more stable
**Cons**: Requires C++ compiler and LAPACK/BLAS

### Julia Backend (Alternative)

```python
fc = fermionchain.Spinful_Fermionic_Chain(L)
fc.setup_julia()  # Switch to ITensors.jl
```

**Pros**: Modern, actively developed
**Cons**: Requires Julia installation, slower startup

---

## Troubleshooting

### Installation Issues

**Problem**: `pip install dmrgpy` fails with compiler errors

**Solution**:
1. Ensure g++ ≥ 6 is installed: `g++ --version`
2. Install LAPACK/BLAS:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install liblapack-dev libblas-dev

   # macOS
   brew install lapack
   ```
3. Retry: `pip install dmrgpy`

**Alternative**: Use Julia backend instead of C++

### Runtime Errors

**Problem**: DMRG fails to converge

**Solutions**:
- Increase bond dimension: `maxm=500`
- More sweeps: `nsweeps=20`
- Tighter cutoff: `cutoff=1e-10`

**Problem**: `ImportError: No module named dmrgpy`

**Solution**: Check installation:
```python
import sys
print(sys.path)  # Verify dmrgpy is in path
```

---

## Comparison with Existing Methods

### Exact Diagonalization (ED)
- **Accuracy**: Exact (within numerical precision)
- **Scaling**: Exponential, limited to L ≤ 8
- **Use**: Ground truth reference

### TeNPy DMRG
- **Accuracy**: 1-3% error for t2/t1 ≥ 0.5 (systematic bug)
- **Scaling**: Polynomial in L
- **Use**: Currently **NOT RECOMMENDED** for SSH-Hubbard

### dmrgpy DMRG (This Implementation)
- **Accuracy**: To be determined by validation tests
- **Scaling**: Polynomial in L (same as TeNPy)
- **Use**: Potential replacement for TeNPy if validation succeeds

### VQE (Qiskit)
- **Accuracy**: Optimizer-dependent, typically 0.1-10% error
- **Scaling**: Exponential circuit depth
- **Use**: Algorithm development and comparison

---

## API Reference

### `solve_ssh_hubbard_dmrgpy()`

```python
solve_ssh_hubbard_dmrgpy(
    L: int,                # Number of lattice sites
    t1: float = 1.0,       # Strong hopping
    t2: float = 0.6,       # Weak hopping
    U: float = 1.0,        # Hubbard interaction
    maxm: int = 200,       # Maximum bond dimension
    nsweeps: int = 10,     # Number of DMRG sweeps
    cutoff: float = 1e-8,  # Truncation cutoff
    use_julia: bool = False,  # Use Julia backend
    verbose: bool = True   # Print progress
) -> dict
```

**Returns**:
```python
{
    'energy': float,          # Ground state energy
    'converged': bool,        # Convergence status
    'bond_dimension': int,    # Final bond dimension
    'parameters': dict        # Input parameters
}
```

### `compare_with_exact()`

```python
compare_with_exact(
    L: int,
    t1: float,
    t2: float,
    U: float,
    exact_energy: float,   # From exact diagonalization
    dmrg_energy: float,    # From dmrgpy
    verbose: bool = True
) -> dict
```

**Returns**:
```python
{
    'exact_energy': float,
    'dmrg_energy': float,
    'error_abs': float,        # Absolute error
    'error_rel_pct': float,    # Relative error (%)
    't2_t1_ratio': float
}
```

---

## Future Work

### Short Term
1. **Run validation suite** on systems with dmrgpy installed
2. **Document results** in this file
3. **Compare performance** (runtime) vs TeNPy

### Medium Term
1. **Extend to larger systems** (L = 6, 8, 10)
2. **Parameter sweeps** in δ (dimerization)
3. **Observable calculations** (correlations, entanglement)

### Long Term
1. **2D SSH-Hubbard** (if dmrgpy supports 2D)
2. **Finite temperature** DMRG
3. **Dynamics** (time evolution)

---

## References

### dmrgpy
- **Repository**: https://github.com/joselado/dmrgpy
- **Documentation**: See GitHub README and examples

### ITensor
- **C++ Version**: https://itensor.org/
- **Julia Version**: https://github.com/ITensor/ITensors.jl
- **Paper**: Fishman et al., "The ITensor Software Library for Tensor Network Calculations", SciPost Phys. Codebases 4 (2022)

### SSH-Hubbard Model
- Su, Schrieffer, Heeger, Phys. Rev. Lett. 42, 1698 (1979) - SSH model
- Hubbard, Proc. R. Soc. A 276, 238 (1963) - Hubbard model
- Various studies of SSH-Hubbard (see repository references)

---

## Status

**Implementation**: ✓ Complete
**Documentation**: ✓ Complete
**Validation**: ⏳ Pending (requires dmrgpy installation)
**Recommendation**: ⏳ Awaiting validation results

---

*Last updated: 2025-11-17*
