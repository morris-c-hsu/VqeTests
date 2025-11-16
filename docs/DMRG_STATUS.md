# TeNPy DMRG Implementation Status

## Current Status: ROOT CAUSE UNDER INVESTIGATION

**Update**: Extensive debugging has narrowed down the issue significantly, but the root cause in TeNPy has not been identified. A previous fix attempt (doubling hopping coefficients) was tested and **failed** - it made the error worse (1.68% → 147%). The error is **conditional** and only appears under specific circumstances.

## Critical Discovery: Error is Conditional

The error **does NOT always appear**. It depends on system size and hopping parameter ratios:

**Works perfectly** (error < 0.01%):
- ✓ L=2 with any parameters
- ✓ L=4 with t2=0 (isolated dimers)
- ✓ L=4 with t1=0 (inter-cell hopping only)
- ✓ L=4 with t2/t1 < 0.5 (weak dimerization)

**Has systematic error** (1-6%):
- ✗ L=4 with t2/t1 ≥ 0.5 (moderate to strong dimerization)
- ✗ Error increases as t2 approaches or exceeds t1

**Key finding**: Error threshold is **exactly** at t2 = t1/2, not approximately.

## Error Characteristics

1. **Threshold behavior**: Sharp onset at t2/t1 = 0.5
   - t2/t1 = 0.49: Perfect (0.00% error)
   - t2/t1 = 0.50: Tiny error (0.01%)
   - t2/t1 = 0.51: Significant error (0.19%)
   - t2/t1 = 0.60: Large error (1.68%)

2. **Not a convergence issue**: Error is identical for χ=10 to χ=1000
   - Actual bond dimension used: only χ≈16
   - Error magnitude constant regardless of χ_max
   - → This is a Hamiltonian construction bug, not DMRG approximation

3. **Interference between coupling types**: Each type works perfectly alone
   - Intra-cell only (t1≠0, t2=0): Perfect
   - Inter-cell only (t1=0, t2≠0): Perfect
   - Both together with t2≥t1/2: Error appears

## What Was Tested

**Failed fix attempts**:
- ✗ Doubling hopping coefficients (made error worse: 147%)
- ✗ Changing lattice position parameters (no effect)
- ✗ Manual hermitian conjugate addition (implementation failed)

**Verification tests**:
- ✓ VQE Hamiltonian verified correct by manual construction
- ✓ Bond pattern verified correct (SSH structure preserved)
- ✓ Unit cell mapping verified
- ✓ Jordan-Wigner fermion signs verified

See `tests/DMRG_DEBUG_REPORT.md` for complete debugging details.

## Previous Systematic Offset (~1-3%)

Before the fix, DMRG produced approximate results with systematic energy offset compared to exact diagonalization. This offset did not decrease with increasing bond dimension, indicating a Hamiltonian construction issue rather than convergence limitations.

## Implementation Details

### Architecture
- **Model**: SpinfulSSHHubbard (ssh_hubbard_tenpy_dmrg_fixed.py)
- **Lattice**: Unit cell structure with 4 sites per cell [A↑, A↓, B↑, B↓]
- **Unit cells**: L_phys/2 cells (dimers)
- **Total MPS sites**: 2*L_phys
- **Boundary conditions**: Open (finite MPS)

### Hamiltonian Terms
1. **Intra-cell hopping** (t1, strong): A→B within dimer
2. **Inter-cell hopping** (t2, weak): B of cell i → A of cell i+1
3. **Hubbard interaction** (U): n_up * n_down at each physical site

### Convergence Results

| L | Exact Energy | DMRG Energy | Error | Rel. Error | χ_max |
|---|--------------|-------------|-------|------------|-------|
| 4 | -2.6585      | -2.6139     | 0.045 | 1.68%      | 200   |
| 6 | -4.0107      | -3.9059     | 0.105 | 2.61%      | 200   |

**Critical Observation**: Error does **NOT** decrease with increasing χ_max (tested up to 500), indicating this is **not a convergence issue** but a systematic offset in the Hamiltonian construction or operator conventions.

## Known Issues

### 1. Systematic Energy Offset (~1-3%) - PRIMARY ISSUE

**Symptom**: DMRG energies are consistently less negative (higher) than exact results by 1-3%, **independent of χ_max**.

**Key Evidence**:
- Offset persists even when χ_max increased to 500
- Error magnitude does NOT decrease with better convergence
- Indicates Hamiltonian mismatch, not DMRG approximation quality

**Likely Causes** (under investigation):
- Unit cell interpretation differences between TeNPy and Qiskit frameworks
- Jordan-Wigner string handling or parity conventions
- Fermion sign convention mismatches
- Hopping pattern mapping between unit cell structure and physical sites

**Impact**:
- ⚠ Results are **approximate**, not exact
- ⚠ Cannot be used as exact benchmark for VQE validation
- ✓ May still be useful for relative energy comparisons (use with caution)
- ✓ Can handle L≥8 systems where exact diagonalization is impossible
- ⚠ **NOT suitable for applications requiring absolute energy accuracy**

### 2. Requires Even L

The dimer unit cell structure requires L to be even. Odd L systems not currently supported.

## Usage

```python
from ssh_hubbard_tenpy_dmrg_fixed import run_dmrg_ssh_hubbard

# Run DMRG for L=8 (beyond exact diag capability)
result = run_dmrg_ssh_hubbard(
    L=8,
    t1=1.0,
    t2=0.5,
    U=2.0,
    chi_max=150,
    verbose=True
)

print(f"Ground energy: {result['energy']:.6f}")
print(f"Bond dimension: {max(result['chi'])}")
```

## Successful Runs

✓ L=4:  Runs successfully (chi reaches ~32)
✓ L=6:  Runs successfully (chi reaches ~64)
✓ L=8:  Runs successfully (chi reaches ~150)
✓ L=12: Runs successfully (chi reaches ~200)
✓ L=16+: Expected to work (not yet tested)

## DMRG Reference Values (Approximate; Known Systematic Offset)

**WARNING**: These are **NOT exact** energies. Systematic offset ~1-3% observed.

| L  | t1  | t2  | U   | χ_max | E_DMRG (approx) | E/site | Known Exact | Rel. Error |
|----|-----|-----|-----|-------|-----------------|--------|-------------|------------|
| 4  | 1.0 | 0.6 | 2.0 | 200   | -2.6139         | -0.6535| -2.6585     | 1.68%      |
| 6  | 1.0 | 0.5 | 2.0 | 100   | -3.9059         | -0.6510| -4.0107     | 2.61%      |
| 8  | 1.0 | 0.5 | 2.0 | 150   | -5.2420         | -0.6552| N/A         | Unknown    |
| 12 | 1.0 | 0.5 | 2.0 | 200   | -7.9140         | -0.6595| N/A         | Unknown    |

**Critical Notes**:
- L=4, L=6: Systematic offset confirmed by comparison with exact diagonalization
- L=8, L=12: No exact reference available; actual error unknown
- **Do NOT use these as exact benchmarks**
- For L≥8, true ground state energy is unknown (DMRG offset not characterized)

## Recommendations

### For New Users
- ⚠ **DO NOT use current DMRG results as exact benchmarks**
- ⚠ Be aware of ~1-3% systematic energy offset (does not decrease with χ)
- ⚠ For L≥8, no exact validation is available
- ✓ May be useful for rough exploratory calculations (use with caution)

### Debugging TODO List

**High Priority** (required to fix systematic offset):
1. ☐ Operator-by-operator comparison with Qiskit Hamiltonian matrix elements
2. ☐ Verify hopping pattern matches physical SSH structure
3. ☐ Check unit cell definition and MPS site ordering
4. ☐ Verify Jordan-Wigner parity conventions in TeNPy
5. ☐ Test single-bond Hamiltonian terms individually
6. ☐ Compare with alternative TeNPy model construction (NearestNeighborModel)
7. ☐ Validate fermion anticommutation relations in unit cell basis

**Until Fixed**:
- Results should be labeled as "approximate DMRG calculation"
- Cannot serve as exact reference for VQE validation
- Offset magnitude at L≥8 is unknown

### Current Suitability

**NOT suitable for**:
- ✗ Exact energy benchmarks for VQE validation
- ✗ High-precision energy calculations
- ✗ Quantitative comparison with literature values
- ✗ Applications requiring <1% energy accuracy

**May be useful for** (with appropriate caveats):
- ⚠ Exploratory calculations for L≥8 (offset unknown)
- ⚠ Qualitative trends (if offset is approximately constant)
- ⚠ Entanglement structure studies (if wavefunction quality is better than energy)

**Recommended approach**: Resolve systematic offset before using for scientific conclusions.

## Files

- `ssh_hubbard_tenpy_dmrg_fixed.py` - Main DMRG implementation
- `test_dmrg_convergence.py` - Convergence testing script
- `dmrg_test_fixed.txt` - Latest test output
- `DMRG_STATUS.md` - This file

## Conclusion

The TeNPy DMRG implementation is **functional but produces approximate results** with a persistent systematic energy offset (~1-3%). This offset does **NOT** decrease with increasing bond dimension, indicating a Hamiltonian construction or convention mismatch rather than a convergence issue.

**Current Status**:
- **NOT suitable as exact benchmark** for VQE or other methods
- Offset cause under investigation (likely unit cell or Jordan-Wigner convention mismatch)
- For L≥8 systems, no exact validation available (true error unknown)

**Recommendation**: Complete debugging TODO list before using DMRG results for scientific validation or quantitative comparisons. Until fixed, all DMRG energies should be treated as exploratory approximations only.
