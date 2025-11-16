# Test Suite Organization

This directory contains comprehensive tests for the SSH-Hubbard VQE implementation.

## Main Tests

### Sparse Lanczos Diagonalization

- **test_sparse_lanczos.py** - Comprehensive validation of sparse Lanczos implementation
  - Verifies sparse and dense methods agree for L ≤ 6
  - Tests sparse method for L = 7, 8
  - Performance benchmarks

- **test_L7_benchmark.py** - Integration test for L=7 system
  - Tests exact diagonalization with benchmark functions
  - Validates physically reasonable results

## DMRG Error Investigation (dmrg/)

Systematic tests investigating the DMRG 1-3% systematic error.
See `DMRG_DEBUG_REPORT.md` for complete findings.

### Test Sequence

1. **01_hamiltonian_mismatch.py** - Main DMRG validation test
   - Compares DMRG vs exact diagonalization for L=4
   - Documents the 1.68% error for standard parameters

2. **02_verify_L2.py** - L=2 verification
   - Shows DMRG works perfectly for L=2
   - Validates VQE Hamiltonian construction
   - Manual Hamiltonian construction for verification

3. **03_isolated_dimers.py** - Test with t2=0
   - Shows DMRG works perfectly with isolated dimers
   - Rules out intra-cell coupling as error source

4. **04_only_intercell.py** - Test with t1=0
   - Shows DMRG works perfectly with only inter-cell hopping
   - Rules out inter-cell coupling alone as error source

5. **05_find_threshold.py** - Find exact error threshold
   - Scans t2 values to find where error appears
   - **Key finding**: Error threshold at exactly t2 = t1/2

6. **06_varying_ratio.py** - Systematic parameter scan
   - Tests multiple t1/t2 ratios
   - Documents error pattern vs parameters

7. **07_convergence_test.py** - χ-independence test
   - Shows error constant for χ = 10 to 1000
   - Proves error is NOT a convergence issue

### Key Findings

**Error Conditions:**
- ✓ L=2: Always perfect
- ✓ L=4 with t2=0: Perfect (isolated dimers)
- ✓ L=4 with t1=0: Perfect (inter-cell only)
- ✓ L=4 with t2/t1 < 0.5: Perfect
- ✗ L=4 with t2/t1 ≥ 0.5: Error (1-6%)

**Error Characteristics:**
- Sharp threshold at t2 = t1/2 (not gradual)
- Independent of bond dimension χ
- Both coupling types work perfectly in isolation
- Error only appears when both are present and t2 ≥ t1/2

**Conclusion:**
- NOT a convergence issue (χ-independent)
- NOT a single coupling bug (each works alone)
- Likely a TeNPy bug in handling multiple coupling types of comparable strength
- Root cause still under investigation

## Documentation

- **DMRG_DEBUG_REPORT.md** - Comprehensive debugging report (300+ lines)
- **DMRG_FIX_TEST_REPORT.md** - Failed fix attempt documentation
- **DMRG_STATUS.md** - Current status summary
- **SPARSE_LANCZOS.md** - Sparse diagonalization documentation

## Running Tests

### All DMRG tests
```bash
python tests/dmrg/01_hamiltonian_mismatch.py
python tests/dmrg/05_find_threshold.py
```

### Sparse Lanczos tests
```bash
python tests/test_sparse_lanczos.py
python tests/test_L7_benchmark.py
```

## Test Philosophy

- **Keep**: Essential validation tests and valuable diagnostic tools
- **Archive**: Failed attempts documented in reports, code removed
- **Document**: Key findings preserved in markdown reports

## Notes

- DMRG tests require TeNPy installation
- Sparse Lanczos tests require scipy
- All tests use parameters from the main VQE implementation
