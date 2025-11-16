# DMRG Error Investigation

This directory contains systematic tests investigating the TeNPy DMRG implementation's 1-3% systematic energy offset.

## Quick Summary

**Status**: Root cause not yet identified, but extensively narrowed down.

**Key Finding**: Error only appears when:
- L ≥ 4 (multiple unit cells)
- Both t1 and t2 are nonzero
- t2/t1 ≥ 0.5 (exact threshold)

## Test Sequence

Run these tests in order to reproduce the investigation:

### 1. Confirm the Error Exists
```bash
python 01_hamiltonian_mismatch.py
```
**Expected**: ~1.68% error for L=4, t1=1.0, t2=0.6, U=2.0

### 2. Verify L=2 Works
```bash
python 02_verify_L2.py
```
**Expected**: Perfect agreement (0.00% error)
**Conclusion**: Unit cell structure is correct, error is size-dependent

### 3. Test Isolated Dimers (t2=0)
```bash
python 03_isolated_dimers.py
```
**Expected**: Perfect agreement (0.00% error)
**Conclusion**: Intra-cell coupling alone is correct

### 4. Test Inter-Cell Only (t1=0)
```bash
python 04_only_intercell.py
```
**Expected**: Perfect agreement (0.00% error)
**Conclusion**: Inter-cell coupling alone is correct

### 5. Find Exact Threshold
```bash
python 05_find_threshold.py
```
**Expected**: Error appears exactly at t2 = 0.50 (when t1 = 1.0)
**Conclusion**: Sharp threshold at t2/t1 = 0.5, not gradual

### 6. Scan Parameter Space
```bash
python 06_varying_ratio.py
```
**Expected**: Error increases with t2/t1 ratio above threshold
**Conclusion**: Error magnitude scales with how far t2 exceeds t1/2

### 7. Test Convergence Independence
```bash
python 07_convergence_test.py
```
**Expected**: Error constant for χ = 10 to 1000
**Conclusion**: NOT a convergence issue, it's a Hamiltonian bug

## Key Results

| Test | L | t1 | t2 | Result | Implication |
|------|---|----|----|--------|-------------|
| 02 | 2 | any | any | ✓ Perfect | Unit cell OK |
| 03 | 4 | 1.0 | 0.0 | ✓ Perfect | Intra-cell OK |
| 04 | 4 | 0.0 | 1.0 | ✓ Perfect | Inter-cell OK |
| 05 | 4 | 1.0 | <0.5 | ✓ Perfect | Threshold found |
| 05 | 4 | 1.0 | ≥0.5 | ✗ Error | Above threshold |
| 07 | 4 | 1.0 | 0.6 | ✗ 1.68% | χ-independent |

## Detailed Reports

- **DEBUG_REPORT.md** - Complete investigation (300+ lines)
  - All test results
  - What was ruled out
  - Hypotheses about root cause
  - Next steps for debugging

- **FIX_TEST_REPORT.md** - Failed fix attempt
  - Hypothesis: TeNPy adds automatic 1/2 factor
  - Fix: Doubled hopping coefficients
  - Result: Made error WORSE (1.68% → 147%)
  - Conclusion: Hypothesis was wrong

## Interpretation

The error appears to be a **TeNPy bug or limitation** in how it handles multiple coupling types of comparable strength in unit cell models.

Evidence:
1. **Exact threshold** at t2 = t1/2 suggests hardcoded logic
2. **Each coupling type works alone** → not a single bug
3. **χ-independent** → not convergence, but Hamiltonian
4. **Size-dependent** (L=2 works, L=4 fails) → unit cell interaction

Most likely causes:
- TeNPy has different code paths for "strong dimerization" vs "weak dimerization"
- Automatic neighbor detection activates at certain coupling ratios
- Bug in how multiple `add_coupling` calls interact for unit cell models

## Recommendations

**For VQE validation:**
- ✓ Use DMRG for t2/t1 < 0.5 (weak dimerization)
- ✗ Do NOT use DMRG for t2/t1 ≥ 0.5
- ✓ Always verify against exact diag when possible (L ≤ 6)

**For further debugging:**
- Consult TeNPy documentation/community
- Examine TeNPy source code for threshold logic
- Try alternative model construction methods
- Compare with other DMRG implementations

## Files Modified During Investigation

All failed attempts have been removed. Only essential diagnostic tests remain:
- ✗ Removed: test_without_plus_hc.py (manual h.c. failed)
- ✗ Removed: test_fixed_positions.py (lattice positions didn't help)
- ✗ Removed: inspect_tenpy_model.py (incomplete debugging)
- ✗ Removed: print_tenpy_bonds.py (incomplete debugging)
- ✗ Removed: check_lattice_neighbors.py (not relevant)
- ✗ Removed: diagnose_dmrg_hamiltonian.py (redundant)
- ✗ Removed: verify_dmrg_fix_logic.py (for failed fix)

## Quick Reference

```bash
# Run all tests
for f in 0*.py; do echo "=== $f ==="; python $f 2>&1 | tail -20; done

# Just check the threshold
python 05_find_threshold.py 2>&1 | grep -E "t2|Error"

# Verify χ-independence
python 07_convergence_test.py 2>&1 | grep -E "χ_max|Error"
```
