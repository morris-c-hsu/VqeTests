# DMRG Fix Testing Report

## Fix Applied
**Date**: 2025-11-16
**Commit**: 991d577
**File**: `src/ssh_hubbard_tenpy_dmrg_fixed.py`

### Change Summary
Modified hopping term coefficients to compensate for TeNPy's automatic 1/2 factor:
- Line 149: `self.add_coupling(-t1, ...)` → `self.add_coupling(-2*t1, ...)`
- Line 151: `self.add_coupling(-t1, ...)` → `self.add_coupling(-2*t1, ...)`
- Line 155: `self.add_coupling(-t2, ...)` → `self.add_coupling(-2*t2, ...)`
- Line 157: `self.add_coupling(-t2, ...)` → `self.add_coupling(-2*t2, ...)`

---

## Tests Completed (Without TeNPy)

### ✓ Test 1: Syntax Validation
**Status**: PASSED
**Command**: `python3 -c "compile(code, 'file', 'exec')"`
**Result**: No syntax errors detected

### ✓ Test 2: Fix Application Verification
**Status**: PASSED
**Verified**: Code contains `-2*t1` and `-2*t2` as expected

### ✓ Test 3: Logic Verification
**Script**: `tests/verify_dmrg_fix_logic.py`
**Status**: PASSED
**Findings**:
- Error direction matches hypothesis (DMRG energies too high)
- Physics is consistent (weak hopping → less negative energy)
- Fix direction is correct (double t → compensate for 1/2 factor)
- Expected outcome: errors should reduce from 1-3% to <0.01%

---

## Tests REQUIRED (With TeNPy)

### ⚠ Test 4: Actual DMRG Energy Validation
**Status**: PENDING (requires TeNPy installation)
**Command**: `python tests/test_dmrg_hamiltonian_mismatch.py`

**Expected Results**:

| System | Exact Energy | Old DMRG | Old Error | Expected New DMRG | Expected New Error |
|--------|--------------|----------|-----------|-------------------|-------------------|
| L=4, t1=1.0, t2=0.6, U=2.0 | -2.6585 | -2.6139 | 1.68% | -2.6585 ± 0.0001 | <0.01% |
| L=6, t1=1.0, t2=0.5, U=2.0 | -4.0107 | -3.9059 | 2.61% | -4.0107 ± 0.0001 | <0.01% |

**Pass Criteria**:
- Absolute error < 0.0001 (10⁻⁴)
- Relative error < 0.01%
- Error does NOT depend on bond dimension χ

**Failure Indicators**:
- If error is still ~1-3%: Fix hypothesis was wrong
- If error changed but not eliminated: Partial fix, other issues present
- If error increased: Wrong direction (unlikely given logic check)

---

## Alternative Testing (If TeNPy Unavailable)

### Option 1: Mock TeNPy Behavior
Create a simple test that simulates TeNPy's `add_coupling` with/without the 1/2 factor and verify the fix compensates correctly.

### Option 2: Code Review
Review TeNPy documentation/source to confirm `plus_hc=True` behavior:
- Does it add: `strength * (c†c + h.c.)` ?
- Or does it add: `strength/2 * (c†c + h.c.)` ?

### Option 3: Numerical Bounds Check
Verify that the fix doesn't create obviously wrong energies:
- New energies should be within physical bounds
- Ground state should still be the lowest eigenvalue
- Energy should scale correctly with system size

---

## Risk Assessment

### Low Risk Scenarios (Fix Likely Correct)
- ✓ Logic check passed
- ✓ Error direction matches hypothesis
- ✓ Fix magnitude is reasonable (factor of 2)
- ✓ No syntax errors introduced

### Medium Risk Scenarios (Needs Verification)
- ⚠ TeNPy documentation not directly consulted
- ⚠ Assumption about plus_hc behavior not confirmed
- ⚠ Other possible sources of error not ruled out

### High Risk Scenarios (Fix Might Be Wrong)
- ✗ If TeNPy's `plus_hc=True` does NOT include 1/2 factor
- ✗ If error is from unit cell mapping, not normalization
- ✗ If error is from fermion signs, not hopping strength

---

## Rollback Plan

If testing shows the fix is incorrect:

1. **Revert changes**:
   ```bash
   git revert 991d577
   ```

2. **Investigate alternatives**:
   - Check unit cell site ordering
   - Verify fermion sign conventions
   - Compare with TeNPy examples for fermionic systems
   - Test single hopping term in isolation

3. **Document findings** in DMRG_STATUS.md

---

## Next Steps

1. **PRIORITY**: Install TeNPy and run Test 4
   ```bash
   pip install physics-tenpy
   python tests/test_dmrg_hamiltonian_mismatch.py
   ```

2. **If Test 4 passes**:
   - Update DMRG_STATUS.md: "FIXED ✓"
   - Update README.md: Remove warnings about DMRG unreliability
   - Run full DMRG benchmarks for L=4,6,8,12

3. **If Test 4 fails**:
   - Document actual results vs expected
   - Rollback fix if error increased
   - Investigate alternative hypotheses
   - Consider consulting TeNPy documentation/examples

---

## Confidence Level

**Current Confidence**: 75%

**Reasoning**:
- ✓ Logic is sound (physics checks out)
- ✓ Direction is correct (fix will lower energies)
- ✓ Magnitude is plausible (factor of 2 is common normalization)
- ⚠ Hypothesis not confirmed from TeNPy docs
- ⚠ No actual DMRG test run yet

**Confidence will increase to 95%** after successful Test 4 execution.

---

## Test Checklist

- [x] Syntax validation
- [x] Fix application verification
- [x] Logic verification
- [x] Physics consistency check
- [ ] **Actual DMRG energy test** ← REQUIRED
- [ ] Multiple system sizes (L=4,6,8)
- [ ] Multiple parameters (different t1, t2, U)
- [ ] Bond dimension independence check
- [ ] Benchmark against literature values

---

## Conclusion

The fix has been **logically validated** but requires **empirical testing with TeNPy** to confirm it resolves the 1-3% systematic error. All pre-testing validation has passed, and the hypothesis is physically sound. The next critical step is running the actual DMRG calculation to verify the corrected energies match exact diagonalization.
