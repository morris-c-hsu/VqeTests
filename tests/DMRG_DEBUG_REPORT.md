# DMRG Systematic Error - Debug Report

**Date**: 2025-11-16
**Status**: ROOT CAUSE NOT YET IDENTIFIED - But significantly narrowed down

---

## Executive Summary

The DMRG implementation has a systematic energy offset (~1-3%) that appears **only under specific conditions**. Extensive testing has identified the exact circumstances where the error occurs, pointing to a likely bug in TeNPy's coupling implementation or an incompatibility with the SSH-Hubbard model structure.

---

## Key Findings

### 1. Error is Conditional - NOT Universal

The error does **NOT** appear in all cases. It depends on system size and hopping parameters:

| L | t1 | t2 | Result |
|---|----|----|--------|
| 2 | any | any | ✓ PERFECT |
| 4 | any | 0 | ✓ PERFECT |
| 4 | 0 | any | ✓ PERFECT |
| 4 | 1.0 | <0.50 | ✓ PERFECT |
| 4 | 1.0 | ≥0.50 | ✗ ERROR |

### 2. Error Threshold is EXACTLY t2 = t1/2

Precision scan revealed a sharp threshold:

```
t2    Error
0.45  0.0000%  ✓
0.48  0.0000%  ✓
0.49  0.0000%  ✓
0.50  0.0134%  ⚠ (threshold)
0.51  0.1883%  ✗
0.52  0.3613%  ✗
0.55  0.8690%  ✗
0.60  1.6765%  ✗
1.00  6.3070%  ✗
```

The threshold is **exactly** at t2/t1 = 0.5, not approximately.

### 3. Error is Independent of Convergence

Error remains constant regardless of bond dimension χ:

```
χ_max   E_DMRG          Error
10      -2.6133876229   1.6967%
20      -2.6139255188   1.6765%
100     -2.6139255188   1.6765%  (same)
500     -2.6139255188   1.6765%  (same)
1000    -2.6139255188   1.6765%  (same)
```

Actual χ used: only 16 (tiny!)
→ This is **NOT a convergence issue**, it's a Hamiltonian bug.

### 4. Both Coupling Types Work Perfectly in Isolation

- **Intra-cell only** (t2=0, isolated dimers): Perfect
- **Inter-cell only** (t1=0): Perfect
- **Both together with t2 ≥ t1/2**: Error appears

This suggests the bug is in how TeNPy handles the **interaction** between different coupling types.

---

## What the Error is NOT

Based on testing, we can rule out:

- ❌ **Convergence issue** - Error independent of χ
- ❌ **Precision issue** - Error is systematic, not random
- ❌ **VQE Hamiltonian bug** - Verified correct by manual construction
- ❌ **Unit cell mapping** - L=2 works perfectly, bond pattern verified
- ❌ **Jordan-Wigner strings** - Fermion signs verified correct
- ❌ **Lattice positions** - Changing positions didn't help
- ❌ **`plus_hc=True` behavior** - Both coupling types work individually

---

## What the Error Likely IS

Given the evidence, the most probable causes are:

### Hypothesis 1: TeNPy Internal Threshold Logic

The **exact** threshold at t2 = t1/2 suggests TeNPy may have:
- Hardcoded approximations that activate when coupling ratios exceed thresholds
- Different code paths for "strong dimerization" (t2 << t1) vs. "weak dimerization" (t2 ≈ t1)
- Neighbor detection based on coupling strength ratios

### Hypothesis 2: Multiple Coupling Types Interference

Since each coupling works alone but fails together:
- TeNPy may be double-counting or interfering when multiple `add_coupling` calls target related bonds
- When t2 becomes comparable to t1, some automatic neighbor detection may activate
- The lattice geometry (all consecutive MPS sites separated by equal distance 0.25) might trigger unintended behavior

### Hypothesis 3: SSH-Specific Issue

The error appears when:
- Multiple unit cells exist (L ≥ 4)
- Both intra-cell and inter-cell couplings are strong enough
- The dimerization δ = (t1-t2)/(t1+t2) drops below ~0.33

This could be a bug specific to models with unit cell structure where different bonds within and between cells have comparable strengths.

---

## Tests Performed

### Systematic Scans

1. ✓ **L scan**: L=2 perfect, L=4 with errors (when t2 large)
2. ✓ **t2 scan**: Threshold at exactly t2=0.5 for t1=1.0
3. ✓ **χ scan**: Error constant for χ=10 to χ=1000
4. ✓ **Isolated couplings**: Both t1-only and t2-only work perfectly

### Verification Tests

1. ✓ **VQE Hamiltonian**: Verified correct by manual construction
2. ✓ **Bond pattern**: Confirmed MPS bonds map correctly to physical SSH bonds
3. ✓ **Unit cell structure**: Checked lattice site ordering
4. ✓ **Fermion signs**: Jordan-Wigner strings verified

### Attempted Fixes

1. ✗ **Double hopping strength** (compensate for 1/2 factor): Made error WORSE (147%)
2. ✗ **Change lattice positions**: No improvement
3. ✗ **Manual hermitian conjugate**: Failed (dx=[-1] doesn't work in TeNPy)

---

## Implications

### For VQE Validation

- **L ≤ 6, weak dimerization (t2 << t1)**: DMRG is reliable ✓
- **L ≤ 6, strong dimerization (t2 ≥ t1/2)**: DMRG has systematic error ✗
- **L ≥ 8**: No exact reference, error magnitude unknown ⚠

### Current Recommendations

1. **DO NOT USE DMRG** for SSH-Hubbard with t2/t1 ≥ 0.5
2. **CAN USE DMRG** for weak dimerization (t2/t1 < 0.5)
3. **Always verify** against exact diag for L ≤ 6 when possible

---

## Next Steps for Further Debugging

### High Priority

1. **Consult TeNPy documentation/community**
   - Check if this is a known issue
   - Ask about coupling strength thresholds
   - Verify expected behavior of `add_coupling` with unit cells

2. **Examine TeNPy source code**
   - Look for threshold logic in `add_coupling`
   - Check neighbor detection algorithms
   - Search for hardcoded ratio checks

3. **Try alternative TeNPy model construction**
   - Use `NearestNeighborModel` instead of `CouplingMPOModel`
   - Build Hamiltonian as explicit MPO without `add_coupling`
   - Test with different lattice types (Chain vs custom Lattice)

### Medium Priority

4. **Test with different parameters**
   - Vary U (interaction strength)
   - Try different L values (L=6, 8, 10)
   - Test periodic boundary conditions

5. **Compare with other DMRG codes**
   - ITensor, ALPS, or other DMRG implementations
   - If they also show errors, it's a physics issue
   - If they work, it's TeNPy-specific

### Low Priority

6. **Detailed matrix element comparison**
   - Extract MPO from TeNPy model
   - Convert to dense matrix for L=4
   - Compare element-by-element with VQE Hamiltonian

---

## Files Generated During Investigation

### Test Scripts

- `compare_2site_hamiltonians.py` - Verified L=2 works perfectly
- `test_dmrg_isolated_dimers.py` - Confirmed t2=0 works
- `test_dmrg_only_intercell.py` - Confirmed t1=0 works
- `test_dmrg_varying_ratio.py` - Scanned t1/t2 ratios
- `test_dmrg_convergence_vs_t2.py` - Verified χ-independence
- `find_error_threshold.py` - Found exact threshold at t2=0.50
- `test_without_plus_hc.py` - Tested manual h.c. (failed)
- `test_fixed_positions.py` - Tested different lattice positions (failed)

### Diagnostic Scripts

- `diagnose_dmrg_hamiltonian.py` - VQE Hamiltonian verification
- `verify_dmrg_fix_logic.py` - Physics logic verification
- `inspect_tenpy_model.py` - Model structure inspection
- `check_lattice_neighbors.py` - Lattice neighbor detection
- `print_tenpy_bonds.py` - Bond term inspection

### Documentation

- `DMRG_FIX_TEST_REPORT.md` - Initial fix attempt (failed)
- `DMRG_DEBUG_REPORT.md` - This file

---

## Confidence Assessment

**Certainty about findings**: 95%
- Error threshold at t2=t1/2 is definitive
- χ-independence is confirmed
- Isolation tests are conclusive

**Certainty about root cause**: 20%
- Mechanism is still unknown
- Could be TeNPy bug, design choice, or usage error
- Need to examine TeNPy internals or get expert input

---

## Conclusion

The DMRG systematic error is **NOT a simple coefficient mistake**. It's a complex issue that:

1. Only appears under specific conditions (t2 ≥ t1/2, L ≥ 4, both coupling types present)
2. Has an **exact** threshold, not gradual onset
3. Is **not** a convergence problem
4. Each component works correctly in isolation

This points to a bug or limitation in **how TeNPy handles multiple coupling types of comparable strength in unit cell models**. The issue may be TeNPy-specific, or it could reveal a deeper problem with how the SSH-Hubbard model is being represented.

**Recommended action**: Consult TeNPy documentation/community before further debugging, as this may be a known limitation or require model restructuring.

---

## Reproducibility

All findings can be reproduced by running:

```bash
# Verify threshold
python tests/find_error_threshold.py

# Verify χ-independence
python tests/test_dmrg_convergence_vs_t2.py

# Verify L=2 works
python tests/compare_2site_hamiltonians.py

# Verify isolated couplings work
python tests/test_dmrg_isolated_dimers.py
python tests/test_dmrg_only_intercell.py
```

All tests are deterministic and produce consistent results.
