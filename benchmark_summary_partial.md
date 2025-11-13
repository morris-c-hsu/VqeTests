# Large System Benchmarks: L=6 and L=8 (Partial Results)

## Overview
Comprehensive benchmarking of 8 ansätze (hea, hva, topoinsp, topo_rn, dqap, np_hva, tn_mps, tn_mps_np)
across multiple system sizes and parameter regimes for the SSH-Hubbard model.

**Status**: In progress (completed 2/6 tests as of last check)

---

## Test 1: L=6, Standard Parameters (δ=0.33, U=2.0)
**Ground State Energy (exact):** -4.0107137460

### Rankings by Accuracy:
| Rank | Ansatz    | Rel. Error | Abs. Error | Params | Runtime (s) |
|------|-----------|------------|------------|--------|-------------|
| 1    | np_hva    | 6.97%      | 2.797e-01  | 52     | 181.23      |
| 2    | tn_mps    | 18.59%     | 7.457e-01  | 312    | 476.12      |
| 3    | hva       | 21.46%     | 8.605e-01  | 32     | 65.82       |
| 4    | topo_rn   | 22.09%     | 8.858e-01  | 72     | 209.67      |
| 5    | hea       | 23.44%     | 9.403e-01  | 72     | 92.81       |
| 6    | tn_mps_np | 23.68%     | 9.497e-01  | 68     | 84.86       |
| 7    | dqap      | 23.83%     | 9.556e-01  | 6      | 4.38        |
| 8    | topoinsp  | 32.09%     | 1.287e+00  | 48     | 51.16       |

### Best Performers:
- **Most Accurate**: np_hva (6.97% error)
- **Fastest**: dqap (4.38s)
- **Most Efficient**: tn_mps (2.390e-03 error/param)

### Key Observations:
- **NP_HVA** achieved best accuracy at 6.97% - significantly outperforming all others
- **TN_MPS** ranked 2nd despite 312 parameters - good parameter efficiency
- **DQAP** extremely fast (4.38s) with only 6 parameters, though moderate accuracy
- **TN ansätze** show promise but require long runtimes (476s for TN_MPS)

---

## Test 2: L=6, Weak SSH (δ=0.11, U=2.0)
**Ground State Energy (exact):** -4.5470219361

### Rankings by Accuracy:
| Rank | Ansatz    | Rel. Error | Abs. Error | Params | Runtime (s) |
|------|-----------|------------|------------|--------|-------------|
| 1    | np_hva    | 17.75%     | 8.071e-01  | 52     | 165.18      |
| 2    | hva       | 21.92%     | 9.969e-01  | 32     | 61.50       |
| 3    | tn_mps_np | 22.84%     | 1.039e+00  | 68     | 103.72      |
| 4    | tn_mps    | 26.00%     | 1.182e+00  | 312    | 470.29      |
| 5    | topo_rn   | 27.02%     | 1.229e+00  | 72     | 152.52      |
| 6    | dqap      | 30.16%     | 1.371e+00  | 6      | 3.24        |
| 7    | hea       | 37.92%     | 1.724e+00  | 72     | 166.86      |
| 8    | topoinsp  | 39.12%     | 1.779e+00  | 48     | 57.84       |

### Best Performers:
- **Most Accurate**: np_hva (17.75% error)
- **Fastest**: dqap (3.24s)
- **Most Efficient**: tn_mps (3.788e-03 error/param)

### Key Observations:
- **NP_HVA** again dominated with 17.75% error
- Weak SSH regime (δ=0.11) appears more challenging - higher errors across all ansätze
- **HVA** performed well at 21.92% with only 32 parameters
- **TN_MPS_NP** (number-preserving) outperformed TN_MPS in this regime

---

## Test 3: L=6, Strong SSH (δ=0.67, U=2.0) [IN PROGRESS]
**Ground State Energy (exact):** -3.7391916223

### Partial Results:
| Ansatz | Status     | Rel. Error | Abs. Error | Runtime (s) |
|--------|------------|------------|------------|-------------|
| hea    | ✓ Complete | 19.18%     | 7.173e-01  | 128.61      |
| hva    | ✓ Complete | 19.23%     | 7.192e-01  | 64.55       |
| topoinsp | ✓ Complete | 44.98%   | 1.682e+00  | 70.00       |
| topo_rn | ✓ Complete | 29.49%    | 1.103e+00  | 242.03      |
| dqap   | ✓ Complete | 19.63%     | 7.341e-01  | 7.75        |
| np_hva | ✓ Complete | **0.77%**  | 2.881e-02  | 167.50      |
| tn_mps | Running... | -          | -          | -           |
| tn_mps_np | Pending | -          | -          | -           |

### Key Observations:
- **NP_HVA** achieved remarkable 0.77% error - near-exact accuracy!
- Strong SSH regime (δ=0.67) appears easier for number-preserving ansätze
- **HEA, HVA, DQAP** all achieved ~19% error - consistent performance
- **TOPOINSP** struggled significantly (44.98% error)

---

## Tests 4-6: L=8 Systems [PENDING]
- Test 4: L=8, Standard (δ=0.33, U=2.0)
- Test 5: L=8, Weak SSH (δ=0.11, U=2.0)
- Test 6: L=8, Strong SSH (δ=0.67, U=2.0)

---

## Overall Insights (So Far)

### Best Ansatz:
**NP_HVA (Number-Preserving HVA)**
- Consistently best accuracy across all regimes
- Errors: 6.97% (standard), 17.75% (weak SSH), 0.77% (strong SSH)
- Moderate parameter count (52) and runtime (~165-180s)
- **Excellent for strong dimerization regimes**

### Parameter Efficiency:
**TN_MPS (Tensor Network MPS)**
- Best error/parameter ratio: 2.39e-03 to 3.79e-03
- Demonstrates that TN structure efficiently uses its 312 parameters
- However, long runtime (~470s) may limit practical utility

### Speed Champion:
**DQAP (Differentiable Quantum Architecture Plus)**
- Extremely fast: 3-8 seconds
- Only 6 parameters
- Moderate accuracy (19-30% error)
- **Best for quick approximate solutions**

### Regime-Dependent Behavior:
- **Strong SSH (δ=0.67)**: NP_HVA excels (0.77% error!)
- **Weak SSH (δ=0.11)**: All ansätze struggle more (higher errors)
- **Standard (δ=0.33)**: Balanced performance across ansätze

### Tensor Network Ansätze:
- **TN_MPS**: Good accuracy but very slow (312 params)
- **TN_MPS_NP**: Faster with fewer params (68), competitive accuracy
- Both show promise but need runtime optimization

---

## Next Steps:
1. Complete L=6 Strong SSH test (TN ansätze remaining)
2. Run all L=8 tests (16 qubits) - more challenging
3. Generate final comparison table across all 6 test configurations
4. Analyze scaling behavior L=4 → L=6 → L=8

**Benchmark started:** 10:59 UTC
**Estimated completion:** ~3-4 hours total
