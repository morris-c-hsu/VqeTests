# Multi-Start VQE Benchmark Results

Generated: 2025-11-16 09:15:01

## Configuration

- **Optimizers**: L-BFGS-B, COBYLA, SLSQP
- **Seeds**: [0, 1, 2, 3, 4]
- **Ansätze**: HEA, HVA, NP_HVA
- **Method**: Multi-start VQE (5 random initializations per optimizer)

## Summary Table

| Test | Ansatz | Optimizer | Best Energy | Mean ± Std | Error (Best) | Rel % |
|------|--------|-----------|-------------|------------|--------------|-------|
| L=4, δ=0.333, U=2.00 | HEA | L_BFGS_B | -2.03935317 | -2.03420385 ± 4.20e-03 | 5.31e-01 | 20.66% |
| L=4, δ=0.333, U=2.00 | HEA | COBYLA | -1.16622394 | -1.05406572 ± 1.12e-01 | 1.40e+00 | 54.63% |
| L=4, δ=0.333, U=2.00 | HEA | SLSQP | -2.03935238 | -2.03420653 ± 4.20e-03 | 5.31e-01 | 20.66% |
| L=4, δ=0.333, U=2.00 | HVA | L_BFGS_B | -2.07919200 | -2.06873758 ± 6.92e-03 | 4.91e-01 | 19.11% |
| L=4, δ=0.333, U=2.00 | HVA | COBYLA | -1.98939457 | -1.86842231 ± 1.09e-01 | 5.81e-01 | 22.60% |
| L=4, δ=0.333, U=2.00 | HVA | SLSQP | -2.10432695 | -2.08048689 ± 1.63e-02 | 4.66e-01 | 18.13% |
| L=4, δ=0.333, U=2.00 | NP_HVA | L_BFGS_B | -2.47790198 | -2.47029086 ± 1.45e-02 | 9.25e-02 | 3.60% |
| L=4, δ=0.333, U=2.00 | NP_HVA | COBYLA | -1.22405265 | -1.17988766 ± 4.03e-02 | 1.35e+00 | 52.38% |
| L=4, δ=0.333, U=2.00 | NP_HVA | SLSQP | -2.47781765 | -2.41596772 ± 1.23e-01 | 9.26e-02 | 3.60% |

---

# Detailed Results


## L=4, δ=0.333, U=2.00


### System: L=4, δ=0.333, U=2.00

**Exact Energy**: `-2.5703727848`

| Ansatz | Optimizer | Best Energy | Mean ± Std | Min | Max | Error (Best) | Rel % |
|--------|-----------|-------------|------------|-----|-----|--------------|-------|
| HEA | L_BFGS_B | -2.03935317 | -2.03420385 ± 4.20e-03 | -2.03935317 | -2.03077639 | 5.31e-01 | 20.66% |
| HEA | COBYLA | -1.16622394 | -1.05406572 ± 1.12e-01 | -1.16622394 | -0.84185977 | 1.40e+00 | 54.63% |
| HEA | SLSQP | -2.03935238 | -2.03420653 ± 4.20e-03 | -2.03935238 | -2.03077558 | 5.31e-01 | 20.66% |
| HVA | L_BFGS_B | -2.07919200 | -2.06873758 ± 6.92e-03 | -2.07919200 | -2.06204968 | 4.91e-01 | 19.11% |
| HVA | COBYLA | -1.98939457 | -1.86842231 ± 1.09e-01 | -1.98939457 | -1.67435047 | 5.81e-01 | 22.60% |
| HVA | SLSQP | -2.10432695 | -2.08048689 ± 1.63e-02 | -2.10432695 | -2.06206888 | 4.66e-01 | 18.13% |
| NP_HVA | L_BFGS_B | -2.47790198 | -2.47029086 ± 1.45e-02 | -2.47790198 | -2.44125387 | 9.25e-02 | 3.60% |
| NP_HVA | COBYLA | -1.22405265 | -1.17988766 ± 4.03e-02 | -1.22405265 | -1.11236059 | 1.35e+00 | 52.38% |
| NP_HVA | SLSQP | -2.47781765 | -2.41596772 ± 1.23e-01 | -2.47781765 | -2.16993065 | 9.26e-02 | 3.60% |

## Per-Seed Details


### HEA - L_BFGS_B

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.0307763856 | 3675 | 22.76 | 3675 |
| 1 | -2.0307763949 | 2499 | 15.63 | 2499 |
| 2 | -2.0393531700 | 5684 | 35.43 | 5684 |
| 3 | -2.0307763982 | 3185 | 19.46 | 3185 |
| 4 | -2.0393368810 | 5831 | 36.38 | 5831 |

### HEA - COBYLA

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -1.1160577457 | 100 | 0.81 | 100 |
| 1 | -1.0933206125 | 100 | 0.82 | 100 |
| 2 | -0.8418597728 | 100 | 0.81 | 100 |
| 3 | -1.0528665064 | 100 | 0.84 | 100 |
| 4 | -1.1662239410 | 100 | 0.76 | 100 |

### HEA - SLSQP

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.0307755807 | 1483 | 9.76 | 1483 |
| 1 | -2.0393523823 | 2658 | 16.63 | 2658 |
| 2 | -2.0307759943 | 1773 | 11.70 | 1773 |
| 3 | -2.0307763227 | 1821 | 11.57 | 1821 |
| 4 | -2.0393523643 | 3294 | 20.54 | 3294 |

### HVA - L_BFGS_B

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.0621297265 | 2541 | 12.43 | 2541 |
| 1 | -2.0620496839 | 2457 | 12.84 | 2457 |
| 2 | -2.0744842272 | 2352 | 11.91 | 2352 |
| 3 | -2.0658322752 | 2457 | 12.04 | 2457 |
| 4 | -2.0791919997 | 2436 | 11.83 | 2436 |

### HVA - COBYLA

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -1.8342006015 | 100 | 0.62 | 100 |
| 1 | -1.6743504669 | 100 | 0.64 | 100 |
| 2 | -1.9893945701 | 100 | 0.62 | 100 |
| 3 | -1.9069218262 | 100 | 0.60 | 100 |
| 4 | -1.9372440983 | 100 | 0.62 | 100 |

### HVA - SLSQP

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.0621347574 | 1188 | 5.75 | 1188 |
| 1 | -2.0620688797 | 881 | 4.37 | 881 |
| 2 | -2.0870500607 | 1450 | 6.99 | 1450 |
| 3 | -2.1043269455 | 2150 | 10.52 | 2150 |
| 4 | -2.0868538245 | 1599 | 7.83 | 1599 |

### NP_HVA - L_BFGS_B

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.4778802004 | 3729 | 27.48 | 3729 |
| 1 | -2.4777773950 | 3861 | 28.67 | 3861 |
| 2 | -2.4412538686 | 3828 | 28.81 | 3828 |
| 3 | -2.4766408655 | 3927 | 29.83 | 3927 |
| 4 | -2.4779019802 | 3762 | 29.54 | 3762 |

### NP_HVA - COBYLA

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -1.1572961017 | 100 | 0.83 | 100 |
| 1 | -1.1123605933 | 100 | 0.81 | 100 |
| 2 | -1.2240526521 | 100 | 0.82 | 100 |
| 3 | -1.1989858874 | 100 | 0.84 | 100 |
| 4 | -1.2067430852 | 100 | 0.85 | 100 |

### NP_HVA - SLSQP

| Seed | Energy | Evaluations | Runtime (s) | Convergence Steps |
|------|--------|-------------|-------------|-------------------|
| 0 | -2.4778176516 | 2226 | 16.40 | 2226 |
| 1 | -2.4777700792 | 3360 | 24.74 | 3360 |
| 2 | -2.4765328467 | 3353 | 24.14 | 3353 |
| 3 | -2.1699306451 | 3362 | 25.10 | 3362 |
| 4 | -2.4777873862 | 2459 | 18.18 | 2459 |

### Convergence Plots

- [HEA - L_BFGS_B](../docs/images/convergence_hea_L_BFGS_B_L4.png)
- [HEA - COBYLA](../docs/images/convergence_hea_COBYLA_L4.png)
- [HEA - SLSQP](../docs/images/convergence_hea_SLSQP_L4.png)
- [HVA - L_BFGS_B](../docs/images/convergence_hva_L_BFGS_B_L4.png)
- [HVA - COBYLA](../docs/images/convergence_hva_COBYLA_L4.png)
- [HVA - SLSQP](../docs/images/convergence_hva_SLSQP_L4.png)
- [NP_HVA - L_BFGS_B](../docs/images/convergence_np_hva_L_BFGS_B_L4.png)
- [NP_HVA - COBYLA](../docs/images/convergence_np_hva_COBYLA_L4.png)
- [NP_HVA - SLSQP](../docs/images/convergence_np_hva_SLSQP_L4.png)

---

# Best Performers


## L=4, δ=0.333, U=2.00

**Best Accuracy**: NP_HVA + L_BFGS_B (3.60% error)
**Most Consistent**: HEA + L_BFGS_B (std = 4.198e-03)

---

## Notes

- **Best Energy**: Lowest energy found across all 5 seeds
- **Mean ± Std**: Statistics across 5 random initializations
- **Error**: Absolute and relative error compared to exact diagonalization
- **Convergence plots**: Show all seed trajectories (gray), best seed (blue), and mean ± std (red band)