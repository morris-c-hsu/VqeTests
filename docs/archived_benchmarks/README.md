# Archived Benchmark Scripts

This directory contains legacy benchmark scripts that have been superseded by newer implementations but are preserved for historical reference.

## Files

### compare_all_ansatze.py
**Status**: Archived
**Date Archived**: 2025-11-17
**Reason**: Superseded by `benchmarks/run_multistart_benchmark.py`

This was an older comprehensive comparison tool that tested multiple ansätze variants including archived ones (TopoInspired, Topo_RN, DQAP, TN_MPS, TN_MPS_NP).

**Modern Replacement**: Use `benchmarks/run_multistart_benchmark.py` which provides:
- Multi-start VQE with configurable seeds
- Three primary ansätze: HEA, HVA, NP_HVA
- Enhanced statistical analysis
- Better plotting and convergence tracking

**Why Kept**:
- Historical reference for alternative ansatz designs
- Contains comparison methodology that may be useful
- Documents early benchmark approach

**Usage**: Not recommended for new work. Refer to `benchmarks/run_multistart_benchmark.py` instead.

---

## Archived Ansätze

The following ansätze are implemented in `src/ansatze/archived_ansatze.py` but are not actively benchmarked:

1. **TopoInspired** - Topological/problem-inspired ansatz with dimer pattern + edge links
2. **Topo_RN** - RN-Topological ansatz using number-conserving RN gates
3. **DQAP** - Digital-adiabatic QAOA-style ansatz with Hamiltonian splitting
4. **TN_MPS** - Tensor-network brick-wall MPS ansatz
5. **TN_MPS_NP** - Number-preserving tensor-network brick-wall ansatz

These represent legitimate alternative designs preserved for research purposes but are not part of the standard benchmark suite.

**Active Ansätze** (benchmarked regularly):
- **HEA** - Hardware-efficient ansatz (baseline)
- **HVA** - Hamiltonian variational ansatz (problem-aware)
- **NP_HVA** - Number-preserving HVA (strict particle conservation)

---

For current benchmarking, see `benchmarks/README.md`.
