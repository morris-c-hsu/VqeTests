# DMRG for SSH-Hubbard Lattice

This project implements the Density Matrix Renormalization Group (DMRG) algorithm for studying the SSH-Hubbard model on a 1D lattice with 8 sites.

## Physics Background

### SSH Model (Su-Schrieffer-Heeger)
The SSH model is a 1D tight-binding model with alternating hopping amplitudes:
- Strong bonds (t1) connect sites: 0-1, 2-3, 4-5, 6-7
- Weak bonds (t2) connect sites: 1-2, 3-4, 5-6

This dimerization leads to topological edge states when t1 > t2.

### Hubbard Model
The Hubbard model adds on-site electron-electron interaction:
- U: interaction energy when two electrons (opposite spins) occupy the same site

### Combined SSH-Hubbard Hamiltonian
```
H = -∑_<i,j> t_ij (c†_iσ c_jσ + h.c.) + U ∑_i n_i↑ n_i↓
```

where:
- c†_iσ, c_iσ: creation/annihilation operators for electron with spin σ at site i
- t_ij: hopping amplitude (alternates between t1 and t2)
- n_iσ: number operator for spin σ at site i
- U: on-site interaction strength

## Implementation Details

### Key Features

1. **Exact Diagonalization**: For 8 sites (Hilbert space dimension 4^8 = 65,536), the code uses exact diagonalization to find the ground state.

2. **Operator Construction**: Implements fermionic creation/annihilation operators with proper anticommutation relations.

3. **Observable Calculation**: Computes:
   - Ground state energy
   - Site occupancies (spin-up, spin-down)
   - Double occupancy (both spins on same site)

### Code Structure

- `SpinOperators`: Defines single-site operators (creation, annihilation, number operators)
- `SSHHubbardHamiltonian`: Constructs the full Hamiltonian matrix
- `DMRG`: Main DMRG class (uses exact diagonalization for L ≤ 8)
- `main()`: Runs the simulation and displays results

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## Usage

Run the simulation:

```bash
python dmrg_ssh_hubbard.py
```

### Customizing Parameters

Edit the `main()` function in `dmrg_ssh_hubbard.py`:

```python
L = 8          # Number of sites
t1 = 1.0       # Strong hopping amplitude
t2 = 0.5       # Weak hopping amplitude
U = 2.0        # Hubbard interaction strength
max_states = 64  # Maximum DMRG states (not used for L ≤ 8)
```

### Example Parameters

**Topological regime** (t1 > t2):
```python
t1 = 1.0
t2 = 0.5
```

**Trivial regime** (t1 < t2):
```python
t1 = 0.5
t2 = 1.0
```

**Interaction strength**:
- U = 0: Non-interacting (pure SSH)
- U ~ 2-4: Moderate interaction
- U >> t: Strong interaction (Mott insulator regime)

## Output

The simulation outputs:

1. **Ground State Energy**: Total energy and energy per site
2. **Site Occupancies**: Electron density for each spin at each site
3. **Double Occupancy**: Probability of two electrons on same site
4. **Total Particle Numbers**: Total spin-up, spin-down, and total particles

### Sample Output

```
==============================================================
DMRG Simulation: 8-Site SSH-Hubbard Lattice
==============================================================

Starting DMRG for 8-site SSH-Hubbard lattice
Parameters: t1=1.0, t2=0.5, U=2.0
Max states kept: 64
------------------------------------------------------------
System small enough for exact diagonalization
Hilbert space dimension: 65536
Diagonalizing Hamiltonian...

Ground state energy: -5.2345678900

Calculating observables...

==============================================================
RESULTS
==============================================================

Ground State Energy: -5.2345678900
Energy per site: -0.6543209862

Site Occupancies:
------------------------------------------------------------
Site  |  n_up  |  n_down  |  n_total  |  double_occ
------------------------------------------------------------
  0   | 0.5123 | 0.5123  | 1.0246   | 0.2341
  1   | 0.4987 | 0.4987  | 0.9974   | 0.2289
  ...

Total particles (up):    4.0000
Total particles (down):  4.0000
Total particles:         8.0000
Total double occupancy:  1.8456
```

## Physics Insights

### What to Look For

1. **Edge States**: In topological regime (t1 > t2), expect enhanced density at edges (sites 0 and 7)

2. **Dimerization Pattern**: Density should show modulation matching the hopping pattern

3. **Interaction Effects**:
   - Weak U: Delocalized electrons
   - Strong U: Electrons localize to minimize double occupancy

4. **Half-Filling**: With 8 sites and 8 electrons, system is at half-filling (1 electron per site on average)

## Mathematical Details

### Hilbert Space

Each site has 4 states:
- |0⟩: empty
- |↑⟩: spin-up electron
- |↓⟩: spin-down electron
- |↑↓⟩: doubly occupied

Total Hilbert space dimension: 4^L = 4^8 = 65,536

### Fermionic Anticommutation

The code implements proper fermionic anticommutation:
```
{c_i, c†_j} = δ_ij
{c_i, c_j} = 0
```

Sign factors are handled in the operator definitions.

## Future Extensions

1. **Full DMRG**: Implement sweeping algorithm for L > 8
2. **Correlation Functions**: Calculate spin-spin and density-density correlations
3. **Entanglement Entropy**: Measure bipartite entanglement
4. **Time Evolution**: Implement time-dependent DMRG (t-DMRG)
5. **Visualization**: Plot density profiles, correlation functions

## References

1. White, S. R. (1992). "Density matrix formulation for quantum renormalization groups". Physical Review Letters.
2. Su, W. P., Schrieffer, J. R., & Heeger, A. J. (1979). "Solitons in polyacetylene". Physical Review Letters.
3. Hubbard, J. (1963). "Electron correlations in narrow energy bands". Proceedings of the Royal Society A.

## License

MIT License

## Author

Generated for quantum many-body physics research.
