# Reproducibility datasets

The repository uses two Git LFS HDF5 files:

| File | Samples | Phases | Preprocessing | SHA-256 |
|---|---:|---|---|---|
| `data/battery_200k.h5` | 200,000 | LCO, Graphite, Al, Cu | full-range max-pool to 64 bins | `689390bf181397d1dfc2b4ad1a9a938d6a45d7f0b7816b60723054c03b4b25a7` |
| `data/sofc_200k.h5` | 200,000 | LSM, YSZ, NiO | indices 33:327, interpolation to 64 bins | `7452204c746c3c10f2c22dc51c1201a81e55973167b0e3f81b6e4e67ba7e0c85` |

Each file contains raw and encoded patterns, mixture coefficients, normalized
scale-factor targets, physical effective scale factors, physically recalculated
weight fractions, unit-cell volumes, source-library row indices, and a
deterministic 160,000/20,000/20,000 train/validation/test split.
