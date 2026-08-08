from __future__ import annotations

import numpy as np


def unit_cell_volume(
    phase_index: int,
    lpa: float,
    lpc: float | None,
    crystal_system: str | None = None,
) -> float:
    """Unit-cell volume used by the Rietveld weight conversion."""
    system = crystal_system or ("hexagonal" if phase_index in (0, 1) else "cubic")
    if system in ("hexagonal", "trigonal"):
        if lpc is None or not np.isfinite(lpc):
            raise ValueError(f"{system} phases require a finite c lattice parameter")
        return float(lpa) ** 2 * float(lpc) * np.sin(np.deg2rad(120.0))
    if system == "tetragonal":
        if lpc is None or not np.isfinite(lpc):
            raise ValueError("Tetragonal phases require a finite c lattice parameter")
        return float(lpa) ** 2 * float(lpc)
    if system == "cubic":
        return float(lpa) ** 3
    raise ValueError(f"Unsupported crystal system: {system}")


def effective_sf(coefficients: np.ndarray, fixed_sf: np.ndarray) -> np.ndarray:
    return np.asarray(coefficients, dtype=np.float64) * np.asarray(fixed_sf, dtype=np.float64)


def scale_factors_to_weight_fraction(
    scale_factors: np.ndarray,
    volumes: np.ndarray,
    unit_cell_mass: np.ndarray,
) -> np.ndarray:
    """Convert effective scale factors to normalized physical weight fractions.

    This follows the project's Rietveld implementation:
    relative_weight_i = effective_sf_i * unit_cell_volume_i * unit_cell_mass_i.
    """
    relative = (
        np.asarray(scale_factors, dtype=np.float64)
        * np.asarray(volumes, dtype=np.float64)
        * np.asarray(unit_cell_mass, dtype=np.float64)
    )
    total = relative.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("At least one effective phase scale factor must be positive")
    return relative / total
