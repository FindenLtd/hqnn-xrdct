from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .physics import effective_sf, scale_factors_to_weight_fraction, unit_cell_volume
from .preprocessing import l2_normalize, maxpool_1d


@dataclass
class PhasePool:
    patterns: np.ndarray
    lpa: np.ndarray
    lpc: np.ndarray


def _valid_indices(pool: PhasePool, phase_index: int) -> np.ndarray:
    valid = np.isfinite(pool.lpa) & (np.max(pool.patterns, axis=1) > 1e-12)
    if phase_index in (0, 1):
        valid &= np.isfinite(pool.lpc)
    return np.flatnonzero(valid)


def load_library(path: str | Path, config: dict) -> tuple[np.ndarray, list[PhasePool]]:
    pools: list[PhasePool] = []
    with h5py.File(path, "r") as source:
        tth = np.asarray(source["twotheta"], dtype=np.float64)
        for index, pattern_key in enumerate(config["library_keys"]):
            patterns = np.asarray(source[pattern_key], dtype=np.float64)
            lpa = np.asarray(source[config["lpa_keys"][index]], dtype=np.float64)
            lpc_key = config["lpc_keys"][index]
            lpc = (
                np.asarray(source[lpc_key], dtype=np.float64)
                if lpc_key
                else np.full_like(lpa, np.nan)
            )
            pool = PhasePool(patterns=patterns, lpa=lpa, lpc=lpc)
            valid = _valid_indices(pool, index)
            if valid.size == 0:
                raise ValueError(f"No valid rows for {config['phases'][index]}")
            pools.append(pool)
    return tth, pools


def add_measurement_noise(pattern: np.ndarray, rng: np.random.Generator, config: dict) -> np.ndarray:
    """Poisson plus small Gaussian detector noise; no synthetic background."""
    maximum = float(np.max(pattern))
    if maximum <= 1e-12:
        return pattern
    counts = rng.uniform(*config["dataset"]["photon_counts"])
    poisson = rng.poisson(np.clip(pattern / maximum * counts, 0.0, None)) / counts * maximum
    sigma = rng.uniform(*config["dataset"]["gaussian_fraction"]) * maximum
    return np.clip(poisson + rng.normal(0.0, sigma, size=pattern.shape), 0.0, None)


def sample_coefficients(rng: np.random.Generator, config: dict) -> np.ndarray:
    """Sample in the scale-coefficient domain, never in the wt% domain."""
    n_phases = len(config["phases"])
    dataset = config["dataset"]
    count = int(rng.choice(np.arange(1, n_phases + 1), p=dataset["phase_count_probability"]))
    selected = rng.choice(
        n_phases,
        size=count,
        replace=False,
        p=np.asarray(dataset["phase_selection_probability"], dtype=np.float64),
    )
    coefficients = np.zeros(n_phases, dtype=np.float64)
    for order, phase in enumerate(selected):
        if order == 0:
            bounds = dataset["anchor_coefficient"]
        elif rng.random() < dataset["strong_nonanchor_probability"]:
            bounds = dataset["strong_coefficient"]
        else:
            bounds = dataset["ghost_coefficient"]
        coefficients[phase] = rng.uniform(*bounds)
    return coefficients


def sample_mixture(pools: list[PhasePool], rng: np.random.Generator, config: dict):
    coefficients = sample_coefficients(rng, config)
    fixed_sf = np.asarray(config["fixed_sf"], dtype=np.float64)
    mass = np.asarray(config["unit_cell_mass"], dtype=np.float64)
    pattern = np.zeros(pools[0].patterns.shape[1], dtype=np.float64)
    volumes = np.ones(len(pools), dtype=np.float64)
    source_rows = np.full(len(pools), -1, dtype=np.int64)

    for phase, coefficient in enumerate(coefficients):
        if coefficient <= 0:
            continue
        valid = _valid_indices(pools[phase], phase)
        row = int(rng.choice(valid))
        source_rows[phase] = row
        pattern += coefficient * pools[phase].patterns[row]
        lpc = None if phase not in (0, 1) else float(pools[phase].lpc[row])
        volumes[phase] = unit_cell_volume(phase, float(pools[phase].lpa[row]), lpc)

    sf = effective_sf(coefficients, fixed_sf)
    wt = scale_factors_to_weight_fraction(sf, volumes, mass)
    return pattern, coefficients, sf, wt, volumes, source_rows


def generate_dataset(library_path: str | Path, output_path: str | Path, config: dict) -> Path:
    rng = np.random.default_rng(config["seed"])
    tth, pools = load_library(library_path, config)
    count = int(config["dataset"]["samples"])
    bins = int(config["dataset"]["target_bins"])
    raw = np.empty((count, tth.size), dtype=np.float32)
    encoded = np.empty((count, bins), dtype=np.float64)
    pattern_max = np.empty((count, 1), dtype=np.float64)
    coefficients = np.empty((count, len(pools)), dtype=np.float64)
    scale_factors = np.empty_like(coefficients)
    weights = np.empty_like(coefficients)
    volumes = np.empty_like(coefficients)
    source_rows = np.empty((count, len(pools)), dtype=np.int64)

    for row in range(count):
        pattern, coeff, sf, wt, vol, rows = sample_mixture(pools, rng, config)
        raw[row] = pattern
        pattern_max[row, 0] = float(pattern.max()) * float(config["model"]["pattern_max_scale"])
        noisy = add_measurement_noise(pattern, rng, config)
        encoded[row] = l2_normalize(maxpool_1d(noisy, bins))
        coefficients[row], scale_factors[row], weights[row] = coeff, sf, wt
        volumes[row], source_rows[row] = vol, rows

    order = rng.permutation(count)
    train_end = int(config["dataset"]["split"][0] * count)
    val_end = train_end + int(config["dataset"]["split"][1] * count)
    split = np.full(count, 2, dtype=np.int8)
    split[order[:train_end]] = 0
    split[order[train_end:val_end]] = 1
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as target:
        target.create_dataset("twotheta", data=tth)
        target.create_dataset("raw_patterns", data=raw, compression="gzip")
        target.create_dataset("encoded_patterns", data=encoded, compression="gzip")
        target.create_dataset("pattern_max", data=pattern_max)
        target.create_dataset("mix_coefficients", data=coefficients)
        target.create_dataset("effective_scale_factors", data=scale_factors)
        target.create_dataset("weight_fractions", data=weights)
        target.create_dataset("unit_cell_volumes", data=volumes)
        target.create_dataset("source_rows", data=source_rows)
        target.create_dataset("split", data=split)
        target.attrs["phase_order"] = ",".join(config["phases"])
        target.attrs["sampling_domain"] = "scale_coefficient"
    return destination


def load_split(path: str | Path, split: int):
    with h5py.File(path, "r") as source:
        mask = np.asarray(source["split"]) == split
        return tuple(
            np.asarray(source[key][mask], dtype=np.float64)
            for key in ("encoded_patterns", "pattern_max", "weight_fractions", "effective_scale_factors")
        )

