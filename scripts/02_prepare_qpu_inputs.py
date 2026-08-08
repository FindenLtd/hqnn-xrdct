from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from hybrid_qnn_xrdct.preprocessing import l2_normalize, maxpool_1d


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare simulated or experimental QPU amplitudes")
    parser.add_argument("--kind", choices=("simulated", "experimental"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--patterns-key", default="patterns")
    parser.add_argument("--mask-key", default="Mask")
    args = parser.parse_args()
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "simulated":
        with h5py.File(args.input, "r") as source:
            indices = np.flatnonzero(np.asarray(source["split"]) == 0)[: args.count]
            np.savez_compressed(
                destination,
                patterns=np.asarray(source["encoded_patterns"][indices], dtype=np.float64),
                pattern_max=np.asarray(source["pattern_max"][indices], dtype=np.float64),
                weight_fractions=np.asarray(source["weight_fractions"][indices], dtype=np.float64),
                normalized_scale_factors=np.asarray(source["normalized_scale_factors"][indices], dtype=np.float64),
                effective_scale_factors=np.asarray(source["effective_scale_factors"][indices], dtype=np.float64),
                source_indices=indices,
            )
        return
    with h5py.File(args.input, "r") as source:
        volume = np.asarray(source[args.patterns_key], dtype=np.float64)
        mask = np.asarray(source[args.mask_key]) > 0
    spectra = volume[mask]
    maximum = spectra.max(axis=1, keepdims=True)
    encoded = np.vstack([maxpool_1d(pattern, 64) for pattern in spectra])
    coordinates = np.argwhere(mask)
    np.savez_compressed(
        destination,
        patterns=l2_normalize(encoded),
        pattern_max=maximum,
        coordinates=coordinates,
        image_shape=np.asarray(mask.shape),
        mask=mask,
    )


if __name__ == "__main__":
    main()
