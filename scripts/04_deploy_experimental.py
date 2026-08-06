from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hybrid_qnn_xrdct.config import load_config
from scripts_03_support import build_model, load_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply hardware-retrained heads to experimental QPU features")
    parser.add_argument("--config", default="configs/battery.yaml")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/experimental_predictions.npz")
    args = parser.parse_args()
    config = load_config(args.config); inputs = np.load(args.inputs)
    features = load_features(Path(args.features)); count = min(len(features), len(inputs["pattern_max"]))
    model = build_model(config, args.checkpoint); model.eval()
    with torch.no_grad():
        wt, sf = model.heads_from_features(
            torch.tensor(features[:count], dtype=torch.float64),
            torch.tensor(inputs["pattern_max"][:count], dtype=torch.float64),
        )
    shape, coordinates = tuple(inputs["image_shape"]), inputs["coordinates"][:count]
    wt_maps = np.zeros((*shape, len(config["phases"])), dtype=np.float64)
    sf_maps = np.zeros_like(wt_maps)
    wt_maps[coordinates[:, 0], coordinates[:, 1]] = wt.numpy()
    sf_maps[coordinates[:, 0], coordinates[:, 1]] = sf.numpy()
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, wt_maps=wt_maps, sf_maps=sf_maps, mask=inputs["mask"], phases=config["phases"])
    print(destination.resolve())


if __name__ == "__main__":
    main()

