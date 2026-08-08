from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector
from torch.utils.data import DataLoader, TensorDataset

from hybrid_qnn_xrdct.config import load_config
from hybrid_qnn_xrdct.model import DualHeadQNN
from hybrid_qnn_xrdct.quantum import StatevectorQNN


def load_features(directory: Path) -> np.ndarray:
    records = []
    for path in sorted(directory.glob("group_*.npz")):
        payload = np.load(path); records.append((int(payload["start"]), np.asarray(payload["features"])))
    if not records:
        raise FileNotFoundError(f"No group_*.npz in {directory}")
    return np.vstack([values for _, values in sorted(records)])


def build_model(config, checkpoint_path):
    qnn = StatevectorQNN(config["model"]["qubits"], config["model"]["repetitions"], config["model"]["observables"])
    connector = TorchConnector(qnn, initial_weights=np.zeros(qnn.parameter_count))
    model = DualHeadQNN(connector, qnn.feature_count, len(config["phases"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for parameter in model.qlayer.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze QNN and recalibrate both heads on QPU features")
    parser.add_argument("--config", default="configs/battery.yaml")
    parser.add_argument("--inputs", required=True, help="Simulated QPU input NPZ with labels")
    parser.add_argument("--features", required=True, help="Downloaded simulated QPU feature directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/retrained_heads")
    args = parser.parse_args()
    config = load_config(args.config); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.inputs); features = load_features(Path(args.features))
    count = min(len(features), len(source["pattern_max"]))
    features = features[:count]; maximum = source["pattern_max"][:count]
    wt, sf = source["weight_fractions"][:count], source["normalized_scale_factors"][:count]
    train_end, validation_end = int(0.8 * count), int(0.9 * count)
    model = build_model(config, args.checkpoint)
    model.pmn.fit(torch.tensor(features[:train_end], dtype=torch.float64))
    wt_parameters = [p for name, p in model.named_parameters() if name.startswith("wt_")]
    sf_parameters = [p for name, p in model.named_parameters() if name.startswith("sf_")]
    wt_optimizer = torch.optim.Adam(wt_parameters, lr=config["training"]["head_learning_rate"])
    sf_optimizer = torch.optim.Adam(sf_parameters, lr=config["training"]["head_learning_rate"])
    dataset = TensorDataset(*(torch.tensor(value[:train_end], dtype=torch.float64) for value in (features, maximum, wt, sf)))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    history = []
    for epoch in range(1, config["training"]["head_epochs"] + 1):
        model.train()
        for q, intensity, target_wt, target_sf in loader:
            wt_optimizer.zero_grad(set_to_none=True)
            pred_wt, _ = model.heads_from_features(q, intensity); F.mse_loss(pred_wt, target_wt).backward(); wt_optimizer.step()
            sf_optimizer.zero_grad(set_to_none=True)
            _, pred_sf = model.heads_from_features(q, intensity); F.mse_loss(pred_sf, target_sf).backward(); sf_optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_wt, pred_sf = model.heads_from_features(
                torch.tensor(features[train_end:validation_end], dtype=torch.float64),
                torch.tensor(maximum[train_end:validation_end], dtype=torch.float64),
            )
        row = {"epoch": epoch, "wt_rmse": float(torch.sqrt(F.mse_loss(pred_wt, torch.tensor(wt[train_end:validation_end])))),
               "sf_rmse": float(torch.sqrt(F.mse_loss(pred_sf, torch.tensor(sf[train_end:validation_end]))))}
        history.append(row); print(row, flush=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, output / "hardware_retrained_dual_head.pt")
    np.savez_compressed(output / "test_predictions.npz", features=features[validation_end:], pattern_max=maximum[validation_end:],
                        target_wt=wt[validation_end:], target_sf=sf[validation_end:])
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
