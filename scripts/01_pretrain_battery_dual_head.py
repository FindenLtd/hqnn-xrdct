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
from hybrid_qnn_xrdct.data import load_split
from hybrid_qnn_xrdct.model import DualHeadQNN
from hybrid_qnn_xrdct.quantum import StatevectorQNN


def loader(dataset: str, split: int, batch: int, shuffle: bool) -> DataLoader:
    x, maximum, wt, sf = load_split(dataset, split)
    return DataLoader(
        TensorDataset(*(torch.tensor(value, dtype=torch.float64) for value in (x, maximum, wt, sf))),
        batch_size=batch,
        shuffle=shuffle,
    )


def evaluate(model, data):
    model.eval(); predictions = [[], []]; targets = [[], []]
    with torch.no_grad():
        for x, maximum, wt, sf in data:
            pred_wt, pred_sf = model(x, maximum)
            predictions[0].append(pred_wt.numpy()); predictions[1].append(pred_sf.numpy())
            targets[0].append(wt.numpy()); targets[1].append(sf.numpy())
    metrics = {}
    for name, pred, target in zip(("wt", "sf"), predictions, targets):
        pred, target = np.vstack(pred), np.vstack(target)
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((pred - target) ** 2)))
        denominator = np.sum((target - target.mean(0)) ** 2, axis=0)
        metrics[f"{name}_r2"] = float(np.mean(1 - np.sum((target - pred) ** 2, axis=0) / np.maximum(denominator, 1e-12)))
    return metrics


@torch.no_grad()
def fit_pmn(model, data):
    features = []
    model.eval()
    for x, _, _, _ in data:
        features.append(model.qlayer(x).to(torch.float64))
    model.pmn.fit(torch.vstack(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise-injected dual-head QNN pre-training")
    parser.add_argument("--config", default="configs/battery.yaml")
    parser.add_argument("--dataset", default="data/battery_200k.h5")
    parser.add_argument("--output", default="outputs/pretraining")
    args = parser.parse_args()
    config = load_config(args.config); training = config["training"]; model_config = config["model"]
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    train = loader(args.dataset, 0, training["batch_size"], True)
    validation = loader(args.dataset, 1, 256, False)

    qnn = StatevectorQNN(
        model_config["qubits"], model_config["repetitions"], model_config["observables"],
        p1q=training["gate_noise_p1q"], p2q=training["gate_noise_p2q"],
    )
    connector = TorchConnector(qnn, initial_weights=0.01 * np.random.randn(qnn.parameter_count))
    model = DualHeadQNN(connector, qnn.feature_count, len(config["phases"]), training["effective_shots"])
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("qlayer.")]
    optimizer = torch.optim.Adam([
        {"params": model.qlayer.parameters(), "lr": training["qnn_learning_rate"]},
        {"params": head_parameters, "lr": training["head_learning_rate"]},
    ])
    history = []
    for epoch in range(1, training["joint_epochs"] + 1):
        model.train()
        for x, maximum, wt, sf in train:
            optimizer.zero_grad(set_to_none=True)
            pred_wt, pred_sf = model(x, maximum)
            loss = F.mse_loss(pred_wt, wt) + F.mse_loss(pred_sf, sf)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        row = {"stage": "joint", "epoch": epoch, **evaluate(model, validation)}
        history.append(row); print(row, flush=True)

    for parameter in model.qlayer.parameters():
        parameter.requires_grad_(False)
    model.effective_shots = 0
    fit_pmn(model, loader(args.dataset, 0, 512, False))
    wt_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("wt_")]
    sf_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("sf_")]
    wt_optimizer = torch.optim.Adam(wt_parameters, lr=training["head_learning_rate"])
    sf_optimizer = torch.optim.Adam(sf_parameters, lr=training["head_learning_rate"])
    for epoch in range(1, training["head_epochs"] + 1):
        model.train()
        for x, maximum, wt, sf in train:
            wt_optimizer.zero_grad(set_to_none=True)
            pred_wt, _ = model(x, maximum); F.mse_loss(pred_wt, wt).backward(); wt_optimizer.step()
            sf_optimizer.zero_grad(set_to_none=True)
            _, pred_sf = model(x, maximum); F.mse_loss(pred_sf, sf).backward(); sf_optimizer.step()
        row = {"stage": "heads", "epoch": epoch, **evaluate(model, validation)}
        history.append(row); print(row, flush=True)

    checkpoint = output / "battery_dual_head.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, checkpoint)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(checkpoint.resolve())


if __name__ == "__main__":
    main()

