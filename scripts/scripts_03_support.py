"""Shared import shim for stage 3 and stage 4 command-line scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from qiskit_machine_learning.connectors import TorchConnector

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

