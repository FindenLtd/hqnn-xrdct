from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from qiskit import qpy
from qiskit_ibm_runtime import Batch, QiskitRuntimeService, SamplerV2

from hybrid_qnn_xrdct.config import load_config
from hybrid_qnn_xrdct.qpu import atomic_json, compile_replicated, decode_pub, select_disjoint_paths


def service() -> QiskitRuntimeService:
    token, instance = os.getenv("QISKIT_IBM_TOKEN"), os.getenv("QISKIT_IBM_INSTANCE")
    if not token or not instance:
        raise RuntimeError("Set QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE; never commit them")
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)


def groups(patterns: np.ndarray, size: int):
    for group, start in enumerate(range(0, len(patterns), size)):
        yield group, start, min(start + size, len(patterns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile, submit, or download IBM QPU features")
    parser.add_argument("stage", choices=("compile", "submit", "download"))
    parser.add_argument("--config", default="configs/battery.yaml")
    parser.add_argument("--inputs", required=True, help="NPZ with patterns and pattern_max")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/qpu")
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    config = load_config(args.config); hardware = config["hardware"]; model = config["model"]
    output = Path(args.output); (output / "compiled").mkdir(parents=True, exist_ok=True)
    (output / "receipts").mkdir(exist_ok=True); (output / "features").mkdir(exist_ok=True)
    payload = np.load(args.inputs); patterns = np.asarray(payload["patterns"], dtype=np.float64)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint["state_dict"]["qlayer.weight"].numpy()
    runtime = service(); backend = runtime.backend(hardware["backend"])
    path_file = output / "physical_paths.json"
    if path_file.exists():
        paths = json.loads(path_file.read_text(encoding="utf-8"))["paths"]
    else:
        paths = select_disjoint_paths(backend, hardware["replicas"], model["qubits"])
        atomic_json(path_file, json.dumps({"backend": backend.name, "paths": paths}, indent=2))
    specs = list(groups(patterns, hardware["patterns_per_group"]))

    if args.stage == "compile":
        for group, start, stop in specs:
            destination = output / "compiled" / f"group_{group:03d}.qpy"
            if destination.exists():
                continue
            circuits = [
                compile_replicated(backend, patterns[index], weights, basis, paths)
                for index in range(start, stop) for basis in hardware["measurement_bases"]
            ]
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                qpy.dump(circuits, stream)
            temporary.replace(destination)
            print(f"compiled group {group}: {stop-start} patterns", flush=True)
        return

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"jobs": {}}
    if args.stage == "submit":
        batch = Batch.from_id(args.batch_id, runtime) if args.batch_id else Batch(backend=backend, max_time=hardware["batch_max_time"])
        manifest["batch_id"] = batch.session_id
        # 18 replicas require 114 physical shots for at least 2048 effective samples.
        physical_shots = int(np.ceil(config["training"]["effective_shots"] / hardware["replicas"]))
        for group, start, stop in specs:
            key = f"group_{group:03d}"
            if key in manifest["jobs"]:
                continue
            with (output / "compiled" / f"{key}.qpy").open("rb") as stream:
                circuits = list(qpy.load(stream))
            job = SamplerV2(mode=batch).run(circuits, shots=physical_shots)
            receipt = {"job_id": job.job_id(), "start": start, "stop": stop, "physical_shots": physical_shots,
                       "effective_shots": physical_shots * hardware["replicas"], "pubs": len(circuits)}
            manifest["jobs"][key] = receipt
            atomic_json(output / "receipts" / f"{key}.json", json.dumps(receipt, indent=2))
            atomic_json(manifest_path, json.dumps(manifest, indent=2))
            print(f"submitted {key}: {job.job_id()}", flush=True)
        return

    for key, receipt in manifest["jobs"].items():
        destination = output / "features" / f"{key}.npz"
        if destination.exists():
            continue
        job = runtime.job(receipt["job_id"])
        if str(job.status()).upper() not in {"DONE", "COMPLETED"}:
            print(f"{key}: {job.status()}", flush=True); continue
        result = job.result(); count = receipt["stop"] - receipt["start"]
        features = np.zeros((count, model["qnn_features"]), dtype=np.float64)
        cursor = 0
        for index in range(count):
            blocks = []
            for _basis in hardware["measurement_bases"]:
                blocks.append(decode_pub(result[cursor], hardware["replicas"], model["qubits"]).mean(0)); cursor += 1
            features[index] = np.concatenate(blocks)
        np.savez_compressed(destination, features=features, start=receipt["start"], stop=receipt["stop"], job_id=receipt["job_id"])
        print(f"downloaded {key}", flush=True)


if __name__ == "__main__":
    main()

