from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.transpiler import CouplingMap


def _error(backend, operation: str, qargs: tuple[int, ...], fallback=1.0) -> float:
    try:
        value = backend.target[operation][qargs]
        return fallback if value is None or value.error is None else float(value.error)
    except (KeyError, TypeError):
        return fallback


def select_disjoint_paths(backend, count: int, length: int) -> list[dict]:
    graph = {index: set() for index in range(backend.num_qubits)}
    for left, right in backend.coupling_map.get_edges():
        graph[int(left)].add(int(right)); graph[int(right)].add(int(left))
    candidates: dict[tuple[int, ...], float] = {}

    def visit(path: list[int]) -> None:
        if len(path) == length:
            canonical = min(tuple(path), tuple(reversed(path)))
            readout = np.mean([_error(backend, "measure", (node,), 0.1) for node in path])
            edges = [min(_error(backend, "cz", (a, b)), _error(backend, "cz", (b, a))) for a, b in zip(path, path[1:])]
            candidates[canonical] = float(np.mean(edges) + 0.25 * readout)
            return
        for neighbor in graph[path[-1]]:
            if neighbor not in path:
                visit(path + [neighbor])

    for start in graph:
        visit([start])
    selected, used = [], set()
    for path, score in sorted(candidates.items(), key=lambda item: item[1]):
        if used.isdisjoint(path):
            selected.append({"qubits": list(path), "score": score}); used.update(path)
            if len(selected) == count:
                return selected
    raise RuntimeError(f"Only {len(selected)} disjoint {length}-qubit paths are available")


def logical_circuit(pattern: np.ndarray, weights: np.ndarray, basis: str) -> QuantumCircuit:
    qubits = int(np.log2(len(pattern)))
    qreg, creg = QuantumRegister(qubits, "q"), ClassicalRegister(qubits, "meas")
    circuit = QuantumCircuit(qreg, creg)
    circuit.append(StatePreparation(pattern), list(qreg))
    cursor = 0
    for _ in range(len(weights) // (3 * qubits)):
        for qubit in qreg:
            circuit.rx(float(weights[cursor]), qubit); cursor += 1
            circuit.ry(float(weights[cursor]), qubit); cursor += 1
            circuit.rz(float(weights[cursor]), qubit); cursor += 1
        for index in range(qubits):
            circuit.cx(qreg[index], qreg[(index + 1) % qubits])
    for qubit in qreg:
        if basis == "X":
            circuit.h(qubit)
        elif basis == "Y":
            circuit.sdg(qubit); circuit.h(qubit)
        elif basis != "Z":
            raise ValueError(basis)
    circuit.measure(qreg, creg)
    return circuit


def compile_replicated(backend, pattern, weights, basis, paths, optimization_level=3):
    qubits = int(np.log2(len(pattern)))
    coupling = CouplingMap(
        [(index, index + 1) for index in range(qubits - 1)]
        + [(index + 1, index) for index in range(qubits - 1)]
    )
    local = transpile(
        logical_circuit(pattern, weights, basis),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=coupling,
        initial_layout=list(range(qubits)),
        optimization_level=optimization_level,
        seed_transpiler=13013,
    )
    master = QuantumCircuit(backend.num_qubits, len(paths) * qubits, name=f"hqnn_{basis}_rep{len(paths)}")
    for replica, path in enumerate(paths):
        master.compose(
            local,
            qubits=path["qubits"],
            clbits=list(range(replica * qubits, (replica + 1) * qubits)),
            inplace=True,
        )
    return master


def decode_pub(pub_result, replicas: int, qubits: int) -> np.ndarray:
    register_name = next(iter(pub_result.data.keys()))
    bitstrings = getattr(pub_result.data, register_name).get_bitstrings()
    counts = Counter(bitstrings)
    values = np.zeros((replicas, qubits), dtype=np.float64)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]
        for replica in range(replicas):
            for qubit in range(qubits):
                values[replica, qubit] += (1 if bits[replica * qubits + qubit] == "0" else -1) * count
    return values / max(sum(counts.values()), 1)


def atomic_json(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)

