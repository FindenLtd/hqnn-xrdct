from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator
from qiskit_machine_learning.neural_networks import NeuralNetwork

X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
OBSERVABLES = {"X": X, "Y": Y, "Z": Z}


def single_qubit_observable(qubits: int, target: int, operator: np.ndarray) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    matrices = [operator if index == target else identity for index in range(qubits)]
    result = matrices[-1]
    for matrix in matrices[-2::-1]:
        result = np.kron(result, matrix)
    return result


def build_ansatz(qubits: int, repetitions: int, *, noisy: bool = False, p1q: float = 0.0, p2q: float = 0.0):
    circuit = QuantumCircuit(qubits)
    parameters = ParameterVector("theta", 3 * qubits * repetitions)
    rng = np.random.default_rng()

    def pauli_error(target: int, probability: float) -> None:
        if noisy and rng.random() < probability:
            getattr(circuit, str(rng.choice(["x", "y", "z"])))(target)

    cursor = 0
    for _ in range(repetitions):
        for qubit in range(qubits):
            circuit.rx(parameters[cursor], qubit); cursor += 1; pauli_error(qubit, p1q)
            circuit.ry(parameters[cursor], qubit); cursor += 1; pauli_error(qubit, p1q)
            circuit.rz(parameters[cursor], qubit); cursor += 1; pauli_error(qubit, p1q)
        for qubit in range(qubits):
            other = (qubit + 1) % qubits
            circuit.cx(qubit, other)
            pauli_error(qubit, p2q)
            pauli_error(other, p2q)
    return circuit, parameters


class StatevectorQNN(NeuralNetwork):
    """Fast statevector feature extractor with exact parameter-shift gradients."""

    def __init__(self, qubits: int, repetitions: int, bases=("Z", "X", "Y"), p1q=0.0, p2q=0.0):
        self.qubits = qubits
        self.repetitions = repetitions
        self.bases = tuple(bases)
        self.p1q, self.p2q = float(p1q), float(p2q)
        self.noisy = self.p1q > 0 or self.p2q > 0
        self.parameter_count = 3 * qubits * repetitions
        self.feature_count = qubits * len(self.bases)
        self.observables = np.stack([
            single_qubit_observable(qubits, qubit, OBSERVABLES[basis])
            for basis in self.bases for qubit in range(qubits)
        ])
        self.clean_circuit, self.clean_parameters = build_ansatz(qubits, repetitions)
        super().__init__(
            num_inputs=2**qubits,
            num_weights=self.parameter_count,
            sparse=False,
            output_shape=(self.feature_count,),
            input_gradients=False,
        )

    def _unitary(self, weights: np.ndarray) -> np.ndarray:
        if self.noisy:
            circuit, parameters = build_ansatz(
                self.qubits, self.repetitions, noisy=True, p1q=self.p1q, p2q=self.p2q
            )
        else:
            circuit, parameters = self.clean_circuit, self.clean_parameters
        return Operator(circuit.assign_parameters({parameters: weights})).data

    def _evaluate(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        states = np.asarray(inputs, dtype=np.float64).astype(np.complex128) @ self._unitary(weights).T
        return np.einsum("bi,kij,bj->bk", states.conj(), self.observables, states).real

    def _forward(self, inputs, weights):
        return self._evaluate(np.asarray(inputs), np.asarray(weights, dtype=np.float64))

    def _backward(self, inputs, weights, _output_gradient=None):
        values = np.asarray(inputs)
        weights = np.asarray(weights, dtype=np.float64)
        jacobian = np.empty((len(values), self.feature_count, self.parameter_count), dtype=np.float64)
        for index in range(self.parameter_count):
            plus, minus = weights.copy(), weights.copy()
            plus[index] += np.pi / 2
            minus[index] -= np.pi / 2
            jacobian[:, :, index] = 0.5 * (self._evaluate(values, plus) - self._evaluate(values, minus))
        return None, jacobian

    def backward(self, inputs, weights, output_gradient=None):
        return self._backward(inputs, weights, output_gradient)

