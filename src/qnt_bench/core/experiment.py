"""
QNT-SYS-02-MPS-DIVI-BENCH — Core Experiment Module

This script runs a minimal working quantum simulation pipeline for benchmarking purposes.

What it does:
1. Constructs a Transverse Field Ising Model (TFIM) Hamiltonian as a simple test system.
2. Uses a Variational Quantum Eigensolver (VQE) to approximate the ground state energy.
3. Optimises a parameterised quantum circuit (ansatz) using classical gradient descent.
4. Outputs a single scalar: the approximate ground state energy.

Purpose in the overall project:
- This is the baseline "physics + solver" layer of the benchmarking framework.
- It provides a reference measurement before introducing:
  (a) entanglement entropy analysis
  (b) Matrix Product State (MPS) compression limits
  (c) distributed circuit execution (Divi/Qoro layer)
- Later modules will compare how classical simulation scales vs entanglement growth,
  identifying when tensor-network methods break down.

In short:
This file answers: "Can we solve small quantum systems efficiently using VQE?"
Everything else in the project builds on scaling this up and measuring when it stops working.
"""

import pennylane as qml
from pennylane import numpy as np


def tfim_hamiltonian(n_qubits):
    coeffs = []
    ops = []

    for i in range(n_qubits - 1):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    for i in range(n_qubits):
        coeffs.append(-1.0)
        ops.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, ops)


def run_vqe(n_qubits=6):
    dev = qml.device("default.qubit", wires=n_qubits)

    H = tfim_hamiltonian(n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(H)

    params = np.random.randn(n_qubits, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    for _ in range(50):
        params = opt.step(circuit, params)

    return circuit(params)


if __name__ == "__main__":
    result = run_vqe(6)
    print("VQE energy:", result)