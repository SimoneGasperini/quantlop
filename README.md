## Introduction
`quantlop` is a high-performance simulator for the time evolution of quantum systems whose Hamiltonians can be written as sparse sums of Pauli words.
It integrates with [PennyLane](https://github.com/PennyLaneAI/pennylane), so Hamiltonians can be defined with familiar Python operators while the compute-intensive work runs in C++.

Rather than constructing the full Hamiltonian matrix, `quantlop` applies each Pauli word as a linear operator directly to the state vector.
It then uses a Krylov method to numerically approximate the action of the matrix exponential.
This matrix-free approach dramatically reduces memory usage and avoids costly dense-matrix operations, making larger simulations more practical.

## Installation
The project requires Python 3.11 or later and a C++20-compatible compiler.
```bash
pip install quantlop
```

## Usage example
```python
import numpy as np
import pennylane as qp
import quantlop

# set number of qubits
nq = 3

# define Hamiltonian in Pauli basis
op = 0.5 * qp.Z(0) @ qp.Z(1) + 0.2 * qp.Y(0) @ qp.X(2)
ham = quantlop.Hamiltonian.from_pennylane(op, nq)

# prepare initial state vector
psi = np.zeros(2**nq, dtype=complex)
psi[0] = 1.0

# evolve state vector
evolved_psi = quantlop.evolve(ham, psi)
```

The function `quantlop.evolve` returns a new state vector corresponding to
$$
\lvert \psi(t) \rangle = e^{-iH}\lvert \psi(0) \rangle.
$$
