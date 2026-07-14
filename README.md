# quantlop
High-performance quantum simulator for matrix-free Hamiltonian evolution

[![Build and test](https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml/badge.svg)](https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/quantlop.svg)](https://pypi.org/project/quantlop/)
[![Python 3.11+](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/SimoneGasperini/quantlop.svg)](LICENSE)


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


## Development
The Python package is built with scikit-build-core, while the numerical C++ code is kept in the standalone `quantlop_core` CMake target.
The pybind11 extension is a thin private module named `_quantlop`.

Run the Python test suite with:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Run the native C++ test suite with:

```bash
cmake -S . -B build -DQUANTLOP_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
