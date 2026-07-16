# quantlop
### High-performance quantum simulator for matrix-free Hamiltonian evolution

<p align="center">
    <img src="light_logo.png#gh-light-mode-only" alt="quantlop">
    <img src="dark_logo.png#gh-dark-mode-only" alt="quantlop">
</p>

<p align="center">
    <a href="https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml"><img src="https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml/badge.svg" alt="Build and test"></a>
    <a href="https://pypi.org/project/quantlop/"><img src="https://img.shields.io/pypi/v/quantlop.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg" alt="Python 3.11+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/SimoneGasperini/quantlop.svg" alt="License"></a>
</p>


## Introduction
`quantlop` is a high-performance simulator for the time evolution of quantum systems whose Hamiltonians can be written as sparse sums of Pauli words.

Rather than constructing the full Hamiltonian matrix, `quantlop` applies each Pauli word as a linear operator directly to the state vector.
It then uses a Krylov method to numerically approximate the action of the matrix exponential.
This matrix-free approach dramatically reduces memory usage and avoids costly dense-matrix operations, making larger simulations more practical.


## Installation
The project requires Python 3.11 or later and a C++20-compatible compiler.
```bash
pip install quantlop
```

On macOS, install the OpenMP runtime before installing `quantlop`:

```bash
brew install libomp
pip install quantlop
```


## Quick example
Here is a simple code example using  `quantlop` native data structures:
```python
import numpy as np
import quantlop as ql

num_qubits = 3

# define Hamiltonian in Pauli basis
pwords = [
    ql.PauliWord(coeff=0.5, string="ZZI"),
    ql.PauliWord(coeff=0.2, string="YIX"),
]
ham = ql.Hamiltonian(pwords=pwords)

# set initial state vector
psi = np.zeros(2**num_qubits, dtype=complex)
psi[0] = 1.0

# evolve state vector
evolved_psi = ql.evolve(ham, psi)
```


## Multi-threading
Evolution is serial by default.
Set `num_threads` to a positive integer to use that many OpenMP threads, or to `"auto"` to use the CPU count reported by the operating system:
```python
evolved_psi = ql.evolve(ham, psi, num_threads="auto")
```


## Development
The Python package is built with scikit-build-core, while the numerical C++ code is kept in the standalone `quantlop_core` CMake target.

Run the Python test suite with:
```bash
python -m pip install -e .[dev]
python -m pytest -v
```

Run the native C++ test suite with:
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Install the pre-commit hook to format Python and C++ files automatically:
```bash
pre-commit install
```
