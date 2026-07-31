# quantlop
### High-performance quantum simulator for matrix-free Hamiltonian evolution

<p align="center">
    <img src="assets/light_logo.png#gh-light-mode-only" alt="quantlop">
    <img src="assets/dark_logo.png#gh-dark-mode-only" alt="quantlop">
</p>

<p align="center">
    <a href="https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml"><img src="https://github.com/SimoneGasperini/quantlop/actions/workflows/ci.yml/badge.svg" alt="Build and test"></a>
    <a href="https://simonegasperini.github.io/quantlop/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation"></a>
    <a href="https://pypi.org/project/quantlop/"><img src="https://img.shields.io/pypi/v/quantlop.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg" alt="Python 3.11+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/SimoneGasperini/quantlop.svg" alt="License"></a>
</p>


## Introduction

`quantlop` is a Python package, backed by a native C++ core, for simulating the evolution of
quantum states under Hamiltonians expressed as weighted sums of Pauli words $P_k$. For an
$n$-qubit Hamiltonian
<p align="center">
    <img src="assets/light_ham_eq.png#gh-light-mode-only" alt="quantlop">
    <img src="assets/dark_ham_eq.png#gh-dark-mode-only" alt="quantlop">
</p>

`quantlop` computes the action
<p align="center">
    <img src="assets/light_expm_eq.png#gh-light-mode-only" alt="quantlop">
    <img src="assets/dark_expm_eq.png#gh-dark-mode-only" alt="quantlop">
</p>
without constructing either the full Hamiltonian matrix or its exponential. Each Pauli word is 
applied directly to the dense state vector, while either an adaptively scaled Taylor series or a
Lanczos-Krylov subspace projection evaluates the matrix-exponential action.

A dense Hamiltonian for $n$ qubits requires $O(4^n)$ storage, whereas the matrix-free evolution
works only with its compact Pauli representation. The dense state vector still grows exponentially
with the number of qubits, but avoiding the dense operator substantially lowers the memory
requirement for Hamiltonians with Pauli decompositions.


## Installation
Install the latest release of the package directly from PyPI with:
```bash
pip install quantlop
```

## Quick example
Here is a simple code example using `quantlop` native data structures:
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
evolved_psi = ql.evolve_higham(ham, psi)
# or
evolved_psi = ql.evolve_krylov(ham, psi)
```

Both algorithms select their Taylor truncation or Krylov dimension automatically.
The default relative tolerance is `1e-9`.
Smaller values generally improve accuracy at the cost of more computation.
The tolerance guides the internal approximation rather than measuring the
final error directly.

The library also provides class methods to import Hamiltonians directly from other quantum computing frameworks:
- `ql.Hamiltonian.from_pennylane` to build from PennyLane [`Hamiltonian`](https://docs.pennylane.ai/en/stable/code/api/pennylane.Hamiltonian.html) objects
- `ql.Hamiltonian.from_qiskit` to build from Qiskit [`SparsePauliOp`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SparsePauliOp) objects


## Multi-threading
By default, evolution is serial. To enable multi-threading, pass a non-zero positive integer
as ``num_threads`` to request that many OpenMP threads. Passing ``"auto"`` selects the thread
count reported by the operating system.

```python
evolved_psi = ql.evolve_higham(ham, psi, num_threads=4)
```


## Development
The Python package is built with scikit-build-core, while the numerical C++ code is kept in the standalone `quantlop_core` CMake target.
See the [Development](https://simonegasperini.github.io/quantlop/development.html) section in the documentation for more details.

Build the project from source in editable mode and run Python tests with:
```bash
pip install -e .[dev,docs]
pytest -v
```
