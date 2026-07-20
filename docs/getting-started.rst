Getting started
===============

Installation
------------

``quantlop`` requires Python 3.11 or later and a C++20-compatible compiler.
Install the latest release from PyPI:

.. code-block:: console

   pip install quantlop

On macOS, install the OpenMP runtime first:

.. code-block:: console

   brew install libomp
   pip install quantlop

Basic evolution
---------------

Create a Hamiltonian from weighted Pauli words, prepare a state vector, and
call :func:`quantlop.evolve`:

.. code-block:: python

   import numpy as np
   import quantlop as ql

   num_qubits = 3
   ham = ql.Hamiltonian(
       pwords=[
           ql.PauliWord(coeff=0.5, string="ZZI"),
           ql.PauliWord(coeff=0.2, string="YIX"),
       ]
   )

   psi = np.zeros(2**num_qubits, dtype=complex)
   psi[0] = 1.0
   evolved_psi = ql.evolve(ham, psi)

Hamiltonians can also be imported from PennyLane operators with
:meth:`quantlop.Hamiltonian.from_pennylane` or from Qiskit
``SparsePauliOp`` objects with :meth:`quantlop.Hamiltonian.from_qiskit`.

Multi-threading
---------------

Evolution is serial by default. Pass a positive integer to ``num_threads`` to
request that many OpenMP threads, or pass ``"auto"`` to use the logical CPU
count reported by the operating system:

.. code-block:: python

   evolved_psi = ql.evolve(ham, psi, num_threads="auto")
