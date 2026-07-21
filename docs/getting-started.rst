Getting started
===============

``quantlop`` is a high-performance quantum simulator for evolving systems whose
Hamiltonians can be expressed as sparse sums of Pauli words. It applies those
operators directly to the state vector and uses a Krylov method to approximate
the matrix-exponential action, avoiding the memory cost of constructing a dense
Hamiltonian.

This page walks you through installing ``quantlop``, defining a qubit
Hamiltonian in the Pauli basis, preparing an initial state, and running your
first matrix-free simulation.

Installation
------------

Create or activate a Python virtual environment, then install the latest
release from PyPI:

.. code-block:: console

   pip install quantlop

.. admonition:: Working on quantlop itself?
   :class: ql-note

   Clone the repository and install the package with its development tools:

   .. code-block:: console

      pip install -e .[dev]


Quick example
-------------

Here is a simple code example using ``quantlop`` native data structures:

.. testcode::

   import numpy as np
   import quantlop as ql

   num_qubits = 3

   pwords = [
       ql.PauliWord(coeff=0.5, string="ZZI"),
       ql.PauliWord(coeff=0.2, string="YIX"),
   ]
   ham = ql.Hamiltonian(pwords=pwords)

   psi = np.zeros(2**num_qubits, dtype=complex)
   psi[0] = 1.0

   evolved_psi = ql.evolve(ham, psi)

The interface allows to import Hamiltonians from other quantum computing
frameworks using :meth:`~quantlop.Hamiltonian.from_pennylane` and
:meth:`~quantlop.Hamiltonian.from_qiskit`.

Multi-threaded execution is also available by passing ``num_threads``
to the :func:`quantlop.evolve` function.
