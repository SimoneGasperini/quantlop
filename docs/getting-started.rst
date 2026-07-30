Getting started
===============

``quantlop`` is a high-performance quantum simulator for evolving systems whose
Hamiltonians can be expressed as sparse sums of Pauli words. It applies those
operators directly to the state vector and evaluates the matrix-exponential
action with either an adaptively scaled, truncated Taylor series or a
Lanczos-Krylov subspace projection, avoiding the memory cost of constructing a
dense Hamiltonian.

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

   Clone the repository and install with its development and documentation tools:

   .. code-block:: console

      pip install -e .[dev,docs]


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

   evolved_psi = ql.evolve_higham(ham, psi)
   # or
   evolved_psi = ql.evolve_krylov(ham, psi)

Both algorithms select their Taylor truncation or Krylov dimension automatically.
The default relative tolerance is `1e-9`.
Smaller values generally improve accuracy at the cost of more computation.
The tolerance guides the internal approximation rather than measuring the final error directly.

The interface allows to import Hamiltonians from other quantum computing frameworks using
:meth:`~quantlop.Hamiltonian.from_pennylane` and :meth:`~quantlop.Hamiltonian.from_qiskit`.

By default, evolution uses one fewer than the logical CPU count reported by the operating system.
The thread count can be overridden by passing a positive integer as ``num_threads`` to
:func:`quantlop.evolve_higham` or :func:`quantlop.evolve_krylov`.
