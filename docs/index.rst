quantlop
========

High-performance quantum simulation for matrix-free Hamiltonian evolution.

``quantlop`` evolves quantum systems whose Hamiltonians are sparse sums of
Pauli words. It applies those terms directly to a state vector and uses a
Krylov method to approximate the matrix exponential, avoiding construction of
the exponentially large dense Hamiltonian matrix.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api
