quantlop
========

.. rst-class:: lead

   High-performance quantum simulation for matrix-free Hamiltonian evolution

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Memory-efficient
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Directly apply Pauli words without storing the dense representation of the Hamiltonian matrix.

   .. grid-item-card:: Multi-threaded
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Run serially by default or enable OpenMP parallel execution with an explicit thread count.

   .. grid-item-card:: PennyLane and Qiskit
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Build Hamiltonians directly from PennyLane ``Hamiltonian`` and Qiskit ``SparsePauliOp`` objects.


Explore the documentation
-------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started 💡
      :link: getting-started
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Install ``quantlop`` and run your first matrix-free simulation of Hamiltonian evolution.

      +++
      Start here →

   .. grid-item-card:: Algorithms 🧮
      :link: algorithms
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Learn how ``quantlop`` evaluates the exponential action without forming a dense operator.

      +++
      Read the guide →

   .. grid-item-card:: API reference 📝
      :link: api-reference
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Browse ``quantlop`` Python interface for Paulis, Hamiltonians, and evolution.

      +++
      Browse the API →

   .. grid-item-card:: Benchmarks 🚀
      :link: benchmarks
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Explore how ``quantlop`` runtime and memory performance scale with system size.

      +++
      Contribute →


.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   getting-started
   algorithms
   benchmarks
   api-reference
   development
