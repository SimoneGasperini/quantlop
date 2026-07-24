quantlop
========

.. rst-class:: lead

   High-performance quantum simulation for matrix-free Hamiltonian evolution

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Matrix-free
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Apply Pauli words directly to the state vector and avoid storing
      the exponentially large Hamiltonian matrix.

   .. grid-item-card:: Krylov-powered
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Approximate time evolution through the action of the matrix exponential,
      using methods for large sparse problems.

   .. grid-item-card:: Multi-threaded
      :class-card: sd-shadow-sm sd-outline-primary
      :class-title: sd-text-primary

      Run serially by default or enable OpenMP execution with an explicit
      thread count or ``num_threads="auto"``.


Explore the documentation
-------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting-started
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Install ``quantlop`` and run your first matrix-free simulation of Hamiltonian evolution.

      +++
      Start here →

   .. grid-item-card:: Overview
      :link: overview
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Learn more about Krylov methods and understand the algorithm implemented in ``quantlop``.

      +++
      Read the guide →

   .. grid-item-card:: API reference
      :link: api-reference
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Browse the Python interface for Pauli words, Hamiltonians, and evolution.

      +++
      Browse the API →

   .. grid-item-card:: Development
      :link: development
      :link-type: doc
      :class-card: sd-shadow-sm sd-outline-info
      :class-title: sd-text-info
      :class-footer: sd-text-info

      Build from source, run the test suite, and contribute to the project.

      +++
      Contribute →


.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   getting-started
   overview
   api-reference
   development
