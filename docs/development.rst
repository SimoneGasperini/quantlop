Development
===========

The Python package is built with ``scikit-build-core`` and uses ``nanobind``
for the Python bindings, while the numerical C++ code is available as the
standalone ``quantlop_core`` CMake target.

Project setup
-------------

Clone the repository, then install the package in editable mode together with
the development and documentation dependencies:

.. code-block:: console

   python -m pip install -e .[dev,docs]

.. WARNING:: 
   If you are on macOS, you need first to install ``libomp`` with either 
   `Homebrew <https://brew.sh/>`_ or `MacPorts <https://www.macports.org/>`_:

   .. code-block:: console

      brew install libomp
      # or
      sudo port install libomp +top_level

Run the Python test suite with:

.. code-block:: console

   python -m pytest -v

To configure and build the native C++ target directly:

.. code-block:: console

   cmake -S . -B build
   cmake --build build

Format code
-----------

Format Python files and run the Ruff checks with:

.. code-block:: console

   python -m ruff format .
   python -m ruff check .

Install the pre-commit hooks to format Python and C++ files automatically:

.. code-block:: console

   python -m pre_commit install

Build docs
----------

Check the examples in API docstrings, then build the documentation with
warnings treated as errors:

.. code-block:: console

   python -m sphinx -b doctest -W docs site
   python -m sphinx -W docs site
