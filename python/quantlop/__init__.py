"""Matrix-free evolution of Hamiltonians expressed as sums of Pauli words.

``quantlop`` provides a small Python API backed by a native C++ implementation.
Use :class:`PauliWord` to describe weighted tensor products of single-qubit
Pauli operators, combine those terms in a :class:`Hamiltonian`, and apply the
corresponding time-evolution operator with :func:`evolve`.

The usual workflow is::

    import numpy as np
    import quantlop as ql

    ham = ql.Hamiltonian(
        pwords=[
            ql.PauliWord(coeff=0.5, string="ZI"),
            ql.PauliWord(coeff=-0.25, string="XX"),
        ]
    )
    psi = np.array([1, 0, 0, 0], dtype=np.complex128)
    evolved = ql.evolve(ham, psi, coeff=0.1)

The evolution routine applies the Hamiltonian directly to state vectors. It
does not construct the exponentially large dense Hamiltonian matrix.
"""

from .pauliword import PauliWord
from .hamiltonian import Hamiltonian
from .evolution import evolve

__all__ = ["Hamiltonian", "PauliWord", "evolve"]
