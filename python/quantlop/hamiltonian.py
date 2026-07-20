"""Hamiltonians represented as finite sums of weighted Pauli words."""

import numpy as np

from ._quantlop import _PauliWord
from ._quantlop import _Hamiltonian

char2mat = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


class Hamiltonian(_Hamiltonian):
    r"""A Hamiltonian expressed as a sum of :class:`~quantlop.PauliWord` terms.

    If the supplied words are :math:`c_k P_k`, this object represents

    .. math::

        H = \sum_k c_k P_k.

    This representation lets :func:`quantlop.evolve` apply ``H`` to a state
    vector without materializing a dense :math:`2^n \times 2^n` matrix.

    Parameters
    ----------
    pwords : sequence of PauliWord
        Non-empty collection of Pauli terms. All words must act on the same
        number of qubits. The evolution algorithm assumes their sum is
        Hermitian, which normally means using real coefficients for individual
        Pauli words (or otherwise arranging terms so the full sum is
        Hermitian).

    Attributes
    ----------
    num_qubits : int
        Number of qubits acted on by every term.
    num_terms : int
        Number of Pauli words in the sum, including duplicate words and terms
        with zero coefficients.

    Notes
    -----
    The constructor does not combine like terms. Keeping the representation
    sparse is usually preferable because the cost of a matrix-free
    Hamiltonian application scales with both ``num_terms`` and
    :math:`2^{\mathtt{num_qubits}}`.

    The native implementation expects at least one word and equal-length,
    valid Pauli strings. Callers should enforce those conditions before
    construction.

    Examples
    --------
    Build :math:`H = 0.5 Z \otimes I - 0.25 X \otimes X`:

    >>> import quantlop as ql
    >>> ham = ql.Hamiltonian(
    ...     pwords=[
    ...         ql.PauliWord(0.5, "ZI"),
    ...         ql.PauliWord(-0.25, "XX"),
    ...     ]
    ... )
    >>> ham.num_qubits, ham.num_terms
    (2, 2)
    """

    @property
    def num_qubits(self):
        """Number of qubits acted on by the Hamiltonian.

        Returns
        -------
        int
            Number inferred from the first Pauli word. All terms are expected
            to have this length. The property is read-only.
        """
        return self._num_qubits()

    @property
    def num_terms(self):
        """Number of Pauli-word terms stored in the Hamiltonian.

        Returns
        -------
        int
            Number of terms as supplied. Identical terms are not merged. The
            property is read-only.
        """
        return self._num_terms()

    @classmethod
    def from_pennylane(cls, operator, num_qubits):
        """Construct a Hamiltonian from a PennyLane Pauli operator.

        The conversion reads ``operator.pauli_rep`` and expands each sparse
        Pauli sentence entry into a full-length word. Qubits absent from an
        entry are filled with identity operators. The resulting word order is
        ``0, 1, ..., num_qubits - 1``, matching a PennyLane matrix requested
        with ``wire_order=range(num_qubits)``.

        Parameters
        ----------
        operator : pennylane.operation.Operator
            PennyLane operator with a defined ``pauli_rep``. Its wire labels
            must be integer indices in ``range(num_qubits)``. Operators that
            do not have a Pauli representation cannot be converted by this
            method.
        num_qubits : int
            Total number of qubits in the returned Hamiltonian. This can be
            larger than the number of wires used by ``operator``; unused wires
            are represented by identities.

        Returns
        -------
        Hamiltonian
            Native quantlop Hamiltonian containing one term per entry in the
            PennyLane Pauli representation.

        Examples
        --------
        >>> import pennylane as qml
        >>> import quantlop as ql
        >>> op = 0.5 * qml.Z(0) + 0.2 * (qml.X(0) @ qml.X(1))
        >>> ham = ql.Hamiltonian.from_pennylane(op, num_qubits=2)
        >>> ham.num_terms
        2

        See Also
        --------
        from_qiskit : Construct from a Qiskit ``SparsePauliOp``.
        matrix : Materialize the converted operator for verification.
        """
        pwords = []
        for pw, coeff in operator.pauli_rep.items():
            string = "".join(pw.get(i, "I") for i in range(num_qubits))
            pwords.append(_PauliWord(coeff=coeff, string=string))
        return cls(pwords=pwords)

    @classmethod
    def from_qiskit(cls, operator):
        """Construct a Hamiltonian from a Qiskit ``SparsePauliOp``.

        Each label and coefficient is copied from the input in its existing
        order. Qiskit's Pauli labels place the highest-index qubit on the left;
        this is the same tensor-product order used by :meth:`matrix`, so the
        returned dense matrices agree directly.

        Parameters
        ----------
        operator : qiskit.quantum_info.SparsePauliOp
            Qiskit sparse Pauli operator. All labels are expected to have the
            same width, as guaranteed by ``SparsePauliOp``.

        Returns
        -------
        Hamiltonian
            Native quantlop Hamiltonian with the input terms and coefficients.

        Examples
        --------
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> import quantlop as ql
        >>> op = SparsePauliOp(["ZI", "XX"], coeffs=[0.5, -0.25])
        >>> ham = ql.Hamiltonian.from_qiskit(op)
        >>> ham.num_qubits, ham.num_terms
        (2, 2)

        See Also
        --------
        from_pennylane : Construct from a PennyLane Pauli operator.
        matrix : Materialize the converted operator for verification.
        """
        pwords = []
        for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs):
            pwords.append(_PauliWord(coeff=coeff, string=label))
        return cls(pwords=pwords)

    def matrix(self):
        """Return the dense matrix represented by this Hamiltonian.

        The matrix is assembled as a sum of Kronecker products. Characters are
        processed from left to right, so the first character in each Pauli
        word is the leftmost (most-significant) tensor factor.

        Returns
        -------
        numpy.ndarray
            Complex array of shape ``(2**num_qubits, 2**num_qubits)`` with
            dtype :class:`numpy.complex128`.

        Notes
        -----
        This method requires :math:`O(4^n)` memory and work for ``n`` qubits.
        It is intended for inspection, validation, and small systems. Use
        :func:`quantlop.evolve` for matrix-free evolution of larger systems.

        Examples
        --------
        >>> import numpy as np
        >>> import quantlop as ql
        >>> ham = ql.Hamiltonian([ql.PauliWord(1.0, "Z")])
        >>> np.array_equal(ham.matrix(), np.diag([1.0, -1.0]))
        True
        """
        dim = 2 ** self._num_qubits()
        matrix = np.zeros(shape=(dim, dim), dtype=np.complex128)
        for pw in self._get_pwords():
            mat = np.ones(shape=(1, 1), dtype=np.complex128)
            for char in pw._get_string():
                pauli = char2mat.get(char, char2mat["I"])
                mat = np.kron(mat, pauli)
            matrix += pw._get_coeff() * mat
        return matrix
