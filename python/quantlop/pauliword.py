"""Python representation of a weighted Pauli word."""

from ._quantlop import _PauliWord


class PauliWord(_PauliWord):
    r"""A coefficient multiplying a tensor product of Pauli operators.

    A Pauli word represents

    .. math::

        c\,(P_0 \otimes P_1 \otimes \cdots \otimes P_{n-1}),

    where ``c`` is ``coeff`` and each ``P_i`` is selected by one character of
    ``string``. The leftmost character acts on the most-significant qubit in
    the computational-basis index. For example, ``"XI"`` represents
    :math:`X \otimes I` and maps ``|00>`` to ``|10>``.

    Parameters
    ----------
    coeff : complex
        Scalar coefficient of the Pauli term. Real values are accepted and
        converted to double-precision complex values by the native binding.
        Use real coefficients when the term is part of a Hermitian
        Hamiltonian.
    string : str
        Non-empty Pauli string. Each character should be one of ``"I"``,
        ``"X"``, ``"Y"``, or ``"Z"``. Its length determines the number of
        qubits represented by the word.

    Attributes
    ----------
    num_qubits : int
        Number of characters, and therefore qubits, in the Pauli word.
    coeff : complex
        Read-only coefficient supplied at construction.
    string : str
        Read-only Pauli string supplied at construction.

    Notes
    -----
    Instances are implemented by the :mod:`quantlop._quantlop` C++ extension.
    The public attributes expose immutable construction data; create a new
    instance to change either the coefficient or the operator string.

    Examples
    --------
    Construct a three-qubit Pauli term and inspect its components:

    >>> import quantlop as ql
    >>> word = ql.PauliWord(coeff=-0.5j, string="YIZ")
    >>> word.num_qubits
    3
    >>> word.coeff == -0.5j
    True
    >>> word.string
    'YIZ'
    """

    @property
    def num_qubits(self):
        """Number of qubits on which the Pauli word acts.

        Returns
        -------
        int
            Length of :attr:`string`. The property is read-only.
        """
        return self._num_qubits()

    @property
    def coeff(self):
        """Scalar coefficient multiplying the Pauli tensor product.

        Returns
        -------
        complex
            Coefficient stored in double-precision complex form by the native
            implementation. The property is read-only.
        """
        return self._get_coeff()

    @property
    def string(self):
        """Pauli operators in most-significant-qubit-first order.

        Returns
        -------
        str
            Operator string in which character ``i`` acts on qubit axis
            ``i``. The property is read-only.
        """
        return self._get_string()
