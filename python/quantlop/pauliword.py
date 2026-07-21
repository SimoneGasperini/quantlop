from ._quantlop import _PauliWord


class PauliWord(_PauliWord):
    r"""Represent a weighted tensor product of single-qubit Pauli operators

    .. math::

        P = c\,\bigotimes_i \sigma_i

    where :math:`c` is a complex ``coeff`` and each :math:`\sigma_i`
    is a single-qubit Pauli operator applied to qubit :math:`i`.
    The leftmost character of ``string`` acts on the most-significant qubit in
    the computational-basis index. For example, ``"XII"`` represents
    :math:`X \otimes I \otimes I` and maps :math:`|000\rangle` to :math:`|100\rangle`.

    Parameters
    ----------
    coeff : complex
        Scalar complex coefficient of the Pauli word.
    string : str
        Non-empty string of single-qubit Pauli operators.
        Each character should be one of ``"I"``, ``"X"``, ``"Y"``, or ``"Z"``.
        Its length determines the total number of qubits.

    Attributes
    ----------
    num_qubits : int
        Number of qubits in the Pauli word.
    coeff : complex
        Coefficient supplied at construction.
    string : str
        Pauli string supplied at construction.
    """

    @property
    def num_qubits(self):
        return self._num_qubits()

    @property
    def coeff(self):
        return self._get_coeff()

    @property
    def string(self):
        return self._get_string()
