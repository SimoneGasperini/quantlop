from functools import reduce
from operator import add
from scipy.sparse.linalg import LinearOperator
from .pauli import PauliWord


def coeffs_and_strings(pauli_sent, num_qubits):
    pauli_strings = []
    for pauli_word, coeff in pauli_sent.items():
        pauli_list = ["I"] * num_qubits
        for qubit, pauli_char in pauli_word.items():
            pauli_list[qubit] = pauli_char
        pauli_str = "".join(pauli_list)
        pauli_strings.append((coeff, pauli_str))
    return pauli_strings


class Hamiltonian(LinearOperator):
    def __new__(cls, *args, **kwargs):
        raise ValueError(
            "Hamiltonian cannot be instantiated directly. "
            "Use Hamiltonian.from_pennylane(...) instead."
        )

    def __repr__(self):
        return self._operator.__repr__()

    @classmethod
    def from_pennylane(cls, operator, num_qubits):
        obj = super().__new__(cls)
        shape = (2**num_qubits, 2**num_qubits)
        obj.__init__(shape=shape, dtype=complex)
        obj._operator = operator
        obj._num_qubits = num_qubits
        return obj

    def to_matrix(self):
        wire_order = range(self._num_qubits)
        matrix = self._operator.matrix(wire_order=wire_order)
        return matrix

    def _matvec(self, vec):
        pauli_sent = self._operator.pauli_rep
        coeffs_strings = coeffs_and_strings(pauli_sent, num_qubits=self._num_qubits)
        pauli_words = (coeff * PauliWord(string) for coeff, string in coeffs_strings)
        linop = reduce(add, pauli_words)
        vec_new = linop._matvec(vec)
        return vec_new
