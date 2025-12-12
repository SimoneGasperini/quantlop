from functools import reduce
from operator import add
from scipy.sparse.linalg import LinearOperator
from .pauli import PauliWord


def get_terms(pauli_sent, num_qubits):
    terms = []
    for pauli_word, coeff in pauli_sent.items():
        pauli_list = ["I"] * num_qubits
        for qubit, pauli_char in pauli_word.items():
            pauli_list[qubit] = pauli_char
        pauli_string = "".join(pauli_list)
        terms.append((coeff, pauli_string))
    return terms


class Hamiltonian(LinearOperator):
    def __new__(cls, *args, allowed=False, **kwargs):
        if not allowed:
            raise ValueError(
                "Hamiltonian cannot be instantiated directly. "
                "Use Hamiltonian.from_pennylane(...) instead."
            )
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, operator, num_qubits):
        shape = (2**num_qubits, 2**num_qubits)
        super().__init__(dtype=complex, shape=shape)
        self._operator = operator
        self._num_qubits = num_qubits

    def __repr__(self):
        return self._operator.__repr__()

    def __mul__(self, scalar):
        new_operator = scalar * self._operator
        num_qubits = self._num_qubits
        return self.from_pennylane(operator=new_operator, num_qubits=num_qubits)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    @classmethod
    def from_pennylane(cls, operator, num_qubits):
        obj = cls.__new__(cls, allowed=True)
        obj.__init__(operator, num_qubits)
        return obj

    def to_matrix(self):
        wire_order = range(self._num_qubits)
        matrix = self._operator.matrix(wire_order=wire_order)
        return matrix

    def _lcu_norm(self):
        pauli_sent = self._operator.pauli_rep
        terms = get_terms(pauli_sent, num_qubits=self._num_qubits)
        norm = sum(abs(coeff) for coeff, _ in terms)
        return norm

    def _matvec(self, vec):
        pauli_sent = self._operator.pauli_rep
        terms = get_terms(pauli_sent, num_qubits=self._num_qubits)
        pauli_words = (coeff * PauliWord(string) for coeff, string in terms)
        linop = reduce(add, pauli_words)
        vec_new = linop._matvec(vec)
        return vec_new
