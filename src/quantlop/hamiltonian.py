from functools import reduce
from operator import add
from scipy.sparse.linalg import LinearOperator
from .pauliword import PauliWord


class Hamiltonian(LinearOperator):
    def __init__(self, pauli_words: list[PauliWord]):
        self._pauli_words = pauli_words
        nq = pauli_words[0].num_qubits
        shape = (2**nq, 2**nq)
        super().__init__(dtype=complex, shape=shape)

    @property
    def pauli_words(self):
        return self._pauli_words

    @property
    def coeffs(self):
        return [pauli_word.coeff for pauli_word in self.pauli_words]

    @property
    def strings(self):
        return [pauli_word.string for pauli_word in self.pauli_words]

    @classmethod
    def from_pennylane(cls, operator_pl, nqubits):
        pauli_words = []
        for pauliword_pl, coeff in operator_pl.pauli_rep.items():
            string = "".join([pauliword_pl.get(i, "I") for i in range(nqubits)])
            pauli_words.append(PauliWord(coeff=coeff, string=string))
        return cls(pauli_words)

    def _matvec(self, vec):
        linop = reduce(add, self.pauli_words)
        return linop._matvec(vec)

    def _adjoint(self):
        return self

    def lcu_norm(self):
        return sum(abs(coeff) for coeff in self.coeffs)
