from functools import reduce
from operator import add
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .pauliword import PauliWord


class Hamiltonian(LinearOperator):
    def __init__(self, pauli_words: list[PauliWord]):
        self._pauli_words = pauli_words
        nq = pauli_words[0].num_qubits
        shape = (2**nq, 2**nq)
        super().__init__(dtype=complex, shape=shape)
        self._linop = reduce(add, self.pauli_words)

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
        pws = []
        for pauliword_pl, coeff in operator_pl.pauli_rep.items():
            string = "".join([pauliword_pl.get(i, "I") for i in range(nqubits)])
            pws.append(PauliWord(coeff=coeff, string=string))
        return cls(pauli_words=pws)

    def __mul__(self, k):
        if np.isscalar(k):
            pws = [PauliWord(k * pw.coeff, pw.string) for pw in self.pauli_words]
            return self.__class__(pauli_words=pws)
        raise NotImplementedError

    def __rmul__(self, k):
        return self.__mul__(k)

    def _matvec(self, vec):
        return self._linop._matvec(vec)

    def lcu_norm(self):
        return sum(abs(coeff) for coeff in self.coeffs)
