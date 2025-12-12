from functools import reduce
from operator import add, mul
import numpy as np
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

    def __mul__(self, other):
        if np.isscalar(other):
            pws = [PauliWord(other * pw.coeff, pw.string) for pw in self.pauli_words]
            return self.__class__(pws)
        if isinstance(other, self.__class__):
            accumulator = {}
            for pw1 in self.pauli_words:
                for pw2 in other.pauli_words:
                    pw = pw1 * pw2
                    accumulator[pw.string] = accumulator.get(pw.string, 0) + pw.coeff
            pws = [PauliWord(coeff, string) for string, coeff in accumulator.items()]
            return self.__class__(pws)
        raise NotImplementedError

    def __rmul__(self, other):
        if np.isscalar(other):
            return self.__mul__(other)
        raise NotImplementedError

    def __pow__(self, p):
        if isinstance(p, int) and p > 0:
            return reduce(mul, (self for _ in range(p)))
        raise NotImplementedError

    def _matvec(self, vec):
        linop = reduce(add, self.pauli_words)
        return linop._matvec(vec)

    def lcu_norm(self):
        return sum(abs(coeff) for coeff in self.coeffs)
