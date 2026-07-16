import numpy as np

from ._quantlop import Hamiltonian as _Hamiltonian

char2mat = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


class Hamiltonian:
    def __init__(self, pauli_words):
        self._pauli_words = pauli_words
        self._native = _Hamiltonian(pauli_words=[pw._native for pw in pauli_words])

    @property
    def pauli_words(self):
        return self._pauli_words

    @property
    def num_qubits(self):
        return self._pauli_words[0].num_qubits

    def matrix(self):
        dim = 2**self.num_qubits
        matrix = np.zeros(shape=(dim, dim), dtype=np.complex128)
        for pw in self._pauli_words:
            mat = np.ones(shape=(1, 1), dtype=np.complex128)
            for char in pw.string:
                pauli = char2mat.get(char, char2mat["I"])
                mat = np.kron(mat, pauli)
            matrix += pw.coeff * mat
        return matrix
