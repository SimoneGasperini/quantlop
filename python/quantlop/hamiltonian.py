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
    @property
    def num_qubits(self):
        return self._num_qubits()

    @property
    def num_terms(self):
        return self._num_terms()

    @classmethod
    def from_pennylane(cls, operator, num_qubits):
        pwords = []
        for pw, coeff in operator.pauli_rep.items():
            string = "".join(pw.get(i, "I") for i in range(num_qubits))
            pwords.append(_PauliWord(coeff=coeff, string=string))
        return cls(pwords=pwords)

    @classmethod
    def from_qiskit(cls, operator):
        pwords = []
        for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs):
            pwords.append(_PauliWord(coeff=coeff, string=label))
        return cls(pwords=pwords)

    def matrix(self):
        dim = 2 ** self._num_qubits()
        matrix = np.zeros(shape=(dim, dim), dtype=np.complex128)
        for pw in self._get_pwords():
            mat = np.ones(shape=(1, 1), dtype=np.complex128)
            for char in pw._get_string():
                pauli = char2mat.get(char, char2mat["I"])
                mat = np.kron(mat, pauli)
            matrix += pw._get_coeff() * mat
        return matrix
