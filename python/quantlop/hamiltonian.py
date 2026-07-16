from ._quantlop import Hamiltonian as _Hamiltonian


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
