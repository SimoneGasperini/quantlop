from functools import reduce
from operator import add
import pylops
from .pauli import Identity, PauliX, PauliY, PauliZ


char2linop = {
    "I": Identity(),
    "X": PauliX(),
    "Y": PauliY(),
    "Z": PauliZ(),
}


class Hamiltonian:
    def __new__(cls, *args, **kwargs):
        raise ValueError(
            "Hamiltonian cannot be instantiated directly. "
            "Use Hamiltonian.from_pennylane(...) instead."
        )

    def __repr__(self):
        return self._lincomb.__repr__()

    @classmethod
    def from_pennylane(cls, lincomb, num_qubits):
        obj = super().__new__(cls)
        obj._lincomb = lincomb
        obj._num_qubits = num_qubits
        return obj

    def to_matrix(self):
        wire_order = range(self._num_qubits)
        matrix = self._lincomb.matrix(wire_order=wire_order)
        return matrix

    def to_linop(self):
        linops = []
        pauli_sentence = self._lincomb.pauli_rep
        wire_order = range(self._num_qubits)
        for pword, coeff in pauli_sentence.items():
            paulis = [char2linop[pword.get(wire, "I")] for wire in wire_order]
            linops.append(coeff * reduce(pylops.Kronecker, paulis))
        linop = reduce(add, linops)
        return linop
