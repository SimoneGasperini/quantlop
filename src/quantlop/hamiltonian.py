from functools import reduce
from operator import add
import pylops
from pauli import Identity, PauliX, PauliY, PauliZ


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
    def from_pennylane(cls, lincomb):
        obj = super().__new__(cls)
        obj._lincomb = lincomb
        return obj

    def to_matrix(self):
        return self._lincomb.matrix()

    def to_linop(self):
        linops = []
        pauli_sentence = self._lincomb.pauli_rep
        qubits = self._lincomb.wires
        for pword, coeff in pauli_sentence.items():
            paulis = [char2linop[pword.get(qubit, "I")] for qubit in qubits]
            linops.append(coeff * reduce(pylops.Kronecker, paulis))
        linop = reduce(add, linops)
        return linop
