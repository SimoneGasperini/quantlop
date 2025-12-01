class Hamiltonian:

    def __init__(self, coeffs, paulis):
        self._coeffs = coeffs
        self._paulis = paulis

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def paulis(self):
        return self._paulis

    @classmethod
    def from_pennylane(cls, pl_ham):
        coeffs = ...
        paulis = ...
        return cls(coeffs=coeffs, paulis=paulis)
