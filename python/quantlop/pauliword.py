from ._quantlop import PauliWord as _PauliWord


class PauliWord:
    def __init__(self, coeff, string):
        self._coeff = coeff
        self._string = string
        self._native = _PauliWord(coeff=coeff, string=string)

    @property
    def coeff(self):
        return self._coeff

    @property
    def string(self):
        return self._string

    @property
    def num_qubits(self):
        return len(self._string)
