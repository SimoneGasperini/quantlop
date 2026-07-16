from ._quantlop import _PauliWord


class PauliWord(_PauliWord):
    @property
    def num_qubits(self):
        return self._num_qubits()

    @property
    def coeff(self):
        return self._get_coeff()

    @property
    def string(self):
        return self._get_string()
