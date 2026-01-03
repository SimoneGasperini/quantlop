from functools import cache
import numpy as np


@cache
def _yz_phase(nq):
    y_phase = []
    z_phase = []
    for axis in range(nq):
        shape = [1] * nq
        shape[axis] = 2
        y_phase.append(np.array([-1j, 1j], dtype=complex).reshape(shape))
        z_phase.append(np.array([1, -1], dtype=complex).reshape(shape))
    return y_phase, z_phase


class PauliWord:
    def __init__(self, coeff: complex, string: str):
        self._coeff = coeff
        self._string = string

    @property
    def coeff(self):
        return self._coeff

    @property
    def string(self):
        return self._string

    @property
    def num_qubits(self):
        return len(self.string)

    def _matvec(self, vec):
        out = vec.reshape((2,) * self.num_qubits)
        y_phase, z_phase = _yz_phase(nq=self.num_qubits)
        for axis, pauli in enumerate(self.string):
            if pauli == "X":
                out = np.flip(out, axis=axis)
            elif pauli == "Y":
                out = np.flip(out, axis=axis)
                out = out * y_phase[axis]
            elif pauli == "Z":
                out = out * z_phase[axis]
        return self.coeff * out.reshape(-1)
