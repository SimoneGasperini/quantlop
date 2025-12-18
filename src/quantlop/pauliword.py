import numpy as np
from scipy.sparse.linalg import LinearOperator


class PauliWord(LinearOperator):
    def __init__(self, coeff: complex, string: str):
        self._coeff = coeff
        self._string = string
        nq = len(string)
        shape = (2**nq, 2**nq)
        super().__init__(dtype=complex, shape=shape)

    @property
    def coeff(self):
        return self._coeff

    @property
    def string(self):
        return self._string

    @property
    def num_qubits(self):
        return len(self.string)

    def __mul__(self, scalar):
        if np.isscalar(scalar):
            return self.__class__(scalar * self.coeff, self.string)
        raise NotImplementedError

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def _matvec(self, vec):
        out = vec.reshape((2,) * self.num_qubits)
        for axis, pauli in enumerate(self.string):
            if pauli == "X":
                out = np.flip(out, axis=axis)
            elif pauli == "Y":
                out = np.flip(out, axis=axis)
                shape = [1] * self.num_qubits
                shape[axis] = 2
                out = -1j * out * np.array([1, -1], dtype=complex).reshape(shape)
            elif pauli == "Z":
                shape = [1] * self.num_qubits
                shape[axis] = 2
                out = out * np.array([1, -1], dtype=complex).reshape(shape)
        return self.coeff * out.reshape(-1)
