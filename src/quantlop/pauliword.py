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

    def __mul__(self, other):
        if np.isscalar(other):
            return self.__class__(other * self.coeff, self.string)
        if isinstance(other, self.__class__):
            phase = 1 + 0j
            plist = []
            for pauli1, pauli2 in zip(self.string, other.string):
                ph, pauli = _paulimul[(pauli1, pauli2)]
                phase *= ph
                plist.append(pauli)
            coeff = self.coeff * other.coeff * phase
            string = "".join(plist)
            return self.__class__(coeff, string)
        raise NotImplementedError

    def __rmul__(self, other):
        if np.isscalar(other):
            return self.__mul__(other)
        raise NotImplementedError

    def _matvec(self, vec):
        nq = self.num_qubits
        out = np.empty(2**nq, dtype=complex)
        for i in range(2**nq):
            b = [(i >> (nq - 1 - k)) & 1 for k in range(nq)]
            b_new = list(b)
            phase = 1 + 0j
            for q, p in enumerate(self.string):
                if p == "X":
                    b_new[q] ^= 1
                elif p == "Y":
                    b_new[q] ^= 1
                    phase *= 1j if b[q] == 0 else -1j
                elif p == "Z":
                    if b[q] == 1:
                        phase *= -1
            j = sum(bit << (nq - 1 - k) for k, bit in enumerate(b_new))
            out[j] = phase * vec[i]
        return self.coeff * out


_paulimul = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}
