import numpy as np
from scipy.sparse.linalg import LinearOperator


class PauliWord(LinearOperator):
    def __init__(self, string):
        n = len(string)
        super().__init__(shape=(2**n, 2**n), dtype=complex)
        self._string = string

    def _matvec(self, vec):
        n = len(self._string)
        vec_new = np.empty(2**n, dtype=complex)
        for i in range(len(vec)):
            b = [(i >> k) & 1 for k in reversed(range(n))]
            b_new = b.copy()
            phase = 1.0 + 0j
            for qubit, pauli in enumerate(self._string):
                if pauli == "X":
                    b_new[qubit] ^= 1
                elif pauli == "Y":
                    b_new[qubit] ^= 1
                    if b[qubit] == 0:
                        phase *= 1j
                    else:
                        phase *= -1j
                elif pauli == "Z":
                    if b[qubit] == 1:
                        phase *= -1
            j = sum(b << k for k, b in enumerate(reversed(b_new)))
            vec_new[j] = phase * vec[i]
        return vec_new
