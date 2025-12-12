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
        for i in range(2**n):
            b = [(i >> (n - 1 - k)) & 1 for k in range(n)]
            b_new = list(b)
            phase = 1.0 + 0j
            for q, p in enumerate(self._string):
                if p == "X":
                    b_new[q] ^= 1
                elif p == "Y":
                    b_new[q] ^= 1
                    phase *= 1j if b[q] == 0 else -1j
                elif p == "Z":
                    if b[q] == 1:
                        phase *= -1
            j = sum(bit << (n - 1 - k) for k, bit in enumerate(b_new))
            vec_new[j] = phase * vec[i]
        return vec_new
