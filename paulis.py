import numpy as np
from pylops import LinearOperator


class PauliX(LinearOperator):

    def __init__(self):
        super().__init__(shape=(2,2), dtype=complex)

    def _matvec(self, psi):
        alpha, beta = psi
        return np.array([beta, alpha])


class PauliY(LinearOperator):

    def __init__(self):
        super().__init__(shape=(2,2), dtype=complex)

    def _matvec(self, psi):
        alpha, beta = psi
        return np.array([-1j * beta, 1j * alpha])


class PauliZ(LinearOperator):

    def __init__(self):
        super().__init__(shape=(2,2), dtype=complex)

    def _matvec(self, psi):
        alpha, beta = psi
        return np.array([alpha, -beta])
