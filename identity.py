import numpy as np
from pylops import LinearOperator


class Identity(LinearOperator):

    def __init__(self, shape):
        super().__init__(shape=shape, dtype=complex)

    def _matvec(self, psi):
        return np.array(psi)


class Id(Identity):

    def __init__(self):
        super().__init__(shape=(2,2))
