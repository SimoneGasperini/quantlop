import os
import numpy as np

from ._quantlop import _evolve


def evolve(ham, psi, coeff=1, num_threads=None):
    if num_threads is None:
        num_threads = 0
    if num_threads == "auto":
        num_threads = os.cpu_count()
    state = np.asarray(psi, dtype=np.complex128, order="C")
    return _evolve(ham, state, coeff, num_threads)
