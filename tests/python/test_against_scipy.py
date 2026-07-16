import pytest
import numpy as np
import scipy as sp
import quantlop as ql

from utils import get_rand_hamiltonian


@pytest.mark.parametrize("num_qubits", range(1, 11))
def test_against_scipy(num_qubits):
    psi = np.zeros(2**num_qubits)
    psi[0] = 1
    ham = get_rand_hamiltonian(num_qubits, num_terms=num_qubits * 5)
    psi_scipy = sp.linalg.expm(-1j * ham.matrix()) @ psi
    psi_linop = ql.evolve(ham, psi)
    assert np.allclose(psi_scipy, psi_linop)
