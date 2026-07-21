import pytest
import numpy as np
import scipy as sp
import quantlop as ql

from utils import get_rand_statevector, get_rand_hamiltonian


@pytest.mark.parametrize("num_qubits", range(1, 6))
def test_identity_evolution(num_qubits):
    psi = get_rand_statevector(num_qubits)
    num_terms = num_qubits * 5
    ham = get_rand_hamiltonian(num_qubits, num_terms=num_terms)
    new_psi = ql.evolve(ham, psi, theta=0.0)
    assert np.allclose(psi, new_psi)


@pytest.mark.parametrize("num_qubits", range(1, 11))
def test_against_scipy(num_qubits):
    psi = get_rand_statevector(num_qubits)
    num_terms = num_qubits * 5
    ham = get_rand_hamiltonian(num_qubits, num_terms=num_terms)
    psi_scipy = sp.linalg.expm(-1j * ham.matrix()) @ psi
    psi_linop = ql.evolve(ham, psi)
    assert np.allclose(psi_scipy, psi_linop)
