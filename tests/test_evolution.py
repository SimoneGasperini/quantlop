import pytest
import numpy as np
import scipy as sp
import quantlop as ql

EVOLVE_FUNCS = [ql.evolve_higham, ql.evolve_krylov]


@pytest.mark.parametrize("evolve_func", EVOLVE_FUNCS)
@pytest.mark.parametrize("num_qubits", range(1, 11))
def test_against_scipy(evolve_func, num_qubits):
    psi = ql.utils.get_rand_statevector(num_qubits)
    num_terms = num_qubits * 5
    ham = ql.utils.get_rand_hamiltonian(num_qubits, num_terms=num_terms)
    psi_scipy = sp.linalg.expm(-1j * ham.matrix()) @ psi
    psi_quantlop = evolve_func(ham, psi)
    assert np.allclose(psi_scipy, psi_quantlop)


@pytest.mark.parametrize("evolve_func", EVOLVE_FUNCS)
@pytest.mark.parametrize("num_qubits", range(1, 6))
def test_identity_evolution(evolve_func, num_qubits):
    psi = ql.utils.get_rand_statevector(num_qubits)
    num_terms = num_qubits * 5
    ham = ql.utils.get_rand_hamiltonian(num_qubits, num_terms=num_terms)
    new_psi = evolve_func(ham, psi, theta=0.0)
    assert np.allclose(psi, new_psi)
