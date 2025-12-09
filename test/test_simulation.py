import pytest
import numpy as np
import scipy as sp
import pennylane as qml
from quantlop import Hamiltonian, evolve
from utils import get_rand_statevector, get_rand_hamiltonian


@pytest.mark.parametrize("num_qubits", (1, 2, 3, 4))
def test_scipy(num_qubits):
    psi = get_rand_statevector(num_qubits=num_qubits)
    num_terms = num_qubits * 4
    lincomb = get_rand_hamiltonian(num_qubits=num_qubits, num_terms=num_terms)
    ham = Hamiltonian.from_pennylane(lincomb, num_qubits=num_qubits)
    psi_linop = evolve(ham, psi)
    psi_dense = sp.linalg.expm(-1j * ham.to_matrix()) @ psi
    assert np.allclose(psi_linop, psi_dense)


@pytest.mark.parametrize("num_qubits", (1, 2, 3, 4))
def test_pennylane(num_qubits):
    psi = get_rand_statevector(num_qubits=num_qubits)
    num_terms = num_qubits * 4
    lincomb = get_rand_hamiltonian(num_qubits=num_qubits, num_terms=num_terms)
    ham = Hamiltonian.from_pennylane(lincomb, num_qubits=num_qubits)
    psi_linop = evolve(ham, psi)

    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.StatePrep(psi, wires=range(num_qubits))
        qml.evolve(lincomb)
        return qml.state()

    assert np.allclose(psi_linop, circuit())
