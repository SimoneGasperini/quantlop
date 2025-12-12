import pytest
import numpy as np
import scipy as sp
import pennylane as qml
from quantlop import Hamiltonian, evolve
from utils import get_rand_statevector, get_rand_hamiltonian


@pytest.mark.parametrize("num_qubits", range(1, 9))
def test_scipy(num_qubits):
    psi = get_rand_statevector(num_qubits=num_qubits)
    op = get_rand_hamiltonian(num_qubits=num_qubits, num_terms=num_qubits * 5)
    ham = Hamiltonian.from_pennylane(op, num_qubits=num_qubits)
    psi_linop = evolve(ham, psi)
    psi_dense = sp.linalg.expm(-1j * ham.to_matrix()) @ psi
    assert np.allclose(psi_linop, psi_dense)


@pytest.mark.parametrize("num_qubits", range(1, 9))
def test_pennylane(num_qubits):
    psi = get_rand_statevector(num_qubits=num_qubits)
    op = get_rand_hamiltonian(num_qubits=num_qubits, num_terms=num_qubits * 5)
    ham = Hamiltonian.from_pennylane(op, num_qubits=num_qubits)
    psi_linop = evolve(ham, psi)

    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.StatePrep(psi, wires=range(num_qubits))
        qml.evolve(op)
        return qml.state()

    assert np.allclose(psi_linop, circuit())
