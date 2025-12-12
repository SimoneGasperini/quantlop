import pytest
import numpy as np
import scipy as sp
import pennylane as qml
from quantlop import Hamiltonian, evolve
from utils import get_rand_statevector, get_rand_hamiltonian


@pytest.mark.parametrize("nqubits", range(1, 9))
def test_scipy(nqubits):
    psi = get_rand_statevector(nqubits=nqubits)
    op = get_rand_hamiltonian(nqubits=nqubits, num_terms=nqubits * 5)
    mat = op.matrix(range(nqubits))
    psi_scipy = sp.linalg.expm(-1j * mat) @ psi
    ham = Hamiltonian.from_pennylane(op, nqubits=nqubits)
    psi_linop = evolve(ham, psi)
    assert np.allclose(psi_scipy, psi_linop)


@pytest.mark.parametrize("nqubits", range(1, 9))
def test_pennylane(nqubits):
    psi = get_rand_statevector(nqubits=nqubits)
    op = get_rand_hamiltonian(nqubits=nqubits, num_terms=nqubits * 5)

    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.StatePrep(psi, wires=range(nqubits))
        qml.evolve(op)
        return qml.state()

    psi_pennylane = circuit()
    ham = Hamiltonian.from_pennylane(op, nqubits=nqubits)
    psi_linop = evolve(ham, psi)
    assert np.allclose(psi_pennylane, psi_linop)
