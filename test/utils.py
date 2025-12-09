import numpy as np
import pennylane as qml


char2qml = {"I": qml.I, "X": qml.X, "Y": qml.Y, "Z": qml.Z}
paulis = ("I", "X", "Y", "Z")


def get_rand_statevector(num_qubits, seed=None):
    rng = np.random.default_rng(seed=seed)
    real = rng.random(2**num_qubits)
    imag = rng.random(2**num_qubits)
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


def get_rand_pauliword(num_qubits, paulis=paulis, seed=None):
    rng = np.random.default_rng(seed=seed)
    pauli_list = rng.choice(paulis, size=num_qubits)
    return "".join(pauli_list)


def get_rand_hamiltonian(num_qubits, num_terms, paulis=paulis, seed=None):
    rng = np.random.default_rng(seed=seed)
    coeffs = rng.random(num_terms)
    observables = []
    for _ in range(num_terms):
        word = get_rand_pauliword(num_qubits=num_qubits, paulis=paulis, seed=seed)
        obs = [char2qml[p](i) for i, p in enumerate(word)]
        observables.append(qml.prod(*obs))
    ham = sum(c * obs for c, obs in zip(coeffs, observables))
    return qml.simplify(ham)
