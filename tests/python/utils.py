import numpy as np
import quantlop as ql

chars = ("I", "X", "Y", "Z")


def get_rand_statevector(num_qubits, seed=None):
    rng = np.random.default_rng(seed=seed)
    real = rng.random(2**num_qubits)
    imag = rng.random(2**num_qubits)
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


def get_rand_pauliword(num_qubits, seed=None):
    rng = np.random.default_rng(seed=seed)
    coeff = rng.random() * 2 - 1
    string = "".join(rng.choice(chars, size=num_qubits))
    return ql.PauliWord(coeff=coeff, string=string)


def get_rand_hamiltonian(num_qubits, num_terms, seed=None):
    pauli_words = [get_rand_pauliword(num_qubits, seed=seed) for _ in range(num_terms)]
    return ql.Hamiltonian(pauli_words=pauli_words)
