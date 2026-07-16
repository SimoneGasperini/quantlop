import numpy as np
import quantlop as ql

chars = ("I", "X", "Y", "Z")


def get_rand_statevector(nqubits, seed=None):
    rng = np.random.default_rng(seed=seed)
    real = rng.random(2**nqubits)
    imag = rng.random(2**nqubits)
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


def get_rand_pauliword(nqubits, seed=None):
    rng = np.random.default_rng(seed=seed)
    coeff = rng.random() * 2 - 1
    string = "".join(rng.choice(chars, size=nqubits))
    return ql.PauliWord(coeff=coeff, string=string)


def get_rand_hamiltonian(nqubits, num_terms, seed=None):
    pauli_words = [get_rand_pauliword(nqubits, seed=seed) for _ in range(num_terms)]
    return ql.Hamiltonian(pauli_words=pauli_words)
