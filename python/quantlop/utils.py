import numpy as np
import quantlop as ql

chars = ("I", "X", "Y", "Z")


def get_rand_statevector(num_qubits, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    real = rng.random(2**num_qubits)
    imag = rng.random(2**num_qubits)
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


def get_rand_pauliword(num_qubits, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    coeff = rng.random()
    string = "".join(rng.choice(chars, size=num_qubits))
    return ql.PauliWord(coeff=coeff, string=string)


def get_rand_hamiltonian(num_qubits, num_terms, rng=None):
    pwords = [get_rand_pauliword(num_qubits, rng=rng) for _ in range(num_terms)]
    return ql.Hamiltonian(pwords=pwords)
