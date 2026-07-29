import time
import json

import numpy as np
import scipy as sp
from tqdm import tqdm
import quantlop as ql
from memory_profiler import memory_usage


def scipy_dense(mat, psi):
    return sp.linalg.expm(-1j * mat) @ psi


def scipy_sparse(mat, psi):
    return sp.sparse.linalg.expm_multiply(-1j * mat, psi, traceA=0)


def quantlop_higham(ham, psi):
    return ql.evolve_higham(ham, psi)


def quantlop_krylov(ham, psi):
    return ql.evolve_krylov(ham, psi)


def runtime_and_memory(func, *args, interval=0.0005):
    t1 = time.perf_counter()
    mem_usage, result = memory_usage(
        (func, (args), {}),
        interval=interval,
        retval=True,
        include_children=True,
        multiprocess=True,
        max_iterations=1,
    )
    t2 = time.perf_counter()
    runtime = t2 - t1
    memory = max(mem_usage) - mem_usage[0]
    return runtime, memory, result


num_qubits = range(1, 21)
num_terms = 100

runtime_data = {
    "Scipy dense": {},
    "Scipy sparse": {},
    "Quantlop Higham": {},
    "Quantlop Krylov": {},
}
memory_data = {
    "Scipy dense": {},
    "Scipy sparse": {},
    "Quantlop Higham": {},
    "Quantlop Krylov": {},
}
for nq in tqdm(num_qubits, desc="Run simulation"):
    ham = ql.utils.get_rand_hamiltonian(num_qubits=nq, num_terms=num_terms)
    psi = np.zeros(2**nq, dtype=complex)
    psi[0] = 1

    dense = ham.matrix()
    runtime, memory, result_dense = runtime_and_memory(scipy_dense, dense, psi)
    runtime_data["Scipy dense"][nq] = runtime
    memory_data["Scipy dense"][nq] = memory
    dense = None

    sparse = ham.sparse_matrix()
    runtime, memory, result_sparse = runtime_and_memory(scipy_sparse, sparse, psi)
    runtime_data["Scipy sparse"][nq] = runtime
    memory_data["Scipy sparse"][nq] = memory
    sparse = None

    runtime, memory, result_higham = runtime_and_memory(quantlop_higham, ham, psi)
    runtime_data["Quantlop Higham"][nq] = runtime
    memory_data["Quantlop Higham"][nq] = memory

    runtime, memory, result_krylov = runtime_and_memory(quantlop_krylov, ham, psi)
    runtime_data["Quantlop Krylov"][nq] = runtime
    memory_data["Quantlop Krylov"][nq] = memory

    assert np.allclose(result_dense, result_sparse)
    assert np.allclose(result_dense, result_higham)
    assert np.allclose(result_dense, result_krylov)

    with open("runtime_vs_qubits.json", "w") as file:
        json.dump(runtime_data, file, indent=4)
    with open("memory_vs_qubits.json", "w") as file:
        json.dump(memory_data, file, indent=4)
