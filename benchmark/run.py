import time
import json
import numpy as np
import scipy as sp
import pennylane as qml
from tqdm import trange
from memory_profiler import memory_usage
from quantlop import Hamiltonian, evolve
from quantlop.utils import get_rand_hamiltonian


def sp_simulation(nq, op, psi):
    mat = op.matrix(range(nq))
    return sp.linalg.expm(-1j * mat) @ psi


def pl_simulation(op):
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.evolve(op)
        return qml.state()

    return circuit()


def qlo_simulation(nq, op, psi):
    ham = Hamiltonian.from_pennylane(op, nqubits=nq)
    return evolve(ham, psi)


def runtime_and_memory(func, *args, reps, interval=0.0005):
    runtime = []
    memory = []
    for _ in trange(reps, ncols=80):
        t1 = time.perf_counter()
        mem, result = memory_usage(
            (func, (args), {}),
            interval=interval,
            retval=True,
            max_iterations=1,
        )
        t2 = time.perf_counter()
        runtime.append(t2 - t1)
        memory.append(max(mem))
    return runtime, memory, result


def run_benchmark(num_qubits, time_fname, mem_fname, reps):
    runtime = {"scipy": {}, "pennylane": {}, "quantlop": {}}
    memory = {"scipy": {}, "pennylane": {}, "quantlop": {}}
    for nq in num_qubits:
        print(f"\nRunning simulation for {nq} qubits:")
        psi = np.zeros(2**nq, dtype=complex)
        psi[0] = 1
        op = get_rand_hamiltonian(nqubits=nq, num_terms=5 * nq)
        time, mem, res1 = runtime_and_memory(sp_simulation, nq, op, psi, reps=reps)
        runtime["scipy"][nq] = time
        memory["scipy"][nq] = mem
        time, mem, res2 = runtime_and_memory(pl_simulation, op, reps=reps)
        runtime["pennylane"][nq] = time
        memory["pennylane"][nq] = mem
        time, mem, res3 = runtime_and_memory(qlo_simulation, nq, op, psi, reps=reps)
        runtime["quantlop"][nq] = time
        memory["quantlop"][nq] = mem
        assert np.allclose(res1, res2)
        assert np.allclose(res2, res3)
        with open(time_fname, "w") as file:
            json.dump(runtime, file, indent=4)
        with open(mem_fname, "w") as file:
            json.dump(memory, file, indent=4)


if __name__ == "__main__":
    run_benchmark(
        num_qubits=range(1, 15),
        time_fname="runtime.json",  # sec
        mem_fname="memory.json",  # MB
        reps=7,
    )
