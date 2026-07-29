import pytest
import numpy as np

import pennylane as qp
import qiskit as qk
import quantlop as ql


def test_positional_args_init():
    pw0 = ql.PauliWord(-1.27j, "YIIX")
    pw1 = ql.PauliWord(0.92, "IIZZ")
    ham = ql.Hamiltonian([pw0, pw1])
    assert ham.num_qubits == 4
    assert ham.num_terms == 2

    assert ham._get_pwords()[0]._num_qubits() == 4
    assert ham._get_pwords()[0]._get_coeff() == -1.27j
    assert ham._get_pwords()[0]._get_string() == "YIIX"

    assert ham._get_pwords()[1]._num_qubits() == 4
    assert ham._get_pwords()[1]._get_coeff() == 0.92
    assert ham._get_pwords()[1]._get_string() == "IIZZ"


def test_keyword_args_init():
    pw0 = ql.PauliWord(-1.27j, "YIIX")
    pw1 = ql.PauliWord(0.92, "IIZZ")
    ham = ql.Hamiltonian(pwords=[pw0, pw1])
    assert ham.num_qubits == 4
    assert ham.num_terms == 2

    assert ham._get_pwords()[0]._num_qubits() == 4
    assert ham._get_pwords()[0]._get_coeff() == -1.27j
    assert ham._get_pwords()[0]._get_string() == "YIIX"

    assert ham._get_pwords()[1]._num_qubits() == 4
    assert ham._get_pwords()[1]._get_coeff() == 0.92
    assert ham._get_pwords()[1]._get_string() == "IIZZ"


def test_num_qubits_read_only():
    pw0 = ql.PauliWord(-1.27j, "YIIX")
    pw1 = ql.PauliWord(0.92, "IIZZ")
    ham = ql.Hamiltonian(pwords=[pw0, pw1])
    with pytest.raises(
        AttributeError, match="property 'num_qubits' of 'Hamiltonian' object has no setter"
    ):
        ham.num_qubits = 6


def test_num_terms_read_only():
    pw0 = ql.PauliWord(-1.27j, "YIIX")
    pw1 = ql.PauliWord(0.92, "IIZZ")
    ham = ql.Hamiltonian(pwords=[pw0, pw1])
    with pytest.raises(
        AttributeError, match="property 'num_terms' of 'Hamiltonian' object has no setter"
    ):
        ham.num_terms = 3


def test_matrix_from_pennylane():
    op = qp.Hamiltonian(
        coeffs=[0.9, -0.6, 1.1],
        observables=[qp.X(1) @ qp.X(2), qp.Z(0) @ qp.Y(2), qp.Y(1) @ qp.Z(3)],
    )
    num_qubits = 5
    ham = ql.Hamiltonian.from_pennylane(op, num_qubits=num_qubits)
    assert np.allclose(op.matrix(range(num_qubits)), ham.matrix())


def test_sparse_matrix_from_pennylane():
    op = qp.Hamiltonian(
        coeffs=[0.9, -0.6, 1.1],
        observables=[qp.X(1) @ qp.X(2), qp.Z(0) @ qp.Y(2), qp.Y(1) @ qp.Z(3)],
    )
    num_qubits = 5
    ham = ql.Hamiltonian.from_pennylane(op, num_qubits=num_qubits)
    pl_sparse = op.sparse_matrix(wire_order=range(num_qubits))
    ql_sparse = ham.sparse_matrix()
    assert np.allclose(pl_sparse.toarray(), ql_sparse.toarray())


def test_matrix_from_qiskit():
    op = qk.quantum_info.SparsePauliOp(
        data=["YIZXI", "XXIIZ", "IZYXY"],
        coeffs=[-0.7, 1.2, 0.3],
    )
    ham = ql.Hamiltonian.from_qiskit(op)
    assert np.allclose(op.to_matrix(), ham.matrix())


def test_sparse_matrix_from_qiskit():
    op = qk.quantum_info.SparsePauliOp(
        data=["YIZXI", "XXIIZ", "IZYXY"],
        coeffs=[-0.7, 1.2, 0.3],
    )
    ham = ql.Hamiltonian.from_qiskit(op)
    qk_sparse = op.to_matrix(sparse=True)
    ql_sparse = ham.sparse_matrix()
    assert np.allclose(qk_sparse.toarray(), ql_sparse.toarray())
