import os
import numpy as np

from ._quantlop import _evolve_higham
from ._quantlop import _evolve_krylov


def evolve_higham(ham, psi, theta=1, num_threads=None):
    r"""Apply Hamiltonian evolution using the Higham exponential-action algorithm.

    .. math::

        |\psi(\theta)\rangle = e^{-i \theta H}|\psi\rangle

    The implementation uses adaptive scaling and a truncated Taylor series.
    It applies Pauli terms directly and never constructs the dense Hamiltonian
    or its exponential.

    See :ref:`Higham method <higham-method>` for algorithm details.

    Parameters
    ----------
    ham : Hamiltonian
        Pauli-sum ``quantlop`` Hamiltonian. The algorithm assumes that the
        operator is Hermitian.
    psi : array_like
        Nonzero one-dimensional input state vector.
    theta : float, optional
        Real parameter in the exponential. The default is 1.
    num_threads : int or "auto" or None, optional
        OpenMP thread selection for Hamiltonian-vector products. ``None``
        selects serial execution, while ``"auto"`` uses the logical CPU count.

    Returns
    -------
    numpy.ndarray
        Evolved dense state vector.

    Examples
    --------
    .. testcode::

        import numpy as np
        import quantlop as ql

        ham = ql.Hamiltonian([ql.PauliWord(1.0, "X")])
        psi = np.array([1.0, 0.0])
        out = ql.evolve_higham(ham, psi, theta=np.pi / 2)
    """
    if num_threads is None:
        num_threads = 1
    elif num_threads == "auto":
        num_threads = os.cpu_count() or 1
    state = np.asarray(psi, dtype=np.complex128, order="C")
    return _evolve_higham(ham, state, theta, num_threads)


def evolve_krylov(ham, psi, theta=1, num_threads=None, dim_krylov=30):
    r"""Apply Hamiltonian evolution using the Lanczos-Krylov subspace algorithm.

    .. math::

        |\psi(\theta)\rangle = e^{-i \theta H}|\psi\rangle

    The implementation projects the Hamiltonian onto a Lanczos basis, evolves
    within that Krylov subspace, and reconstructs the dense state vector.

    See :ref:`Krylov method <krylov-method>` for algorithm details.

    Parameters
    ----------
    ham : Hamiltonian
        Pauli-sum ``quantlop`` Hamiltonian. The algorithm assumes that the
        operator is Hermitian.
    psi : array_like
        Nonzero one-dimensional input state vector.
    theta : float, optional
        Real parameter in the exponential. The default is 1.
    num_threads : int or "auto" or None, optional
        OpenMP thread selection for Hamiltonian-vector products. ``None``
        selects serial execution, while ``"auto"`` uses the logical CPU count.
    dim_krylov : int, optional
        Maximum Krylov-subspace dimension. The default is 30.

    Returns
    -------
    numpy.ndarray
        Evolved dense state vector.

    Examples
    --------
    .. testcode::

        import numpy as np
        import quantlop as ql

        ham = ql.Hamiltonian([ql.PauliWord(1.0, "X")])
        psi = np.array([1.0, 0.0])
        out = ql.evolve_krylov(ham, psi, theta=np.pi / 2)
    """
    if num_threads is None:
        num_threads = 1
    elif num_threads == "auto":
        num_threads = os.cpu_count() or 1
    state = np.asarray(psi, dtype=np.complex128, order="C")
    return _evolve_krylov(ham, state, theta, num_threads, dim_krylov)
