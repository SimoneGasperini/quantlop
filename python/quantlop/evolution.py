import os
import math
from numbers import Integral, Real

from ._quantlop import _evolve_higham
from ._quantlop import _evolve_krylov

DEFAULT_RTOL = 1e-9
DEFAULT_NTHR = os.cpu_count() - 1


def _validate_num_threads(num_threads):
    error_message = "num_threads must be a non-zero positive integer number"
    if not isinstance(num_threads, Integral):
        raise ValueError(error_message)
    if num_threads <= 0:
        raise ValueError(error_message)
    return int(num_threads)


def _validate_theta(theta):
    error_message = "theta must be a finite real floating point number"
    if not isinstance(theta, Real):
        raise ValueError(error_message)
    if not math.isfinite(theta):
        raise ValueError(error_message)
    return float(theta)


def evolve_higham(ham, psi, theta=1.0, rtol=DEFAULT_RTOL, num_threads=DEFAULT_NTHR):
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
        Finite real floating point parameter in the exponential. The default
        is 1.0.
    rtol : float, optional
        Relative accuracy target used to select the approximation. Smaller
        values generally require more computation. The default is ``1e-9``.
    num_threads : int, optional
        OpenMP thread selection for Hamiltonian-vector products. The default is
        one fewer than the logical CPU count.

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
    theta = _validate_theta(theta)
    num_threads = _validate_num_threads(num_threads)
    return _evolve_higham(ham, psi, theta, rtol, num_threads)


def evolve_krylov(ham, psi, theta=1.0, rtol=DEFAULT_RTOL, num_threads=DEFAULT_NTHR):
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
        Finite real floating point parameter in the exponential. The default
        is 1.0.
    rtol : float, optional
        Relative accuracy target used to select the approximation. Smaller
        values generally require more computation. The default is ``1e-9``.
    num_threads : int, optional
        OpenMP thread selection for Hamiltonian-vector products. The default is
        one fewer than the logical CPU count.

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
    theta = _validate_theta(theta)
    num_threads = _validate_num_threads(num_threads)
    return _evolve_krylov(ham, psi, theta, rtol, num_threads)
