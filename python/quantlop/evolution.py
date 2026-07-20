"""Matrix-free Hamiltonian evolution."""

import os
from numbers import Real

import numpy as np

from ._quantlop import _evolve


def evolve(ham, psi, theta=1, num_threads=None):
    r"""Apply a Hamiltonian exponential to a state vector.

    This function computes the Krylov approximation

    .. math::

        |\psi(\theta)\rangle = e^{-i \theta H}|\psi\rangle.

    The implementation applies Pauli terms directly to vectors and never
    constructs the dense Hamiltonian or its exponential.

    Parameters
    ----------
    ham : Hamiltonian
        Pauli-sum Hamiltonian. The Lanczos-based algorithm assumes that the
        represented matrix is Hermitian.
    psi : array_like
        Nonzero one-dimensional input state with exactly
        ``2**ham.num_qubits`` amplitudes. Values are converted to a
        C-contiguous :class:`numpy.complex128` array. The input is not modified
        by the evolution.
    theta : float, optional
        Real scalar parameter in the exponential. The default is ``1``.
    num_threads : int or {"auto"} or None, optional
        OpenMP thread selection for Hamiltonian-vector products. ``None``
        selects the serial implementation. A positive integer requests that
        many threads. ``"auto"`` requests the logical CPU count reported by
        the operating system. The default is ``None``.

    Returns
    -------
    numpy.ndarray
        Newly allocated, C-contiguous state vector with dtype
        :class:`numpy.complex128` and the same one-dimensional shape as the
        converted input.

    Raises
    ------
    TypeError
        If ``theta`` is not a real scalar.

    Notes
    -----
    The Krylov subspace dimension is capped internally, so the result is a
    numerical approximation to the matrix-exponential action. For a Hermitian
    Hamiltonian and real ``theta``, the exact operation is unitary and
    preserves the state norm up to numerical error.

    ``num_threads`` only affects the native Hamiltonian-vector products; it
    does not change the mathematical result. Whether multiple threads improve
    runtime depends on the system size, number of terms, and OpenMP runtime.

    Examples
    --------
    Evolve ``|0>`` under the Pauli-X Hamiltonian for half a rotation:

    .. testcode::

        import numpy as np
        import quantlop as ql

        ham = ql.Hamiltonian([ql.PauliWord(1.0, "X")])
        psi = np.array([1.0, 0.0])
        out = ql.evolve(ham, psi, theta=np.pi / 2)

    Use all logical CPUs reported by the operating system:

    .. testcode::

        out = ql.evolve(ham, psi, theta=0.1, num_threads="auto")

    See Also
    --------
    Hamiltonian.matrix : Construct a dense matrix for small-system checks.
    """
    if not isinstance(theta, Real):
        raise TypeError("theta must be a real scalar")
    if num_threads is None:
        num_threads = 0
    if num_threads == "auto":
        num_threads = os.cpu_count()
    state = np.asarray(psi, dtype=np.complex128, order="C")
    return _evolve(ham, state, theta, num_threads)
