"""Matrix-free Hamiltonian time evolution."""

import os
import numpy as np

from ._quantlop import _evolve


def evolve(ham, psi, coeff=1, num_threads=None):
    r"""Apply a Hamiltonian exponential to a state vector.

    This function computes the Krylov approximation

    .. math::

        |\psi'\rangle = \exp(-i\,c\,H)|\psi\rangle,

    where ``H`` is ``ham`` and ``c`` is ``coeff``. The implementation applies
    Pauli terms directly to vectors and never constructs the dense Hamiltonian
    or its exponential.

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
    coeff : complex, optional
        Scalar ``c`` in the exponential. For real-time evolution this is the
        elapsed time (in units where :math:`\hbar=1`). ``coeff=0`` applies the
        identity. A complex value is accepted, although non-real values make
        the exponential non-unitary even when ``ham`` is Hermitian. The
        default is ``1``.
    num_threads : int or {"auto"} or None, optional
        OpenMP thread selection for Hamiltonian-vector products. ``None``
        selects the serial implementation. A positive integer requests that
        many threads. ``"auto"`` requests the logical CPU count reported by
        :func:`os.cpu_count`. The default is ``None``.

    Returns
    -------
    numpy.ndarray
        Newly allocated, C-contiguous state vector with dtype
        :class:`numpy.complex128` and the same one-dimensional shape as the
        converted input.

    Notes
    -----
    The Krylov subspace dimension is capped internally, so the result is a
    numerical approximation to the matrix-exponential action. For a Hermitian
    Hamiltonian and real ``coeff``, the exact operation is unitary and
    preserves the state norm up to numerical error.

    ``num_threads`` only affects the native Hamiltonian-vector products; it
    does not change the mathematical result. Whether multiple threads improve
    runtime depends on the system size, number of terms, and OpenMP runtime.

    Examples
    --------
    Evolve ``|0>`` under the Pauli-X Hamiltonian for half a rotation:

    >>> import numpy as np
    >>> import quantlop as ql
    >>> ham = ql.Hamiltonian([ql.PauliWord(1.0, "X")])
    >>> psi = np.array([1.0, 0.0])
    >>> out = ql.evolve(ham, psi, coeff=np.pi / 2)
    >>> np.allclose(out, [0.0, -1.0j])
    True

    Use all logical CPUs reported by the operating system:

    >>> out = ql.evolve(ham, psi, coeff=0.1, num_threads="auto")

    See Also
    --------
    Hamiltonian.matrix : Construct a dense matrix for small-system checks.
    """
    if num_threads is None:
        num_threads = 0
    if num_threads == "auto":
        num_threads = os.cpu_count()
    state = np.asarray(psi, dtype=np.complex128, order="C")
    return _evolve(ham, state, coeff, num_threads)
