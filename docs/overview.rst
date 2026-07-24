Overview
========

In quantum mechanics, the Hamiltonian :math:`H` describes the energy of a
system and determines how its quantum state :math:`|\psi\rangle` evolves over
time. For a time-independent Hamiltonian, the Schrödinger equation

.. math::

   i\frac{d}{dt}|\psi(t)\rangle = H|\psi(t)\rangle

has the solution

.. math::

   |\psi(t)\rangle = e^{-itH}|\psi(0)\rangle.

The matrix exponential :math:`e^{-itH}` is therefore the time-evolution
operator. The resulting state can be used to estimate expectation values or
sampled in the computational basis to generate output bitstrings.

Hamiltonian evolution is central to quantum simulation, including the study of
interacting particles and spin models. For an :math:`n`-qubit system, the
Hamiltonian has dimension :math:`2^n\times 2^n`, and constructing the full
matrix exponential quickly becomes impractical because its computational and
storage costs grow exponentially with the number of qubits.

For time evolution, only the action of the exponential on the supplied state
is typically required. :mod:`quantlop` computes this action with a matrix-free
Lanczos-Krylov method, which applies the Hamiltonian directly to vectors and
never forms a dense representation of :math:`H` or :math:`e^{-itH}`. The
method identifies a low-dimensional subspace that captures the relevant action
of the Hamiltonian and solves the evolution problem within that subspace.


Krylov methods
^^^^^^^^^^^^^^

The connection between time evolution and Krylov methods follows from the
exact power-series expansion of the matrix exponential. For a square matrix
:math:`A` and an input vector :math:`v`,

.. math::

   e^A v = \sum_{k=0}^{\infty}\frac{A^k v}{k!} = v + A v + \frac{A^2 v}{2!} + \frac{A^3 v}{3!} + \cdots.

The action of the exponential is therefore determined by the sequence
:math:`v, A v, A^2 v, \ldots`, whose first :math:`m` vectors define the Krylov
subspace

.. math::

   \mathcal{K}_m(A,v) = \operatorname{span}\left\{v,\; A v,\; \ldots,\; A^{m-1} v\right\}.

Since the Hamiltonian is Hermitian, an orthonormal basis for this subspace can
be constructed efficiently using the Lanczos recurrence.

Rather than evaluating or truncating the power series term by term, the Krylov
method projects :math:`H` onto an orthonormal basis of the subspace,
evaluates the exponential of the projected matrix, and maps the result back to
the full state space. Increasing :math:`m` generally improves the numerical
approximation at the cost of additional matrix-vector products.


Lanczos iteration
^^^^^^^^^^^^^^^^^

For a Hermitian matrix, the Lanczos algorithm constructs an orthonormal Krylov
basis through a three-term recurrence involving only the current and previous
basis vectors. Starting from a nonzero input :math:`v`, initialize

.. math::

   q_0 = 0, \qquad \beta_1 = 0, \qquad q_1 = \frac{v}{\lVert v\rVert_2}.

The first iteration computes :math:`H q_1` as a matrix-vector product and
subtracts its projection along :math:`q_1` to obtain the residual

.. math::

   r_1 = H q_1 - \alpha_1 q_1,

where :math:`\alpha_1 = q_1^\dagger H q_1`. The normalized residual then gives
the next basis vector :math:`q_2`.

In general, the residual at iteration :math:`j` is

.. math::

   r_j = H q_j - \beta_j q_{j-1} - \alpha_j q_j,

where :math:`\alpha_j = q_j^\dagger H q_j` is the coefficient of the projection
of :math:`H q_j` onto :math:`q_j`. In exact arithmetic, Hermiticity ensures
that :math:`r_j` is orthogonal to every Lanczos vector constructed so far. Its
norm defines the next :math:`\beta_{j+1}` coefficient, and the normalized
residual provides the next basis vector

.. math::

   \beta_{j+1} = \lVert r_j\rVert_2, \qquad q_{j+1} = \frac{r_j}{\beta_{j+1}}.

Repeating this process produces the basis :math:`q_1,\ldots,q_m`.
When :math:`\beta_{j+1}=0`, the current Krylov subspace is invariant under
:math:`H` and the recurrence terminates exactly. In floating-point arithmetic,
:mod:`quantlop` uses a numerical tolerance and enforces a maximum Krylov
dimension.


Projected evolution
^^^^^^^^^^^^^^^^^^^

After :math:`m` Lanczos steps, the basis vectors form the matrix

.. math::

   Q_m = \begin{bmatrix}q_1 & \cdots & q_m\end{bmatrix}.

For a full state-space dimension :math:`N=2^n`, the matrix :math:`Q_m` has shape
:math:`N\times m`. Projecting the Hamiltonian onto this basis gives

.. math::

   T_m = Q_m^\dagger H Q_m.

In exact arithmetic, the Lanczos recurrence makes :math:`T_m` real symmetric
and tridiagonal, with the :math:`\alpha_j` coefficients on the diagonal and
the :math:`\beta_j` coefficients on the adjacent diagonals:

.. math::

   T_m =
   \begin{bmatrix}
   \alpha_1 & \beta_2 &         &            \\
   \beta_2  & \alpha_2 & \ddots &            \\
            & \ddots   & \ddots & \beta_m    \\
            &          & \beta_m & \alpha_m
   \end{bmatrix}.

Since :math:`q_1=v/\lVert v\rVert_2`, the input vector is represented in the
Krylov basis by :math:`\lVert v\rVert_2 e_1`, where
:math:`e_1=(1,0,\ldots,0)^T` is the first coordinate vector. The
Lanczos-Krylov approximation is therefore

.. math::

   e^{-itH} v \approx \lVert v\rVert_2 Q_m e^{-itT_m} e_1.

The small exponential :math:`e^{-itT_m}` evolves the Krylov coefficients, and
:math:`Q_m` maps the result back to the full state space. Because
:math:`m \ll N`, the method evaluates the exponential of a much smaller
tridiagonal matrix rather than that of the full Hamiltonian.
