Algorithms
==========

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
matrix quickly becomes impractical because its computational and
storage costs grow exponentially with the number of qubits.

Viewed as a **linear operator**, the Hamiltonian maps the state vector space into itself:

.. math::

   H:\mathbb{C}^{2^n}\rightarrow\mathbb{C}^{2^n},
   \qquad v\mapsto Hv.

Time evolution applies the exponential of this operator to an input state,
producing the new state :math:`e^{-itH}v`. Exponential-action algorithms
compute this vector from repeated evaluations of the map :math:`v\mapsto Hv`,
without constructing any explicit representation of the matrices :math:`H` or
:math:`e^{-itH}`. :mod:`quantlop` provides efficient implementations of two
such ideas:

.. |higham-method-link| replace:: **Higham method**
.. _higham-method-link: #higham-method

.. |krylov-method-link| replace:: **Krylov method**
.. _krylov-method-link: #krylov-method

* |higham-method-link|_ uses scaling and truncated Taylor expansions to
  approximate the exponential action through repeated Hamiltonian-vector
  products.
* |krylov-method-link|_ projects the Hamiltonian onto a low-dimensional Krylov
  subspace, evolves the state there, and maps the result back to the full
  state space.


.. _higham-method:

Higham method
-------------

The algorithm of Al-Mohy and Higham [AH11]_ approximates
:math:`e^{-itH}v` by dividing the evolution into scaled steps and evaluating
each step with a finite Taylor polynomial. :mod:`quantlop` selects the scaling
and polynomial degree from the requested relative tolerance, then generates
the Taylor terms through direct applications of the Pauli-sum Hamiltonian.


Scaling and Taylor degree
^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`A=-itH` and define the degree-:math:`m` Taylor polynomial

.. math::

   T_m(X) = \sum_{k=0}^{m}\frac{X^k}{k!}.

Instead of evaluating one high-degree polynomial for :math:`e^A`, the algorithm uses

.. math::

   e^A v \approx \left(T_m(A/s)\right)^s v,

where :math:`s` is a positive scaling factor. Taylor approximation is most
effective when the norm of its argument is moderate. Dividing :math:`A` by
:math:`s` reduces that norm, and applying the same polynomial :math:`s` times
recovers the evolution over the full time interval.

The two parameters express a cost trade-off. Increasing :math:`m` makes each
step more accurate but requires more Hamiltonian-vector products per step;
increasing :math:`s` makes the individual steps easier to approximate but
requires more steps. For a Pauli decomposition

.. math::

   H = \sum_k c_k P_k,

each :math:`P_k` has unit matrix norm, giving the inexpensive bound

.. math::

   \lVert A\rVert \leq |t|\sum_k |c_k|.

This bound can be computed directly from the Pauli coefficients. Parameter
selection therefore needs only the compact Pauli representation and does not
construct or inspect the full matrix.

Using this bound and the requested ``rtol``, :mod:`quantlop` compares suitable
Taylor degrees and scaling factors. It chooses a combination expected to meet
the accuracy target with the fewest Hamiltonian-vector products.


Matrix-free recurrence
^^^^^^^^^^^^^^^^^^^^^^

For each of the :math:`s` scaled steps, the Taylor action is accumulated using

.. math::

   b_0 = v,\qquad
   b_k = \frac{A}{sk}b_{k-1},\qquad
   f_k = f_{k-1} + b_k,

with :math:`f_0=v`. Here :math:`b_k` is the next term of the Taylor series and
:math:`f_k` is the running polynomial approximation.
Within each step, :math:`v` denotes its current input state. To compute
:math:`b_k`, :mod:`quantlop` applies :math:`H` to :math:`b_{k-1}` and
multiplies the result by :math:`-it/(sk)`. It never forms the matrix powers
:math:`H^k`; only the vectors needed for the current recurrence step are kept
in memory. Although :math:`m` sets the maximum degree, evaluation stops
earlier when the next terms are already below the termination threshold.

The result of one scaled step becomes the input to the next. After :math:`s`
steps this produces the full evolution. Only a small fixed number of
state-sized work vectors is needed, so memory usage remains linear in the state
vector dimension and independent of the Taylor degree.


.. _krylov-method:

Krylov method
-------------

The matrix-free Lanczos-Krylov method [Saad92]_ constructs a low-dimensional
subspace from repeated applications of the Hamiltonian to the input state.
It projects the evolution problem onto this subspace, evaluates the
exponential of the resulting small matrix, and maps the evolved state back to
the full state vector space.

Krylov subspace
^^^^^^^^^^^^^^^

The powers of :math:`H` appearing in the Taylor expansion above also motivate
the Krylov method. For :math:`A=-itH`, the exponential action is determined by
the sequence :math:`v,Hv,H^2v,\ldots`. Its first :math:`m` vectors define the
Krylov subspace

.. math::

   \mathcal{K}_m(H,v)
   = \operatorname{span}\left\{v,\; Hv,\; \ldots,\; H^{m-1}v\right\}.

Since the Hamiltonian is Hermitian, an orthonormal basis for this subspace can
be constructed efficiently using the Lanczos recurrence. For a fixed
:math:`m`, this is generally easier over shorter evolution times or a narrower
relevant spectral interval. :mod:`quantlop` handles longer times or broader
spectra by increasing the dimension or using multiple time steps.

Rather than evaluating or truncating the power series term by term, the Krylov
method projects :math:`H` onto the orthonormal basis of the subspace, evaluates
the exponential of the projected matrix, and maps the result back to the full
state space. Increasing :math:`m` generally improves the numerical
approximation at the cost of additional matrix-vector products.


Lanczos iteration
^^^^^^^^^^^^^^^^^

For a Hermitian matrix, the Lanczos algorithm constructs an orthonormal Krylov
basis through a three-term recurrence involving only the current and previous
basis vectors. Starting from a nonzero input :math:`v`, initialize

.. math::

   \beta_1 = 0, \qquad
   q_1 = \frac{v}{\lVert v\rVert_2}.

At iteration :math:`j`, the algorithm applies the Hamiltonian to
:math:`q_j` and removes its components along the current and previous basis
vectors:

.. math::

   r_j = Hq_j - \beta_j q_{j-1} - \alpha_j q_j,
   \qquad
   \alpha_j = q_j^\dagger Hq_j.

In exact arithmetic, Hermiticity ensures that :math:`r_j` is orthogonal to
every Lanczos vector constructed so far. Its norm and normalized direction
provide the next recurrence coefficient and basis vector:

.. math::

   \beta_{j+1} = \lVert r_j\rVert_2,\qquad
   q_{j+1} = \frac{r_j}{\beta_{j+1}}.

Each iteration requires one Hamiltonian-vector product. The coefficients
:math:`\alpha_j` and :math:`\beta_{j+1}` record how :math:`H` acts within the
growing basis and later become the entries of the projected tridiagonal
matrix. Repeating this process produces the basis
:math:`q_1,\ldots,q_m`.
When :math:`\beta_{j+1}=0`, the current Krylov subspace is invariant under
:math:`H`: applying the Hamiltonian cannot generate a new direction, and the
recurrence terminates exactly. In floating-point arithmetic, :mod:`quantlop`
treats a sufficiently small :math:`\beta_{j+1}` as this breakdown condition
and otherwise continues until reaching the automatically selected Krylov
dimension.


Projected evolution
^^^^^^^^^^^^^^^^^^^

After :math:`m` Lanczos steps, the basis vectors form the matrix

.. math::

   Q_m = \begin{bmatrix}q_1 & \cdots & q_m\end{bmatrix}.

For a full state-space dimension :math:`N=2^n`, the matrix :math:`Q_m` has
shape :math:`N\times m`. Projecting the Hamiltonian onto this basis gives

.. math::

   T_m = Q_m^\dagger H Q_m.

In exact arithmetic, the Lanczos recurrence makes :math:`T_m` real symmetric
and tridiagonal, with the :math:`\alpha_j` coefficients on the diagonal and
the :math:`\beta_j` coefficients on the adjacent diagonals:

.. math::

   T_m =
   \begin{bmatrix}
   \alpha_1 & \beta_2 &          &           \\
   \beta_2  & \alpha_2 & \ddots  &           \\
            & \ddots   & \ddots  & \beta_m   \\
            &          & \beta_m & \alpha_m
   \end{bmatrix}.

Since :math:`q_1=v/\lVert v\rVert_2`, the input vector is represented in the
Krylov basis by :math:`\lVert v\rVert_2 e_1`, where
:math:`e_1=(1,0,\ldots,0)^T` is the first coordinate vector. The
Lanczos-Krylov approximation is therefore

.. math::

   e^{-itH}v \approx \lVert v\rVert_2 Q_m e^{-itT_m}e_1.

The small exponential :math:`e^{-itT_m}` evolves the Krylov coefficients, and
:math:`Q_m` maps the result back to the full state space. Because
:math:`m\ll N`, the method evaluates the exponential of a much smaller
tridiagonal matrix rather than that of the full Hamiltonian.

The matrix :math:`Q_m` is useful for expressing the method, but
:mod:`quantlop` does not store all :math:`m` basis vectors simultaneously.
It first runs the Lanczos recurrence to construct :math:`T_m`, computes the
small exponential, and then repeats the recurrence to reconstruct the evolved
state. This second pass uses additional Hamiltonian-vector products in
exchange for keeping only a fixed number of state-sized work vectors. Memory
therefore remains linear in :math:`N`, apart from the small
:math:`m\times m` projected matrix.


References
----------

.. [AH11] Awad H. Al-Mohy and Nicholas J. Higham, "Computing the Action of the
   Matrix Exponential, with an Application to Exponential Integrators,"
   *SIAM Journal on Scientific Computing*, 33(2), 488--511, 2011.
   https://doi.org/10.1137/100788860

.. [Saad92] Yousef Saad, "Analysis of Some Krylov Subspace Approximations to
   the Matrix Exponential Operator," *SIAM Journal on Numerical Analysis*,
   29(1), 209--228, 1992. https://doi.org/10.1137/0729014
