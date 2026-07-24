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
is required. :mod:`quantlop` computes this action with the scaling and
truncated-Taylor algorithm of `Al-Mohy and Higham
<https://doi.org/10.1137/100788860>`_. Pauli words are applied directly to
vectors, so neither :math:`H` nor :math:`e^{-itH}` is stored as a dense matrix.


Identity shift
^^^^^^^^^^^^^^

Write the Hamiltonian as

.. math::

   H = cI + \widetilde{H},

where :math:`c` is the sum of the coefficients of all identity Pauli words.
The identity contribution can be separated exactly:

.. math::

   e^{-itH}v = e^{-itc}e^{-it\widetilde{H}}v.

This reduces the norm of the operator that must be approximated. During the
Taylor recurrence, :mod:`quantlop` applies only the non-identity Pauli words
and accumulates the identity contribution as a scalar phase.


Scaling and Taylor degree
^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`A=-it\widetilde{H}` and define the degree-:math:`m` Taylor
polynomial

.. math::

   T_m(X) = \sum_{k=0}^{m}\frac{X^k}{k!}.

Instead of evaluating one high-degree polynomial for :math:`e^A`, the
algorithm uses

.. math::

   e^A v \approx \left(T_m(A/s)\right)^s v,

where :math:`s` is a positive scaling factor. The supported Taylor degrees
have precomputed double-precision backward-error bounds. :mod:`quantlop`
chooses :math:`m` and :math:`s` to minimize the estimated number
:math:`ms` of Hamiltonian-vector products while satisfying those bounds.

For a Pauli decomposition

.. math::

   \widetilde{H} = \sum_k c_k P_k,

each :math:`P_k` has unit matrix norm, giving the inexpensive bound

.. math::

   \lVert A\rVert \leq |t|\sum_k |c_k|.

The parameter selection therefore needs only the compact Pauli
representation; it does not construct or inspect the full matrix.


Matrix-free recurrence
^^^^^^^^^^^^^^^^^^^^^^

For each of the :math:`s` scaled steps, the Taylor action is accumulated using

.. math::

   b_0 = v,\qquad
   b_k = \frac{A}{sk}b_{k-1},\qquad
   f_k = f_{k-1} + b_k,

with :math:`f_0=v`. Every new term requires one direct application of the
Pauli-sum Hamiltonian. Evaluation stops before degree :math:`m` when the next
terms are already below the double-precision termination threshold.

The result of one scaled step becomes the input to the next. Only a small
fixed number of state-sized work vectors is needed, so memory usage remains
linear in the state-vector dimension and independent of the Taylor degree.
