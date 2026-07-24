#pragma once

#include <quantlop/hamiltonian.hpp>
#include <quantlop/types.hpp>

Complex *evolve(const Hamiltonian &hamiltonian, const Complex *state, Complex theta, int num_threads);
