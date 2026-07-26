#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <quantlop/evolution.hpp>

// see https://epubs.siam.org/doi/10.1137/100788860
static constexpr std::array<std::pair<Size, double>, 35> taylor_bounds = {{
    {1, 2.29e-16}, {2, 2.58e-8},  {3, 1.39e-5},  {4, 3.40e-4},  {5, 2.40e-3},  {6, 9.07e-3},
    {7, 2.38e-2},  {8, 5.00e-2},  {9, 8.96e-2},  {10, 1.44e-1}, {11, 2.14e-1}, {12, 3.00e-1},
    {13, 4.00e-1}, {14, 5.14e-1}, {15, 6.41e-1}, {16, 7.81e-1}, {17, 9.31e-1}, {18, 1.09},
    {19, 1.26},    {20, 1.44},    {21, 1.62},    {22, 1.82},    {23, 2.01},    {24, 2.22},
    {25, 2.43},    {26, 2.64},    {27, 2.86},    {28, 3.08},    {29, 3.31},    {30, 3.54},
    {35, 4.7},     {40, 6.0},     {45, 7.2},     {50, 8.5},     {55, 9.9},
}};

static double inf_norm(const Complex *values, Size dimension)
{
    double max_norm = 0.0;
    for (Index index = 0; index < dimension; ++index)
    {
        max_norm = std::max(max_norm, std::norm(values[index]));
    }
    return std::sqrt(max_norm);
}

static std::pair<Size, Size> select_taylor_degree_and_scaling(double norm_bound)
{
    Size best_degree = 0;
    Size best_scaling = 1;
    double best_cost = std::numeric_limits<double>::infinity();

    for (const auto &[degree, bound] : taylor_bounds)
    {
        const Size scaling = std::max<Size>(1, static_cast<Size>(std::ceil(norm_bound / bound)));
        const double cost = static_cast<double>(degree * scaling);
        if (cost < best_cost)
        {
            best_degree = degree;
            best_scaling = scaling;
            best_cost = cost;
        }
    }

    return {best_degree, best_scaling};
}

Complex *evolve_higham(const Hamiltonian &ham, const Complex *psi, Complex theta, int num_threads)
{
    const Size dimension = ham.dimension();
    const Complex operator_scale = Complex(0.0, -1.0) * theta;
    const Complex mu = operator_scale * ham.identity_coeff();
    const double norm_bound = std::abs(operator_scale) * ham.residual_lcu_norm();
    std::unique_ptr<Complex[]> state = std::make_unique<Complex[]>(dimension);

    if (norm_bound == 0.0)
    {
        const Complex phase = std::exp(mu);
        for (Index row = 0; row < dimension; ++row)
        {
            state[row] = phase * psi[row];
        }
        return state.release();
    }

    const auto [degree, scaling] = select_taylor_degree_and_scaling(norm_bound);
    const double tolerance = 0.5 * std::numeric_limits<double>::epsilon();
    const Complex phase_per_step = std::exp(mu / static_cast<double>(scaling));
    const double inverse_scaling = 1.0 / static_cast<double>(scaling);

    std::copy_n(psi, dimension, state.get());
    std::vector<Complex> term(dimension);
    std::vector<Complex> product(dimension);

    for (Size step = 0; step < scaling; ++step)
    {
        std::copy_n(state.get(), dimension, term.data());
        double previous_norm = inf_norm(term.data(), dimension);

        for (Size order = 1; order <= degree; ++order)
        {
            ham.residual_matvec_into(term.data(), product.data(), num_threads);
            const Complex recurrence_factor =
                operator_scale * (inverse_scaling / static_cast<double>(order));
            double term_norm = 0.0;
            double state_norm = 0.0;
            for (Index row = 0; row < dimension; ++row)
            {
                term[row] = recurrence_factor * product[row];
                state[row] += term[row];
                term_norm = std::max(term_norm, std::norm(term[row]));
                state_norm = std::max(state_norm, std::norm(state[row]));
            }

            term_norm = std::sqrt(term_norm);
            if (previous_norm + term_norm <= tolerance * std::sqrt(state_norm))
            {
                break;
            }
            previous_norm = term_norm;
        }

        for (Index row = 0; row < dimension; ++row)
        {
            state[row] *= phase_per_step;
        }
    }

    return state.release();
}
