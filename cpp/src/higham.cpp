#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <quantlop/evolution.hpp>

#include "utils.cpp"

static constexpr Size max_taylor_degree = 55;

static double inf_norm(const Complex *values, Size dimension)
{
    double max_norm = 0.0;
    for (Index index = 0; index < dimension; ++index)
    {
        max_norm = std::max(max_norm, std::norm(values[index]));
    }
    return std::sqrt(max_norm);
}

static std::pair<Size, Size> select_taylor_degree_and_scaling(double norm_bound, double rtol)
{
    Size best_degree = 1;
    Size best_scaling = 1;
    double best_cost = std::numeric_limits<double>::infinity();
    const double log_norm = std::log(norm_bound);
    const double log_target = std::log(std::log1p(0.5 * rtol));

    for (Size degree = 1; degree <= max_taylor_degree; ++degree)
    {
        const double log_error_at_one =
            static_cast<double>(degree + 1) * log_norm - std::lgamma(degree + 2);
        const Size scaling = minimum_scaling(log_error_at_one, degree, log_target);
        const double cost = static_cast<double>(degree) * static_cast<double>(scaling);
        if (cost < best_cost)
        {
            best_degree = degree;
            best_scaling = scaling;
            best_cost = cost;
        }
    }

    return {best_degree, best_scaling};
}

std::unique_ptr<Complex[]> evolve_higham(
    const Hamiltonian &ham,
    const Complex *psi,
    double theta,
    double rtol,
    int num_threads)
{
    const Size dimension = ham.dimension();
    const Complex operator_scale(0.0, -theta);
    const Complex mu = operator_scale * ham.identity_coeff();
    const double norm_bound = std::abs(theta) * ham.residual_lcu_norm();
    std::unique_ptr<Complex[]> state = std::make_unique<Complex[]>(dimension);

    if (norm_bound == 0.0)
    {
        scale_copy(psi, state.get(), dimension, std::exp(mu));
        return state;
    }

    const auto [degree, scaling] = select_taylor_degree_and_scaling(norm_bound, rtol);
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

        scale_in_place(state.get(), dimension, phase_per_step);
    }

    return state;
}
