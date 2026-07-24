#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <quantlop/evolution.hpp>

// see https://epubs.siam.org/doi/10.1137/100788860
static constexpr std::array<std::pair<Size, double>, 35> taylor_bounds = {{
    {1, 2.29e-16}, {2, 2.58e-8},  {3, 1.39e-5},  {4, 3.40e-4},  {5, 2.40e-3},  {6, 9.07e-3},  {7, 2.38e-2},
    {8, 5.00e-2},  {9, 8.96e-2},  {10, 1.44e-1}, {11, 2.14e-1}, {12, 3.00e-1}, {13, 4.00e-1}, {14, 5.14e-1},
    {15, 6.41e-1}, {16, 7.81e-1}, {17, 9.31e-1}, {18, 1.09},    {19, 1.26},    {20, 1.44},    {21, 1.62},
    {22, 1.82},    {23, 2.01},    {24, 2.22},    {25, 2.43},    {26, 2.64},    {27, 2.86},    {28, 3.08},
    {29, 3.31},    {30, 3.54},    {35, 4.7},     {40, 6.0},     {45, 7.2},     {50, 8.5},     {55, 9.9},
}};

static double inf_norm(const std::vector<Complex> &values)
{
    double norm = 0.0;
    for (const Complex value : values)
    {
        norm = std::max(norm, std::abs(value));
    }
    return norm;
}

static bool is_identity(const PauliWord &word)
{
    const String &paulis = word.string();
    return std::all_of(paulis.begin(), paulis.end(), [](char pauli) { return pauli == 'I'; });
}

static std::pair<Size, Size> select_taylor_degree_and_scaling(double norm_bound)
{
    Size best_degree = 0;
    Size best_scaling = 1;
    long double best_cost = std::numeric_limits<long double>::infinity();

    for (const auto &[degree, bound] : taylor_bounds)
    {
        const long double scaling_real = std::ceil(static_cast<long double>(norm_bound) / bound);
        const Size scaling = std::max<Size>(1, static_cast<Size>(scaling_real));
        const long double cost = static_cast<long double>(degree) * scaling;
        if (cost < best_cost)
        {
            best_degree = degree;
            best_scaling = scaling;
            best_cost = cost;
        }
    }

    return {best_degree, best_scaling};
}

static void scaled_matvec(
    const Hamiltonian &ham,
    const std::vector<Complex> &in,
    std::vector<Complex> &out,
    Complex operator_scale,
    int num_threads)
{
    ham.matvec_into(in.data(), out.data(), num_threads);

    for (Index row = 0; row < in.size(); ++row)
    {
        out[row] *= operator_scale;
    }
}

Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex theta, int num_threads)
{
    const Size dim = Size(1) << ham.num_qubits();
    const Complex operator_scale = Complex(0.0, -1.0) * theta;

    Complex identity_coeff = 0.0;
    std::vector<PauliWord> residual_terms;
    residual_terms.reserve(ham.num_terms());
    for (const PauliWord &term : ham.terms())
    {
        if (is_identity(term))
        {
            identity_coeff += term.coeff();
        }
        else
        {
            residual_terms.push_back(term);
        }
    }

    const Complex mu = operator_scale * identity_coeff;
    const Hamiltonian residual_hamiltonian(std::move(residual_terms));
    const double norm_bound = std::abs(operator_scale) * residual_hamiltonian.lcu_norm();

    if (norm_bound == 0.0)
    {
        Complex *out = new Complex[dim];
        const Complex phase = std::exp(mu);
        for (Index row = 0; row < dim; ++row)
        {
            out[row] = phase * psi[row];
        }
        return out;
    }

    const auto [degree, scaling] = select_taylor_degree_and_scaling(norm_bound);
    const double tolerance = 0.5 * std::numeric_limits<double>::epsilon();
    const Complex phase_per_step = std::exp(mu / static_cast<double>(scaling));
    const double inverse_scaling = 1.0 / static_cast<double>(scaling);

    std::vector<Complex> state(psi, psi + dim);
    std::vector<Complex> term(dim);
    std::vector<Complex> product(dim);
    std::vector<Complex> sum(dim);

    for (Size step = 0; step < scaling; ++step)
    {
        term = state;
        sum = state;
        double previous_norm = inf_norm(term);

        for (Size j = 1; j <= degree; ++j)
        {
            scaled_matvec(residual_hamiltonian, term, product, operator_scale, num_threads);
            const double recurrence_scale = inverse_scaling / static_cast<double>(j);
            for (Index row = 0; row < dim; ++row)
            {
                term[row] = recurrence_scale * product[row];
                sum[row] += term[row];
            }

            const double term_norm = inf_norm(term);
            if (previous_norm + term_norm <= tolerance * inf_norm(sum))
            {
                break;
            }
            previous_norm = term_norm;
        }

        for (Index row = 0; row < dim; ++row)
        {
            state[row] = phase_per_step * sum[row];
        }
    }

    Complex *out = new Complex[dim];
    std::copy(state.begin(), state.end(), out);
    return out;
}
