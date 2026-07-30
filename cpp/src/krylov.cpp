#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <quantlop/evolution.hpp>

#include "utils.cpp"

static constexpr Size max_krylov_dimension = 64;
using DenseMatrix = std::vector<Complex>;

struct LanczosWorkspace
{
    LanczosWorkspace(Size state_dimension, Size max_basis_size)
        : basis_capacity(max_basis_size),
          tridiagonal(max_basis_size * max_basis_size),
          previous(state_dimension),
          current(state_dimension),
          product(state_dimension)
    {
    }

    void reset_tridiagonal() { std::fill(tridiagonal.begin(), tridiagonal.end(), 0.0); }

    Size basis_capacity;
    std::vector<double> tridiagonal;
    std::vector<Complex> previous;
    std::vector<Complex> current;
    std::vector<Complex> product;
};

static double l2_norm(const Complex *values, Size dimension)
{
    double sum = 0.0;
    for (Index row = 0; row < dimension; ++row)
    {
        sum += std::norm(values[row]);
    }
    return std::sqrt(sum);
}

static Complex dot_product(const Complex *lhs, const Complex *rhs, Size dimension)
{
    Complex out = 0.0;
    for (Index row = 0; row < dimension; ++row)
    {
        out += std::conj(lhs[row]) * rhs[row];
    }
    return out;
}

static double one_norm_dense(const DenseMatrix &values, Size dimension)
{
    double best = 0.0;
    for (Index column = 0; column < dimension; ++column)
    {
        double column_sum = 0.0;
        for (Index row = 0; row < dimension; ++row)
        {
            column_sum += std::abs(values[row * dimension + column]);
        }
        best = std::max(best, column_sum);
    }
    return best;
}

static DenseMatrix matmul_dense(const DenseMatrix &lhs, const DenseMatrix &rhs, Size dimension)
{
    DenseMatrix out(dimension * dimension);
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index inner = 0; inner < dimension; ++inner)
        {
            const Complex lhs_value = lhs[row * dimension + inner];
            if (lhs_value == Complex(0.0, 0.0))
            {
                continue;
            }
            for (Index column = 0; column < dimension; ++column)
            {
                out[row * dimension + column] += lhs_value * rhs[inner * dimension + column];
            }
        }
    }
    return out;
}

static void swap_rows(DenseMatrix &matrix, Size first, Size second, Size dimension)
{
    Complex *first_row = matrix.data() + first * dimension;
    Complex *second_row = matrix.data() + second * dimension;
    std::swap_ranges(first_row, first_row + dimension, second_row);
}

static DenseMatrix solve_dense_system(DenseMatrix matrix, DenseMatrix rhs, Size dimension)
{
    for (Index column = 0; column < dimension; ++column)
    {
        Index pivot = column;
        double pivot_abs = std::abs(matrix[column * dimension + column]);
        for (Index row = column + 1; row < dimension; ++row)
        {
            const double candidate = std::abs(matrix[row * dimension + column]);
            if (candidate > pivot_abs)
            {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        if (pivot != column)
        {
            swap_rows(matrix, column, pivot, dimension);
            swap_rows(rhs, column, pivot, dimension);
        }

        const Complex diagonal = matrix[column * dimension + column];
        for (Index entry = 0; entry < dimension; ++entry)
        {
            matrix[column * dimension + entry] /= diagonal;
            rhs[column * dimension + entry] /= diagonal;
        }

        for (Index row = 0; row < dimension; ++row)
        {
            if (row == column)
            {
                continue;
            }
            const Complex factor = matrix[row * dimension + column];
            if (factor == Complex(0.0, 0.0))
            {
                continue;
            }
            for (Index entry = 0; entry < dimension; ++entry)
            {
                matrix[row * dimension + entry] -= factor * matrix[column * dimension + entry];
                rhs[row * dimension + entry] -= factor * rhs[column * dimension + entry];
            }
        }
    }
    return rhs;
}

static void add_to_diagonal(DenseMatrix &matrix, Size dimension, double value)
{
    for (Index diagonal = 0; diagonal < dimension; ++diagonal)
    {
        matrix[diagonal * dimension + diagonal] += value;
    }
}

static DenseMatrix expm_dense(DenseMatrix matrix, Size dimension)
{
    if (dimension == 1)
    {
        return {std::exp(matrix[0])};
    }

    constexpr std::array<double, 14> pade_coefficients = {
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0};

    constexpr double pade_theta = 5.371920351148152;
    const double matrix_norm = one_norm_dense(matrix, dimension);
    const int squarings = matrix_norm > pade_theta
                              ? static_cast<int>(std::ceil(std::log2(matrix_norm / pade_theta)))
                              : 0;
    const double scale = std::ldexp(1.0, squarings);
    for (Complex &value : matrix)
    {
        value /= scale;
    }

    const DenseMatrix matrix_2 = matmul_dense(matrix, matrix, dimension);
    const DenseMatrix matrix_4 = matmul_dense(matrix_2, matrix_2, dimension);
    const DenseMatrix matrix_6 = matmul_dense(matrix_4, matrix_2, dimension);

    const Size matrix_size = dimension * dimension;
    DenseMatrix temporary_1(matrix_size);
    DenseMatrix temporary_2;
    DenseMatrix u_inner(matrix_size);
    DenseMatrix v(matrix_size);

    for (Index index = 0; index < matrix_size; ++index)
    {
        temporary_1[index] = pade_coefficients[13] * matrix_6[index] +
                             pade_coefficients[11] * matrix_4[index] +
                             pade_coefficients[9] * matrix_2[index];
    }
    temporary_2 = matmul_dense(matrix_6, temporary_1, dimension);
    for (Index index = 0; index < matrix_size; ++index)
    {
        u_inner[index] = temporary_2[index] + pade_coefficients[7] * matrix_6[index] +
                         pade_coefficients[5] * matrix_4[index] +
                         pade_coefficients[3] * matrix_2[index];
    }
    add_to_diagonal(u_inner, dimension, pade_coefficients[1]);
    const DenseMatrix u = matmul_dense(matrix, u_inner, dimension);

    for (Index index = 0; index < matrix_size; ++index)
    {
        temporary_1[index] = pade_coefficients[12] * matrix_6[index] +
                             pade_coefficients[10] * matrix_4[index] +
                             pade_coefficients[8] * matrix_2[index];
    }
    temporary_2 = matmul_dense(matrix_6, temporary_1, dimension);
    for (Index index = 0; index < matrix_size; ++index)
    {
        v[index] = temporary_2[index] + pade_coefficients[6] * matrix_6[index] +
                   pade_coefficients[4] * matrix_4[index] + pade_coefficients[2] * matrix_2[index];
    }
    add_to_diagonal(v, dimension, pade_coefficients[0]);

    DenseMatrix denominator(matrix_size);
    DenseMatrix numerator(matrix_size);
    for (Index index = 0; index < matrix_size; ++index)
    {
        denominator[index] = v[index] - u[index];
        numerator[index] = v[index] + u[index];
    }
    DenseMatrix result =
        solve_dense_system(std::move(denominator), std::move(numerator), dimension);

    for (int step = 0; step < squarings; ++step)
    {
        result = matmul_dense(result, result, dimension);
    }
    return result;
}

static DenseMatrix extract_scaled_dense(
    const std::vector<double> &matrix,
    Size leading_dimension,
    Size dimension,
    Complex scale)
{
    DenseMatrix out(dimension * dimension);
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index column = 0; column < dimension; ++column)
        {
            out[row * dimension + column] = scale * matrix[row * leading_dimension + column];
        }
    }
    return out;
}

static Size build_lanczos_tridiagonal(
    const Hamiltonian &ham,
    const Complex *psi,
    double state_norm,
    int num_threads,
    LanczosWorkspace &workspace)
{
    const Size dimension = ham.dimension();
    const Size max_basis_size = workspace.basis_capacity;
    const double norm_tolerance = std::numeric_limits<double>::epsilon() * 1e2;
    std::vector<double> &tridiagonal = workspace.tridiagonal;
    std::vector<Complex> &previous = workspace.previous;
    std::vector<Complex> &current = workspace.current;
    std::vector<Complex> &product = workspace.product;
    double previous_beta = 0.0;

    scale_copy(psi, current.data(), dimension, 1.0 / state_norm);

    for (Index basis_index = 0; basis_index < max_basis_size; ++basis_index)
    {
        ham.residual_matvec_into(current.data(), product.data(), num_threads);

        if (basis_index > 0)
        {
            for (Index row = 0; row < dimension; ++row)
            {
                product[row] -= previous_beta * previous[row];
            }
        }

        const double alpha = dot_product(current.data(), product.data(), dimension).real();
        tridiagonal[basis_index * max_basis_size + basis_index] = alpha;

        for (Index row = 0; row < dimension; ++row)
        {
            product[row] -= alpha * current[row];
        }

        const double next_beta = l2_norm(product.data(), dimension);
        if (basis_index + 1 < max_basis_size)
        {
            tridiagonal[(basis_index + 1) * max_basis_size + basis_index] = next_beta;
            tridiagonal[basis_index * max_basis_size + (basis_index + 1)] = next_beta;
        }

        if (next_beta < norm_tolerance || basis_index + 1 == max_basis_size)
        {
            return basis_index + 1;
        }

        previous.swap(current);
        const double inverse_beta = 1.0 / next_beta;
        for (Index row = 0; row < dimension; ++row)
        {
            current[row] = inverse_beta * product[row];
        }
        previous_beta = next_beta;
    }
    return max_basis_size;
}

static void reconstruct_lanczos_state(
    const Hamiltonian &ham,
    const Complex *psi,
    double state_norm,
    Size basis_size,
    double theta,
    Complex *out,
    int num_threads,
    LanczosWorkspace &workspace)
{
    const Size dimension = ham.dimension();
    const Size leading_dimension = workspace.basis_capacity;
    const std::vector<double> &tridiagonal = workspace.tridiagonal;
    std::vector<Complex> &previous = workspace.previous;
    std::vector<Complex> &current = workspace.current;
    std::vector<Complex> &product = workspace.product;
    scale_copy(psi, current.data(), dimension, 1.0 / state_norm);

    const Complex operator_scale(0.0, -theta);
    const DenseMatrix exponential = expm_dense(
        extract_scaled_dense(tridiagonal, leading_dimension, basis_size, operator_scale),
        basis_size);

    for (Index row = 0; row < dimension; ++row)
    {
        out[row] = psi[row] * exponential[0];
    }

    for (Index basis_index = 1; basis_index < basis_size; ++basis_index)
    {
        const double beta = tridiagonal[(basis_index - 1) * leading_dimension + basis_index];
        const double alpha = tridiagonal[(basis_index - 1) * leading_dimension + (basis_index - 1)];
        ham.residual_matvec_into(current.data(), product.data(), num_threads);

        for (Index row = 0; row < dimension; ++row)
        {
            product[row] -= alpha * current[row];
        }
        if (basis_index > 1)
        {
            const double previous_previous_beta =
                tridiagonal[(basis_index - 2) * leading_dimension + (basis_index - 1)];
            for (Index row = 0; row < dimension; ++row)
            {
                product[row] -= previous_previous_beta * previous[row];
            }
        }

        previous.swap(current);
        const double inverse_beta = 1.0 / beta;
        const Complex coefficient = state_norm * exponential[basis_index * basis_size];
        for (Index row = 0; row < dimension; ++row)
        {
            current[row] = inverse_beta * product[row];
            out[row] += coefficient * current[row];
        }
    }
}

static std::pair<Size, Size> select_krylov_dimension_and_scaling(
    double norm_bound,
    Size dimension,
    double rtol)
{
    if (dimension == 1)
    {
        return {1, 1};
    }

    const Size largest_basis = std::min(max_krylov_dimension, dimension);
    const double log_norm = std::log(norm_bound);
    const double log_target = std::log(0.5 * rtol);
    Size best_basis = 2;
    Size best_scaling = 1;
    double best_cost = std::numeric_limits<double>::infinity();

    for (Size basis_size = 2; basis_size <= largest_basis; ++basis_size)
    {
        Size scaling;
        if (basis_size == dimension)
        {
            scaling = 1;
        }
        else
        {
            const double log_error_at_one = std::log(2.0) +
                                            static_cast<double>(basis_size) * log_norm -
                                            std::lgamma(basis_size + 1);
            scaling = minimum_scaling(log_error_at_one, basis_size - 1, log_target);
        }

        const double cost = static_cast<double>(scaling) * static_cast<double>(2 * basis_size - 1);
        if (cost < best_cost)
        {
            best_basis = basis_size;
            best_scaling = scaling;
            best_cost = cost;
        }
    }

    return {best_basis, best_scaling};
}

std::unique_ptr<Complex[]> evolve_krylov(
    const Hamiltonian &ham,
    const Complex *psi,
    double theta,
    double rtol,
    int num_threads)
{
    const Size dimension = ham.dimension();
    const double norm_bound = std::abs(theta) * ham.residual_lcu_norm();
    std::unique_ptr<Complex[]> state = std::make_unique<Complex[]>(dimension);

    if (norm_bound == 0.0)
    {
        const Complex phase = std::exp(Complex(0.0, -theta) * ham.identity_coeff());
        scale_copy(psi, state.get(), dimension, phase);
        return state;
    }

    const auto [max_basis_size, scaling] =
        select_krylov_dimension_and_scaling(norm_bound, dimension, rtol);
    const double step_theta = theta / static_cast<double>(scaling);
    const Complex phase_per_step = std::exp(Complex(0.0, -step_theta) * ham.identity_coeff());

    std::copy_n(psi, dimension, state.get());
    std::unique_ptr<Complex[]> next_state = std::make_unique<Complex[]>(dimension);
    LanczosWorkspace workspace(dimension, max_basis_size);

    for (Size step = 0; step < scaling; ++step)
    {
        const double state_norm = l2_norm(state.get(), dimension);
        workspace.reset_tridiagonal();
        const Size basis_size =
            build_lanczos_tridiagonal(ham, state.get(), state_norm, num_threads, workspace);
        reconstruct_lanczos_state(
            ham,
            state.get(),
            state_norm,
            basis_size,
            step_theta,
            next_state.get(),
            num_threads,
            workspace);

        scale_in_place(next_state.get(), dimension, phase_per_step);
        state.swap(next_state);
    }

    return state;
}
