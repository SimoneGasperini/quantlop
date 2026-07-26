#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <quantlop/evolution.hpp>

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

static double one_norm_dense(const std::vector<Complex> &values, Size dimension)
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

static std::vector<Complex> matmul_dense(
    const std::vector<Complex> &lhs,
    const std::vector<Complex> &rhs,
    Size dimension)
{
    std::vector<Complex> out(dimension * dimension, Complex(0.0, 0.0));
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

static std::vector<Complex> solve_dense_system(
    std::vector<Complex> matrix,
    std::vector<Complex> rhs,
    Size dimension)
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
            for (Index rhs_column = 0; rhs_column < dimension; ++rhs_column)
            {
                std::swap(
                    matrix[column * dimension + rhs_column],
                    matrix[pivot * dimension + rhs_column]);
                std::swap(
                    rhs[column * dimension + rhs_column],
                    rhs[pivot * dimension + rhs_column]);
            }
        }

        const Complex diagonal = matrix[column * dimension + column];
        for (Index rhs_column = 0; rhs_column < dimension; ++rhs_column)
        {
            matrix[column * dimension + rhs_column] /= diagonal;
            rhs[column * dimension + rhs_column] /= diagonal;
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
            for (Index rhs_column = 0; rhs_column < dimension; ++rhs_column)
            {
                matrix[row * dimension + rhs_column] -=
                    factor * matrix[column * dimension + rhs_column];
                rhs[row * dimension + rhs_column] -= factor * rhs[column * dimension + rhs_column];
            }
        }
    }
    return rhs;
}

static std::vector<Complex> expm_dense(std::vector<Complex> matrix, Size dimension)
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

    constexpr double theta_13 = 5.371920351148152;
    const double matrix_norm = one_norm_dense(matrix, dimension);
    int squarings = 0;
    if (matrix_norm > theta_13)
    {
        squarings = static_cast<int>(std::ceil(std::log2(matrix_norm / theta_13)));
    }
    const double scale = std::ldexp(1.0, squarings);
    for (Complex &value : matrix)
    {
        value /= scale;
    }

    const std::vector<Complex> matrix_2 = matmul_dense(matrix, matrix, dimension);
    const std::vector<Complex> matrix_4 = matmul_dense(matrix_2, matrix_2, dimension);
    const std::vector<Complex> matrix_6 = matmul_dense(matrix_4, matrix_2, dimension);

    std::vector<Complex> temporary_1(dimension * dimension, Complex(0.0, 0.0));
    std::vector<Complex> temporary_2;
    std::vector<Complex> u_inner(dimension * dimension, Complex(0.0, 0.0));
    std::vector<Complex> v(dimension * dimension, Complex(0.0, 0.0));

    for (Index index = 0; index < dimension * dimension; ++index)
    {
        temporary_1[index] = pade_coefficients[13] * matrix_6[index] +
                             pade_coefficients[11] * matrix_4[index] +
                             pade_coefficients[9] * matrix_2[index];
    }
    temporary_2 = matmul_dense(matrix_6, temporary_1, dimension);
    for (Index index = 0; index < dimension * dimension; ++index)
    {
        u_inner[index] = temporary_2[index] + pade_coefficients[7] * matrix_6[index] +
                         pade_coefficients[5] * matrix_4[index] +
                         pade_coefficients[3] * matrix_2[index];
    }
    for (Index diagonal = 0; diagonal < dimension; ++diagonal)
    {
        u_inner[diagonal * dimension + diagonal] += pade_coefficients[1];
    }
    const std::vector<Complex> u = matmul_dense(matrix, u_inner, dimension);

    for (Index index = 0; index < dimension * dimension; ++index)
    {
        temporary_1[index] = pade_coefficients[12] * matrix_6[index] +
                             pade_coefficients[10] * matrix_4[index] +
                             pade_coefficients[8] * matrix_2[index];
    }
    temporary_2 = matmul_dense(matrix_6, temporary_1, dimension);
    for (Index index = 0; index < dimension * dimension; ++index)
    {
        v[index] = temporary_2[index] + pade_coefficients[6] * matrix_6[index] +
                   pade_coefficients[4] * matrix_4[index] + pade_coefficients[2] * matrix_2[index];
    }
    for (Index diagonal = 0; diagonal < dimension; ++diagonal)
    {
        v[diagonal * dimension + diagonal] += pade_coefficients[0];
    }

    std::vector<Complex> denominator(dimension * dimension);
    std::vector<Complex> numerator(dimension * dimension);
    for (Index index = 0; index < dimension * dimension; ++index)
    {
        denominator[index] = v[index] - u[index];
        numerator[index] = v[index] + u[index];
    }
    std::vector<Complex> result =
        solve_dense_system(std::move(denominator), std::move(numerator), dimension);

    for (int step = 0; step < squarings; ++step)
    {
        result = matmul_dense(result, result, dimension);
    }
    return result;
}

static std::vector<Complex> extract_scaled_dense(
    const std::vector<double> &matrix,
    Size leading_dimension,
    Size dimension,
    Complex scale)
{
    std::vector<Complex> out(dimension * dimension, Complex(0.0, 0.0));
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
    std::vector<double> &tridiagonal,
    Size max_basis_size,
    int num_threads,
    std::vector<Complex> &previous,
    std::vector<Complex> &current,
    std::vector<Complex> &product)
{
    const Size dimension = ham.dimension();
    const double norm_tolerance = std::numeric_limits<double>::epsilon() * 1e2;
    const double inverse_state_norm = 1.0 / state_norm;
    double previous_beta = 0.0;

    for (Index row = 0; row < dimension; ++row)
    {
        current[row] = inverse_state_norm * psi[row];
    }

    for (Index basis_index = 0; basis_index < max_basis_size; ++basis_index)
    {
        ham.matvec_into(current.data(), product.data(), num_threads);

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
    const std::vector<double> &tridiagonal,
    Size leading_dimension,
    Size basis_size,
    Complex theta,
    Complex *out,
    int num_threads,
    std::vector<Complex> &previous,
    std::vector<Complex> &current,
    std::vector<Complex> &product)
{
    const Size dimension = ham.dimension();
    const double inverse_state_norm = 1.0 / state_norm;

    for (Index row = 0; row < dimension; ++row)
    {
        current[row] = inverse_state_norm * psi[row];
    }

    const Complex operator_scale = Complex(0.0, -1.0) * theta;
    const std::vector<Complex> exponential = expm_dense(
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
        ham.matvec_into(current.data(), product.data(), num_threads);

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

Complex *evolve_krylov(
    const Hamiltonian &ham,
    const Complex *psi,
    Complex theta,
    int num_threads,
    int dim_krylov)
{
    const Size dimension = ham.dimension();
    std::unique_ptr<Complex[]> out = std::make_unique<Complex[]>(dimension);
    const double state_norm = l2_norm(psi, dimension);
    const Size max_basis_size = std::min<Size>(static_cast<Size>(dim_krylov), dimension);
    std::vector<double> tridiagonal(max_basis_size * max_basis_size, 0.0);
    std::vector<Complex> previous(dimension);
    std::vector<Complex> current(dimension);
    std::vector<Complex> product(dimension);
    const Size basis_size = build_lanczos_tridiagonal(
        ham,
        psi,
        state_norm,
        tridiagonal,
        max_basis_size,
        num_threads,
        previous,
        current,
        product);
    reconstruct_lanczos_state(
        ham,
        psi,
        state_norm,
        tridiagonal,
        max_basis_size,
        basis_size,
        theta,
        out.get(),
        num_threads,
        previous,
        current,
        product);

    return out.release();
}
