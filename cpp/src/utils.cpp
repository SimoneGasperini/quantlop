#include <algorithm>
#include <cmath>

#include <quantlop/types.hpp>

inline Size minimum_scaling(double log_error_at_one, Size scaling_power, double log_target)
{
    const double log_scaling =
        std::max(0.0, (log_error_at_one - log_target) / static_cast<double>(scaling_power));
    Size scaling = std::max<Size>(1, static_cast<Size>(std::ceil(std::exp(log_scaling))));
    while (log_error_at_one - static_cast<double>(scaling_power) * std::log(scaling) > log_target)
    {
        ++scaling;
    }
    return scaling;
}

inline void scale_copy(const Complex *source, Complex *destination, Size size, Complex factor)
{
    for (Index index = 0; index < size; ++index)
    {
        destination[index] = factor * source[index];
    }
}

inline void scale_in_place(Complex *values, Size size, Complex factor)
{
    for (Index index = 0; index < size; ++index)
    {
        values[index] *= factor;
    }
}
