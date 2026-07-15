#pragma once
#include <quantlop/types.hpp>

namespace quantlop
{

class MatVec
{
public:
    MatVec(Complex c, String str);
    void operator()(const Complex *in, Complex *out) const;
    void operator()(const Complex *in, Complex *out, const int num_threads) const;

private:
    Complex coeff;
    String string;
    Mask flip_mask, y_mask, z_mask;
};

}
