#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <quantlop/hamiltonian.hpp>
#include <quantlop/pauliword.hpp>
#include <quantlop/simulation.hpp>
#include <quantlop/types.hpp>

namespace nb = nanobind;
using quantlop::Complex;
using quantlop::Hamiltonian;
using quantlop::PauliWord;
using quantlop::Size;
using quantlop::String;

using ComplexArray = nb::ndarray<const Complex, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NumpyComplexArray = nb::ndarray<nb::numpy, Complex, nb::ndim<1>, nb::c_contig>;

static NumpyComplexArray
evolve_py(const Hamiltonian &ham,
          ComplexArray psi,
          Complex coeff,
          int num_threads)
{
    const Size dim = psi.shape(0);
    Complex *out_ptr = quantlop::evolve(ham, psi.data(), coeff, num_threads);
    nb::capsule owner(out_ptr, [](void *p) noexcept
                      { delete[] static_cast<Complex *>(p); });
    return NumpyComplexArray(out_ptr, {dim}, owner);
}

NB_MODULE(_quantlop, module_py)
{
    module_py.doc() = "Quantlop C++ core bindings";

    nb::class_<PauliWord>(module_py, "PauliWord")
        .def(nb::init<Complex, String>(), nb::arg("coeff"), nb::arg("string"));

    nb::class_<Hamiltonian>(module_py, "Hamiltonian")
        .def(nb::init<std::vector<PauliWord>>(), nb::arg("pauli_words"));

    module_py.def("evolve", &evolve_py, nb::arg("ham"), nb::arg("psi"),
                  nb::arg("coeff"), nb::arg("num_threads"));
}
