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

static NumpyComplexArray evolve_py(const Hamiltonian &ham, ComplexArray psi, Complex coeff, int num_threads)
{
    const Size dim = psi.shape(0);
    Complex *out_ptr = quantlop::evolve(ham, psi.data(), coeff, num_threads);
    nb::capsule owner(out_ptr, [](void *p) noexcept { delete[] static_cast<Complex *>(p); });
    return NumpyComplexArray(out_ptr, {dim}, owner);
}

NB_MODULE(_quantlop, module_py)
{
    module_py.doc() = "Quantlop C++ core bindings";

    nb::class_<PauliWord>(module_py, "_PauliWord")
        .def(nb::init<Complex, String>(), nb::arg("coeff"), nb::arg("string"))
        .def("_num_qubits", &PauliWord::num_qubits)
        .def("_get_coeff", &PauliWord::get_coeff)
        .def("_get_string", &PauliWord::get_string);

    nb::class_<Hamiltonian>(module_py, "_Hamiltonian")
        .def(nb::init<std::vector<PauliWord>>(), nb::arg("pwords"))
        .def("_num_qubits", &Hamiltonian::num_qubits)
        .def("_num_terms", &Hamiltonian::num_terms)
        .def("_get_pwords", &Hamiltonian::get_pwords);

    module_py.def("_evolve", &evolve_py, nb::arg("ham"), nb::arg("psi"), nb::arg("coeff"), nb::arg("num_threads"));
}
