cmake_minimum_required(VERSION 3.18)

project(quantlop VERSION 0.0.3 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

add_library(
    quantlop_core STATIC
    cpp/src/pauliword.cpp
    cpp/src/matvec.cpp
    cpp/src/hamiltonian.cpp
    cpp/src/krylov.cpp
    cpp/src/simulation.cpp
)
add_library(quantlop::core ALIAS quantlop_core)

target_link_libraries(quantlop_core PRIVATE OpenMP::OpenMP_CXX)

target_compile_features(quantlop_core PUBLIC cxx_std_20)
set_target_properties(
    quantlop_core
    PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    quantlop_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
)

find_package(
    Python 3.11
    REQUIRED COMPONENTS Interpreter Development.Module
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_quantlop cpp/bindings/python_bindings.cpp)
target_link_libraries(_quantlop PRIVATE quantlop::core)

install(
    TARGETS _quantlop
    LIBRARY DESTINATION quantlop
    RUNTIME DESTINATION quantlop
)
