cmake_minimum_required(VERSION 3.18)

project(quantlop VERSION 0.0.1 LANGUAGES CXX)

option(QUANTLOP_BUILD_PYTHON "Build the Python extension" ON)
option(BUILD_TESTING "Build the native C++ tests" OFF)

add_library(
    quantlop_core STATIC
    cpp/src/pauliword.cpp
    cpp/src/matvec.cpp
    cpp/src/hamiltonian.cpp
    cpp/src/krylov.cpp
    cpp/src/simulation.cpp
)
add_library(quantlop::core ALIAS quantlop_core)

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

if(QUANTLOP_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_quantlop cpp/bindings/python_bindings.cpp)
    target_link_libraries(_quantlop PRIVATE quantlop::core)

    install(
        TARGETS _quantlop
        LIBRARY DESTINATION quantlop
        RUNTIME DESTINATION quantlop
    )
endif()

if(BUILD_TESTING)
    enable_testing()

    add_executable(quantlop_core_tests tests/cpp/test_core.cpp)
    target_link_libraries(quantlop_core_tests PRIVATE quantlop::core)
    add_test(NAME quantlop_core_tests COMMAND quantlop_core_tests)
endif()
