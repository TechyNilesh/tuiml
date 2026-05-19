cmake_minimum_required(VERSION 3.15...3.30)
project(tuiml_cpp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter NumPy REQUIRED)

option(USE_OPENMP "Enable OpenMP parallelism" ON)

if(USE_OPENMP)
    find_package(OpenMP)
endif()

pybind11_add_module(_cpp_ext
    tuiml/_cpp/module.cpp
    tuiml/_cpp/tree/splitters.cpp
    tuiml/_cpp/tree/criteria.cpp
    tuiml/_cpp/tree/predict.cpp
    tuiml/_cpp/tree/builder.cpp
    tuiml/_cpp/distance/metrics.cpp
    tuiml/_cpp/neighbors/knn.cpp
    tuiml/_cpp/neighbors/kd_tree.cpp
    tuiml/_cpp/neighbors/ball_tree.cpp
    tuiml/_cpp/svm/kernels.cpp
    tuiml/_cpp/svm/smo_solver.cpp
    tuiml/_cpp/svm/svc.cpp
    tuiml/_cpp/svm/svr.cpp
    tuiml/_cpp/linear/sgd.cpp
)

target_include_directories(_cpp_ext PRIVATE
    ${Python_NumPy_INCLUDE_DIRS}
    tuiml/_cpp
)

target_compile_features(_cpp_ext PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_cpp_ext PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_cpp_ext PRIVATE TUIML_USE_OPENMP)
endif()

install(TARGETS _cpp_ext DESTINATION tuiml)
