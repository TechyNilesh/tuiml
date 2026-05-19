#pragma once
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

namespace tuiml {
namespace clustering {

// Agglomerative hierarchical clustering using Lance-Williams updates.
// Returns cluster label array of shape (n_samples,).
// linkage: "ward" | "complete" | "average" | "single"
py::array_t<int> hierarchical_fit(
    py::array_t<double> X,
    int                 n_clusters,
    std::string         linkage
);

} // namespace clustering
} // namespace tuiml
