#pragma once
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace tuiml {
namespace clustering {

struct EMResult {
    std::vector<double> weights;      // (n_components,)
    std::vector<double> means;        // (n_components, n_features)
    std::vector<double> covariances;  // (n_components, n_features, n_features) for "full"
                                      // (n_components, n_features) for "diag"/"spherical"
    int    n_iter;
    double log_likelihood;
    std::string cov_type;
};

// Fit a Gaussian Mixture Model via EM.
// covariance_type: "full" | "diag" | "spherical"
// Returns EMResult; caller extracts weights/means/covs and computes labels.
EMResult em_fit(
    py::array_t<double> X,
    int                 n_components,
    std::string         covariance_type,
    int                 max_iter,
    double              tol,
    double              reg_covar,      // regularisation added to diagonal
    int                 n_init,
    int                 random_seed
);

// Predict cluster labels given fitted parameters.
py::array_t<int> em_predict(
    py::array_t<double> X,
    py::array_t<double> weights,
    py::array_t<double> means,
    py::array_t<double> covariances,
    std::string         covariance_type
);

// Compute per-sample log responsibilities (n_samples, n_components).
py::array_t<double> em_log_resp(
    py::array_t<double> X,
    py::array_t<double> weights,
    py::array_t<double> means,
    py::array_t<double> covariances,
    std::string         covariance_type
);

} // namespace clustering
} // namespace tuiml
