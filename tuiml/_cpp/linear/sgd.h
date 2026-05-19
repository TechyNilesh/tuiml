#pragma once

#include <vector>
#include <string>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace linear {

// ── Loss types ──────────────────────────────────────────────────────
// Classification: 0=hinge, 1=log (logistic), 2=modified_huber
// Regression:     3=squared_error, 4=huber, 5=epsilon_insensitive
enum class LossType : int {
    HINGE             = 0,
    LOG               = 1,
    MODIFIED_HUBER    = 2,
    SQUARED_ERROR     = 3,
    HUBER             = 4,
    EPSILON_INSENSITIVE = 5,
};

// ── Regularization types ────────────────────────────────────────────
enum class PenaltyType : int {
    NONE       = 0,
    L2         = 1,
    L1         = 2,
    ELASTICNET = 3,
};

// ── Result structs ──────────────────────────────────────────────────
struct SGDResult {
    std::vector<double> weights;  // shape (n_classes, n_features) flattened (classifier) or (n_features,) (regressor)
    std::vector<double> bias;     // shape (n_classes,) or (1,)
    int n_iter;                   // actual epochs run
};

// ── Training functions ───────────────────────────────────────────────

// Classifier: supports binary and multiclass (OvR)
// X: (n_samples, n_features) C-contiguous float64
// y: (n_samples,) int32 class labels 0..n_classes-1
// weights_init: (n_classes, n_features) or empty → zero init
// bias_init:    (n_classes,) or empty → zero init
SGDResult sgd_fit_classifier(
    py::array_t<double> X,
    py::array_t<int>    y,
    int                 n_classes,
    int                 loss_type,
    int                 penalty_type,
    double              alpha,        // regularization strength
    double              l1_ratio,     // for elasticnet: l1_ratio * l1 + (1-l1_ratio) * l2
    double              eta0,         // initial learning rate
    std::string         lr_schedule,  // "constant", "optimal", "invscaling", "adaptive"
    double              power_t,      // for invscaling: eta = eta0 / t^power_t
    int                 n_epochs,
    int                 batch_size,
    double              tol,
    int                 patience,     // early stopping: epochs without improvement
    bool                shuffle,
    int                 random_seed,
    py::array_t<double> weights_init,
    py::array_t<double> bias_init
);

// Regressor: single output
// y: (n_samples,) float64
SGDResult sgd_fit_regressor(
    py::array_t<double> X,
    py::array_t<double> y,
    int                 loss_type,
    int                 penalty_type,
    double              alpha,
    double              l1_ratio,
    double              eta0,
    std::string         lr_schedule,
    double              power_t,
    double              epsilon,      // for huber / epsilon_insensitive
    int                 n_epochs,
    int                 batch_size,
    double              tol,
    int                 patience,
    bool                shuffle,
    int                 random_seed,
    py::array_t<double> weights_init,
    py::array_t<double> bias_init
);

// Predict (linear dot product + bias)
py::array_t<double> sgd_decision_function(
    py::array_t<double> X,
    py::array_t<double> weights,   // (n_outputs, n_features) or (n_features,)
    py::array_t<double> bias
);

}  // namespace linear
}  // namespace tuiml
