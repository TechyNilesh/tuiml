#include "linear/sgd.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tuiml {
namespace linear {

// ── Helpers ──────────────────────────────────────────────────────────

static inline double clip(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Soft-threshold shrinkage for L1/elasticnet
static inline double soft_threshold(double w, double thresh) {
    if (w > thresh)  return w - thresh;
    if (w < -thresh) return w + thresh;
    return 0.0;
}

// Compute learning rate at step t (1-indexed)
static double compute_lr(const std::string& schedule, double eta0,
                          double alpha, double power_t, long long t) {
    if (schedule == "constant") {
        return eta0;
    } else if (schedule == "optimal") {
        // sklearn's optimal: eta = 1 / (alpha * (t + t0))
        // t0 chosen so that the expected initial step equals 0.01
        // simplified: 1 / (alpha * t) with floor
        double t0 = 1.0 / (0.01 * alpha + 1e-12);
        return 1.0 / (alpha * (t + t0) + 1e-12);
    } else if (schedule == "invscaling") {
        return eta0 / std::pow(static_cast<double>(t), power_t);
    } else {  // adaptive — caller adjusts eta externally; return eta0 here
        return eta0;
    }
}

// Apply regularization penalty (weight update after gradient step)
// Modifies weights in-place for L2 (weight decay) and L1/elasticnet (soft threshold)
static void apply_penalty(double* w, int n_features, int penalty_type,
                           double alpha, double l1_ratio, double lr) {
    if (penalty_type == static_cast<int>(PenaltyType::NONE)) return;

    if (penalty_type == static_cast<int>(PenaltyType::L2)) {
        double decay = 1.0 - lr * alpha;
        if (decay < 0.0) decay = 0.0;
        for (int j = 0; j < n_features; ++j) w[j] *= decay;
    } else if (penalty_type == static_cast<int>(PenaltyType::L1)) {
        double thresh = lr * alpha;
        for (int j = 0; j < n_features; ++j)
            w[j] = soft_threshold(w[j], thresh);
    } else {  // elasticnet
        // L2 decay first, then L1 shrinkage
        double decay = 1.0 - lr * alpha * (1.0 - l1_ratio);
        if (decay < 0.0) decay = 0.0;
        double thresh = lr * alpha * l1_ratio;
        for (int j = 0; j < n_features; ++j) {
            w[j] = soft_threshold(w[j] * decay, thresh);
        }
    }
}

// ── Loss gradient functions ───────────────────────────────────────────
// Returns dloss/d(score) for a single sample

// Hinge: max(0, 1 - y*score), y in {-1, +1}
static inline double hinge_grad(double score, double label) {
    double margin = label * score;
    return (margin < 1.0) ? -label : 0.0;
}

// Log (logistic): log(1 + exp(-y*score)), y in {-1, +1}
static inline double log_grad(double score, double label) {
    double margin = label * score;
    // sigmoid(-margin) = 1 / (1 + exp(margin))
    if (margin > 18.0) return -label * std::exp(-margin);
    if (margin < -18.0) return -label;
    return -label / (1.0 + std::exp(margin));
}

// Modified huber: if margin >= -1: hinge-like, else clipped
static inline double modified_huber_grad(double score, double label) {
    double margin = label * score;
    if (margin >= 1.0)  return 0.0;
    if (margin >= -1.0) return -2.0 * (1.0 - margin) * label;
    return -4.0 * label;
}

// Squared error for regression: 0.5*(y-score)^2 → grad = score - y
static inline double squared_error_grad(double score, double y) {
    return score - y;
}

// Huber loss gradient (epsilon-insensitive squared)
static inline double huber_grad(double score, double y, double epsilon) {
    double r = score - y;
    if (std::abs(r) <= epsilon) return 0.0;
    return (r > 0) ? 1.0 : -1.0;  // sign(r), outer clipping is huber behavior
}

// Epsilon-insensitive: |y-score| - epsilon if > 0
static inline double eps_insensitive_grad(double score, double y, double epsilon) {
    double r = score - y;
    if (r > epsilon)  return 1.0;
    if (r < -epsilon) return -1.0;
    return 0.0;
}

// ── Core SGD loop (classifier, OvR) ──────────────────────────────────

SGDResult sgd_fit_classifier(
    py::array_t<double> X,
    py::array_t<int>    y,
    int                 n_classes,
    int                 loss_type,
    int                 penalty_type,
    double              alpha,
    double              l1_ratio,
    double              eta0,
    std::string         lr_schedule,
    double              power_t,
    int                 n_epochs,
    int                 batch_size,
    double              tol,
    int                 patience,
    bool                shuffle,
    int                 random_seed,
    py::array_t<double> weights_init,
    py::array_t<double> bias_init
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    if (X_buf.ndim != 2) throw std::runtime_error("X must be 2-D");
    if (y_buf.ndim != 1) throw std::runtime_error("y must be 1-D");

    int n_samples  = static_cast<int>(X_buf.shape[0]);
    int n_features = static_cast<int>(X_buf.shape[1]);
    const double* Xp = static_cast<double*>(X_buf.ptr);
    const int*    yp = static_cast<int*>(y_buf.ptr);

    // Binary case uses 1 output with labels -1/+1; multiclass uses n_classes outputs (OvR)
    bool binary = (n_classes == 2);
    int  n_out  = binary ? 1 : n_classes;

    // Initialize weights
    std::vector<double> W(n_out * n_features, 0.0);
    std::vector<double> b(n_out, 0.0);

    auto wi_buf = weights_init.request();
    auto bi_buf = bias_init.request();
    if (wi_buf.size == n_out * n_features) {
        const double* wp = static_cast<double*>(wi_buf.ptr);
        std::copy(wp, wp + n_out * n_features, W.begin());
    }
    if (bi_buf.size == static_cast<size_t>(n_out)) {
        const double* bp = static_cast<double*>(bi_buf.ptr);
        std::copy(bp, bp + n_out, b.begin());
    }

    std::mt19937 rng(random_seed);
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    long long global_step = 1;
    double    best_loss   = 1e18;
    int       no_improve  = 0;
    double    cur_eta     = eta0;
    int       actual_epochs = 0;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        ++actual_epochs;
        if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;

        for (int start = 0; start < n_samples; start += batch_size) {
            int end = std::min(start + batch_size, n_samples);
            int bsz = end - start;

            // Accumulate gradients for the mini-batch
            std::vector<double> grad_W(n_out * n_features, 0.0);
            std::vector<double> grad_b(n_out, 0.0);
            double batch_loss = 0.0;

            for (int bi = start; bi < end; ++bi) {
                int i = indices[bi];
                const double* xi = Xp + i * n_features;
                int yi_label = yp[i];

                if (binary) {
                    // Map 0→-1, 1→+1
                    double label = (yi_label == 1) ? 1.0 : -1.0;
                    // score
                    double score = b[0];
                    for (int j = 0; j < n_features; ++j) score += W[j] * xi[j];

                    double g = 0.0;
                    if (loss_type == static_cast<int>(LossType::HINGE))
                        g = hinge_grad(score, label);
                    else if (loss_type == static_cast<int>(LossType::LOG))
                        g = log_grad(score, label);
                    else
                        g = modified_huber_grad(score, label);

                    grad_b[0] += g;
                    for (int j = 0; j < n_features; ++j)
                        grad_W[j] += g * xi[j];
                    batch_loss += std::abs(g);
                } else {
                    // One-vs-rest
                    for (int c = 0; c < n_classes; ++c) {
                        double label = (yi_label == c) ? 1.0 : -1.0;
                        double score = b[c];
                        const double* Wc = W.data() + c * n_features;
                        for (int j = 0; j < n_features; ++j) score += Wc[j] * xi[j];

                        double g = 0.0;
                        if (loss_type == static_cast<int>(LossType::HINGE))
                            g = hinge_grad(score, label);
                        else if (loss_type == static_cast<int>(LossType::LOG))
                            g = log_grad(score, label);
                        else
                            g = modified_huber_grad(score, label);

                        grad_b[c] += g;
                        double* gWc = grad_W.data() + c * n_features;
                        for (int j = 0; j < n_features; ++j)
                            gWc[j] += g * xi[j];
                        batch_loss += std::abs(g);
                    }
                }
            }

            // LR for this step
            if (lr_schedule != "adaptive")
                cur_eta = compute_lr(lr_schedule, eta0, alpha, power_t, global_step);
            double scale = cur_eta / bsz;

            // Update weights
            for (int c = 0; c < n_out; ++c) {
                b[c] -= scale * grad_b[c];
                double* Wc = W.data() + c * n_features;
                for (int j = 0; j < n_features; ++j)
                    Wc[j] -= scale * grad_W[c * n_features + j];
                apply_penalty(Wc, n_features, penalty_type, alpha, l1_ratio, cur_eta);
            }

            epoch_loss += batch_loss;
            ++global_step;
        }

        epoch_loss /= n_samples;

        // Early stopping (adaptive LR: halve eta on no improvement)
        if (epoch_loss < best_loss - tol) {
            best_loss  = epoch_loss;
            no_improve = 0;
        } else {
            ++no_improve;
            if (lr_schedule == "adaptive") cur_eta *= 0.5;
            if (no_improve >= patience) break;
        }
    }

    SGDResult result;
    result.weights = std::move(W);
    result.bias    = std::move(b);
    result.n_iter  = actual_epochs;
    return result;
}

// ── Core SGD loop (regressor) ─────────────────────────────────────────

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
    double              epsilon,
    int                 n_epochs,
    int                 batch_size,
    double              tol,
    int                 patience,
    bool                shuffle,
    int                 random_seed,
    py::array_t<double> weights_init,
    py::array_t<double> bias_init
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    if (X_buf.ndim != 2) throw std::runtime_error("X must be 2-D");

    int n_samples  = static_cast<int>(X_buf.shape[0]);
    int n_features = static_cast<int>(X_buf.shape[1]);
    const double* Xp = static_cast<double*>(X_buf.ptr);
    const double* yp = static_cast<double*>(y_buf.ptr);

    std::vector<double> W(n_features, 0.0);
    double              b_val = 0.0;

    auto wi_buf = weights_init.request();
    auto bi_buf = bias_init.request();
    if (wi_buf.size == static_cast<size_t>(n_features)) {
        const double* wp = static_cast<double*>(wi_buf.ptr);
        std::copy(wp, wp + n_features, W.begin());
    }
    if (bi_buf.size >= 1) {
        b_val = static_cast<double*>(bi_buf.ptr)[0];
    }

    std::mt19937 rng(random_seed);
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    long long global_step = 1;
    double    best_loss   = 1e18;
    int       no_improve  = 0;
    double    cur_eta     = eta0;
    int       actual_epochs = 0;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        ++actual_epochs;
        if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;

        for (int start = 0; start < n_samples; start += batch_size) {
            int end = std::min(start + batch_size, n_samples);
            int bsz = end - start;

            std::vector<double> grad_W(n_features, 0.0);
            double grad_b = 0.0;
            double batch_loss = 0.0;

            for (int bi = start; bi < end; ++bi) {
                int i = indices[bi];
                const double* xi = Xp + i * n_features;
                double yi = yp[i];

                double score = b_val;
                for (int j = 0; j < n_features; ++j) score += W[j] * xi[j];

                double g = 0.0;
                if (loss_type == static_cast<int>(LossType::SQUARED_ERROR))
                    g = squared_error_grad(score, yi);
                else if (loss_type == static_cast<int>(LossType::HUBER))
                    g = huber_grad(score, yi, epsilon);
                else
                    g = eps_insensitive_grad(score, yi, epsilon);

                grad_b += g;
                for (int j = 0; j < n_features; ++j)
                    grad_W[j] += g * xi[j];
                batch_loss += g * g;
            }

            if (lr_schedule != "adaptive")
                cur_eta = compute_lr(lr_schedule, eta0, alpha, power_t, global_step);
            double scale = cur_eta / bsz;

            b_val -= scale * grad_b;
            for (int j = 0; j < n_features; ++j)
                W[j] -= scale * grad_W[j];
            apply_penalty(W.data(), n_features, penalty_type, alpha, l1_ratio, cur_eta);

            epoch_loss += batch_loss;
            ++global_step;
        }

        epoch_loss = std::sqrt(epoch_loss / n_samples);

        if (epoch_loss < best_loss - tol) {
            best_loss  = epoch_loss;
            no_improve = 0;
        } else {
            ++no_improve;
            if (lr_schedule == "adaptive") cur_eta *= 0.5;
            if (no_improve >= patience) break;
        }
    }

    SGDResult result;
    result.weights = std::move(W);
    result.bias    = {b_val};
    result.n_iter  = actual_epochs;
    return result;
}

// ── Decision function (linear score) ─────────────────────────────────

py::array_t<double> sgd_decision_function(
    py::array_t<double> X,
    py::array_t<double> weights,
    py::array_t<double> bias
) {
    auto X_buf = X.request();
    auto w_buf = weights.request();
    auto b_buf = bias.request();

    int n_samples  = static_cast<int>(X_buf.shape[0]);
    int n_features = static_cast<int>(X_buf.shape[1]);
    const double* Xp = static_cast<double*>(X_buf.ptr);
    const double* Wp = static_cast<double*>(w_buf.ptr);
    const double* bp = static_cast<double*>(b_buf.ptr);

    int n_out = static_cast<int>(b_buf.size);
    // weights shape: (n_out, n_features) or (n_features,) if n_out==1

    py::array_t<double> out;
    if (n_out == 1) {
        out = py::array_t<double>({n_samples});
        double* op = static_cast<double*>(out.request().ptr);
        for (int i = 0; i < n_samples; ++i) {
            double s = bp[0];
            const double* xi = Xp + i * n_features;
            for (int j = 0; j < n_features; ++j) s += Wp[j] * xi[j];
            op[i] = s;
        }
    } else {
        out = py::array_t<double>({n_samples, n_out});
        double* op = static_cast<double*>(out.request().ptr);
        for (int i = 0; i < n_samples; ++i) {
            const double* xi = Xp + i * n_features;
            for (int c = 0; c < n_out; ++c) {
                double s = bp[c];
                const double* Wc = Wp + c * n_features;
                for (int j = 0; j < n_features; ++j) s += Wc[j] * xi[j];
                op[i * n_out + c] = s;
            }
        }
    }
    return out;
}

}  // namespace linear
}  // namespace tuiml
