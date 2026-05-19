#include "clustering/em.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tuiml {
namespace clustering {

static constexpr double LOG_2PI = 1.8378770664093453;  // log(2*pi)

// ── Cholesky: L lower-triangular s.t. A = L Lᵀ ───────────────────────
static bool cholesky(const double* A, double* L, int d) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < i; ++j) {
            double s = A[i * d + j];
            for (int k = 0; k < j; ++k) s -= L[i * d + k] * L[j * d + k];
            L[i * d + j] = s / L[j * d + j];
            L[j * d + i] = 0.0;
        }
        double s = A[i * d + i];
        for (int k = 0; k < i; ++k) s -= L[i * d + k] * L[i * d + k];
        if (s <= 0.0) return false;
        L[i * d + i] = std::sqrt(s);
    }
    return true;
}

// ── Forward-substitution: solve Lx = b in-place ──────────────────────
static void fwd_sub(const double* L, double* x, int d) {
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < i; ++j) x[i] -= L[i * d + j] * x[j];
        x[i] /= L[i * d + i];
    }
}

// ── log|L| = sum of log(diagonal) ────────────────────────────────────
static double log_det_L(const double* L, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) s += std::log(L[i * d + i]);
    return s;
}

// ── E-step ────────────────────────────────────────────────────────────
// Fills log_resp (n, K) and returns mean log-likelihood.
// diff_buf: pre-allocated scratch of size d (reused across all i,k).
static double e_step(
    const double* X, int n, int d, int K,
    const double* log_w,
    const double* means,
    const double* covs,
    const std::string& cov_type,
    double* log_resp,
    const std::vector<std::vector<double>>& chol,
    const std::vector<double>& log_det_L_vec,
    double* diff_buf        // scratch (d,)
) {
    double total_ll = 0.0;

    for (int i = 0; i < n; ++i) {
        const double* xi = X + i * d;
        double max_lp = -std::numeric_limits<double>::infinity();

        for (int k = 0; k < K; ++k) {
            const double* mk = means + k * d;
            double lp;

            if (cov_type == "full") {
                for (int j = 0; j < d; ++j) diff_buf[j] = xi[j] - mk[j];
                fwd_sub(chol[k].data(), diff_buf, d);
                double maha = 0.0;
                for (int j = 0; j < d; ++j) maha += diff_buf[j] * diff_buf[j];
                lp = log_w[k] - 0.5 * (d * LOG_2PI + 2.0 * log_det_L_vec[k] + maha);
            } else if (cov_type == "diag") {
                const double* vk = covs + k * d;
                double s = 0.0, log_det = 0.0;
                for (int j = 0; j < d; ++j) {
                    double diff = xi[j] - mk[j];
                    s += diff * diff / vk[j];
                    log_det += std::log(vk[j]);
                }
                lp = log_w[k] - 0.5 * (d * LOG_2PI + log_det + s);
            } else {  // spherical
                double var = covs[k];
                double s = 0.0;
                for (int j = 0; j < d; ++j) { double v = xi[j] - mk[j]; s += v * v; }
                lp = log_w[k] - 0.5 * (d * (LOG_2PI + std::log(var)) + s / var);
            }

            log_resp[i * K + k] = lp;
            if (lp > max_lp) max_lp = lp;
        }

        // log-sum-exp normalisation
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) sum_exp += std::exp(log_resp[i * K + k] - max_lp);
        double log_norm = max_lp + std::log(sum_exp);
        total_ll += log_norm;
        for (int k = 0; k < K; ++k) log_resp[i * K + k] -= log_norm;
    }
    return total_ll / n;
}

// ── M-step ────────────────────────────────────────────────────────────
// resp_buf: pre-allocated (n, K) scratch for exp(log_resp).
static void m_step(
    const double* X, int n, int d, int K,
    const double* log_resp,
    const std::string& cov_type,
    double reg_covar,
    double* log_w,
    double* means,
    double* covs,
    std::vector<std::vector<double>>& chol,
    std::vector<double>& log_det_L_vec,
    double* resp_buf        // scratch (n, K)
) {
    // Materialise responsibilities
    std::vector<double> Nk(K, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < K; ++k) {
            double r = std::exp(log_resp[i * K + k]);
            resp_buf[i * K + k] = r;
            Nk[k] += r;
        }

    // Weights
    for (int k = 0; k < K; ++k)
        log_w[k] = std::log(std::max(Nk[k] / n, 1e-300));

    // Means: mk = (sum_i r[i,k] * xi) / Nk
    std::fill(means, means + K * d, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* xi = X + i * d;
        for (int k = 0; k < K; ++k) {
            double r = resp_buf[i * K + k];
            double* mk = means + k * d;
            for (int j = 0; j < d; ++j) mk[j] += r * xi[j];
        }
    }
    for (int k = 0; k < K; ++k) {
        double inv_Nk = 1.0 / std::max(Nk[k], 1e-10);
        double* mk = means + k * d;
        for (int j = 0; j < d; ++j) mk[j] *= inv_Nk;
    }

    // Covariances
    if (cov_type == "full") {
        // For each k: Sk = sum_i r[i,k] * (xi-mk)(xi-mk)^T / Nk
        // Implemented via: form wd = sqrt(r[i,k])*(xi-mk), then Sk = wd^T wd / Nk
        // We accumulate Sk directly with symmetric outer products.
        for (int k = 0; k < K; ++k) {
            double* Sk = covs + k * d * d;
            std::fill(Sk, Sk + d * d, 0.0);
            const double* mk = means + k * d;
            double inv_Nk = 1.0 / std::max(Nk[k], 1e-10);

            for (int i = 0; i < n; ++i) {
                const double* xi = X + i * d;
                double r = resp_buf[i * K + k] * inv_Nk;
                // Upper/lower triangular outer product (unrolled for cache)
                for (int a = 0; a < d; ++a) {
                    double da = r * (xi[a] - mk[a]);
                    for (int b = 0; b <= a; ++b) {
                        double val = da * (xi[b] - mk[b]);
                        Sk[a * d + b] += val;
                    }
                }
            }
            // Symmetrise and add regularisation
            for (int a = 0; a < d; ++a) {
                Sk[a * d + a] += reg_covar;
                for (int b = 0; b < a; ++b) Sk[b * d + a] = Sk[a * d + b];
            }

            chol[k].assign(d * d, 0.0);
            if (!cholesky(Sk, chol[k].data(), d)) {
                for (int j = 0; j < d; ++j) Sk[j * d + j] += 1e-6;
                cholesky(Sk, chol[k].data(), d);
            }
            log_det_L_vec[k] = log_det_L(chol[k].data(), d);
        }
    } else if (cov_type == "diag") {
        for (int k = 0; k < K; ++k) {
            double* vk = covs + k * d;
            std::fill(vk, vk + d, 0.0);
            const double* mk = means + k * d;
            double inv_Nk = 1.0 / std::max(Nk[k], 1e-10);
            for (int i = 0; i < n; ++i) {
                const double* xi = X + i * d;
                double r = resp_buf[i * K + k] * inv_Nk;
                for (int j = 0; j < d; ++j) { double v = xi[j] - mk[j]; vk[j] += r * v * v; }
            }
            for (int j = 0; j < d; ++j) vk[j] += reg_covar;
            double ld = 0.0;
            for (int j = 0; j < d; ++j) ld += 0.5 * std::log(vk[j]);
            log_det_L_vec[k] = ld;
        }
    } else {  // spherical
        for (int k = 0; k < K; ++k) {
            const double* mk = means + k * d;
            double s = 0.0, inv_Nk = 1.0 / std::max(Nk[k], 1e-10);
            for (int i = 0; i < n; ++i) {
                const double* xi = X + i * d;
                double r = resp_buf[i * K + k] * inv_Nk;
                double maha = 0.0;
                for (int j = 0; j < d; ++j) { double v = xi[j] - mk[j]; maha += v * v; }
                s += r * maha;
            }
            covs[k] = s / d + reg_covar;
            log_det_L_vec[k] = 0.5 * d * std::log(covs[k]);
        }
    }
}

// ── Public fit ────────────────────────────────────────────────────────
EMResult em_fit(
    py::array_t<double> X,
    int                 n_components,
    std::string         covariance_type,
    int                 max_iter,
    double              tol,
    double              reg_covar,
    int                 n_init,
    int                 random_seed
) {
    auto Xb = X.request();
    if (Xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    int n = static_cast<int>(Xb.shape[0]);
    int d = static_cast<int>(Xb.shape[1]);
    const double* Xp = static_cast<const double*>(Xb.ptr);
    int K = n_components;
    if (K < 1 || K > n) throw std::runtime_error("n_components out of range");

    int cov_size = (covariance_type == "full") ? d * d :
                   (covariance_type == "diag") ? d : 1;

    std::mt19937 rng(random_seed);

    // Pre-allocate scratch buffers shared across all inits/iterations
    std::vector<double> log_resp(static_cast<size_t>(n) * K);
    std::vector<double> resp_buf(static_cast<size_t>(n) * K);
    std::vector<double> diff_buf(d);

    EMResult best;
    best.log_likelihood = -std::numeric_limits<double>::infinity();
    best.n_iter = 0;

    for (int init = 0; init < n_init; ++init) {
        std::vector<double> log_w(K), means(K * d), covs(K * cov_size);
        std::vector<std::vector<double>> chol(K);
        std::vector<double> log_det_L_vec(K, 0.0);

        // ── k-means++ initialisation ──────────────────────────────────
        std::uniform_int_distribution<int> uni(0, n - 1);
        std::vector<int> center_idx;
        center_idx.push_back(uni(rng));

        std::vector<double> min_dist_sq(n, std::numeric_limits<double>::infinity());
        for (int c = 0; c < K; ++c) {
            if (c > 0) {
                double total = 0.0;
                for (double v : min_dist_sq) total += v;
                std::uniform_real_distribution<double> udist(0.0, total);
                double r = udist(rng);
                double cum = 0.0;
                int chosen = 0;
                for (int i = 0; i < n; ++i) {
                    cum += min_dist_sq[i];
                    if (cum >= r) { chosen = i; break; }
                }
                center_idx.push_back(chosen);
            }
            const double* cc = Xp + center_idx.back() * d;
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                const double* xi = Xp + i * d;
                for (int j = 0; j < d; ++j) { double v = xi[j] - cc[j]; s += v * v; }
                if (s < min_dist_sq[i]) min_dist_sq[i] = s;
            }
        }
        for (int k = 0; k < K; ++k) {
            const double* src = Xp + center_idx[k] * d;
            std::copy(src, src + d, means.data() + k * d);
        }

        double log_w0 = std::log(1.0 / K);
        std::fill(log_w.begin(), log_w.end(), log_w0);

        if (covariance_type == "full") {
            for (int k = 0; k < K; ++k) {
                double* Sk = covs.data() + k * d * d;
                std::fill(Sk, Sk + d * d, 0.0);
                for (int j = 0; j < d; ++j) Sk[j * d + j] = 1.0 + reg_covar;
                chol[k].assign(d * d, 0.0);
                cholesky(Sk, chol[k].data(), d);
                log_det_L_vec[k] = log_det_L(chol[k].data(), d);
            }
        } else if (covariance_type == "diag") {
            for (int k = 0; k < K; ++k) {
                double* vk = covs.data() + k * d;
                std::fill(vk, vk + d, 1.0 + reg_covar);
                log_det_L_vec[k] = 0.5 * d * std::log(1.0 + reg_covar);
            }
        } else {
            std::fill(covs.begin(), covs.end(), 1.0 + reg_covar);
            double ld = 0.5 * d * std::log(1.0 + reg_covar);
            std::fill(log_det_L_vec.begin(), log_det_L_vec.end(), ld);
        }

        // ── EM loop ───────────────────────────────────────────────────
        double prev_ll = -std::numeric_limits<double>::infinity();
        int actual_iter = 0;

        for (int it = 0; it < max_iter; ++it) {
            ++actual_iter;
            double ll = e_step(Xp, n, d, K, log_w.data(), means.data(), covs.data(),
                               covariance_type, log_resp.data(), chol, log_det_L_vec,
                               diff_buf.data());
            m_step(Xp, n, d, K, log_resp.data(), covariance_type, reg_covar,
                   log_w.data(), means.data(), covs.data(), chol, log_det_L_vec,
                   resp_buf.data());
            if (std::abs(ll - prev_ll) < tol) break;
            prev_ll = ll;
        }

        double final_ll = e_step(Xp, n, d, K, log_w.data(), means.data(), covs.data(),
                                 covariance_type, log_resp.data(), chol, log_det_L_vec,
                                 diff_buf.data());

        if (final_ll > best.log_likelihood) {
            best.log_likelihood = final_ll;
            best.n_iter = actual_iter;
            best.cov_type = covariance_type;
            best.weights.resize(K);
            for (int k = 0; k < K; ++k) best.weights[k] = std::exp(log_w[k]);
            best.means = std::move(means);
            best.covariances = std::move(covs);
        }
    }

    return best;
}

// ── Predict labels ────────────────────────────────────────────────────
py::array_t<int> em_predict(
    py::array_t<double> X,
    py::array_t<double> weights,
    py::array_t<double> means_arr,
    py::array_t<double> covariances_arr,
    std::string         covariance_type
) {
    auto lr = em_log_resp(X, weights, means_arr, covariances_arr, covariance_type);
    auto Xb = X.request();
    auto lb = lr.request();
    int n = static_cast<int>(Xb.shape[0]);
    int K = static_cast<int>(lb.shape[1]);
    const double* lrp = static_cast<const double*>(lb.ptr);

    auto out = py::array_t<int>(n);
    int* outp = static_cast<int*>(out.request().ptr);
    for (int i = 0; i < n; ++i) {
        int best = 0; double best_v = lrp[i * K];
        for (int k = 1; k < K; ++k) if (lrp[i * K + k] > best_v) { best_v = lrp[i * K + k]; best = k; }
        outp[i] = best;
    }
    return out;
}

// ── Log responsibilities ──────────────────────────────────────────────
py::array_t<double> em_log_resp(
    py::array_t<double> X,
    py::array_t<double> weights,
    py::array_t<double> means_arr,
    py::array_t<double> covariances_arr,
    std::string         covariance_type
) {
    auto Xb = X.request();
    auto wb = weights.request();
    auto mb = means_arr.request();
    auto cb = covariances_arr.request();

    int n = static_cast<int>(Xb.shape[0]);
    int d = static_cast<int>(Xb.shape[1]);
    int K = static_cast<int>(wb.shape[0]);
    const double* Xp = static_cast<const double*>(Xb.ptr);
    const double* wp = static_cast<const double*>(wb.ptr);
    const double* mp = static_cast<const double*>(mb.ptr);
    const double* cp = static_cast<const double*>(cb.ptr);

    std::vector<double> log_w(K);
    for (int k = 0; k < K; ++k) log_w[k] = std::log(std::max(wp[k], 1e-300));

    std::vector<std::vector<double>> chol(K);
    std::vector<double> log_det_L_vec(K, 0.0);
    if (covariance_type == "full") {
        for (int k = 0; k < K; ++k) {
            chol[k].assign(d * d, 0.0);
            cholesky(cp + k * d * d, chol[k].data(), d);
            log_det_L_vec[k] = log_det_L(chol[k].data(), d);
        }
    } else if (covariance_type == "diag") {
        for (int k = 0; k < K; ++k) {
            double ld = 0.0;
            const double* vk = cp + k * d;
            for (int j = 0; j < d; ++j) ld += 0.5 * std::log(vk[j]);
            log_det_L_vec[k] = ld;
        }
    } else {
        for (int k = 0; k < K; ++k) log_det_L_vec[k] = 0.5 * d * std::log(cp[k]);
    }

    auto out = py::array_t<double>({n, K});
    double* outp = static_cast<double*>(out.request().ptr);
    std::vector<double> diff_buf(d);

    e_step(Xp, n, d, K, log_w.data(), mp, cp, covariance_type,
           outp, chol, log_det_L_vec, diff_buf.data());

    return out;
}

} // namespace clustering
} // namespace tuiml
