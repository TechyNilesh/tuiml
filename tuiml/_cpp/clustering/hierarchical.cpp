#include "clustering/hierarchical.h"

#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tuiml {
namespace clustering {

// ── Min-heap entry ────────────────────────────────────────────────────
struct HeapEntry {
    double dist;
    int i, j;
    bool operator>(const HeapEntry& o) const { return dist > o.dist; }
};

py::array_t<int> hierarchical_fit(
    py::array_t<double> X,
    int                 n_clusters,
    std::string         linkage
) {
    auto Xb = X.request();
    if (Xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    int n = static_cast<int>(Xb.shape[0]);
    int d = static_cast<int>(Xb.shape[1]);
    const double* Xp = static_cast<const double*>(Xb.ptr);

    if (n_clusters < 1 || n_clusters > n)
        throw std::runtime_error("n_clusters out of range");

    // ── Condensed pairwise Euclidean distance matrix ──────────────────
    // D[i,j] (i<j) stored as flat vector; D(i,j) = D[idx(i,j)]
    // idx(i,j) with i<j: i*n - i*(i+1)/2 + j - i - 1
    auto idx = [n](int i, int j) -> size_t {
        if (i > j) std::swap(i, j);
        return static_cast<size_t>(i) * (2 * n - i - 1) / 2 + (j - i - 1);
    };

    size_t npairs = static_cast<size_t>(n) * (n - 1) / 2;
    std::vector<double> D(npairs);

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            const double* a = Xp + i * d;
            const double* b = Xp + j * d;
            for (int k = 0; k < d; ++k) { double v = a[k] - b[k]; s += v * v; }
            D[idx(i, j)] = std::sqrt(s);
        }

    // Cluster sizes and active flags
    std::vector<int>  sz(n, 1);
    std::vector<bool> active(n, true);
    std::vector<int>  label(n);
    std::iota(label.begin(), label.end(), 0);

    // Build initial min-heap of all pairwise distances
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    // heap does not have reserve(); pre-fill via constructor
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            heap.push({D[idx(i, j)], i, j});

    int n_active = n;

    while (n_active > n_clusters) {
        // ── Pop minimum valid entry ───────────────────────────────────
        HeapEntry top{};
        while (!heap.empty()) {
            top = heap.top(); heap.pop();
            if (active[top.i] && active[top.j]) break;
        }
        if (!active[top.i] || !active[top.j]) break;

        int a = top.i, b = top.j;
        double dist_ab = top.dist;
        int na = sz[a], nb = sz[b];

        // Merge b into a
        active[b] = false;
        --n_active;
        for (int i = 0; i < n; ++i) if (label[i] == b) label[i] = a;

        sz[a] = na + nb;

        // ── Lance-Williams update: push new distances for cluster a ───
        for (int k = 0; k < n; ++k) {
            if (!active[k] || k == a) continue;
            double d_ak = D[idx(a, k)];
            double d_bk = D[idx(b, k)];
            int nk = sz[k];
            double new_dist;

            if (linkage == "single") {
                new_dist = (d_ak < d_bk) ? d_ak : d_bk;
            } else if (linkage == "complete") {
                new_dist = (d_ak > d_bk) ? d_ak : d_bk;
            } else if (linkage == "average") {
                new_dist = (d_ak * na + d_bk * nb) / (na + nb);
            } else {
                // Ward
                double n_total = static_cast<double>(na + nb + nk);
                double sq = ((na + nk) * d_ak * d_ak
                           + (nb + nk) * d_bk * d_bk
                           - nk * dist_ab * dist_ab) / n_total;
                new_dist = (sq > 0.0) ? std::sqrt(sq) : 0.0;
            }

            // Update condensed matrix and push new entry
            int lo = (a < k) ? a : k, hi = (a < k) ? k : a;
            D[idx(lo, hi)] = new_dist;
            heap.push({new_dist, lo, hi});
        }
    }

    // ── Compact labels to 0..n_clusters-1 ────────────────────────────
    std::vector<int> reps;
    reps.reserve(n_clusters);
    for (int i = 0; i < n; ++i) if (active[i]) reps.push_back(i);

    std::vector<int> rep_map(n, -1);
    for (int ci = 0; ci < static_cast<int>(reps.size()); ++ci)
        rep_map[reps[ci]] = ci;

    auto out = py::array_t<int>(n);
    int* outp = static_cast<int*>(out.request().ptr);
    for (int i = 0; i < n; ++i) outp[i] = rep_map[label[i]];

    return out;
}

} // namespace clustering
} // namespace tuiml
