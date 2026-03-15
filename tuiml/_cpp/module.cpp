#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tree/splitters.h"
#include "tree/criteria.h"
#include "tree/predict.h"
#include "tree/builder.h"
#include "distance/metrics.h"
#include "neighbors/knn.h"
#include "neighbors/kd_tree.h"
#include "neighbors/ball_tree.h"
#include "common/parallel.h"
#include "svm/svc.h"
#include "svm/svr.h"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_ext, m) {
    m.doc() = "TuiML C++ acceleration backend";

    // ── tree submodule ─────────────────────────────────────────────────
    auto tree = m.def_submodule("tree", "Tree algorithm kernels");

    tree.def("best_split_classifier",
             &tuiml::tree::best_split_classifier,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("n_classes"),
             py::arg("min_samples_leaf"), py::arg("random_seed"),
             py::arg("max_features"),
             "Find the best binary split for classification.");

    tree.def("best_split_regressor",
             &tuiml::tree::best_split_regressor,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("min_samples_leaf"),
             py::arg("random_seed"), py::arg("max_features"),
             "Find the best binary split for regression.");

    tree.def("predict_batch",
             &tuiml::tree::predict_batch,
             py::arg("feature"), py::arg("threshold"),
             py::arg("children_left"), py::arg("children_right"),
             py::arg("value"), py::arg("X"),
             "Batch prediction over a flattened tree.");

    tree.def("build_classifier_tree",
             &tuiml::tree::build_classifier_tree,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("n_classes"),
             py::arg("max_depth"), py::arg("min_samples_split"),
             py::arg("min_samples_leaf"), py::arg("min_impurity_decrease"),
             py::arg("random_seed"), py::arg("max_features"),
             "Build a complete classification tree, returning flat arrays.");

    tree.def("build_regressor_tree",
             &tuiml::tree::build_regressor_tree,
             py::arg("X"), py::arg("y"),
             py::arg("criterion"), py::arg("max_depth"),
             py::arg("min_samples_split"), py::arg("min_samples_leaf"),
             py::arg("min_impurity_decrease"), py::arg("random_seed"),
             py::arg("max_features"),
             "Build a complete regression tree, returning flat arrays.");

    // ── distance submodule ─────────────────────────────────────────────
    auto dist = m.def_submodule("distance", "Distance computation kernels");

    dist.def("euclidean", &tuiml::distance::euclidean_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise Euclidean distance matrix.");

    dist.def("manhattan", &tuiml::distance::manhattan_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise Manhattan distance matrix.");

    dist.def("cosine", &tuiml::distance::cosine_distance,
             py::arg("X"), py::arg("Y"),
             "Pairwise cosine distance matrix.");

    // ── neighbors submodule ────────────────────────────────────────────
    auto nn = m.def_submodule("neighbors", "Neighbor search kernels");

    nn.def("brute_knn_query",
           &tuiml::neighbors::brute_knn_query,
           py::arg("X_train"), py::arg("X_query"), py::arg("k"),
           "Batch brute-force k-nearest-neighbor query.");

    nn.def("brute_knn_classify",
           &tuiml::neighbors::brute_knn_classify,
           py::arg("X_train"), py::arg("y_train"), py::arg("X_query"),
           py::arg("k"), py::arg("n_classes"), py::arg("weight_mode"),
           "Brute-force KNN classification (returns int labels).");

    nn.def("brute_knn_regress",
           &tuiml::neighbors::brute_knn_regress,
           py::arg("X_train"), py::arg("y_train"), py::arg("X_query"),
           py::arg("k"), py::arg("weight_mode"),
           "Brute-force KNN regression (returns predicted values).");

    py::class_<tuiml::neighbors::CppKDTree>(nn, "KDTree")
        .def(py::init<int>(), py::arg("leaf_size") = 30)
        .def("build", &tuiml::neighbors::CppKDTree::build,
             py::arg("X"), "Build the KD-tree from training data.")
        .def("query", &tuiml::neighbors::CppKDTree::query,
             py::arg("X_query"), py::arg("k"),
             "Batch k-nearest-neighbor query.")
        .def("num_nodes", &tuiml::neighbors::CppKDTree::num_nodes);

    py::class_<tuiml::neighbors::CppBallTree>(nn, "BallTree")
        .def(py::init<int>(), py::arg("leaf_size") = 30)
        .def("build", &tuiml::neighbors::CppBallTree::build,
             py::arg("X"), "Build the Ball-tree from training data.")
        .def("query", &tuiml::neighbors::CppBallTree::query,
             py::arg("X_query"), py::arg("k"),
             "Batch k-nearest-neighbor query.")
        .def("num_nodes", &tuiml::neighbors::CppBallTree::num_nodes);

    // ── svm submodule ─────────────────────────────────────────────────
    auto svm = m.def_submodule("svm", "SVM solver kernels");

    py::class_<tuiml::svm::SVCModel>(svm, "SVCModel")
        .def_readonly("n_classes", &tuiml::svm::SVCModel::n_classes)
        .def_readonly("classes", &tuiml::svm::SVCModel::classes)
        .def_readonly("n_features", &tuiml::svm::SVCModel::n_features)
        .def_readonly("all_sv_indices", &tuiml::svm::SVCModel::all_sv_indices)
        .def_readonly("all_sv_data", &tuiml::svm::SVCModel::all_sv_data)
        .def_readonly("all_dual_coef", &tuiml::svm::SVCModel::all_dual_coef)
        .def_readonly("intercept", &tuiml::svm::SVCModel::intercept)
        .def_readonly("n_support", &tuiml::svm::SVCModel::n_support);

    py::class_<tuiml::svm::SVRModel>(svm, "SVRModel")
        .def_readonly("sv_indices", &tuiml::svm::SVRModel::sv_indices)
        .def_readonly("dual_coef", &tuiml::svm::SVRModel::dual_coef)
        .def_readonly("rho", &tuiml::svm::SVRModel::rho)
        .def_readonly("n_iter", &tuiml::svm::SVRModel::n_iter)
        .def_readonly("n_features", &tuiml::svm::SVRModel::n_features);

    svm.def("svc_train", &tuiml::svm::svc_train,
            py::arg("X"), py::arg("y"),
            py::arg("kernel_type"), py::arg("C"),
            py::arg("gamma"), py::arg("degree"), py::arg("coef0"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train a C-SVC using native SMO.");

    svm.def("svc_train_precomputed", &tuiml::svm::svc_train_precomputed,
            py::arg("K"), py::arg("y"),
            py::arg("C"), py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train a C-SVC using a precomputed kernel matrix.");

    svm.def("svc_predict", &tuiml::svm::svc_predict,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Predict labels using a trained SVC model.");

    svm.def("svc_predict_precomputed", &tuiml::svm::svc_predict_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Predict labels using a precomputed kernel test matrix.");

    svm.def("svc_decision_function", &tuiml::svm::svc_decision_function,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Compute OvO decision function values.");

    svm.def("svc_decision_function_precomputed",
            &tuiml::svm::svc_decision_function_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Decision function for precomputed kernel.");

    svm.def("svr_train", &tuiml::svm::svr_train,
            py::arg("X"), py::arg("y"),
            py::arg("kernel_type"), py::arg("C"), py::arg("epsilon"),
            py::arg("gamma"), py::arg("degree"), py::arg("coef0"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train an epsilon-SVR using native SMO.");

    svm.def("svr_train_precomputed", &tuiml::svm::svr_train_precomputed,
            py::arg("K"), py::arg("y"),
            py::arg("C"), py::arg("epsilon"),
            py::arg("tol"), py::arg("max_iter"),
            py::arg("cache_mb") = 200,
            py::arg("shrinking") = true,
            "Train an epsilon-SVR using a precomputed kernel matrix.");

    svm.def("svr_predict", &tuiml::svm::svr_predict,
            py::arg("model"), py::arg("X_train"), py::arg("X_test"),
            py::arg("kernel_type"), py::arg("gamma"),
            py::arg("degree"), py::arg("coef0"),
            "Predict values using a trained SVR model.");

    svm.def("svr_predict_precomputed", &tuiml::svm::svr_predict_precomputed,
            py::arg("model"), py::arg("K_test"),
            "Predict values using a precomputed kernel test matrix.");

    // ── utility functions ──────────────────────────────────────────────
    m.def("get_num_threads", &tuiml::get_num_threads,
          "Get current number of OpenMP threads.");
    m.def("set_num_threads", &tuiml::set_num_threads,
          py::arg("n"), "Set number of OpenMP threads.");

    m.attr("__version__") = "0.1.1";

#ifdef TUIML_USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
