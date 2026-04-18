#!/usr/bin/env python3
"""
TuiML vs Sklearn Comprehensive Benchmark

Side-by-side comparison of algorithms across:
- Classification (multiple datasets)
- Regression (multiple datasets)
- Clustering (multiple datasets)

Results saved to benchmark_results.json and benchmark_results.csv

Usage:
    cd tuiml/
    python benchmark_algorithms.py
"""

import os
import sys
import time
import tracemalloc
import json
import csv
import signal
import warnings
from datetime import datetime
from typing import Dict, List, Any, Callable, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings('ignore')

# Flush stdout after every print so progress is visible in real-time
import functools
print = functools.partial(print, flush=True)

# Per-algorithm timeout in seconds (skip if exceeded)
ALGORITHM_TIMEOUT = 300


class AlgorithmTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise AlgorithmTimeout("Algorithm exceeded time limit")

# ============================================================================
# DATASETS CONFIGURATION (10k+ samples for credible benchmarks)
# ============================================================================

CLASSIFICATION_DATASETS = [
    # (name, loader_func, params, n_samples, n_features, n_classes)
    ("synthetic_10k", "sklearn.datasets:make_classification",
     {"n_samples": 10000, "n_features": 20, "n_classes": 5,
      "n_informative": 15, "n_redundant": 3, "random_state": 42},
     10000, 20, 5),
]

REGRESSION_DATASETS = [
    # (name, loader_func, params, n_samples, n_features)
    ("synthetic_10k", "sklearn.datasets:make_regression",
     {"n_samples": 10000, "n_features": 20, "n_informative": 15,
      "noise": 10.0, "random_state": 42},
     10000, 20),
]

MAX_REGRESSION_SAMPLES = 10000

CLUSTERING_DATASETS = [
    # (name, generator_func, params)
    ("blobs_10k", "sklearn.datasets:make_blobs",
     {"n_samples": 10000, "n_features": 10, "centers": 5, "random_state": 42}),
]

# ============================================================================
# ALGORITHM PAIRS: Classification
# ============================================================================

CLASSIFICATION_ALGORITHMS = [
    # (Category, TuiML Config, Sklearn Config)
    (
        "Naive Bayes",
        {"name": "NaiveBayesClassifier", "module": "tuiml.algorithms", "params": {}},
        {"name": "GaussianNB", "module": "sklearn.naive_bayes", "class": "GaussianNB", "params": {}},
    ),
    (
        "Decision Tree",
        {"name": "C45TreeClassifier", "module": "tuiml.algorithms", "params": {}},
        {"name": "DecisionTreeClassifier", "module": "sklearn.tree", "class": "DecisionTreeClassifier", "params": {"random_state": 42}},
    ),
    (
        "Random Forest",
        {"name": "RandomForestClassifier", "module": "tuiml.algorithms", "params": {"n_estimators": 10, "random_state": 42}},
        {"name": "RandomForestClassifier", "module": "sklearn.ensemble", "class": "RandomForestClassifier", "params": {"n_estimators": 10, "random_state": 42}},
    ),
    (
        "K-Nearest Neighbors",
        {"name": "KNearestNeighborsClassifier", "module": "tuiml.algorithms", "params": {"k": 5}},
        {"name": "KNeighborsClassifier", "module": "sklearn.neighbors", "class": "KNeighborsClassifier", "params": {"n_neighbors": 5}},
    ),
    (
        "Logistic Regression",
        {"name": "LogisticRegression", "module": "tuiml.algorithms", "params": {}},
        {"name": "LogisticRegression", "module": "sklearn.linear_model", "class": "LogisticRegression", "params": {"max_iter": 200, "random_state": 42}},
    ),
    (
        "SVM",
        {"name": "SVC", "module": "tuiml.algorithms", "params": {}},
        {"name": "SVC", "module": "sklearn.svm", "class": "SVC", "params": {"random_state": 42}},
    ),
    (
        "Neural Network (MLP)",
        {"name": "MultilayerPerceptronClassifier", "module": "tuiml.algorithms", "params": {}},
        {"name": "MLPClassifier", "module": "sklearn.neural_network", "class": "MLPClassifier", "params": {"hidden_layer_sizes": (50,), "max_iter": 200, "random_state": 42}},
    ),
    (
        "AdaBoost",
        {"name": "AdaBoostClassifier", "module": "tuiml.algorithms", "params": {}},
        {"name": "AdaBoostClassifier", "module": "sklearn.ensemble", "class": "AdaBoostClassifier", "params": {"random_state": 42}},
    ),
    (
        "Gradient Boosting",
        {"name": "LogitBoostClassifier", "module": "tuiml.algorithms", "params": {}},
        {"name": "GradientBoostingClassifier", "module": "sklearn.ensemble", "class": "GradientBoostingClassifier", "params": {"random_state": 42}},
    ),
]

# ============================================================================
# ALGORITHM PAIRS: Regression
# ============================================================================

REGRESSION_ALGORITHMS = [
    # (Category, TuiML Config, Sklearn Config)
    (
        "Linear Regression",
        {"name": "LinearRegression", "module": "tuiml.algorithms", "params": {}},
        {"name": "LinearRegression", "module": "sklearn.linear_model", "class": "LinearRegression", "params": {}},
    ),
    (
        "Ridge Regression",
        {"name": "LinearRegression", "module": "tuiml.algorithms", "params": {"ridge": 1.0}},
        {"name": "Ridge", "module": "sklearn.linear_model", "class": "Ridge", "params": {"alpha": 1.0}},
    ),
    (
        "Decision Tree Regressor",
        {"name": "M5ModelTreeRegressor", "module": "tuiml.algorithms", "params": {"min_samples_leaf": 2}},
        {"name": "DecisionTreeRegressor", "module": "sklearn.tree", "class": "DecisionTreeRegressor", "params": {"random_state": 42}},
    ),
    (
        "Random Forest Regressor",
        {"name": "RandomForestRegressor", "module": "tuiml.algorithms", "params": {"n_estimators": 10, "random_state": 42, "max_depth": 10}},
        {"name": "RandomForestRegressor", "module": "sklearn.ensemble", "class": "RandomForestRegressor", "params": {"n_estimators": 10, "random_state": 42, "max_depth": 10}},
    ),
    (
        "SGD Regressor",
        {"name": "SGDRegressor", "module": "tuiml.algorithms", "params": {}},
        {"name": "SGDRegressor", "module": "sklearn.linear_model", "class": "SGDRegressor", "params": {"random_state": 42, "max_iter": 1000}},
    ),
    (
        "K-Neighbors Regressor",
        {"name": "KNearestNeighborsRegressor", "module": "tuiml.algorithms", "params": {"k": 5}},
        {"name": "KNeighborsRegressor", "module": "sklearn.neighbors", "class": "KNeighborsRegressor", "params": {"n_neighbors": 5}},
    ),
    (
        "SVR",
        {"name": "SVR", "module": "tuiml.algorithms", "params": {}},
        {"name": "SVR", "module": "sklearn.svm", "class": "SVR", "params": {}},
    ),
]

# ============================================================================
# ALGORITHM PAIRS: Clustering
# ============================================================================

# Max samples for O(n^3) clustering algorithms (agglomerative) — tuiml's
# implementation is pure Python, so cap at 2000 to keep benchmarks viable.
MAX_CLUSTERING_SAMPLES_CUBIC = 2000

CLUSTERING_ALGORITHMS = [
    # (Category, TuiML Config, Sklearn Config, max_samples_override_or_None)
    (
        "K-Means",
        {"name": "KMeansClusterer", "module": "tuiml.algorithms", "params": {"n_clusters": 4, "random_state": 42}},
        {"name": "KMeans", "module": "sklearn.cluster", "class": "KMeans", "params": {"n_clusters": 4, "random_state": 42, "n_init": 10}},
        None,
    ),
    (
        "DBSCAN",
        {"name": "DBSCANClusterer", "module": "tuiml.algorithms", "params": {"eps": 0.3, "min_samples": 5}},
        {"name": "DBSCAN", "module": "sklearn.cluster", "class": "DBSCAN", "params": {"eps": 0.3, "min_samples": 5}},
        None,
    ),
    (
        "Hierarchical",
        {"name": "AgglomerativeClusterer", "module": "tuiml.algorithms", "params": {"n_clusters": 4}},
        {"name": "AgglomerativeClustering", "module": "sklearn.cluster", "class": "AgglomerativeClustering", "params": {"n_clusters": 4}},
        MAX_CLUSTERING_SAMPLES_CUBIC,
    ),
    (
        "Gaussian Mixture (EM)",
        {"name": "GaussianMixtureClusterer", "module": "tuiml.algorithms", "params": {"n_components": 4, "random_state": 42}},
        {"name": "GaussianMixture", "module": "sklearn.mixture", "class": "GaussianMixture", "params": {"n_components": 4, "random_state": 42}},
        None,
    ),
]


def import_class(module_name: str, class_name: str):
    """Dynamically import a class from a module."""
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_dataset(loader_str: str, **kwargs):
    """Load dataset from loader string (module:function)."""
    module_name, func_name = loader_str.split(":")
    import importlib
    module = importlib.import_module(module_name)
    loader = getattr(module, func_name)
    data = loader(**kwargs)
    if hasattr(data, 'data') and hasattr(data, 'target'):
        return data.data, data.target
    return data


def measure_memory(func: Callable):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024  # MB


def manual_cross_val_classification(clf_class, params: Dict, X: np.ndarray, y: np.ndarray, cv: int = 5) -> List[float]:
    """Manual cross-validation for classification."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_class(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    return scores


def manual_cross_val_regression(reg_class, params: Dict, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[List[float], List[float]]:
    """Manual cross-validation for regression. Returns (r2_scores, rmse_scores)."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        reg = reg_class(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2)
        rmse_scores.append(rmse)

    return r2_scores, rmse_scores


def benchmark_classifier(
    name: str,
    clf_class,
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Benchmark a single classifier."""
    result = {
        "name": name,
        "accuracy": None,
        "accuracy_std": None,
        "train_time_sec": None,
        "memory_mb": None,
        "status": "success",
        "error": None
    }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        def train_func():
            clf = clf_class(**params)
            clf.fit(X_train, y_train)
            return clf

        start_time = time.perf_counter()
        trained_clf, memory_mb = measure_memory(train_func)
        train_time = time.perf_counter() - start_time

        result["train_time_sec"] = round(train_time, 6)
        result["memory_mb"] = round(memory_mb, 4)

        cv_scores = manual_cross_val_classification(clf_class, params, X, y, cv=cv_folds)
        result["accuracy"] = round(np.mean(cv_scores), 4)
        result["accuracy_std"] = round(np.std(cv_scores), 4)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:100]

    return result


def benchmark_regressor(
    name: str,
    reg_class,
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Benchmark a single regressor."""
    result = {
        "name": name,
        "r2_score": None,
        "r2_std": None,
        "rmse": None,
        "rmse_std": None,
        "train_time_sec": None,
        "memory_mb": None,
        "status": "success",
        "error": None
    }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        def train_func():
            reg = reg_class(**params)
            reg.fit(X_train, y_train)
            return reg

        start_time = time.perf_counter()
        trained_reg, memory_mb = measure_memory(train_func)
        train_time = time.perf_counter() - start_time

        result["train_time_sec"] = round(train_time, 6)
        result["memory_mb"] = round(memory_mb, 4)

        r2_scores, rmse_scores = manual_cross_val_regression(reg_class, params, X, y, cv=cv_folds)
        result["r2_score"] = round(np.mean(r2_scores), 4)
        result["r2_std"] = round(np.std(r2_scores), 4)
        result["rmse"] = round(np.mean(rmse_scores), 4)
        result["rmse_std"] = round(np.std(rmse_scores), 4)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:100]

    return result


def benchmark_clusterer(
    name: str,
    clusterer_class,
    params: Dict,
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Benchmark a single clusterer."""
    result = {
        "name": name,
        "silhouette": None,
        "ari": None,
        "train_time_sec": None,
        "memory_mb": None,
        "status": "success",
        "error": None
    }

    try:
        def train_func():
            clusterer = clusterer_class(**params)
            if hasattr(clusterer, 'fit_predict'):
                labels = clusterer.fit_predict(X)
            else:
                clusterer.fit(X)
                if hasattr(clusterer, 'labels_'):
                    labels = clusterer.labels_
                elif hasattr(clusterer, 'predict'):
                    labels = clusterer.predict(X)
                else:
                    labels = None
            return clusterer, labels

        start_time = time.perf_counter()
        (trained_clusterer, labels), memory_mb = measure_memory(train_func)
        train_time = time.perf_counter() - start_time

        result["train_time_sec"] = round(train_time, 6)
        result["memory_mb"] = round(memory_mb, 4)

        if labels is not None:
            # Filter out noise points (label=-1) for silhouette
            mask = labels >= 0
            if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
                result["silhouette"] = round(silhouette_score(X[mask], labels[mask]), 4)

            # ARI needs ground truth
            if y_true is not None:
                result["ari"] = round(adjusted_rand_score(y_true, labels), 4)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:100]

    return result


def run_classification_benchmark() -> List[Dict]:
    """Run classification benchmarks across all datasets."""
    all_results = []

    print("\n" + "=" * 100)
    print("CLASSIFICATION BENCHMARKS")
    print("=" * 100)

    for dataset_name, loader, loader_params, n_samples, n_features, n_classes in CLASSIFICATION_DATASETS:
        print(f"\n{'─' * 100}")
        print(f"  Dataset: {dataset_name} ({n_samples} samples, {n_features} features, {n_classes} classes)")
        print(f"{'─' * 100}")

        try:
            X, y = load_dataset(loader, **loader_params)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            continue

        for category, weka_config, sklearn_config in CLASSIFICATION_ALGORITHMS:
            print(f"\n  [{category}]")
            # Benchmark TuiML
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                clf_class = import_class(weka_config["module"], weka_config["name"])
                weka_result = benchmark_classifier(weka_config["name"], clf_class, weka_config["params"], X_scaled, y)
                signal.alarm(0)
                weka_result["source"] = "tuiml"
                weka_result["category"] = category
                weka_result["dataset"] = dataset_name
                weka_result["task"] = "classification"
                all_results.append(weka_result)

                status = f"Acc={weka_result['accuracy']:.4f} ({weka_result['train_time_sec']:.2f}s)" if weka_result["status"] == "success" else f"FAILED"
                print(f"    TuiML  {weka_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: ERROR - {str(e)[:60]}")

            # Benchmark sklearn
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                clf_class = import_class(sklearn_config["module"], sklearn_config["class"])
                sklearn_result = benchmark_classifier(sklearn_config["name"], clf_class, sklearn_config["params"], X_scaled, y)
                signal.alarm(0)
                sklearn_result["source"] = "sklearn"
                sklearn_result["category"] = category
                sklearn_result["dataset"] = dataset_name
                sklearn_result["task"] = "classification"
                all_results.append(sklearn_result)

                status = f"Acc={sklearn_result['accuracy']:.4f} ({sklearn_result['train_time_sec']:.2f}s)" if sklearn_result["status"] == "success" else f"FAILED"
                print(f"    sklearn {sklearn_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: ERROR - {str(e)[:60]}")

    return all_results


def run_regression_benchmark() -> List[Dict]:
    """Run regression benchmarks across all datasets."""
    all_results = []

    print("\n" + "=" * 100)
    print("REGRESSION BENCHMARKS")
    print("=" * 100)

    for dataset_name, loader, loader_params, n_samples, n_features in REGRESSION_DATASETS:
        print(f"\n{'─' * 100}")
        print(f"  Dataset: {dataset_name} ({n_samples} samples, {n_features} features)")
        print(f"{'─' * 100}")

        try:
            X, y = load_dataset(loader, **loader_params)
            
            # Subsample large datasets for pure-Python implementations
            actual_samples = X.shape[0]
            if actual_samples > MAX_REGRESSION_SAMPLES:
                rng = np.random.RandomState(42)
                indices = rng.choice(actual_samples, MAX_REGRESSION_SAMPLES, replace=False)
                X = X[indices]
                y = y[indices]
                print(f"  (Note: Subsampled from {actual_samples} to {MAX_REGRESSION_SAMPLES} samples)")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            continue

        for category, weka_config, sklearn_config in REGRESSION_ALGORITHMS:
            print(f"\n  [{category}]")
            # Benchmark TuiML
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                reg_class = import_class(weka_config["module"], weka_config["name"])
                weka_result = benchmark_regressor(weka_config["name"], reg_class, weka_config["params"], X_scaled, y)
                signal.alarm(0)
                weka_result["source"] = "tuiml"
                weka_result["category"] = category
                weka_result["dataset"] = dataset_name
                weka_result["task"] = "regression"
                all_results.append(weka_result)

                status = f"R²={weka_result['r2_score']:.4f} ({weka_result['train_time_sec']:.2f}s)" if weka_result["status"] == "success" else f"FAILED"
                print(f"    TuiML  {weka_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: ERROR - {str(e)[:60]}")

            # Benchmark sklearn
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                reg_class = import_class(sklearn_config["module"], sklearn_config["class"])
                sklearn_result = benchmark_regressor(sklearn_config["name"], reg_class, sklearn_config["params"], X_scaled, y)
                signal.alarm(0)
                sklearn_result["source"] = "sklearn"
                sklearn_result["category"] = category
                sklearn_result["dataset"] = dataset_name
                sklearn_result["task"] = "regression"
                all_results.append(sklearn_result)

                status = f"R²={sklearn_result['r2_score']:.4f} ({sklearn_result['train_time_sec']:.2f}s)" if sklearn_result["status"] == "success" else f"FAILED"
                print(f"    sklearn {sklearn_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: ERROR - {str(e)[:60]}")

    return all_results


def run_clustering_benchmark() -> List[Dict]:
    """Run clustering benchmarks across all datasets."""
    all_results = []

    print("\n" + "=" * 100)
    print("CLUSTERING BENCHMARKS")
    print("=" * 100)

    for dataset_name, loader, params in CLUSTERING_DATASETS:
        print(f"\n{'─' * 100}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'─' * 100}")

        try:
            result = load_dataset(loader, **params)
            if isinstance(result, tuple):
                X, y_true = result
            else:
                X, y_true = result, None
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            continue

        for category, weka_config, sklearn_config, max_samples_override in CLUSTERING_ALGORITHMS:
            print(f"\n  [{category}]")

            # Subsample for O(n^3) algorithms
            if max_samples_override and X_scaled.shape[0] > max_samples_override:
                rng_sub = np.random.RandomState(42)
                idx = rng_sub.choice(X_scaled.shape[0], max_samples_override, replace=False)
                X_algo = X_scaled[idx]
                y_algo = y_true[idx] if y_true is not None else None
                print(f"    (Subsampled to {max_samples_override} for O(n³) algorithm)")
            else:
                X_algo = X_scaled
                y_algo = y_true
            # Benchmark TuiML
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                clusterer_class = import_class(weka_config["module"], weka_config["name"])
                weka_result = benchmark_clusterer(weka_config["name"], clusterer_class, weka_config["params"], X_algo, y_algo)
                signal.alarm(0)
                weka_result["source"] = "tuiml"
                weka_result["category"] = category
                weka_result["dataset"] = dataset_name
                weka_result["task"] = "clustering"
                all_results.append(weka_result)

                status = f"Sil={weka_result['silhouette']:.4f} ({weka_result['train_time_sec']:.2f}s)" if weka_result["status"] == "success" and weka_result["silhouette"] else "FAILED"
                print(f"    TuiML  {weka_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    TuiML  {weka_config['name']:<30}: ERROR - {str(e)[:60]}")

            # Benchmark sklearn
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(ALGORITHM_TIMEOUT)
                clusterer_class = import_class(sklearn_config["module"], sklearn_config["class"])
                sklearn_result = benchmark_clusterer(sklearn_config["name"], clusterer_class, sklearn_config["params"], X_algo, y_algo)
                sklearn_result["source"] = "sklearn"
                sklearn_result["category"] = category
                sklearn_result["dataset"] = dataset_name
                sklearn_result["task"] = "clustering"
                all_results.append(sklearn_result)

                signal.alarm(0)
                status = f"Sil={sklearn_result['silhouette']:.4f} ({sklearn_result['train_time_sec']:.2f}s)" if sklearn_result["status"] == "success" and sklearn_result["silhouette"] else "FAILED"
                print(f"    sklearn {sklearn_config['name']:<30}: {status}")
            except AlgorithmTimeout:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: SKIPPED (>{ALGORITHM_TIMEOUT}s timeout)")
            except Exception as e:
                signal.alarm(0)
                print(f"    sklearn {sklearn_config['name']:<30}: ERROR - {str(e)[:60]}")

    return all_results


def print_classification_summary(results: List[Dict]):
    """Print classification summary."""
    clf_results = [r for r in results if r.get("task") == "classification" and r.get("status") == "success"]
    if not clf_results:
        return

    print("\n" + "=" * 110)
    print("CLASSIFICATION SUMMARY")
    print("=" * 110)

    # Group by dataset
    datasets = list(set(r["dataset"] for r in clf_results))

    for dataset in sorted(datasets):
        print(f"\n  {dataset.upper()}")
        print(f"  {'-' * 100}")
        print(f"  {'Algorithm':<30} {'TuiML Acc':<15} {'sklearn Acc':<15} {'Winner':<15}")
        print(f"  {'-' * 100}")

        categories = list(set(r["category"] for r in clf_results if r["dataset"] == dataset))
        for category in categories:
            weka_r = next((r for r in clf_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "tuiml"), None)
            sklearn_r = next((r for r in clf_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "sklearn"), None)

            if weka_r and sklearn_r:
                diff = weka_r["accuracy"] - sklearn_r["accuracy"]
                winner = "TuiML ✓" if diff > 0.001 else "sklearn ✓" if diff < -0.001 else "TIE"
                print(f"  {category:<30} {weka_r['accuracy']:<15.4f} {sklearn_r['accuracy']:<15.4f} {winner:<15}")


def print_regression_summary(results: List[Dict]):
    """Print regression summary."""
    reg_results = [r for r in results if r.get("task") == "regression" and r.get("status") == "success"]
    if not reg_results:
        return

    print("\n" + "=" * 110)
    print("REGRESSION SUMMARY")
    print("=" * 110)

    datasets = list(set(r["dataset"] for r in reg_results))

    for dataset in sorted(datasets):
        print(f"\n  {dataset.upper()}")
        print(f"  {'-' * 100}")
        print(f"  {'Algorithm':<30} {'TuiML R²':<15} {'sklearn R²':<15} {'Winner':<15}")
        print(f"  {'-' * 100}")

        categories = list(set(r["category"] for r in reg_results if r["dataset"] == dataset))
        for category in categories:
            weka_r = next((r for r in reg_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "tuiml"), None)
            sklearn_r = next((r for r in reg_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "sklearn"), None)

            if weka_r and sklearn_r and weka_r.get("r2_score") is not None and sklearn_r.get("r2_score") is not None:
                diff = weka_r["r2_score"] - sklearn_r["r2_score"]
                winner = "TuiML ✓" if diff > 0.01 else "sklearn ✓" if diff < -0.01 else "TIE"
                print(f"  {category:<30} {weka_r['r2_score']:<15.4f} {sklearn_r['r2_score']:<15.4f} {winner:<15}")


def print_clustering_summary(results: List[Dict]):
    """Print clustering summary."""
    clust_results = [r for r in results if r.get("task") == "clustering" and r.get("status") == "success"]
    if not clust_results:
        return

    print("\n" + "=" * 110)
    print("CLUSTERING SUMMARY")
    print("=" * 110)

    datasets = list(set(r["dataset"] for r in clust_results))

    for dataset in sorted(datasets):
        print(f"\n  {dataset.upper()}")
        print(f"  {'-' * 100}")
        print(f"  {'Algorithm':<30} {'TuiML Sil':<15} {'sklearn Sil':<15} {'Winner':<15}")
        print(f"  {'-' * 100}")

        categories = list(set(r["category"] for r in clust_results if r["dataset"] == dataset))
        for category in categories:
            weka_r = next((r for r in clust_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "tuiml"), None)
            sklearn_r = next((r for r in clust_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "sklearn"), None)

            if weka_r and sklearn_r:
                weka_sil = weka_r.get("silhouette") or 0
                sklearn_sil = sklearn_r.get("silhouette") or 0
                diff = weka_sil - sklearn_sil
                winner = "TuiML ✓" if diff > 0.01 else "sklearn ✓" if diff < -0.01 else "TIE"
                print(f"  {category:<30} {weka_sil:<15.4f} {sklearn_sil:<15.4f} {winner:<15}")


def print_overall_summary(results: List[Dict]):
    """Print overall comparison summary."""
    print("\n" + "=" * 110)
    print("OVERALL COMPARISON SUMMARY")
    print("=" * 110)

    successful = [r for r in results if r.get("status") == "success"]

    # Count wins by task
    for task in ["classification", "regression", "clustering"]:
        task_results = [r for r in successful if r.get("task") == task]
        if not task_results:
            continue

        metric = "accuracy" if task == "classification" else "r2_score" if task == "regression" else "silhouette"

        weka_wins = 0
        sklearn_wins = 0
        ties = 0

        # Group by dataset and category
        combos = set((r["dataset"], r["category"]) for r in task_results)
        for dataset, category in combos:
            weka_r = next((r for r in task_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "tuiml"), None)
            sklearn_r = next((r for r in task_results if r["dataset"] == dataset and r["category"] == category and r["source"] == "sklearn"), None)

            if weka_r and sklearn_r:
                weka_val = weka_r.get(metric) or 0
                sklearn_val = sklearn_r.get(metric) or 0
                threshold = 0.001 if task == "classification" else 0.01

                if weka_val - sklearn_val > threshold:
                    weka_wins += 1
                elif sklearn_val - weka_val > threshold:
                    sklearn_wins += 1
                else:
                    ties += 1

        print(f"\n  {task.upper()}: TuiML wins={weka_wins}, sklearn wins={sklearn_wins}, ties={ties}")


def save_results(results: List[Dict], output_prefix: str = "benchmark_results"):
    """Save benchmark results to JSON and CSV."""
    # Always save next to this script (benchmarks/ folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Organize results by task
    classification_results = [r for r in results if r.get("task") == "classification"]
    regression_results = [r for r in results if r.get("task") == "regression"]
    clustering_results = [r for r in results if r.get("task") == "clustering"]

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "cv_folds": 5,
            "datasets": {
                "classification": [d[0] for d in CLASSIFICATION_DATASETS],
                "regression": [d[0] for d in REGRESSION_DATASETS],
                "clustering": [d[0] for d in CLUSTERING_DATASETS],
            }
        },
        "classification": classification_results,
        "regression": regression_results,
        "clustering": clustering_results,
        "results": results  # Flat list for backward compatibility
    }

    json_file = os.path.join(script_dir, f"{output_prefix}.json")
    with open(json_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved {json_file}")

    csv_file = os.path.join(script_dir, f"{output_prefix}.csv")
    fieldnames = ["task", "dataset", "category", "name", "source", "accuracy", "accuracy_std",
                  "r2_score", "r2_std", "rmse", "rmse_std", "silhouette", "ari",
                  "train_time_sec", "memory_mb", "status", "error"]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    print(f"✓ Saved {csv_file}")


def main():
    """Main entry point."""
    print("=" * 100)
    print("  TUIML vs SKLEARN COMPREHENSIVE BENCHMARK")
    print("  Classification • Regression • Clustering")
    print("=" * 100)

    all_results = []

    # Run all benchmarks
    all_results.extend(run_classification_benchmark())
    all_results.extend(run_regression_benchmark())
    all_results.extend(run_clustering_benchmark())

    # Print summaries
    print_classification_summary(all_results)
    print_regression_summary(all_results)
    print_clustering_summary(all_results)
    print_overall_summary(all_results)

    # Save results
    save_results(all_results)

    print("\n" + "=" * 100)
    print("  Benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
