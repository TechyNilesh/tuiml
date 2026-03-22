"""C++ backend wrappers for tree operations.

This module provides Python wrappers around the compiled C++ tree functions
(splitting, building, prediction). All tree algorithms should import these
wrappers instead of calling ``_cpp_ext.tree`` directly.

C++ is a **hard requirement** -- there are no Python fallbacks here.
Pure-Python tree utilities (criteria helpers, pruning, etc.) remain in
``_core/`` and are used for leaf-node logic, pruning, and criteria that
the C++ backend does not yet support.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from tuiml._cpp_ext import tree as _cpp_tree

from ._core.nodes import TreeNode


def has_cpp_backend() -> bool:
    """Return whether the compiled tree backend is available."""
    return _cpp_tree is not None


# ── Splitter wrappers ──────────────────────────────────────────────────

def best_split_classifier(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    n_classes: int,
    min_samples_leaf: int,
    rng: np.random.RandomState,
    max_features: Optional[int] = None,
) -> Tuple[int, float, float]:
    """Find the best binary split for classification via C++.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Integer-encoded labels.
    criterion : str
        ``"gini"`` or ``"entropy"``.
    n_classes : int
        Number of classes.
    min_samples_leaf : int
        Minimum samples at a leaf.
    rng : np.random.RandomState
        Random number generator.
    max_features : int or None
        Number of features to consider (None = all).

    Returns
    -------
    best_feature : int
        Index of the best feature (``-1`` if no valid split).
    best_threshold : float
        Threshold value for the best split.
    best_gain : float
        Impurity reduction of the best split.
    """
    seed = int(rng.randint(0, 2**31))
    mf = max_features if max_features is not None else X.shape[1]
    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.intc)
    return _cpp_tree.best_split_classifier(
        X_c, y_c, criterion, n_classes, min_samples_leaf, seed, mf
    )


def best_split_regressor(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    min_samples_leaf: int,
    rng: np.random.RandomState,
    max_features: Optional[int] = None,
) -> Tuple[int, float, float]:
    """Find the best binary split for regression via C++.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    criterion : str
        ``"squared_error"`` or ``"friedman_mse"``.
    min_samples_leaf : int
        Minimum samples at a leaf.
    rng : np.random.RandomState
        Random number generator.
    max_features : int or None
        Number of features to consider (None = all).

    Returns
    -------
    best_feature : int
        Index of the best feature (``-1`` if no valid split).
    best_threshold : float
        Threshold value for the best split.
    best_gain : float
        Impurity reduction of the best split.
    """
    seed = int(rng.randint(0, 2**31))
    mf = max_features if max_features is not None else X.shape[1]
    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    return _cpp_tree.best_split_regressor(
        X_c, y_c, criterion, min_samples_leaf, seed, mf
    )


# ── Full tree building wrappers ───────────────────────────────────────

def _unflatten_classifier_tree(feature, threshold, children_left,
                               children_right, value) -> TreeNode:
    """Convert C++ flat arrays back to a TreeNode tree.

    Parameters
    ----------
    feature : np.ndarray of shape (n_nodes,)
        Feature index per node (-1 for leaves).
    threshold : np.ndarray of shape (n_nodes,)
        Split threshold per node.
    children_left : np.ndarray of shape (n_nodes,)
        Left child index (-1 for leaves).
    children_right : np.ndarray of shape (n_nodes,)
        Right child index (-1 for leaves).
    value : np.ndarray of shape (n_nodes, n_classes)
        Class distribution per node.

    Returns
    -------
    root : TreeNode
        Root of the reconstructed tree.
    """
    n_nodes = len(feature)
    nodes = [None] * n_nodes

    # Build nodes bottom-up by processing in reverse order
    for i in range(n_nodes - 1, -1, -1):
        is_leaf = (feature[i] == -1)
        left = nodes[children_left[i]] if not is_leaf and children_left[i] >= 0 else None
        right = nodes[children_right[i]] if not is_leaf and children_right[i] >= 0 else None
        nodes[i] = TreeNode(
            is_leaf=is_leaf,
            feature_index=int(feature[i]) if not is_leaf else -1,
            threshold=float(threshold[i]) if not is_leaf else 0.0,
            value=value[i].copy(),
            left=left,
            right=right,
            n_samples=0,
            impurity=0.0,
        )

    return nodes[0]


def _unflatten_regressor_tree(feature, threshold, children_left,
                              children_right, value) -> TreeNode:
    """Convert C++ flat arrays back to a regressor TreeNode tree.

    Parameters
    ----------
    feature : np.ndarray of shape (n_nodes,)
        Feature index per node (-1 for leaves).
    threshold : np.ndarray of shape (n_nodes,)
        Split threshold per node.
    children_left : np.ndarray of shape (n_nodes,)
        Left child index (-1 for leaves).
    children_right : np.ndarray of shape (n_nodes,)
        Right child index (-1 for leaves).
    value : np.ndarray of shape (n_nodes, 1)
        Mean target value per node.

    Returns
    -------
    root : TreeNode
        Root of the reconstructed tree.
    """
    n_nodes = len(feature)
    nodes = [None] * n_nodes

    for i in range(n_nodes - 1, -1, -1):
        is_leaf = (feature[i] == -1)
        left = nodes[children_left[i]] if not is_leaf and children_left[i] >= 0 else None
        right = nodes[children_right[i]] if not is_leaf and children_right[i] >= 0 else None
        nodes[i] = TreeNode(
            is_leaf=is_leaf,
            feature_index=int(feature[i]) if not is_leaf else -1,
            threshold=float(threshold[i]) if not is_leaf else 0.0,
            value=value[i].copy(),
            left=left,
            right=right,
            n_samples=0,
            impurity=0.0,
        )

    return nodes[0]


def build_classifier_tree_cpp(X, y, config, rng):
    """Build a full classification tree in C++ and return (TreeNode, flat_arrays).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Integer-encoded labels.
    config : TreeConfig
        Tree building configuration.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    root : TreeNode
        Root of the tree (unflattened for traversal).
    flat : tuple
        (feature, threshold, children_left, children_right, value) arrays
        for C++ batch prediction.
    """
    seed = int(rng.randint(0, 2**31))
    mf = config.max_features if config.max_features is not None else X.shape[1]
    max_depth = config.max_depth if config.max_depth is not None else -1

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.intc)

    feat_arr, thresh_arr, left_arr, right_arr, val_arr, n_nodes = \
        _cpp_tree.build_classifier_tree(
            X_c, y_c, config.criterion, config.n_classes,
            max_depth, config.min_samples_split, config.min_samples_leaf,
            config.min_impurity_decrease, seed, mf,
        )

    flat = (feat_arr, thresh_arr, left_arr, right_arr, val_arr)
    root = _unflatten_classifier_tree(*flat)
    return root, flat


def build_regressor_tree_cpp(X, y, config, rng):
    """Build a full regression tree in C++ and return (TreeNode, flat_arrays).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    config : TreeConfig
        Tree building configuration.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    root : TreeNode
        Root of the tree (unflattened for traversal).
    flat : tuple
        (feature, threshold, children_left, children_right, value) arrays.
    """
    seed = int(rng.randint(0, 2**31))
    mf = config.max_features if config.max_features is not None else X.shape[1]
    max_depth = config.max_depth if config.max_depth is not None else -1

    X_c = np.ascontiguousarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)

    feat_arr, thresh_arr, left_arr, right_arr, val_arr, n_nodes = \
        _cpp_tree.build_regressor_tree(
            X_c, y_c, config.criterion, max_depth,
            config.min_samples_split, config.min_samples_leaf,
            config.min_impurity_decrease, seed, mf,
        )

    flat = (feat_arr, thresh_arr, left_arr, right_arr, val_arr)
    root = _unflatten_regressor_tree(*flat)
    return root, flat


def predict_batch_cpp(flat, X):
    """Batch prediction using C++ tree traversal.

    Parameters
    ----------
    flat : tuple
        (feature, threshold, children_left, children_right, value) from
        ``build_*_tree_cpp``.
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    result : np.ndarray of shape (n_samples, value_width)
        Leaf values for each sample.
    """
    feat_arr, thresh_arr, left_arr, right_arr, val_arr = flat
    X_c = np.ascontiguousarray(X, dtype=np.float64)
    return _cpp_tree.predict_batch(feat_arr, thresh_arr, left_arr, right_arr,
                                   val_arr, X_c)
