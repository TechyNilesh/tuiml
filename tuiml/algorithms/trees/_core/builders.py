"""Tree configuration and recursive tree builders."""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .nodes import TreeNode, flatten_tree
from .criteria import classifier_node_impurity, regressor_node_impurity

# C++ full-tree builders (called directly, C++ is required)
from .._core_dispatch import (
    build_classifier_tree_cpp as _build_classifier_tree_cpp,
    build_regressor_tree_cpp as _build_regressor_tree_cpp,
)

# Python splitters for criteria NOT yet implemented in C++
# (e.g. gain_ratio, log_loss, absolute_error, sdr)
from .splitters import (
    best_split_classifier as _py_best_split_classifier,
    best_split_regressor as _py_best_split_regressor,
)

# Criteria supported by the C++ full-tree builder
_CPP_CLASSIFIER_CRITERIA = ("gini", "entropy", "log_loss")
_CPP_REGRESSOR_CRITERIA = ("squared_error", "friedman_mse")


@dataclass
class TreeConfig:
    """Configuration for tree building.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree.
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    min_impurity_decrease : float
        Minimum impurity decrease for a split.
    criterion : str
        Splitting criterion name.
    max_features : int or None
        Number of features to consider at each split.
    n_classes : int or None
        Number of classes (classification only).
    """

    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_impurity_decrease: float = 0.0
    criterion: str = "gini"
    max_features: Optional[int] = None
    n_classes: Optional[int] = None


def build_classifier_tree(
    X: np.ndarray,
    y: np.ndarray,
    config: TreeConfig,
    rng: np.random.RandomState,
    depth: int = 0,
) -> TreeNode:
    """Build a classification tree.

    When called at ``depth == 0`` with a C++-supported criterion (``gini``
    or ``entropy``), the entire tree is built in C++. For other criteria,
    a Python recursive builder is used and the resulting tree is flattened
    so that C++ batch prediction is always available.

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
    depth : int
        Current depth in the tree.

    Returns
    -------
    node : TreeNode
        Root of the (sub)tree. At ``depth == 0``, the root always has a
        ``_cpp_flat`` attribute for C++ batch prediction.
    """
    # C++ full builder for supported criteria (top-level call only)
    if depth == 0 and config.criterion in _CPP_CLASSIFIER_CRITERIA:
        root, flat = _build_classifier_tree_cpp(X, y, config, rng)
        root._cpp_flat = flat
        return root

    # Python recursive builder for unsupported criteria
    root = _build_classifier_subtree(X, y, config, rng, depth)

    # Flatten at top level so C++ batch prediction works for all trees
    if depth == 0:
        flat_tree = flatten_tree(root, config.n_classes)
        root._cpp_flat = (
            flat_tree.feature,
            flat_tree.threshold,
            flat_tree.children_left,
            flat_tree.children_right,
            flat_tree.value,
        )

    return root


def _build_classifier_subtree(
    X: np.ndarray,
    y: np.ndarray,
    config: TreeConfig,
    rng: np.random.RandomState,
    depth: int,
) -> TreeNode:
    """Recursively build a classification subtree in Python.

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
    depth : int
        Current depth in the tree.

    Returns
    -------
    node : TreeNode
        Root of the subtree.
    """
    n_samples = len(y)
    n_classes = config.n_classes
    distribution = np.bincount(y, minlength=n_classes).astype(np.float64)
    distribution = distribution / n_samples
    impurity = classifier_node_impurity(y, config.criterion, n_classes)

    # Leaf conditions
    if (
        (config.max_depth is not None and depth >= config.max_depth)
        or n_samples < config.min_samples_split
        or impurity == 0.0
    ):
        return TreeNode(
            is_leaf=True,
            value=distribution,
            n_samples=n_samples,
            impurity=impurity,
        )

    best_feature, best_threshold, best_gain = _py_best_split_classifier(
        X, y, config.criterion, n_classes, config.min_samples_leaf, rng, config.max_features
    )

    if best_feature == -1 or best_gain < config.min_impurity_decrease:
        return TreeNode(
            is_leaf=True,
            value=distribution,
            n_samples=n_samples,
            impurity=impurity,
        )

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_node = _build_classifier_subtree(X[left_mask], y[left_mask], config, rng, depth + 1)
    right_node = _build_classifier_subtree(X[right_mask], y[right_mask], config, rng, depth + 1)

    return TreeNode(
        is_leaf=False,
        feature_index=best_feature,
        threshold=best_threshold,
        value=distribution,
        left=left_node,
        right=right_node,
        n_samples=n_samples,
        impurity=impurity,
    )


def build_regressor_tree(
    X: np.ndarray,
    y: np.ndarray,
    config: TreeConfig,
    rng: np.random.RandomState,
    depth: int = 0,
) -> TreeNode:
    """Build a regression tree.

    When called at ``depth == 0`` with a C++-supported criterion
    (``squared_error`` or ``friedman_mse``), the entire tree is built in
    C++. For other criteria, a Python recursive builder is used and the
    resulting tree is flattened so that C++ batch prediction is always
    available.

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
    depth : int
        Current depth in the tree.

    Returns
    -------
    node : TreeNode
        Root of the (sub)tree. At ``depth == 0``, the root always has a
        ``_cpp_flat`` attribute for C++ batch prediction.
    """
    # C++ full builder for supported criteria (top-level call only)
    if depth == 0 and config.criterion in _CPP_REGRESSOR_CRITERIA:
        root, flat = _build_regressor_tree_cpp(X, y, config, rng)
        root._cpp_flat = flat
        return root

    # Python recursive builder for unsupported criteria
    root = _build_regressor_subtree(X, y, config, rng, depth)

    # Flatten at top level so C++ batch prediction works for all trees
    if depth == 0:
        flat_tree = flatten_tree(root, 1)
        root._cpp_flat = (
            flat_tree.feature,
            flat_tree.threshold,
            flat_tree.children_left,
            flat_tree.children_right,
            flat_tree.value,
        )

    return root


def _build_regressor_subtree(
    X: np.ndarray,
    y: np.ndarray,
    config: TreeConfig,
    rng: np.random.RandomState,
    depth: int,
) -> TreeNode:
    """Recursively build a regression subtree in Python.

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
    depth : int
        Current depth in the tree.

    Returns
    -------
    node : TreeNode
        Root of the subtree.
    """
    n_samples = len(y)
    leaf_val = np.median(y) if config.criterion == "absolute_error" else np.mean(y)
    impurity = regressor_node_impurity(y, config.criterion)

    if (
        (config.max_depth is not None and depth >= config.max_depth)
        or n_samples < config.min_samples_split
        or impurity == 0.0
    ):
        return TreeNode(
            is_leaf=True,
            value=np.array([leaf_val]),
            n_samples=n_samples,
            impurity=impurity,
        )

    best_feature, best_threshold, best_gain = _py_best_split_regressor(
        X, y, config.criterion, config.min_samples_leaf, rng, config.max_features
    )

    if best_feature == -1 or best_gain < config.min_impurity_decrease:
        return TreeNode(
            is_leaf=True,
            value=np.array([leaf_val]),
            n_samples=n_samples,
            impurity=impurity,
        )

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_node = _build_regressor_subtree(X[left_mask], y[left_mask], config, rng, depth + 1)
    right_node = _build_regressor_subtree(X[right_mask], y[right_mask], config, rng, depth + 1)

    return TreeNode(
        is_leaf=False,
        feature_index=best_feature,
        threshold=best_threshold,
        value=np.array([leaf_val]),
        left=left_node,
        right=right_node,
        n_samples=n_samples,
        impurity=impurity,
    )
