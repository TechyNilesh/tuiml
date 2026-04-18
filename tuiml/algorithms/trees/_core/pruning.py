"""Pruning strategies for decision trees."""

from __future__ import annotations

import numpy as np
from typing import Optional
from collections import Counter

from .nodes import TreeNode


def cost_complexity_prune(node: TreeNode, ccp_alpha: float) -> TreeNode:
    """Apply minimal cost-complexity pruning (CART).

    Parameters
    ----------
    node : TreeNode
        Root of the (sub)tree to prune.
    ccp_alpha : float
        Complexity parameter. Subtrees with effective alpha less
        than ``ccp_alpha`` are pruned.

    Returns
    -------
    node : TreeNode
        Pruned node (may become a leaf).
    """
    if node.is_leaf:
        return node

    node.left = cost_complexity_prune(node.left, ccp_alpha)
    node.right = cost_complexity_prune(node.right, ccp_alpha)

    if node.left.is_leaf and node.right.is_leaf:
        n_total = node.n_samples
        leaf_impurity = node.impurity
        left_imp = (node.left.n_samples / n_total) * node.left.impurity
        right_imp = (node.right.n_samples / n_total) * node.right.impurity
        subtree_impurity = left_imp + right_imp

        n_leaves = 2
        alpha = (leaf_impurity - subtree_impurity) / (n_leaves - 1)
        if alpha <= ccp_alpha:
            return TreeNode(
                is_leaf=True,
                value=node.value,
                n_samples=node.n_samples,
                impurity=node.impurity,
            )

    return node


def reduced_error_prune_classifier(
    node: TreeNode,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classes: np.ndarray,
) -> TreeNode:
    """Prune a classification tree using reduced-error pruning.

    Parameters
    ----------
    node : TreeNode
        Current node to evaluate for pruning.
    X_val : np.ndarray of shape (n_samples, n_features)
        Validation feature matrix.
    y_val : np.ndarray of shape (n_samples,)
        Validation class labels (integer-encoded).
    classes : np.ndarray
        Array of class labels.

    Returns
    -------
    node : TreeNode
        Pruned node (leaf or original subtree).
    """
    if node.is_leaf or len(y_val) == 0:
        return node

    X_col = X_val[:, node.feature_index]
    left_mask = X_col <= node.threshold
    missing_mask = np.isnan(X_col)
    if np.any(missing_mask):
        left_mask = left_mask | missing_mask
    right_mask = ~left_mask

    if np.sum(left_mask) > 0:
        node.left = reduced_error_prune_classifier(
            node.left, X_val[left_mask], y_val[left_mask], classes
        )
    if np.sum(right_mask) > 0:
        node.right = reduced_error_prune_classifier(
            node.right, X_val[right_mask], y_val[right_mask], classes
        )

    # Subtree error
    subtree_preds = np.array([
        _predict_single_cls(node, X_val[i]) for i in range(len(X_val))
    ])
    subtree_errors = np.sum(subtree_preds != y_val)

    # Leaf error
    majority = Counter(y_val).most_common(1)[0][0] if len(y_val) > 0 else 0
    leaf_errors = np.sum(y_val != majority)

    if leaf_errors <= subtree_errors:
        n_classes = len(classes)
        n_val = len(y_val)
        distribution = np.bincount(y_val, minlength=n_classes).astype(np.float64)
        if n_val > 0:
            distribution = distribution / n_val
        class_dist = {cls: distribution[cls] for cls in range(n_classes)}
        return TreeNode(
            is_leaf=True,
            predicted_class=majority,
            class_distribution=class_dist,
            value=distribution,
            n_samples=n_val,
        )

    return node


def reduced_error_prune_regressor(
    node: TreeNode,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> TreeNode:
    """Prune a regression tree using reduced-error pruning.

    Parameters
    ----------
    node : TreeNode
        Current node to evaluate for pruning.
    X_val : np.ndarray of shape (n_samples, n_features)
        Validation feature matrix.
    y_val : np.ndarray of shape (n_samples,)
        Validation target values.

    Returns
    -------
    node : TreeNode
        Pruned node (leaf or original subtree).
    """
    if node.is_leaf or len(y_val) == 0:
        return node

    X_col = X_val[:, node.feature_index]
    left_mask = X_col <= node.threshold
    missing_mask = np.isnan(X_col)
    if np.any(missing_mask):
        left_mask = left_mask | missing_mask
    right_mask = ~left_mask

    if np.sum(left_mask) > 0:
        node.left = reduced_error_prune_regressor(
            node.left, X_val[left_mask], y_val[left_mask]
        )
    if np.sum(right_mask) > 0:
        node.right = reduced_error_prune_regressor(
            node.right, X_val[right_mask], y_val[right_mask]
        )

    # Subtree MSE
    subtree_preds = np.array([
        _predict_single_reg(node, X_val[i]) for i in range(len(X_val))
    ])
    subtree_mse = np.mean((subtree_preds - y_val) ** 2)

    # Leaf MSE
    leaf_value = float(np.mean(y_val))
    leaf_mse = np.mean((y_val - leaf_value) ** 2)

    if leaf_mse <= subtree_mse:
        return TreeNode(
            is_leaf=True,
            predicted_value=leaf_value,
            value=np.array([leaf_value]),
            n_samples=len(y_val),
            impurity=leaf_mse,
        )

    return node


def _count_subtree_errors(node: TreeNode, X: np.ndarray, y: np.ndarray) -> int:
    """Count misclassifications of the subtree on the given data.

    Uses recursive partitioning to avoid per-sample tree traversal.

    Parameters
    ----------
    node : TreeNode
        Current tree node.
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        True class labels.

    Returns
    -------
    errors : int
        Number of misclassified samples.
    """
    if len(y) == 0:
        return 0

    if node.is_leaf:
        if node.predicted_class is not None:
            return int(np.sum(y != node.predicted_class))
        if node.value is not None:
            return int(np.sum(y != int(np.argmax(node.value))))
        return int(np.sum(y != 0))

    # Split data along the node's feature
    if node.is_numeric:
        X_col = X[:, node.feature_index].astype(float)
        left_mask = X_col <= node.threshold
    else:
        X_col = X[:, node.feature_index]
        left_mask = X_col == node.threshold
    right_mask = ~left_mask

    errors = 0
    if np.any(left_mask):
        errors += _count_subtree_errors(node.left, X[left_mask], y[left_mask])
    if np.any(right_mask):
        errors += _count_subtree_errors(node.right, X[right_mask], y[right_mask])
    return errors


def pessimistic_prune(
    node: TreeNode,
    X: np.ndarray,
    y: np.ndarray,
    confidence_factor: float,
    classes: np.ndarray,
) -> TreeNode:
    """Prune using C4.5-style pessimistic error estimation.

    Parameters
    ----------
    node : TreeNode
        Current node to evaluate for pruning.
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels (integer-encoded).
    confidence_factor : float
        Confidence factor for pruning (lower = more pruning).
    classes : np.ndarray
        Array of class labels.

    Returns
    -------
    node : TreeNode
        Pruned node.
    """
    if node.is_leaf:
        return node

    # Split data
    if node.is_numeric:
        X_col = X[:, node.feature_index].astype(float)
        left_mask = X_col <= node.threshold
    else:
        X_col = X[:, node.feature_index]
        left_mask = X_col == node.threshold
    right_mask = ~left_mask

    if np.sum(left_mask) > 0:
        node.left = pessimistic_prune(
            node.left, X[left_mask], y[left_mask], confidence_factor, classes
        )
    if np.sum(right_mask) > 0:
        node.right = pessimistic_prune(
            node.right, X[right_mask], y[right_mask], confidence_factor, classes
        )

    # Subtree error -- partition data through children instead of
    # calling _predict_single_cls per sample (O(n) vs O(n*depth))
    subtree_errors = _count_subtree_errors(node, X, y)

    # Leaf error
    majority = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
    leaf_errors = np.sum(y != majority)

    if leaf_errors <= subtree_errors + len(y) * confidence_factor:
        n_classes = len(classes)
        n_samples = len(y)
        distribution = np.bincount(y, minlength=n_classes).astype(np.float64)
        if n_samples > 0:
            distribution = distribution / n_samples
        class_dist = {cls: distribution[cls] for cls in range(n_classes)}
        return TreeNode(
            is_leaf=True,
            predicted_class=majority,
            class_distribution=class_dist,
            value=distribution,
            n_samples=n_samples,
        )

    return node


# =========================================================================
# Internal helpers for pruning traversal
# =========================================================================

def _predict_single_cls(node: TreeNode, x: np.ndarray):
    """Predict class for a single sample during pruning.

    Parameters
    ----------
    node : TreeNode
        Current tree node.
    x : np.ndarray
        Single sample.

    Returns
    -------
    prediction : int or Any
        Predicted class.
    """
    if node.is_leaf:
        if node.predicted_class is not None:
            return node.predicted_class
        if node.value is not None:
            return int(np.argmax(node.value))
        return 0

    feat_val = x[node.feature_index]
    if node.is_numeric:
        if np.isnan(feat_val):
            if node.left.n_samples >= node.right.n_samples:
                return _predict_single_cls(node.left, x)
            return _predict_single_cls(node.right, x)
        if feat_val <= node.threshold:
            return _predict_single_cls(node.left, x)
        return _predict_single_cls(node.right, x)
    else:
        if feat_val == node.threshold:
            return _predict_single_cls(node.left, x)
        return _predict_single_cls(node.right, x)


def _predict_single_reg(node: TreeNode, x: np.ndarray) -> float:
    """Predict value for a single sample during pruning.

    Parameters
    ----------
    node : TreeNode
        Current tree node.
    x : np.ndarray
        Single sample.

    Returns
    -------
    prediction : float
        Predicted value.
    """
    if node.is_leaf:
        if node.predicted_value is not None:
            return node.predicted_value
        if node.value is not None:
            return float(node.value[0])
        return 0.0

    feat_val = x[node.feature_index]
    if np.isnan(feat_val):
        if node.left.n_samples >= node.right.n_samples:
            return _predict_single_reg(node.left, x)
        return _predict_single_reg(node.right, x)
    if feat_val <= node.threshold:
        return _predict_single_reg(node.left, x)
    return _predict_single_reg(node.right, x)
