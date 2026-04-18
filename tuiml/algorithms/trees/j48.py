"""C4.5 Decision Tree classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from ._core import (
    TreeNode,
    entropy,
    gain_ratio_score,
    pessimistic_prune,
    predict_single_numpy,
    build_classifier_tree,
    build_regressor_tree,
    reduced_error_prune_regressor,
    flatten_tree,
    predict_batch,
    TreeConfig,
)

# Backward compatibility alias
C45TreeNode = TreeNode
RegressionTreeNode = TreeNode

@classifier(tags=["trees", "interpretable", "pruning"], version="1.0.0")
class C45TreeClassifier(Classifier):
    """C4.5 Decision Tree classifier.

    C4.5 is a decision tree algorithm that builds decision trees using
    **information gain ratio** for attribute selection. It extends the ID3
    algorithm by handling both **numeric and nominal attributes**, missing
    values, and post-pruning via **pessimistic error estimation**.

    Overview
    --------
    The C4.5 algorithm builds a decision tree top-down:

    1. Compute the **entropy** of the current node's class distribution
    2. For each candidate attribute, compute the **information gain ratio**
    3. Select the attribute with the highest gain ratio as the split
    4. Partition the data and recursively build child subtrees
    5. Apply **pessimistic pruning** to reduce overfitting by replacing
       subtrees with leaves when the estimated error does not increase

    Theory
    ------
    The entropy of a set :math:`S` with :math:`c` classes is:

    .. math::
        H(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)

    The information gain for attribute :math:`A` is:

    .. math::
        IG(S, A) = H(S) - \\sum_{v \\in \\text{values}(A)} \\frac{|S_v|}{|S|} H(S_v)

    The **gain ratio** normalizes by the intrinsic information (split info):

    .. math::
        GR(S, A) = \\frac{IG(S, A)}{SI(S, A)}

    where the split information is:

    .. math::
        SI(S, A) = -\\sum_{v \\in \\text{values}(A)} \\frac{|S_v|}{|S|} \\log_2 \\frac{|S_v|}{|S|}

    Parameters
    ----------
    min_samples_leaf : int, default=2
        Minimum number of samples required at a leaf node.
    confidence_factor : float, default=0.25
        Confidence factor for pessimistic pruning. Lower values result in
        more aggressive pruning.
    unpruned : bool, default=False
        If True, do not prune the tree.
    binary_splits : bool, default=False
        Force binary splits even for nominal attributes.
    max_depth : int, optional
        Maximum depth of the tree. None means unlimited.

    Attributes
    ----------
    tree_ : TreeNode
        The root node of the decision tree.
    classes_ : np.ndarray
        Unique class labels.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features
    - Prediction: :math:`O(\\log(n))` per sample (average tree depth)

    **When to use C45TreeClassifier:**

    - When you need an **interpretable** model with human-readable rules
    - Datasets with a mix of numeric and nominal attributes
    - When handling missing values natively is important
    - When post-pruning is desired to control model complexity

    References
    ----------
    .. [Quinlan1993] Quinlan, J.R. (1993).
           **C4.5: Programs for Machine Learning.**
           *Morgan Kaufmann Publishers*.

    .. [Quinlan1986] Quinlan, J.R. (1986).
           **Induction of Decision Trees.**
           *Machine Learning*, 1(1), pp. 81-106.
           DOI: `10.1007/BF00116251 <https://doi.org/10.1007/BF00116251>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` : A single-split decision tree (weak learner).
    :class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeClassifier` : Fast tree with reduced-error pruning.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of randomized trees.

    Examples
    --------
    Basic usage for classification with pruning:

    >>> from tuiml.algorithms.trees import C45TreeClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> # Fit a pruned C4.5 tree
    >>> clf = C45TreeClassifier(min_samples_leaf=2, confidence_factor=0.25)
    >>> clf.fit(X, y)
    C45TreeClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, min_samples_leaf: int = 2,
                 confidence_factor: float = 0.25,
                 unpruned: bool = False,
                 binary_splits: bool = False,
                 max_depth: Optional[int] = None):
        """Initialize C45TreeClassifier classifier.

        Parameters
        ----------
        min_samples_leaf : int, default=2
            Minimum samples at leaf node.
        confidence_factor : float, default=0.25
            Confidence factor for pruning.
        unpruned : bool, default=False
            If True, skip pruning.
        binary_splits : bool, default=False
            Force binary splits for nominal attributes.
        max_depth : int or None, default=None
            Maximum tree depth.
        """
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.confidence_factor = confidence_factor
        self.unpruned = unpruned
        self.binary_splits = binary_splits
        self.max_depth = max_depth
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self._is_numeric = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "min_samples_leaf": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Minimum number of samples at a leaf node"
            },
            "confidence_factor": {
                "type": "number",
                "default": 0.25,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence factor for pruning (lower = more pruning)"
            },
            "unpruned": {
                "type": "boolean",
                "default": False,
                "description": "If true, do not prune the tree"
            },
            "binary_splits": {
                "type": "boolean",
                "default": False,
                "description": "Force binary splits for nominal attributes"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "nominal",
            "missing_values",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Quinlan, J.R. (1993). C4.5: Programs for Machine Learning. "
            "Morgan Kaufmann Publishers."
        ]

    def _is_numeric_attr(self, values: np.ndarray) -> bool:
        """Check if attribute values are numeric.

        Parameters
        ----------
        values : np.ndarray
            Attribute values to check.

        Returns
        -------
        is_numeric : bool
            True if all values can be cast to float.
        """
        try:
            np.asarray(values, dtype=float)
            return True
        except (ValueError, TypeError):
            return False

    def _find_best_numeric_split(self, X_col: np.ndarray, y: np.ndarray,
                                  n_classes: int) -> Tuple[float, float]:
        """Find best threshold for a numeric attribute using gain ratio.

        Uses vectorized cumulative class-count computation to evaluate all
        candidate thresholds in a single pass with NumPy operations,
        avoiding per-threshold Python loops.

        Parameters
        ----------
        X_col : np.ndarray
            Single feature column values.
        y : np.ndarray
            Corresponding class labels (integer-encoded).
        n_classes : int
            Number of classes.

        Returns
        -------
        best_threshold : float
            Optimal split threshold.
        best_gain_ratio : float
            Gain ratio achieved at the optimal threshold.
        """
        n = len(y)
        sorted_indices = np.argsort(X_col)
        sorted_values = X_col[sorted_indices]
        sorted_y = y[sorted_indices]

        # Build a one-hot matrix (n, n_classes) and cumsum to get
        # left_counts[i] = class counts for sorted_y[:i+1]
        onehot = np.zeros((n, n_classes), dtype=np.int64)
        onehot[np.arange(n), sorted_y] = 1
        cum_counts = np.cumsum(onehot, axis=0)  # (n, n_classes)

        total_counts = cum_counts[-1]  # total class counts

        # Valid split positions: where value changes and min_samples_leaf met
        # Split after index i means left = [:i+1], right = [i+1:]
        value_changes = sorted_values[:-1] != sorted_values[1:]  # (n-1,)
        n_left_arr = np.arange(1, n)  # sizes 1..n-1
        n_right_arr = n - n_left_arr
        leaf_ok = ((n_left_arr >= self.min_samples_leaf) &
                   (n_right_arr >= self.min_samples_leaf))
        valid = value_changes & leaf_ok

        valid_idx = np.nonzero(valid)[0]
        if len(valid_idx) == 0:
            return sorted_values[0], -np.inf

        # Extract counts at valid split positions
        left_counts = cum_counts[valid_idx]  # (k, n_classes)
        right_counts = total_counts - left_counts
        n_left_v = n_left_arr[valid_idx].astype(np.float64)  # (k,)
        n_right_v = n_right_arr[valid_idx].astype(np.float64)

        # Compute entropy for left and right partitions (vectorized)
        left_probs = left_counts / n_left_v[:, None]
        right_probs = right_counts / n_right_v[:, None]

        # Entropy: -sum(p * log2(p)) for p > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            log_left = np.where(left_probs > 0,
                                np.log2(left_probs), 0.0)
            log_right = np.where(right_probs > 0,
                                 np.log2(right_probs), 0.0)

        H_left = -np.sum(left_probs * log_left, axis=1)  # (k,)
        H_right = -np.sum(right_probs * log_right, axis=1)

        # Information gain
        parent_H = entropy(y, n_classes)
        w_left = n_left_v / n
        w_right = n_right_v / n
        info_gain = parent_H - (w_left * H_left + w_right * H_right)

        # Split information
        with np.errstate(divide='ignore', invalid='ignore'):
            si_left = np.where(w_left > 0,
                               -w_left * np.log2(w_left + 1e-10), 0.0)
            si_right = np.where(w_right > 0,
                                -w_right * np.log2(w_right + 1e-10), 0.0)
        split_info = si_left + si_right
        split_info = np.maximum(split_info, 1e-10)

        gain_ratios = info_gain / split_info

        best_local = np.argmax(gain_ratios)
        best_gain_ratio = gain_ratios[best_local]
        i = valid_idx[best_local]
        best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0

        return best_threshold, best_gain_ratio

    def _find_best_split(self, X: np.ndarray, y: np.ndarray,
                         n_classes: int) -> Tuple[int, float, bool, float]:
        """Find the best attribute and split point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Class labels (integer-encoded).
        n_classes : int
            Number of classes.

        Returns
        -------
        feature_index : int
            Index of the best splitting feature.
        threshold : float
            Best split threshold or nominal value.
        is_numeric : bool
            Whether the best feature is numeric.
        gain_ratio : float
            Gain ratio of the best split.
        """
        n_samples, n_features = X.shape
        best_gain_ratio = -np.inf
        best_feature = 0
        best_threshold = None
        best_is_numeric = True
        parent_H = entropy(y, n_classes)

        for feature_idx in range(n_features):
            X_col = X[:, feature_idx]
            is_numeric = self._is_numeric[feature_idx]

            if is_numeric:
                X_col = X_col.astype(float)
                valid_mask = ~np.isnan(X_col)
                if np.sum(valid_mask) < 2 * self.min_samples_leaf:
                    continue
                X_valid = X_col[valid_mask]
                y_valid = y[valid_mask]
                threshold, gr = self._find_best_numeric_split(
                    X_valid, y_valid, n_classes
                )
                if gr > best_gain_ratio:
                    best_gain_ratio = gr
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_is_numeric = True
            else:
                unique_values = np.unique(X_col)
                for val in unique_values:
                    mask = X_col == val
                    y_left = y[mask]
                    y_right = y[~mask]

                    if len(y_left) < self.min_samples_leaf or \
                       len(y_right) < self.min_samples_leaf:
                        continue

                    H_left = entropy(y_left, n_classes)
                    H_right = entropy(y_right, n_classes)
                    gr = gain_ratio_score(
                        parent_H,
                        np.array([H_left, H_right]),
                        np.array([len(y_left), len(y_right)]),
                        len(y),
                    )

                    if gr > best_gain_ratio:
                        best_gain_ratio = gr
                        best_feature = feature_idx
                        best_threshold = val
                        best_is_numeric = False

        return best_feature, best_threshold, best_is_numeric, best_gain_ratio

    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                    n_classes: int, depth: int = 0) -> TreeNode:
        """Recursively build the decision tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current node.
        y : np.ndarray of shape (n_samples,)
            Class labels (integer-encoded).
        n_classes : int
            Number of classes.
        depth : int, default=0
            Current depth in the tree.

        Returns
        -------
        node : TreeNode
            Root of the constructed subtree.
        """
        n_samples = len(y)
        distribution = np.bincount(y, minlength=n_classes).astype(np.float64)
        distribution = distribution / n_samples
        majority_class = int(np.argmax(distribution))

        # Check stopping conditions
        if (n_samples < 2 * self.min_samples_leaf or
            len(set(y)) == 1 or
            (self.max_depth is not None and depth >= self.max_depth)):
            return TreeNode(
                is_leaf=True,
                predicted_class=majority_class,
                value=distribution,
                n_samples=n_samples,
            )

        # Find best split
        feature_idx, threshold, is_numeric, gr = self._find_best_split(
            X, y, n_classes
        )

        if gr <= 0:
            return TreeNode(
                is_leaf=True,
                predicted_class=majority_class,
                value=distribution,
                n_samples=n_samples,
            )

        # Create internal node
        node = TreeNode(
            is_leaf=False,
            feature_index=feature_idx,
            threshold=threshold,
            is_numeric=is_numeric,
            value=distribution,
            n_samples=n_samples,
        )

        # Split data and build subtrees
        if is_numeric:
            X_col = X[:, feature_idx].astype(float)
            left_mask = X_col <= threshold
            right_mask = X_col > threshold
            # Distribute missing values to both branches
            missing_mask = np.isnan(X_col)
            if np.any(missing_mask):
                left_mask = left_mask | missing_mask
                right_mask = right_mask | missing_mask
        else:
            X_col = X[:, feature_idx]
            left_mask = X_col == threshold
            right_mask = X_col != threshold

        node.left = self._build_tree(X[left_mask], y[left_mask], n_classes, depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], n_classes, depth + 1)

        return node

    @staticmethod
    def _populate_n_samples(node: TreeNode, X: np.ndarray) -> None:
        """Populate n_samples on tree nodes built by the C++ backend."""
        stack = [(node, np.arange(X.shape[0]))]
        while stack:
            nd, indices = stack.pop()
            nd.n_samples = len(indices)
            if not nd.is_leaf and nd.feature_index >= 0:
                left_mask = X[indices, nd.feature_index] <= nd.threshold
                stack.append((nd.left, indices[left_mask]))
                stack.append((nd.right, indices[~left_mask]))

    def _predict_single(self, node: TreeNode, x: np.ndarray) -> int:
        """Predict a single sample.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        prediction : int
            Predicted class index.
        """
        if node.is_leaf:
            if node.predicted_class is not None:
                return node.predicted_class
            return int(np.argmax(node.value))

        if node.is_numeric:
            value = float(x[node.feature_index])
            if np.isnan(value):
                if node.left.n_samples >= node.right.n_samples:
                    return self._predict_single(node.left, x)
                return self._predict_single(node.right, x)
            if value <= node.threshold:
                return self._predict_single(node.left, x)
            return self._predict_single(node.right, x)
        else:
            if x[node.feature_index] == node.threshold:
                return self._predict_single(node.left, x)
            return self._predict_single(node.right, x)

    def _predict_proba_single(self, node: TreeNode, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for a single sample.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Class probability distribution.
        """
        if node.is_leaf:
            return node.value

        if node.is_numeric:
            value = float(x[node.feature_index])
            if np.isnan(value):
                left_proba = self._predict_proba_single(node.left, x)
                right_proba = self._predict_proba_single(node.right, x)
                total = node.left.n_samples + node.right.n_samples
                return (node.left.n_samples * left_proba +
                        node.right.n_samples * right_proba) / total
            if value <= node.threshold:
                return self._predict_proba_single(node.left, x)
            return self._predict_proba_single(node.right, x)
        else:
            if x[node.feature_index] == node.threshold:
                return self._predict_proba_single(node.left, x)
            return self._predict_proba_single(node.right, x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "C45TreeClassifier":
        """Fit the C45TreeClassifier decision tree classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : C45TreeClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Determine which attributes are numeric
        self._is_numeric = np.array([self._is_numeric_attr(X[:, i])
                                     for i in range(self.n_features_)])

        # Convert y to integer indices (vectorized)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        sorter = np.argsort(self.classes_)
        y_idx = sorter[np.searchsorted(self.classes_, y, sorter=sorter)]

        all_numeric = np.all(self._is_numeric)

        if all_numeric:
            rng = np.random.RandomState(42)
            config = TreeConfig(
                max_depth=self.max_depth,
                min_samples_split=max(2 * self.min_samples_leaf, 10),
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=0.001,
                criterion="entropy",
                n_classes=n_classes,
            )
            self.tree_ = build_classifier_tree(X, y_idx, config, rng)
            self._populate_n_samples(self.tree_, X)
            self._cpp_path = True
        else:
            self.tree_ = self._build_tree(X, y_idx, n_classes)

        # Prune tree if requested (skip for C++ path which uses min_impurity_decrease)
        if not self.unpruned and not getattr(self, '_cpp_path', False):
            self.tree_ = pessimistic_prune(
                self.tree_, X, y_idx, self.confidence_factor, self.classes_
            )

        # Flatten for batch prediction
        self._flat_tree = flatten_tree(self.tree_, value_width=n_classes)

        self._class_to_idx = class_to_idx
        self._idx_to_class = {i: c for c, i in class_to_idx.items()}

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if hasattr(self, '_flat_tree') and self._flat_tree is not None:
            proba = predict_batch(self._flat_tree, X)
            indices = np.argmax(proba, axis=1)
        else:
            n_samples = X.shape[0]
            indices = np.array([self._predict_single(self.tree_, X[i])
                                for i in range(n_samples)])
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))

        for i in range(n_samples):
            proba[i] = self._predict_proba_single(self.tree_, X[i])

        return proba

    def _tree_to_string(self, node: TreeNode, depth: int = 0) -> str:
        """Convert tree to string representation.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        depth : int, default=0
            Current indentation depth.

        Returns
        -------
        text : str
            Human-readable tree representation.
        """
        indent = "  " * depth
        if node.is_leaf:
            cls = node.predicted_class
            if cls is not None:
                cls = self._idx_to_class.get(cls, cls)
            return f"{indent}Predict: {cls} (n={node.n_samples})"

        lines = []
        if node.is_numeric:
            lines.append(f"{indent}If feature[{node.feature_index}] <= {node.threshold:.4f}:")
        else:
            lines.append(f"{indent}If feature[{node.feature_index}] == {node.threshold}:")
        lines.append(self._tree_to_string(node.left, depth + 1))
        lines.append(f"{indent}Else:")
        lines.append(self._tree_to_string(node.right, depth + 1))

        return "\n".join(lines)

    def get_tree_description(self) -> str:
        """Get human-readable description of the tree."""
        if not self._is_fitted:
            return "Model not fitted"
        return self._tree_to_string(self.tree_)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"C45TreeClassifier(n_features={self.n_features_}, classes={list(self.classes_)})"
        return f"C45TreeClassifier(min_samples_leaf={self.min_samples_leaf})"


@regressor(tags=["trees", "interpretable", "pruning"], version="1.0.0")
class C45TreeRegressor(Regressor):
    """C4.5-style Decision Tree regressor using variance reduction.

    A CART-style regression tree that uses **variance reduction** (MSE
    reduction) instead of information gain ratio for selecting splits.
    Each leaf stores the **mean target value** of its samples. Optional
    **pruning** replaces subtrees with leaves when the held-out MSE does
    not increase.

    Overview
    --------
    The algorithm builds a regression tree top-down:

    1. Compute the **MSE** (variance) of the current node's target values
    2. For each candidate feature and threshold, compute the **variance
       reduction** from the proposed split
    3. Select the split that maximizes variance reduction
    4. Recursively partition the data and build child subtrees
    5. Optionally **prune** using held-out MSE to reduce overfitting

    Theory
    ------
    The variance (MSE) at a node with targets :math:`y_1, \\dots, y_n` is:

    .. math::
        \\text{MSE}(S) = \\frac{1}{|S|} \\sum_{i=1}^{|S|} (y_i - \\bar{y})^2

    The variance reduction for a split into :math:`S_L` and :math:`S_R` is:

    .. math::
        \\Delta \\text{MSE} = \\text{MSE}(S)
            - \\frac{|S_L|}{|S|} \\text{MSE}(S_L)
            - \\frac{|S_R|}{|S|} \\text{MSE}(S_R)

    Parameters
    ----------
    min_samples_leaf : int, default=2
        Minimum number of samples required at a leaf node.
    max_depth : int, optional
        Maximum depth of the tree. None means unlimited.
    unpruned : bool, default=False
        If True, do not prune the tree.

    Attributes
    ----------
    tree_ : TreeNode
        The root node of the regression tree.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features
    - Prediction: :math:`O(\\log(n))` per sample

    **When to use C45TreeRegressor:**

    - When you need an **interpretable** regression model
    - When the relationship between features and target is non-linear
    - When post-pruning is desired to control model complexity

    References
    ----------
    .. [Breiman1984] Breiman, L., Friedman, J.H., Olshen, R.A. & Stone, C.J.
           (1984). **Classification and Regression Trees.** *Wadsworth*.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : C4.5 classifier counterpart.
    :class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeRegressor` : REP tree for regression.
    :class:`~tuiml.algorithms.trees.RandomForestRegressor` : Ensemble of regression trees.

    Examples
    --------
    >>> from tuiml.algorithms.trees import C45TreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    >>> reg = C45TreeRegressor(min_samples_leaf=2)
    >>> reg.fit(X, y)
    C45TreeRegressor(...)
    >>> predictions = reg.predict(X)
    """

    def __init__(self, min_samples_leaf: int = 2,
                 max_depth: Optional[int] = None,
                 unpruned: bool = False):
        """Initialize C45TreeRegressor.

        Parameters
        ----------
        min_samples_leaf : int, default=2
            Minimum samples at leaf node.
        max_depth : int or None, default=None
            Maximum tree depth.
        unpruned : bool, default=False
            If True, skip pruning.
        """
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.unpruned = unpruned
        self.tree_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "min_samples_leaf": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Minimum number of samples at a leaf node"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree"
            },
            "unpruned": {
                "type": "boolean",
                "default": False,
                "description": "If true, do not prune the tree"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L., Friedman, J.H., Olshen, R.A. & Stone, C.J. (1984). "
            "Classification and Regression Trees. Wadsworth."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "C45TreeRegressor":
        """Fit the C45TreeRegressor regression tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : C45TreeRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        rng = np.random.RandomState(0)

        config = TreeConfig(
            criterion="squared_error",
            max_depth=self.max_depth,
            min_samples_split=2 * self.min_samples_leaf,
            min_samples_leaf=self.min_samples_leaf,
        )

        self.tree_ = build_regressor_tree(X, y, config, rng)

        if not self.unpruned:
            self.tree_ = reduced_error_prune_regressor(self.tree_, X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.array([float(predict_single_numpy(self.tree_, X[i])[0])
                         for i in range(X.shape[0])])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R-squared score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"C45TreeRegressor(n_features={self.n_features_})"
        return f"C45TreeRegressor(min_samples_leaf={self.min_samples_leaf})"


# Backward compat aliases
C45DecisionTreeClassifier = C45TreeClassifier
C45DecisionTreeRegressor = C45TreeRegressor
