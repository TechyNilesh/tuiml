"""LogitBoostClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier
from tuiml.hub import registry

@classifier(tags=["ensemble", "boosting", "logistic"], version="1.0.0")
class LogitBoostClassifier(Classifier):
    """LogitBoostClassifier meta-classifier.

    LogitBoostClassifier performs additive logistic regression using boosting. It 
    iteratively fits regression functions to working responses derived 
    from the logistic loss function.

    Parameters
    ----------
    base_classifier : str or class, default='DecisionStumpClassifier'
        The base classifier to use.
    n_iterations : int, default=100
        The number of boosting iterations.
    shrinkage : float, default=0.1
        Learning rate (shrinkage parameter).
    max_depth : int, default=3
        Maximum depth of the regression trees used as weak learners.
        Depth 1 produces stumps; depth 3 captures feature interactions.
    use_resampling : bool, default=False
        Whether to use resampling instead of weighting.
    weight_threshold : float, default=100
        Weight mass percentage for resampling.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    estimators_ : list of list
        The collection of fitted estimators per class.
    classes_ : np.ndarray
        The unique class labels.
    n_classes_ : int
        The number of classes.

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import LogitBoostClassifier
    >>> clf = LogitBoostClassifier(n_iterations=100)
    >>> clf.fit(X_train, y_train)
    LogitBoostClassifier(...)
    >>> predictions = clf.predict(X_test)

    References
    ----------
    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive 
           Logistic Regression: A Statistical View of Boosting. 
           Annals of Statistics, 28(2), 337-407.
    """

    def __init__(self, base_classifier: Any = 'DecisionStumpClassifier',
                 n_iterations: int = 100,
                 shrinkage: float = 0.1,
                 max_depth: int = 3,
                 use_resampling: bool = False,
                 weight_threshold: float = 100,
                 random_state: Optional[int] = None):
        super().__init__()
        self.base_classifier = base_classifier
        self.n_iterations = n_iterations
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.use_resampling = use_resampling
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.estimators_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "DecisionStump",
                "description": "Base classifier name"
            },
            "n_iterations": {
                "type": "integer", "default": 100, "minimum": 1,
                "description": "Number of boosting iterations"
            },
            "shrinkage": {
                "type": "number", "default": 0.1, "minimum": 0.01, "maximum": 1.0,
                "description": "Learning rate (shrinkage)"
            },
            "max_depth": {
                "type": "integer", "default": 3, "minimum": 1,
                "description": "Maximum depth of regression tree weak learners"
            },
            "use_resampling": {
                "type": "boolean", "default": False,
                "description": "Use resampling instead of weighting"
            },
            "weight_threshold": {
                "type": "number", "default": 100, "minimum": 1,
                "description": "Weight mass percentage for resampling"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * n_iterations * base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Friedman, J., Hastie, T., & Tibshirani, R. (2000). "
            "Additive Logistic Regression: A Statistical View of Boosting. "
            "Annals of Statistics, 28(2), 337-407."
        ]

    def _get_base_class(self) -> Type[Classifier]:
        """Get the base classifier class."""
        if isinstance(self.base_classifier, str):
            return registry.get(self.base_classifier)
        elif isinstance(self.base_classifier, type):
            return self.base_classifier
        elif isinstance(self.base_classifier, Classifier):
            # Handle classifier instances by extracting their class
            return type(self.base_classifier)
        else:
            raise ValueError(f"Invalid base_classifier: {self.base_classifier}")

    def _softmax(self, F: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities.

        Parameters
        ----------
        F : np.ndarray of shape (n_samples, n_classes)
            Log-odds values.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Softmax probabilities.
        """
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / (np.sum(exp_F, axis=1, keepdims=True) + 1e-10)

    def _find_best_split(self, X: np.ndarray, z: np.ndarray,
                          weights: np.ndarray):
        """Find the best weighted regression split across all features."""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0
        total_wz = np.sum(weights * z)
        total_w = np.sum(weights)

        for feat in range(n_features):
            order = np.argsort(X[:, feat])
            x_sorted = X[order, feat]
            z_sorted = z[order]
            w_sorted = weights[order]

            cum_wz = np.cumsum(w_sorted * z_sorted)[:-1]
            cum_w = np.cumsum(w_sorted)[:-1]
            right_w = total_w - cum_w
            right_wz = total_wz - cum_wz

            valid = (x_sorted[:-1] != x_sorted[1:]) & (cum_w > 1e-10) & (right_w > 1e-10)
            if not np.any(valid):
                continue

            gain = (cum_wz[valid] ** 2 / cum_w[valid] +
                    right_wz[valid] ** 2 / right_w[valid])
            best_idx = np.argmax(gain)
            if gain[best_idx] > best_gain:
                best_gain = gain[best_idx]
                best_feature = feat
                pos = np.nonzero(valid)[0][best_idx]
                best_threshold = (x_sorted[pos] + x_sorted[pos + 1]) / 2.0

        return best_feature, best_threshold, best_gain

    def _build_regression_tree(self, X: np.ndarray, z: np.ndarray,
                                weights: np.ndarray, depth: int = 0) -> dict:
        """Build a weighted regression tree up to max_depth."""
        leaf_val = np.average(z, weights=weights) if np.sum(weights) > 1e-10 else 0.0

        if depth >= self.max_depth or len(z) < 4:
            return {'leaf': True, 'value': leaf_val}

        feat, thresh, gain = self._find_best_split(X, z, weights)
        if gain <= -np.inf:
            return {'leaf': True, 'value': leaf_val}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            return {'leaf': True, 'value': leaf_val}

        left = self._build_regression_tree(X[left_mask], z[left_mask],
                                            weights[left_mask], depth + 1)
        right = self._build_regression_tree(X[right_mask], z[right_mask],
                                             weights[right_mask], depth + 1)
        return {'leaf': False, 'feature': feat, 'threshold': thresh,
                'left': left, 'right': right}

    def _predict_tree(self, tree: dict, X: np.ndarray) -> np.ndarray:
        """Predict using a regression tree."""
        if tree['leaf']:
            return np.full(X.shape[0], tree['value'])

        predictions = np.empty(X.shape[0])
        left_mask = X[:, tree['feature']] <= tree['threshold']
        right_mask = ~left_mask

        if np.any(left_mask):
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.any(right_mask):
            predictions[right_mask] = self._predict_tree(tree['right'], X[right_mask])
        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogitBoostClassifier":
        """Fit the LogitBoostClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : LogitBoostClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Convert y to class indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # One-hot encode targets
        Y = np.zeros((n_samples, self.n_classes_))
        Y[np.arange(n_samples), y_idx] = 1

        # Initialize F (log-odds) to zero
        F = np.zeros((n_samples, self.n_classes_))

        # Initialize estimators list for each class
        self.estimators_ = [[] for _ in range(self.n_classes_)]

        rng = np.random.RandomState(self.random_state)

        for iteration in range(self.n_iterations):
            # Compute probabilities
            P = self._softmax(F)

            # For each class, fit a regression model
            for k in range(self.n_classes_):
                # Working response (Newton-Raphson step)
                p_k = P[:, k]
                y_k = Y[:, k]

                # Weights for weighted least squares
                weights = p_k * (1 - p_k)
                weights = np.clip(weights, 1e-6, None)

                # Working response
                z = (y_k - p_k) / (weights + 1e-6)
                z = np.clip(z, -4, 4)  # Truncate extreme values

                tree = self._build_regression_tree(X, z, weights)
                self.estimators_[k].append(tree)

                predictions = self._predict_tree(tree, X)
                F[:, k] += self.shrinkage * predictions

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_classes_))

        # Sum predictions from all estimators
        for k in range(self.n_classes_):
            for tree in self.estimators_[k]:
                predictions = self._predict_tree(tree, X)
                F[:, k] += self.shrinkage * predictions

        return self._softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"LogitBoostClassifier(base={self.base_classifier}, n_iter={self.n_iterations})"
        return f"LogitBoostClassifier(base_classifier={self.base_classifier})"
