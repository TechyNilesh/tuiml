"""AdaBoostClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.hub import registry

@classifier(tags=["ensembles", "boosting", "meta"], version="1.0.0")
class AdaBoostClassifier(Classifier):
    """AdaBoostClassifier for **adaptive boosting** in multiclass classification.

    AdaBoostClassifier (also known as AdaBoost.M1) builds an ensemble by iteratively
    training **weak learners** on weighted versions of the data, focusing on
    previously misclassified instances, and combining their predictions using
    **weighted majority voting**.

    Overview
    --------
    The algorithm proceeds as follows:

    1. Initialize uniform sample weights :math:`w_i = 1/n` for all training instances
    2. For each boosting iteration :math:`t = 1, \\ldots, T`:
       a. Train a weak learner :math:`h_t` on the weighted training data
       b. Compute the weighted error :math:`\\epsilon_t`
       c. Calculate the estimator weight :math:`\\alpha_t`
       d. Update sample weights, increasing weights on misclassified instances
    3. Combine all weak learners via weighted majority vote

    Theory
    ------
    The weighted classification error at iteration :math:`t` is:

    .. math::
        \\epsilon_t = \\sum_{i: h_t(x_i) \\neq y_i} w_i

    The estimator weight for the multiclass case (SAMME) is:

    .. math::
        \\alpha_t = \\ln\\!\\left(\\frac{1 - \\epsilon_t}{\\epsilon_t}\\right) + \\ln(K - 1)

    where :math:`K` is the number of classes. Sample weights are updated as:

    .. math::
        w_i \\leftarrow w_i \\cdot \\exp\\!\\bigl(\\alpha_t \\cdot \\mathbb{1}[h_t(x_i) \\neq y_i]\\bigr)

    The final prediction is obtained by weighted majority vote:

    .. math::
        H(x) = \\arg\\max_{k} \\sum_{t=1}^{T} \\alpha_t \\cdot \\mathbb{1}[h_t(x) = k]

    Parameters
    ----------
    base_classifier : str or class, default='DecisionStumpClassifier'
        The base classifier to use as weak learner. Typically a simple
        classifier such as a decision stump.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        More estimators can improve accuracy but may lead to overfitting.

    weight_threshold : float, default=100
        Weight threshold for triggering resampling. When the ratio between
        maximum and minimum sample weights exceeds this value, the data
        is resampled according to the current weight distribution.

    random_state : int or None, default=None
        Random seed for reproducibility. Controls the resampling process.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted weak learners.

    estimator_weights_ : np.ndarray
        Weights :math:`\\alpha_t` assigned to each estimator.

    classes_ : np.ndarray
        The unique class labels discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot C_{\\text{base}})` where :math:`T` = n_estimators,
      :math:`n` = number of samples, :math:`C_{\\text{base}}` = base classifier complexity
    - Prediction: :math:`O(T \\cdot C_{\\text{predict}})` per sample

    **When to use AdaBoostClassifier:**

    - When a simple base learner (e.g., decision stump) needs to be boosted
    - Binary or multiclass classification tasks with moderate noise
    - When you want an interpretable ensemble (weighted sum of simple rules)
    - When training data is relatively clean (AdaBoost is sensitive to outliers)

    References
    ----------
    .. [Freund1996] Freund, Y. and Schapire, R.E. (1996).
           **Experiments with a New Boosting Algorithm.**
           *Proceedings of the 13th International Conference on Machine Learning*,
           pp. 148-156.

    .. [Freund1997] Freund, Y. and Schapire, R.E. (1997).
           **A Decision-Theoretic Generalization of On-Line Learning and an
           Application to Boosting.**
           *Journal of Computer and System Sciences*, 55(1), 119-139.
           DOI: `10.1006/jcss.1997.1504 <https://doi.org/10.1006/jcss.1997.1504>`_

    .. [Hastie2009] Hastie, T., Tibshirani, R. and Friedman, J. (2009).
           **The Elements of Statistical Learning.**
           *Springer*, Chapter 10.

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.BaggingClassifier` : Bootstrap aggregating ensemble method.
    :class:`~tuiml.algorithms.ensemble.VotingClassifier` : Combines classifiers via voting rules.
    :class:`~tuiml.algorithms.ensemble.StackingClassifier` : Stacked generalization meta-classifier.

    Examples
    --------
    Basic usage for multiclass classification with AdaBoost:

    >>> from tuiml.algorithms.ensemble import AdaBoostClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the AdaBoost classifier
    >>> clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    >>> clf.fit(X_train, y_train)
    AdaBoostClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, base_classifier: Any = 'DecisionStumpClassifier',
                 n_estimators: int = 50,
                 weight_threshold: float = 100,
                 random_state: Optional[int] = None):
        """Initialize AdaBoostClassifier.

        Parameters
        ----------
        base_classifier : str or class, default='DecisionStumpClassifier'
            The base classifier to use as weak learner.
        n_estimators : int, default=50
            Maximum number of boosting iterations.
        weight_threshold : float, default=100
            Weight ratio threshold for triggering resampling.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.estimators_ = None
        self.estimator_weights_ = None
        self.classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {"type": "string", "default": "DecisionStump",
                               "description": "Base classifier name"},
            "n_estimators": {"type": "integer", "default": 50, "minimum": 1,
                            "description": "Number of boosting iterations"},
            "weight_threshold": {"type": "number", "default": 100, "minimum": 1,
                                "description": "Weight threshold for resampling"},
            "random_state": {"type": "integer", "default": None,
                            "description": "Random seed"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * base_complexity * n_estimators)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Freund, Y., & Schapire, R.E. (1996). Experiments with a New "
                "Boosting Algorithm. Proceedings of ICML, 148-156."]

    def _get_base_class(self) -> Type[Classifier]:
        """Resolve the base classifier specification to a class.

        Parameters
        ----------
        None

        Returns
        -------
        base_class : Type[Classifier]
            The resolved classifier class.
        """
        if isinstance(self.base_classifier, str):
            return registry.get(self.base_classifier)
        elif isinstance(self.base_classifier, type):
            return self.base_classifier
        elif isinstance(self.base_classifier, Classifier):
            # Handle classifier instances by extracting their class
            return type(self.base_classifier)
        else:
            raise ValueError(f"Invalid base_classifier: {self.base_classifier}")

    def _resample(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  rng: np.random.RandomState) -> tuple:
        """Resample training data according to the current weight distribution.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        weights : np.ndarray of shape (n_samples,)
            Sample weights used as sampling probabilities.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        X_resampled : np.ndarray of shape (n_samples, n_features)
            Resampled training data.
        y_resampled : np.ndarray of shape (n_samples,)
            Resampled target labels.
        """
        # Normalize weights
        weights = weights / np.sum(weights)

        # Resample
        indices = rng.choice(len(y), size=len(y), replace=True, p=weights)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the AdaBoostClassifier.M1 classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : AdaBoostClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self._base_class = self._get_base_class()

        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        sample_weight = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []

        for i in range(self.n_estimators):
            # Check if weights need resampling
            if np.max(sample_weight) > self.weight_threshold * np.min(sample_weight):
                X_train, y_train = self._resample(X, y, sample_weight, rng)
                sample_weight = np.ones(n_samples) / n_samples
            else:
                X_train, y_train = X, y

            # Train weak learner
            estimator = self._base_class()

            # Some classifiers support sample_weight in fit
            try:
                estimator.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                estimator.fit(X_train, y_train)

            # Get predictions
            predictions = estimator.predict(X)

            # Calculate weighted error
            incorrect = predictions != y
            weighted_error = np.sum(sample_weight * incorrect)

            # Check if error is too high
            if weighted_error >= 1 - 1 / n_classes:
                if len(self.estimators_) == 0:
                    # Keep at least one estimator
                    self.estimators_.append(estimator)
                    self.estimator_weights_.append(1.0)
                break

            # Calculate estimator weight (for multiclass)
            if weighted_error <= 0:
                alpha = 10  # Cap at large value
            else:
                alpha = np.log((1 - weighted_error) / weighted_error) + np.log(n_classes - 1)

            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)

            if weighted_error <= 0:
                break

            # Update sample weights
            sample_weight = sample_weight * np.exp(alpha * incorrect)
            sample_weight = sample_weight / np.sum(sample_weight)

        self.estimator_weights_ = np.array(self.estimator_weights_)
        self._is_fitted = True
        return self

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
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Weighted vote
        class_votes = np.zeros((n_samples, n_classes))

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            for i, pred in enumerate(predictions):
                idx = np.where(self.classes_ == pred)[0]
                if len(idx) > 0:
                    class_votes[i, idx[0]] += weight

        return self.classes_[np.argmax(class_votes, axis=1)]

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
        n_classes = len(self.classes_)

        class_votes = np.zeros((n_samples, n_classes))

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            try:
                proba = estimator.predict_proba(X)
                class_votes += weight * proba
            except NotImplementedError:
                predictions = estimator.predict(X)
                for i, pred in enumerate(predictions):
                    idx = np.where(self.classes_ == pred)[0]
                    if len(idx) > 0:
                        class_votes[i, idx[0]] += weight

        # Normalize to probabilities
        proba = class_votes / class_votes.sum(axis=1, keepdims=True)
        return proba

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"AdaBoostClassifier(base={self.base_classifier}, "
                   f"n_estimators={len(self.estimators_)})")
        return f"AdaBoostClassifier(n_estimators={self.n_estimators})"


@regressor(tags=["ensembles", "boosting", "meta"], version="1.0.0")
class AdaBoostRegressor(Regressor):
    """AdaBoost.R2 regressor for **adaptive boosting** in regression tasks.

    AdaBoostRegressor implements the AdaBoost.R2 algorithm, which builds an
    ensemble by iteratively training **base regressors** on weighted versions
    of the data. Instances with larger prediction errors receive higher
    weights in subsequent iterations, and the final prediction is computed
    as a **weighted median** of the individual estimator predictions.

    Overview
    --------
    The algorithm proceeds as follows:

    1. Initialize uniform sample weights :math:`w_i = 1/n`
    2. For each boosting iteration :math:`t = 1, \\ldots, T`:
       a. Fit a base regressor :math:`h_t` on the weighted training data
       b. Compute predictions and the maximum absolute error :math:`D`
       c. Compute relative losses :math:`L_i = |y_i - h_t(x_i)| / D`
       d. Compute weighted average loss :math:`\\bar{L}`
       e. If :math:`\\bar{L} \\geq 0.5`, stop boosting
       f. Compute :math:`\\beta_t = \\bar{L} / (1 - \\bar{L})`
       g. Update weights: :math:`w_i \\leftarrow w_i \\cdot \\beta_t^{1 - L_i}`
       h. Normalize weights
    3. Combine estimators using the **weighted median**

    Theory
    ------
    The relative loss for each sample at iteration :math:`t` is:

    .. math::
        L_i = \\frac{|y_i - h_t(x_i)|}{D_t}

    where :math:`D_t = \\max_i |y_i - h_t(x_i)|` is the maximum absolute error.
    The weighted average loss is:

    .. math::
        \\bar{L}_t = \\sum_{i=1}^{n} w_i \\cdot L_i

    The estimator confidence is:

    .. math::
        \\beta_t = \\frac{\\bar{L}_t}{1 - \\bar{L}_t}

    Weights are updated as:

    .. math::
        w_i \\leftarrow w_i \\cdot \\beta_t^{1 - L_i}

    The final prediction is the **weighted median** of all estimator
    predictions, using :math:`\\log(1/\\beta_t)` as the estimator weight.

    Parameters
    ----------
    base_regressor : str or class, default='AdditiveRegression'
        The base regressor to use as weak learner. Can be a string name
        (resolved via the hub registry) or a class/instance.
    n_estimators : int, default=10
        The maximum number of boosting iterations.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimators_ : list
        The collection of fitted base regressors.
    estimator_weights_ : np.ndarray
        Log-confidence weights :math:`\\log(1/\\beta_t)` for each estimator.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot C_{\\text{base}})` where :math:`T` = n_estimators
    - Prediction: :math:`O(T \\cdot n \\cdot \\log T)` per batch due to weighted median

    **When to use AdaBoostRegressor:**

    - When a simple base regressor needs to be boosted for better accuracy
    - Regression tasks with moderate noise levels
    - When you want an interpretable ensemble of simple models
    - When training data is relatively clean (AdaBoost is sensitive to outliers)

    References
    ----------
    .. [Drucker1997] Drucker, H. (1997).
           **Improving Regressors Using Boosting Techniques.**
           *Proceedings of the 14th International Conference on Machine Learning*,
           pp. 107-115.

    .. [Freund1997] Freund, Y. and Schapire, R.E. (1997).
           **A Decision-Theoretic Generalization of On-Line Learning and an
           Application to Boosting.**
           *Journal of Computer and System Sciences*, 55(1), 119-139.
           DOI: `10.1006/jcss.1997.1504 <https://doi.org/10.1006/jcss.1997.1504>`_

    See Also
    --------
    :class:`~tuiml.algorithms.ensemble.AdaBoostClassifier` : AdaBoost for classification tasks.
    :class:`~tuiml.algorithms.ensemble.AdditiveRegression` : Gradient boosting for regression.
    :class:`~tuiml.algorithms.ensemble.BaggingRegressor` : Bootstrap aggregating for regression.

    Examples
    --------
    Basic usage for regression with AdaBoost.R2:

    >>> from tuiml.algorithms.ensemble import AdaBoostRegressor
    >>> import numpy as np
    >>> X_train = np.array([[1], [2], [3], [4], [5]])
    >>> y_train = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    >>> reg = AdaBoostRegressor(n_estimators=50, random_state=42)
    >>> reg.fit(X_train, y_train)
    AdaBoostRegressor(...)
    """

    def __init__(self, base_regressor: Any = 'AdditiveRegression',
                 n_estimators: int = 10,
                 random_state: Optional[int] = None):
        """Initialize AdaBoostRegressor.

        Parameters
        ----------
        base_regressor : str or class, default='AdditiveRegression'
            The base regressor to use as weak learner.
        n_estimators : int, default=10
            Maximum number of boosting iterations.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.base_regressor = base_regressor
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = None
        self.estimator_weights_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "base_regressor": {"type": "string", "default": "AdditiveRegression",
                              "description": "Base regressor name"},
            "n_estimators": {"type": "integer", "default": 50, "minimum": 1,
                            "description": "Number of boosting iterations"},
            "random_state": {"type": "integer", "default": None,
                            "description": "Random seed"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * base_complexity * n_estimators)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return ["Drucker, H. (1997). Improving Regressors Using Boosting "
                "Techniques. Proceedings of ICML, 107-115."]

    def _get_base_class(self) -> Type:
        """Resolve the base regressor specification to a class.

        Returns
        -------
        base_class : Type[Regressor]
            The resolved regressor class.
        """
        if isinstance(self.base_regressor, str):
            return registry.get(self.base_regressor)
        elif isinstance(self.base_regressor, type):
            return self.base_regressor
        elif isinstance(self.base_regressor, Regressor):
            return type(self.base_regressor)
        else:
            raise ValueError(f"Invalid base_regressor: {self.base_regressor}")

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute the weighted median of a set of values.

        Parameters
        ----------
        values : np.ndarray of shape (n,)
            The values to compute the weighted median over.
        weights : np.ndarray of shape (n,)
            The weights for each value.

        Returns
        -------
        median : float
            The weighted median value.
        """
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumulative_weight = np.cumsum(sorted_weights)
        total_weight = cumulative_weight[-1]
        median_idx = np.searchsorted(cumulative_weight, total_weight / 2.0)

        return sorted_values[min(median_idx, len(sorted_values) - 1)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostRegressor":
        """Fit the AdaBoost.R2 regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : AdaBoostRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(y)
        self._base_class = self._get_base_class()
        rng = np.random.RandomState(self.random_state)

        # Initialize uniform sample weights
        sample_weight = np.ones(n_samples) / n_samples

        self.estimators_ = []
        betas = []

        for t in range(self.n_estimators):
            # Resample according to weights
            indices = rng.choice(n_samples, size=n_samples, replace=True,
                                 p=sample_weight)
            X_resampled = X[indices]
            y_resampled = y[indices]

            # Fit base regressor
            estimator = self._base_class()
            try:
                estimator.fit(X_resampled, y_resampled)
            except Exception:
                break

            # Compute predictions on original data
            predictions = estimator.predict(X)

            # Compute absolute errors and max loss D
            abs_errors = np.abs(y - predictions)
            D = np.max(abs_errors)

            if D == 0:
                # Perfect fit, keep and stop
                self.estimators_.append(estimator)
                betas.append(1e-10)
                break

            # Compute relative losses (linear loss)
            L = abs_errors / D

            # Compute weighted average loss
            loss_avg = np.sum(sample_weight * L)

            if loss_avg >= 0.5:
                # Error too high, stop boosting
                if len(self.estimators_) == 0:
                    self.estimators_.append(estimator)
                    betas.append(1.0)
                break

            # Compute beta
            beta = loss_avg / (1 - loss_avg)

            self.estimators_.append(estimator)
            betas.append(beta)

            # Update weights
            sample_weight = sample_weight * np.power(beta, 1 - L)

            # Normalize weights
            weight_sum = np.sum(sample_weight)
            if weight_sum > 0:
                sample_weight = sample_weight / weight_sum
            else:
                break

        # Compute estimator weights as log(1/beta)
        betas = np.array(betas)
        betas = np.maximum(betas, 1e-10)
        self.estimator_weights_ = np.log(1.0 / betas)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using weighted median of estimators.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Collect predictions from all estimators
        all_predictions = np.array([est.predict(X) for est in self.estimators_])

        # Compute weighted median for each sample
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = self._weighted_median(
                all_predictions[:, i], self.estimator_weights_)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R-squared score.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"AdaBoostRegressor(base={self.base_regressor}, "
                   f"n_estimators={len(self.estimators_)})")
        return f"AdaBoostRegressor(n_estimators={self.n_estimators})"
