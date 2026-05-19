"""Stochastic Gradient Descent (SGD) for linear classification and regression."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml._cpp_ext import linear as _sgd_cpp

# ── Loss / penalty name → int maps (must match sgd.h enums) ───────────
_CLASSIFIER_LOSS = {"hinge": 0, "log": 1, "modified_huber": 2}
_REGRESSOR_LOSS  = {"squared_error": 3, "huber": 4, "epsilon_insensitive": 5}
_PENALTY         = {"none": 0, "l2": 1, "l1": 2, "elasticnet": 3}


@classifier(tags=["linear", "sgd", "online"], version="1.0.0")
class SGDClassifier(Classifier):
    r"""Stochastic Gradient Descent classifier for **large-scale linear classification**.

    Implements **SGD learning** for linear classifiers with support for multiple
    loss functions including hinge (linear SVM), log (logistic regression), and
    modified Huber. It is particularly efficient for **large-scale datasets**
    and **online learning** via the ``partial_fit`` method.

    Overview
    --------
    The algorithm trains a linear classifier using mini-batch SGD:

    1. Initialize weights and bias to zero
    2. For each epoch, optionally shuffle the training data
    3. Process data in mini-batches for vectorized gradient computation
    4. Compute the loss gradient and regularization gradient for each batch
    5. Update weights using the learning rate (with inverse scaling schedule)
    6. For multiclass problems, train one-vs-all binary classifiers

    Theory
    ------
    At each step, the weights are updated using the gradient of the loss
    function with regularization:

    .. math::
        w \leftarrow w - \eta \left(\\nabla_w L(w, x, y) + \lambda \\nabla R(w)\\right)

    The supported loss functions are:

    - **Hinge loss** (linear SVM): :math:`L = \max(0, 1 - y \cdot w^T x)`
    - **Log loss** (logistic regression): :math:`L = \log(1 + e^{-y \cdot w^T x})`
    - **Modified Huber**: a smooth approximation combining hinge and log loss

    The learning rate follows an **inverse scaling** schedule:

    .. math::
        \eta_t = \\frac{\eta_0}{1 + t \cdot 10^{-4}}

    Parameters
    ----------
    loss : {"hinge", "log", "modified_huber"}, default="hinge"
        The loss function to be used:

        - ``"hinge"`` -- Gives a linear SVM.
        - ``"log"`` -- Gives logistic regression, a probabilistic classifier.
        - ``"modified_huber"`` -- Smooth loss that brings tolerance to outliers
          as well as probability estimates.
    learning_rate : float, default=0.01
        Initial learning rate for the weight updates.
    regularization : {"l1", "l2", "elasticnet", "none"}, default="l2"
        The penalty (regularization term) to be used.
    lambda_ : float, default=0.0001
        Regularization strength; must be a positive float.
    n_epochs : int, default=100
        Number of passes over the training data.
    random_state : int or None, default=None
        Seed used by the random number generator.
    shuffle : bool, default=True
        Whether the training data should be shuffled after each epoch.
    batch_size : int, default=256
        Mini-batch size for gradient updates. Larger batches enable better
        vectorization but may converge differently than pure SGD (batch_size=1).

    Attributes
    ----------
    weights_ : np.ndarray
        Weight vector of shape (n_features,) for binary or (n_classes, n_features)
        for multiclass.
    bias_ : float or np.ndarray
        Bias term (intercept).
    classes_ : np.ndarray
        Unique class labels found during training.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m \cdot \\text{n\_epochs})` where :math:`n` is
      the number of samples and :math:`m` is the number of features.
    - Prediction: :math:`O(m)` per sample.

    **When to use SGDClassifier:**

    - Large-scale datasets where batch solvers are too slow
    - Online or streaming learning scenarios (via ``partial_fit``)
    - When you want to switch between SVM and logistic regression by changing the loss
    - When L1 regularization (sparsity) is desired

    References
    ----------
    .. [Bottou2010] Bottou, L. (2010).
           **Large-Scale Machine Learning with Stochastic Gradient Descent.**
           *Proceedings of COMPSTAT 2010*, pp. 177-186.
           DOI: `10.1007/978-3-7908-2604-3_16 <https://doi.org/10.1007/978-3-7908-2604-3_16>`_

    .. [Zhang2004] Zhang, T. (2004).
           **Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms.**
           *Proceedings of the 21st International Conference on Machine Learning (ICML)*.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LogisticRegression` : Batch gradient descent logistic regression with L2 regularization.
    :class:`~tuiml.algorithms.linear.SGDRegressor` : SGD-trained linear regressor for large-scale regression.

    Examples
    --------
    Binary classification with SGD using log loss:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import SGDClassifier
    >>>
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> clf = SGDClassifier(loss='log', learning_rate=0.01)
    >>> clf.fit(X, y)
    >>>
    >>> clf.predict([[2.5, 2.5]])
    array([1])
    """

    def __init__(self, loss: str = 'hinge',
                 learning_rate: float = 0.01,
                 regularization: str = 'l2',
                 lambda_: float = 0.0001,
                 n_epochs: int = 100,
                 random_state: Optional[int] = None,
                 shuffle: bool = True,
                 batch_size: int = 256):
        """Initialize SGDClassifier.

        Parameters
        ----------
        loss : str, default='hinge'
            Loss function ('hinge', 'log', or 'modified_huber').
        learning_rate : float, default=0.01
            Initial learning rate.
        regularization : str, default='l2'
            Regularization type ('l1', 'l2', 'elasticnet', or 'none').
        lambda_ : float, default=0.0001
            Regularization strength.
        n_epochs : int, default=100
            Number of training epochs.
        random_state : int or None, default=None
            Random seed for reproducibility.
        shuffle : bool, default=True
            Whether to shuffle data each epoch.
        batch_size : int, default=256
            Mini-batch size for gradient updates.
        """
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self._n_classes = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "loss": {
                "type": "string", "default": "hinge",
                "enum": ["hinge", "log", "modified_huber"],
                "description": "Loss function"
            },
            "learning_rate": {
                "type": "number", "default": 0.01, "minimum": 0.0001,
                "description": "Initial learning rate"
            },
            "regularization": {
                "type": "string", "default": "l2",
                "enum": ["l1", "l2", "elasticnet", "none"],
                "description": "Regularization type"
            },
            "lambda_": {
                "type": "number", "default": 0.0001, "minimum": 0,
                "description": "Regularization strength"
            },
            "n_epochs": {
                "type": "integer", "default": 100, "minimum": 1,
                "description": "Number of training epochs"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            },
            "shuffle": {
                "type": "boolean", "default": True,
                "description": "Shuffle data each epoch"
            },
            "batch_size": {
                "type": "integer", "default": 256, "minimum": 1,
                "description": "Mini-batch size for gradient updates"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * d * n_epochs)"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Bottou, L. (2010). Large-Scale Machine Learning with "
            "Stochastic Gradient Descent. COMPSTAT 2010."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDClassifier":
        """Fit the SGD classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : SGDClassifier
            Fitted classifier.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int32)

        n_out = 1 if self._n_classes == 2 else self._n_classes
        n_features = X.shape[1]

        result = _sgd_cpp.sgd_fit_classifier(
            X, y_idx,
            n_classes    = self._n_classes,
            loss_type    = _CLASSIFIER_LOSS[self.loss],
            penalty_type = _PENALTY[self.regularization],
            alpha        = self.lambda_,
            l1_ratio     = 0.5,
            eta0         = self.learning_rate,
            lr_schedule  = "invscaling",
            power_t      = 0.25,
            n_epochs     = self.n_epochs,
            batch_size   = self.batch_size,
            tol          = 1e-3,
            patience     = 3,
            shuffle      = self.shuffle,
            random_seed  = self.random_state if self.random_state is not None else 0,
            weights_init = np.empty(0),
            bias_init    = np.empty(0),
        )

        W = np.array(result.weights)
        b = np.array(result.bias)

        if self._n_classes == 2:
            self.weights_ = W  # shape (n_features,)
            self.bias_ = b[0]
        else:
            self.weights_ = W.reshape(n_out, n_features)
            self.bias_ = b

        self._is_fitted = True
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray
            Decision function scores.
        """
        if self._n_classes == 2:
            return X @ self.weights_ + self.bias_
        else:
            return X @ self.weights_.T + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = self._score(X)

        if self._n_classes == 2:
            indices = (scores >= 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = self._score(X)

        if self._n_classes == 2:
            prob_pos = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "SGDClassifier":
        """Update classifier with new samples (online learning).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New training samples.
        y : np.ndarray of shape (n_samples,)
            New target labels.

        Returns
        -------
        self : SGDClassifier
            Updated classifier.
        """
        if not self._is_fitted:
            return self.fit(X, y)

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int32)

        n_out = 1 if self._n_classes == 2 else self._n_classes
        n_features = X.shape[1]

        w_init = self.weights_.ravel().astype(np.float64)
        b_init = np.atleast_1d(self.bias_).astype(np.float64)

        result = _sgd_cpp.sgd_fit_classifier(
            X, y_idx,
            n_classes    = self._n_classes,
            loss_type    = _CLASSIFIER_LOSS[self.loss],
            penalty_type = _PENALTY[self.regularization],
            alpha        = self.lambda_,
            l1_ratio     = 0.5,
            eta0         = self.learning_rate,
            lr_schedule  = "invscaling",
            power_t      = 0.25,
            n_epochs     = 1,
            batch_size   = self.batch_size,
            tol          = 1e-3,
            patience     = 3,
            shuffle      = self.shuffle,
            random_seed  = self.random_state if self.random_state is not None else 0,
            weights_init = w_init,
            bias_init    = b_init,
        )

        W = np.array(result.weights)
        b = np.array(result.bias)
        if self._n_classes == 2:
            self.weights_ = W
            self.bias_ = b[0]
        else:
            self.weights_ = W.reshape(n_out, n_features)
            self.bias_ = b

        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"SGDClassifier(loss={self.loss}, n_classes={self._n_classes})"
        return f"SGDClassifier(loss={self.loss})"


@regressor(tags=["linear", "sgd", "online"], version="1.0.0")
class SGDRegressor(Regressor):
    r"""Stochastic Gradient Descent regressor for **large-scale linear regression**.

    Implements **SGD learning** for linear regression with support for
    squared error and Huber loss functions. It is particularly efficient for
    **large-scale datasets** and supports **online learning** via the
    ``partial_fit`` method.

    Overview
    --------
    The algorithm trains a linear regression model using mini-batch SGD:

    1. Initialize weights and bias to zero
    2. For each epoch, optionally shuffle the training data
    3. Process data in mini-batches for vectorized gradient computation
    4. Compute the loss gradient (squared error or Huber) and regularization gradient
    5. Update weights using the learning rate (with inverse scaling schedule)
    6. Repeat for ``n_epochs`` passes over the data

    Theory
    ------
    The model fits a linear function :math:`f(x) = w^T x + b` by minimizing
    the regularized empirical risk:

    .. math::
        \min_{w, b} \\frac{1}{n} \sum_{i=1}^{n} L(y_i, w^T x_i + b)
        + \lambda R(w)

    **Squared error loss:**

    .. math::
        L(y, \hat{y}) = \\frac{1}{2}(y - \hat{y})^2

    **Huber loss** (robust to outliers):

    .. math::
        L_\delta(y, \hat{y}) = \\begin{cases}
            \\frac{1}{2}(y - \hat{y})^2 & \\text{if } |y - \hat{y}| \leq \epsilon \\
            \epsilon |y - \hat{y}| - \\frac{1}{2}\epsilon^2 & \\text{otherwise}
        \end{cases}

    The weight update rule at each step is:

    .. math::
        w \leftarrow w - \eta_t \left(\\nabla_w L + \lambda \\nabla R(w)\\right)

    with the inverse scaling learning rate schedule
    :math:`\eta_t = \eta_0 / (1 + t \cdot 10^{-4})`.

    Parameters
    ----------
    loss : {"squared_error", "huber"}, default="squared_error"
        The loss function to be used:

        - ``"squared_error"`` -- Ordinary least squares.
        - ``"huber"`` -- Huber loss for robustness to outliers.
    learning_rate : float, default=0.01
        Initial learning rate for the weight updates.
    regularization : {"l1", "l2", "elasticnet", "none"}, default="l2"
        The penalty (regularization term) to be used.
    lambda_ : float, default=0.0001
        Regularization strength; must be a positive float.
    n_epochs : int, default=100
        Number of passes over the training data.
    epsilon : float, default=0.1
        The epsilon threshold for the Huber loss. Residuals smaller than
        epsilon are penalized quadratically; larger residuals are penalized
        linearly.
    random_state : int or None, default=None
        Seed used by the random number generator.
    shuffle : bool, default=True
        Whether the training data should be shuffled after each epoch.
    batch_size : int, default=256
        Mini-batch size for gradient updates. Larger batches enable better
        vectorization but may converge differently than pure SGD (batch_size=1).

    Attributes
    ----------
    weights_ : np.ndarray
        Weight vector of shape (n_features,).
    bias_ : float
        Bias term (intercept).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m \cdot \\text{n\_epochs})` where :math:`n` is
      the number of samples and :math:`m` is the number of features.
    - Prediction: :math:`O(m)` per sample.

    **When to use SGDRegressor:**

    - Large-scale regression datasets where closed-form OLS is too slow
    - Online or streaming regression scenarios (via ``partial_fit``)
    - When Huber loss is needed for robustness to outliers
    - When L1 regularization (sparse solutions) is desired

    References
    ----------
    .. [Bottou2010] Bottou, L. (2010).
           **Large-Scale Machine Learning with Stochastic Gradient Descent.**
           *Proceedings of COMPSTAT 2010*, pp. 177-186.
           DOI: `10.1007/978-3-7908-2604-3_16 <https://doi.org/10.1007/978-3-7908-2604-3_16>`_

    .. [Huber1964] Huber, P.J. (1964).
           **Robust Estimation of a Location Parameter.**
           *The Annals of Mathematical Statistics*, 35(1), 73-101.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LinearRegression` : Closed-form OLS regression with ridge regularization.
    :class:`~tuiml.algorithms.linear.SimpleLinearRegression` : Univariate regression using the single best attribute.
    :class:`~tuiml.algorithms.linear.SGDClassifier` : SGD-trained linear classifier for classification tasks.

    Examples
    --------
    Regression with SGD using squared error loss:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import SGDRegressor
    >>>
    >>> # Generate simple linear data
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>>
    >>> # Fit the model
    >>> reg = SGDRegressor(learning_rate=0.01, n_epochs=1000, random_state=42)
    >>> reg.fit(X, y)
    >>>
    >>> # Predict
    >>> reg.predict([[6]])
    """

    def __init__(self, loss: str = 'squared_error',
                 learning_rate: float = 0.01,
                 regularization: str = 'l2',
                 lambda_: float = 0.0001,
                 n_epochs: int = 100,
                 epsilon: float = 0.1,
                 random_state: Optional[int] = None,
                 shuffle: bool = True,
                 batch_size: int = 256):
        """Initialize SGDRegressor.

        Parameters
        ----------
        loss : str, default='squared_error'
            Loss function ('squared_error' or 'huber').
        learning_rate : float, default=0.01
            Initial learning rate.
        regularization : str, default='l2'
            Regularization type ('l1', 'l2', 'elasticnet', or 'none').
        lambda_ : float, default=0.0001
            Regularization strength.
        n_epochs : int, default=100
            Number of training epochs.
        epsilon : float, default=0.1
            Epsilon threshold for Huber loss.
        random_state : int or None, default=None
            Random seed for reproducibility.
        shuffle : bool, default=True
            Whether to shuffle data each epoch.
        batch_size : int, default=256
            Mini-batch size for gradient updates.
        """
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.random_state = random_state
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "loss": {
                "type": "string", "default": "squared_error",
                "enum": ["squared_error", "huber"],
                "description": "Loss function"
            },
            "learning_rate": {
                "type": "number", "default": 0.01, "minimum": 0.0001,
                "description": "Initial learning rate"
            },
            "regularization": {
                "type": "string", "default": "l2",
                "enum": ["l1", "l2", "elasticnet", "none"],
                "description": "Regularization type"
            },
            "lambda_": {
                "type": "number", "default": 0.0001, "minimum": 0,
                "description": "Regularization strength"
            },
            "n_epochs": {
                "type": "integer", "default": 100, "minimum": 1,
                "description": "Number of training epochs"
            },
            "epsilon": {
                "type": "number", "default": 0.1,
                "description": "Epsilon for Huber loss"
            },
            "random_state": {
                "type": "integer", "default": None,
                "description": "Random seed"
            },
            "shuffle": {
                "type": "boolean", "default": True,
                "description": "Shuffle data each epoch"
            },
            "batch_size": {
                "type": "integer", "default": 256, "minimum": 1,
                "description": "Mini-batch size for gradient updates"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "regression"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegressor":
        """Fit the SGD regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SGDRegressor
            Fitted regressor.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = _sgd_cpp.sgd_fit_regressor(
            X, y,
            loss_type    = _REGRESSOR_LOSS[self.loss],
            penalty_type = _PENALTY[self.regularization],
            alpha        = self.lambda_,
            l1_ratio     = 0.5,
            eta0         = self.learning_rate,
            lr_schedule  = "invscaling",
            power_t      = 0.25,
            epsilon      = self.epsilon,
            n_epochs     = self.n_epochs,
            batch_size   = self.batch_size,
            tol          = 1e-3,
            patience     = 3,
            shuffle      = self.shuffle,
            random_seed  = self.random_state if self.random_state is not None else 0,
            weights_init = np.empty(0),
            bias_init    = np.empty(0),
        )

        self.weights_ = np.array(result.weights)
        self.bias_ = result.bias[0]
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.weights_ + self.bias_

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegressor":
        """Update regressor with new samples (online learning).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New training samples.
        y : np.ndarray of shape (n_samples,)
            New target values.

        Returns
        -------
        self : SGDRegressor
            Updated regressor.
        """
        if not self._is_fitted:
            return self.fit(X, y)

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = _sgd_cpp.sgd_fit_regressor(
            X, y,
            loss_type    = _REGRESSOR_LOSS[self.loss],
            penalty_type = _PENALTY[self.regularization],
            alpha        = self.lambda_,
            l1_ratio     = 0.5,
            eta0         = self.learning_rate,
            lr_schedule  = "invscaling",
            power_t      = 0.25,
            epsilon      = self.epsilon,
            n_epochs     = 1,
            batch_size   = self.batch_size,
            tol          = 1e-3,
            patience     = 3,
            shuffle      = self.shuffle,
            random_seed  = self.random_state if self.random_state is not None else 0,
            weights_init = self.weights_.astype(np.float64),
            bias_init    = np.array([self.bias_], dtype=np.float64),
        )

        self.weights_ = np.array(result.weights)
        self.bias_ = result.bias[0]
        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"SGDRegressor(loss={self.loss})"
        return f"SGDRegressor(loss={self.loss}, not fitted)"
