"""MultilayerPerceptronClassifier (Neural Network) classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor

@classifier(tags=["functions", "neural-network", "deep-learning"], version="1.0.0")
class MultilayerPerceptronClassifier(Classifier):
    """Multilayer Perceptron (Neural Network) classifier.

    A **feedforward artificial neural network** trained using
    **backpropagation**. It supports multiple hidden layers, various
    activation functions, momentum-based updates, and early stopping.

    Overview
    --------
    The Multilayer Perceptron trains by iteratively adjusting weights through
    forward and backward passes:

    1. Initialize weights using **Xavier initialization** for each layer
    2. Standardize input features to zero mean and unit variance
    3. Perform a **forward pass**: propagate inputs through hidden layers
       applying activation functions, then softmax at the output layer
    4. Compute the **cross-entropy loss** between predictions and targets
    5. Perform a **backward pass** (backpropagation): compute gradients of the
       loss with respect to each weight and bias
    6. Update weights using gradient descent with **momentum**
    7. Optionally decay the learning rate over epochs
    8. Stop early if the loss does not improve for ``validation_threshold``
       consecutive epochs

    Theory
    ------
    For a network with :math:`L` layers, the forward pass computes activations
    layer by layer. For hidden layer :math:`l`:

    .. math::
        \\mathbf{z}^{(l)} = \\mathbf{a}^{(l-1)} \\mathbf{W}^{(l)} + \\mathbf{b}^{(l)}

    .. math::
        \\mathbf{a}^{(l)} = g(\\mathbf{z}^{(l)})

    where :math:`g` is the activation function (sigmoid or ReLU). The output
    layer uses softmax:

    .. math::
        \\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}

    The network is trained by minimizing the **cross-entropy loss**:

    .. math::
        \\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{k=1}^{K} y_{ik} \\log(\\hat{y}_{ik})

    Weights are updated using momentum-based gradient descent:

    .. math::
        \\mathbf{v}_t = \\mu \\, \\mathbf{v}_{t-1} - \\eta \\, \\nabla \\mathcal{L}

    .. math::
        \\mathbf{W}_t = \\mathbf{W}_{t-1} + \\mathbf{v}_t

    where :math:`\\mu` is the momentum and :math:`\\eta` is the learning rate.

    Parameters
    ----------
    hidden_layers : list of int, default=[100]
        Sizes of the hidden layers. Each entry in the list represents
        the number of neurons in that hidden layer.
    learning_rate : float, default=0.1
        Learning rate for weight updates.
    momentum : float, default=0.9
        Momentum for gradient descent to accelerate convergence and
        avoid local minima.
    max_epochs : int, default=500
        Maximum number of training epochs.
    validation_threshold : int, default=20
        Number of epochs to wait for improvement in loss before
        stopping (patience).
    decay : bool, default=True
        Whether to decay the learning rate over time.
    activation : {'sigmoid', 'relu'}, default='relu'
        Activation function for hidden layers.
    random_state : int or None, default=None
        Seed for the random number generator.

    Attributes
    ----------
    weights_ : list of np.ndarray
        Weight matrices connecting each layer.
    biases_ : list of np.ndarray
        Bias vectors for each layer.
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels discovered during :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot E \\cdot \\sum_{l=1}^{L} d_l \\cdot d_{l+1})`
      where :math:`n` = number of samples, :math:`E` = number of epochs, and
      :math:`d_l` = size of layer :math:`l`
    - Prediction: :math:`O(n \\cdot \\sum_{l=1}^{L} d_l \\cdot d_{l+1})` per
      batch

    **When to use MultilayerPerceptronClassifier:**

    - When the decision boundary is non-linear
    - When you need a flexible model that can approximate complex functions
    - For problems like XOR that are not linearly separable
    - When sufficient training data is available to fit the network parameters
    - As a lightweight alternative to deep learning frameworks

    References
    ----------
    .. [Rumelhart1986] Rumelhart, D.E., Hinton, G.E. and Williams, R.J. (1986).
           **Learning Representations by Back-Propagating Errors.**
           *Nature*, 323, 533-536.
           DOI: `10.1038/323533a0 <https://doi.org/10.1038/323533a0>`_

    .. [Glorot2010] Glorot, X. and Bengio, Y. (2010).
           **Understanding the Difficulty of Training Deep Feedforward Neural
           Networks.**
           *Proceedings of the Thirteenth International Conference on Artificial
           Intelligence and Statistics (AISTATS)*, pp. 249-256.

    See Also
    --------
    :class:`~tuiml.algorithms.neural.PerceptronClassifier` : Single-layer perceptron for linearly separable problems.
    :class:`~tuiml.algorithms.neural.VotedPerceptronClassifier` : Voted perceptron with weighted voting.
    :class:`~tuiml.algorithms.neural.AveragedPerceptronClassifier` : Averaged perceptron for better generalization.

    Examples
    --------
    Train a Multilayer Perceptron on the XOR problem:

    >>> from tuiml.algorithms.neural import MultilayerPerceptronClassifier
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = np.array([0, 1, 1, 0])  # XOR problem
    >>> clf = MultilayerPerceptronClassifier(hidden_layers=[5], activation='relu')
    >>> clf.fit(X, y)
    MultilayerPerceptronClassifier(layers=[5])
    """

    def __init__(self, hidden_layers: List[int] = None,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 max_epochs: int = 500,
                 validation_threshold: int = 20,
                 decay: bool = True,
                 activation: str = 'relu',
                 random_state: Optional[int] = None):
        super().__init__()
        self.hidden_layers = hidden_layers or [100]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.validation_threshold = validation_threshold
        self.decay = decay
        self.activation = activation
        self.random_state = random_state
        self.weights_ = None
        self.biases_ = None
        self.classes_ = None
        self._n_features = None
        self._rng = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "hidden_layers": {"type": "array", "default": [100],
                             "description": "Sizes of hidden layers"},
            "learning_rate": {"type": "number", "default": 0.1,
                             "minimum": 0, "description": "Learning rate"},
            "momentum": {"type": "number", "default": 0.9,
                        "minimum": 0, "maximum": 1, "description": "Momentum"},
            "max_epochs": {"type": "integer", "default": 500, "minimum": 1,
                          "description": "Maximum training epochs"},
            "validation_threshold": {"type": "integer", "default": 20,
                                    "description": "Early stopping patience"},
            "decay": {"type": "boolean", "default": True,
                     "description": "Decay learning rate"},
            "activation": {"type": "string", "default": "relu",
                          "enum": ["sigmoid", "relu"],
                          "description": "Activation function"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * epochs * network_size)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). "
                "Learning representations by back-propagating errors. "
                "Nature, 323, 533-536."]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.

        Computes :math:`\\sigma(z) = \\frac{1}{1 + e^{-z}}` with clipping for
        numerical stability.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values (linear combination of inputs).

        Returns
        -------
        activations : np.ndarray
            Activated values in the range (0, 1), same shape as ``z``.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, a: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid activation function.

        Computes :math:`\\sigma'(a) = a \\cdot (1 - a)` where :math:`a` is the
        already-activated output of the sigmoid function.

        Parameters
        ----------
        a : np.ndarray
            Activated values (output of ``_sigmoid``).

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative values, same shape as ``a``.
        """
        return a * (1 - a)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation function.

        Computes :math:`\\text{ReLU}(z) = \\max(0, z)`.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values (linear combination of inputs).

        Returns
        -------
        activations : np.ndarray
            Activated values with negative entries set to zero, same shape
            as ``z``.
        """
        return np.maximum(0, z)

    def _relu_derivative(self, a: np.ndarray) -> np.ndarray:
        """Derivative of the ReLU activation function.

        Computes the subgradient: 1 where :math:`a > 0`, and 0 elsewhere.

        Parameters
        ----------
        a : np.ndarray
            Activated values (output of ``_relu``).

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative values (0 or 1), same shape as ``a``.
        """
        return (a > 0).astype(float)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply the configured activation function to pre-activation values.

        Dispatches to ``_sigmoid`` or ``_relu`` based on the ``activation``
        parameter.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values (linear combination of inputs).

        Returns
        -------
        activations : np.ndarray
            Activated values, same shape as ``z``.
        """
        if self.activation == 'relu':
            return self._relu(z)
        return self._sigmoid(z)

    def _activate_derivative(self, a: np.ndarray) -> np.ndarray:
        """Compute the derivative of the configured activation function.

        Dispatches to ``_sigmoid_derivative`` or ``_relu_derivative`` based on
        the ``activation`` parameter.

        Parameters
        ----------
        a : np.ndarray
            Activated values (output of ``_activate``).

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative values, same shape as ``a``.
        """
        if self.activation == 'relu':
            return self._relu_derivative(a)
        return self._sigmoid_derivative(a)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax activation function for the output layer.

        Computes :math:`\\text{softmax}(z_k) = \\frac{e^{z_k}}{\\sum_j e^{z_j}}`
        with numerical stabilization by subtracting the row-wise maximum.

        Parameters
        ----------
        z : np.ndarray of shape (n_samples, n_classes)
            Pre-activation values from the output layer.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Class probability distribution for each sample, summing to 1
            along each row.
        """
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _initialize_weights(self, layer_sizes: List[int]):
        """Initialize weights using Xavier (Glorot) initialization.

        Sets weight values from a uniform distribution scaled by the fan-in and
        fan-out of each layer to maintain stable gradients during training.

        Parameters
        ----------
        layer_sizes : list of int
            Number of neurons in each layer, including input and output layers.

        Returns
        -------
        None
            Populates ``self.weights_`` and ``self.biases_`` in place.
        """
        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            # Xavier initialization
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = self._rng.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros(n_out)

            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through the network.

        Propagates inputs through all layers, applying the configured activation
        function at hidden layers and softmax at the output layer.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features (standardized).

        Returns
        -------
        activations : list of np.ndarray
            Activation values for each layer, starting with the input.
        zs : list of np.ndarray
            Pre-activation values for each layer.
        """
        activations = [X]
        zs = []

        for i in range(len(self.weights_) - 1):
            z = activations[-1] @ self.weights_[i] + self.biases_[i]
            zs.append(z)
            a = self._activate(z)
            activations.append(a)

        # Output layer with softmax
        z_out = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        zs.append(z_out)
        a_out = self._softmax(z_out)
        activations.append(a_out)

        return activations, zs

    def _backward(self, X: np.ndarray, y_onehot: np.ndarray,
                  activations: List[np.ndarray], zs: List[np.ndarray]
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass (backpropagation).

        Computes gradients of the cross-entropy loss with respect to all
        weights and biases by propagating the error signal backwards through
        the network layers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features (standardized).
        y_onehot : np.ndarray of shape (n_samples, n_classes)
            One-hot encoded target labels.
        activations : list of np.ndarray
            Activation values for each layer from the forward pass.
        zs : list of np.ndarray
            Pre-activation values for each layer from the forward pass.

        Returns
        -------
        weight_grads : list of np.ndarray
            Gradients for each weight matrix.
        bias_grads : list of np.ndarray
            Gradients for each bias vector.
        """
        n_samples = X.shape[0]
        weight_grads = [np.zeros_like(w) for w in self.weights_]
        bias_grads = [np.zeros_like(b) for b in self.biases_]

        # Output layer error (cross-entropy with softmax)
        delta = activations[-1] - y_onehot

        for i in range(len(self.weights_) - 1, -1, -1):
            weight_grads[i] = activations[i].T @ delta / n_samples
            bias_grads[i] = np.mean(delta, axis=0)

            if i > 0:
                delta = (delta @ self.weights_[i].T) * self._activate_derivative(activations[i])

        return weight_grads, bias_grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptronClassifier":
        """Fit the MultilayerPerceptronClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : MultilayerPerceptronClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self._n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Standardize features
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1
        X_scaled = (X - self._mean) / self._std

        # One-hot encode targets
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_idx] = 1

        # Initialize network
        self._rng = np.random.RandomState(self.random_state)
        layer_sizes = [self._n_features] + self.hidden_layers + [n_classes]
        self._initialize_weights(layer_sizes)

        # Initialize momentum terms
        weight_velocities = [np.zeros_like(w) for w in self.weights_]
        bias_velocities = [np.zeros_like(b) for b in self.biases_]

        # Training loop
        lr = self.learning_rate
        best_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            # Forward pass
            activations, zs = self._forward(X_scaled)

            # Compute loss
            predictions = activations[-1]
            loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-10), axis=1))

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.validation_threshold:
                    break

            # Backward pass
            weight_grads, bias_grads = self._backward(X_scaled, y_onehot,
                                                       activations, zs)

            # Update weights with momentum
            for i in range(len(self.weights_)):
                weight_velocities[i] = (self.momentum * weight_velocities[i] -
                                        lr * weight_grads[i])
                bias_velocities[i] = (self.momentum * bias_velocities[i] -
                                      lr * bias_grads[i])

                self.weights_[i] += weight_velocities[i]
                self.biases_[i] += bias_velocities[i]

            # Learning rate decay
            if self.decay:
                lr = self.learning_rate / (1 + epoch * 0.01)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self._mean) / self._std
        activations, _ = self._forward(X_scaled)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"MultilayerPerceptronClassifier(layers={self.hidden_layers})"
        return f"MultilayerPerceptronClassifier(hidden_layers={self.hidden_layers})"


@regressor(tags=["functions", "neural-network", "deep-learning"], version="1.0.0")
class MultilayerPerceptronRegressor(Regressor):
    """Multilayer Perceptron (Neural Network) regressor.

    A **feedforward artificial neural network** trained using
    **backpropagation** for regression tasks. It supports multiple hidden
    layers, various activation functions, momentum-based updates, and early
    stopping. The output layer uses a **single linear neuron** to produce
    continuous predictions, trained with **mean squared error** loss.

    Overview
    --------
    The Multilayer Perceptron regressor trains by iteratively adjusting
    weights through forward and backward passes:

    1. Initialize weights using **Xavier initialization** for each layer
    2. Standardize input features and target values to zero mean and unit variance
    3. Perform a **forward pass**: propagate inputs through hidden layers
       applying activation functions, then a linear activation at the output
    4. Compute the **MSE loss** between predictions and targets
    5. Perform a **backward pass** (backpropagation): compute gradients of the
       loss with respect to each weight and bias
    6. Update weights using gradient descent with **momentum**
    7. Optionally decay the learning rate over epochs
    8. Stop early if the loss does not improve for ``validation_threshold``
       consecutive epochs

    Theory
    ------
    For a network with :math:`L` layers, the forward pass computes activations
    layer by layer. For hidden layer :math:`l`:

    .. math::
        \\mathbf{z}^{(l)} = \\mathbf{a}^{(l-1)} \\mathbf{W}^{(l)} + \\mathbf{b}^{(l)}

    .. math::
        \\mathbf{a}^{(l)} = g(\\mathbf{z}^{(l)})

    where :math:`g` is the activation function (sigmoid or ReLU). The output
    layer uses a **linear activation** (identity function):

    .. math::
        \\hat{y} = \\mathbf{a}^{(L-1)} \\mathbf{W}^{(L)} + \\mathbf{b}^{(L)}

    The network is trained by minimizing the **mean squared error** loss:

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2

    Parameters
    ----------
    hidden_layers : list of int, default=[100]
        Sizes of the hidden layers. Each entry represents the number of
        neurons in that hidden layer.
    learning_rate : float, default=0.1
        Learning rate for weight updates.
    momentum : float, default=0.9
        Momentum for gradient descent to accelerate convergence.
    max_epochs : int, default=500
        Maximum number of training epochs.
    validation_threshold : int, default=20
        Number of epochs to wait for improvement in loss before
        stopping (patience).
    decay : bool, default=True
        Whether to decay the learning rate over time.
    activation : {'sigmoid', 'relu'}, default='relu'
        Activation function for hidden layers.
    random_state : int or None, default=None
        Seed for the random number generator.

    Attributes
    ----------
    weights_ : list of np.ndarray
        Weight matrices connecting each layer.
    biases_ : list of np.ndarray
        Bias vectors for each layer.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot E \\cdot \\sum_{l=1}^{L} d_l \\cdot d_{l+1})`
      where :math:`n` = samples, :math:`E` = epochs, :math:`d_l` = layer size
    - Prediction: :math:`O(n \\cdot \\sum_{l=1}^{L} d_l \\cdot d_{l+1})`

    **When to use MultilayerPerceptronRegressor:**

    - When the relationship between features and target is non-linear
    - When you need a flexible model for continuous value prediction
    - When sufficient training data is available to fit the network
    - As a lightweight alternative to deep learning frameworks

    References
    ----------
    .. [Rumelhart1986] Rumelhart, D.E., Hinton, G.E. and Williams, R.J. (1986).
           **Learning Representations by Back-Propagating Errors.**
           *Nature*, 323, 533-536.
           DOI: `10.1038/323533a0 <https://doi.org/10.1038/323533a0>`_

    See Also
    --------
    :class:`~tuiml.algorithms.neural.MultilayerPerceptronClassifier` : MLP for classification tasks.
    :class:`~tuiml.algorithms.linear.LinearRegression` : Linear regression baseline.

    Examples
    --------
    Train a Multilayer Perceptron regressor on a simple problem:

    >>> from tuiml.algorithms.neural import MultilayerPerceptronRegressor
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    >>> reg = MultilayerPerceptronRegressor(hidden_layers=[10], activation='relu')
    >>> reg.fit(X, y)
    MultilayerPerceptronRegressor(layers=[10])
    """

    def __init__(self, hidden_layers: List[int] = None,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 max_epochs: int = 500,
                 validation_threshold: int = 20,
                 decay: bool = True,
                 activation: str = 'relu',
                 random_state: Optional[int] = None):
        """Initialize MultilayerPerceptronRegressor.

        Parameters
        ----------
        hidden_layers : list of int, default=[100]
            Sizes of the hidden layers.
        learning_rate : float, default=0.01
            Learning rate for weight updates.
        momentum : float, default=0.9
            Momentum for gradient descent.
        max_epochs : int, default=500
            Maximum number of training epochs.
        validation_threshold : int, default=20
            Early stopping patience.
        decay : bool, default=True
            Whether to decay the learning rate.
        activation : str, default='relu'
            Activation function for hidden layers.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.hidden_layers = hidden_layers or [100]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.validation_threshold = validation_threshold
        self.decay = decay
        self.activation = activation
        self.random_state = random_state
        self.weights_ = None
        self.biases_ = None
        self._n_features = None
        self._rng = None
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "hidden_layers": {"type": "array", "default": [100],
                             "description": "Sizes of hidden layers"},
            "learning_rate": {"type": "number", "default": 0.1,
                             "minimum": 0, "description": "Learning rate"},
            "momentum": {"type": "number", "default": 0.9,
                        "minimum": 0, "maximum": 1, "description": "Momentum"},
            "max_epochs": {"type": "integer", "default": 500, "minimum": 1,
                          "description": "Maximum training epochs"},
            "validation_threshold": {"type": "integer", "default": 20,
                                    "description": "Early stopping patience"},
            "decay": {"type": "boolean", "default": True,
                     "description": "Decay learning rate"},
            "activation": {"type": "string", "default": "relu",
                          "enum": ["sigmoid", "relu"],
                          "description": "Activation function"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * epochs * network_size)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return ["Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). "
                "Learning representations by back-propagating errors. "
                "Nature, 323, 533-536."]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values.

        Returns
        -------
        activations : np.ndarray
            Activated values in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, a: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid activation function.

        Parameters
        ----------
        a : np.ndarray
            Activated values (output of ``_sigmoid``).

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative values.
        """
        return a * (1 - a)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values.

        Returns
        -------
        activations : np.ndarray
            Activated values with negatives set to zero.
        """
        return np.maximum(0, z)

    def _relu_derivative(self, a: np.ndarray) -> np.ndarray:
        """Derivative of the ReLU activation function.

        Parameters
        ----------
        a : np.ndarray
            Activated values (output of ``_relu``).

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative (0 or 1).
        """
        return (a > 0).astype(float)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply the configured activation function.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values.

        Returns
        -------
        activations : np.ndarray
            Activated values.
        """
        if self.activation == 'relu':
            return self._relu(z)
        return self._sigmoid(z)

    def _activate_derivative(self, a: np.ndarray) -> np.ndarray:
        """Compute the derivative of the configured activation function.

        Parameters
        ----------
        a : np.ndarray
            Activated values.

        Returns
        -------
        derivative : np.ndarray
            Element-wise derivative values.
        """
        if self.activation == 'relu':
            return self._relu_derivative(a)
        return self._sigmoid_derivative(a)

    def _initialize_weights(self, layer_sizes: List[int]):
        """Initialize weights using Xavier (Glorot) initialization.

        Parameters
        ----------
        layer_sizes : list of int
            Number of neurons in each layer, including input and output.

        Returns
        -------
        None
            Populates ``self.weights_`` and ``self.biases_`` in place.
        """
        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = self._rng.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros(n_out)
            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through the network.

        Propagates inputs through all layers, applying the configured
        activation function at hidden layers and linear activation at output.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features (standardized).

        Returns
        -------
        activations : list of np.ndarray
            Activation values for each layer.
        zs : list of np.ndarray
            Pre-activation values for each layer.
        """
        activations = [X]
        zs = []

        for i in range(len(self.weights_) - 1):
            z = activations[-1] @ self.weights_[i] + self.biases_[i]
            zs.append(z)
            a = self._activate(z)
            activations.append(a)

        # Output layer with linear activation
        z_out = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        zs.append(z_out)
        activations.append(z_out)  # linear: identity function

        return activations, zs

    def _backward(self, X: np.ndarray, y: np.ndarray,
                  activations: List[np.ndarray], zs: List[np.ndarray]
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass (backpropagation) for MSE loss.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features (standardized).
        y : np.ndarray of shape (n_samples, 1)
            Target values (standardized).
        activations : list of np.ndarray
            Activation values from the forward pass.
        zs : list of np.ndarray
            Pre-activation values from the forward pass.

        Returns
        -------
        weight_grads : list of np.ndarray
            Gradients for each weight matrix.
        bias_grads : list of np.ndarray
            Gradients for each bias vector.
        """
        n_samples = X.shape[0]
        weight_grads = [np.zeros_like(w) for w in self.weights_]
        bias_grads = [np.zeros_like(b) for b in self.biases_]

        # Output layer error (MSE derivative): 2/N * (pred - y)
        delta = 2.0 * (activations[-1] - y) / n_samples

        for i in range(len(self.weights_) - 1, -1, -1):
            weight_grads[i] = activations[i].T @ delta / n_samples
            bias_grads[i] = np.mean(delta, axis=0)

            if i > 0:
                delta = (delta @ self.weights_[i].T) * self._activate_derivative(activations[i])

        return weight_grads, bias_grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptronRegressor":
        """Fit the MultilayerPerceptronRegressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MultilayerPerceptronRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, self._n_features = X.shape

        # Standardize features
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1
        X_scaled = (X - self._X_mean) / self._X_std

        # Standardize targets
        self._y_mean = np.mean(y, axis=0)
        self._y_std = np.std(y, axis=0)
        if self._y_std == 0:
            self._y_std = 1.0
        y_scaled = (y - self._y_mean) / self._y_std

        # Initialize network (single output neuron)
        self._rng = np.random.RandomState(self.random_state)
        layer_sizes = [self._n_features] + self.hidden_layers + [1]
        self._initialize_weights(layer_sizes)

        # Initialize momentum terms
        weight_velocities = [np.zeros_like(w) for w in self.weights_]
        bias_velocities = [np.zeros_like(b) for b in self.biases_]

        # Training loop
        lr = self.learning_rate
        best_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            activations, zs = self._forward(X_scaled)

            # Compute MSE loss
            loss = np.mean((activations[-1] - y_scaled) ** 2)

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.validation_threshold:
                    break

            # Backward pass
            weight_grads, bias_grads = self._backward(
                X_scaled, y_scaled, activations, zs)

            # Update weights with momentum
            for i in range(len(self.weights_)):
                weight_velocities[i] = (self.momentum * weight_velocities[i] -
                                        lr * weight_grads[i])
                bias_velocities[i] = (self.momentum * bias_velocities[i] -
                                      lr * bias_grads[i])
                self.weights_[i] += weight_velocities[i]
                self.biases_[i] += bias_velocities[i]

            # Learning rate decay
            if self.decay:
                lr = self.learning_rate / (1 + epoch * 0.01)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self._X_mean) / self._X_std
        activations, _ = self._forward(X_scaled)

        # Inverse transform the predictions
        y_pred = activations[-1] * self._y_std + self._y_mean
        return y_pred.ravel()

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
            return f"MultilayerPerceptronRegressor(layers={self.hidden_layers})"
        return f"MultilayerPerceptronRegressor(hidden_layers={self.hidden_layers})"
