"""Sigmoid (Hyperbolic Tangent) Kernel implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import CachedKernel, kernel

@kernel(tags=["sigmoid", "tanh", "neural"], version="1.0.0")
class SigmoidKernel(CachedKernel):
    """Sigmoid (Hyperbolic Tangent) Kernel inspired by **neural networks**.

    The Sigmoid Kernel computes the hyperbolic tangent of a scaled dot product,
    producing a response analogous to a **single hidden-layer neural network**.
    It is sometimes called the **MLP kernel** or **tanh kernel**.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Compute the **dot product** :math:`\\langle x, y \\rangle`
    2. Scale by ``gamma`` and add the independent term ``coef0``
    3. Apply the **hyperbolic tangent** function to obtain a value in :math:`(-1, 1)`

    Theory
    ------
    The sigmoid kernel function is defined as:

    .. math::
        K(x, y) = \\tanh(\\gamma \\, \\langle x, y \\rangle + c_0)

    where:

    - :math:`\\gamma` --- Scaling coefficient for the dot product
    - :math:`c_0` --- Independent (bias) term

    .. warning::
        This kernel is **not positive semi-definite** for all parameter
        values. It satisfies Mercer's condition only for certain combinations
        of :math:`\\gamma > 0` and :math:`c_0 < 0`. Invalid parameters may
        lead to non-convergent SVM solutions.

    Parameters
    ----------
    gamma : float, default=0.01
        Coefficient for the dot product.

    coef0 : float, default=0.0
        Independent (bias) term.

    cache_size : int, default=250007
        Maximum number of cached kernel evaluations.

    Attributes
    ----------
    X\\_ : np.ndarray
        Training data stored after ``build()``.

    n_samples\\_ : int
        Number of training samples.

    Notes
    -----
    **Complexity:**

    - Single evaluation: :math:`O(p)` where :math:`p` = number of features
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples

    **When to use SigmoidKernel:**

    - When a neural-network-like non-linearity is desired
    - As a proxy for a single-layer perceptron in kernel space
    - Experimental comparisons with other kernels (RBF, polynomial)
    - When ``gamma > 0`` and ``coef0 < 0`` for valid Mercer conditions

    References
    ----------
    .. [Lin2003] Lin, H.T. and Lin, C.J. (2003).
           **A Study on Sigmoid Kernels for SVM and the Training of non-PSD Kernels by SMO-type Methods.**
           *National Taiwan University Technical Report*.

    .. [Scholkopf2002] Schoelkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.**
           *MIT Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Gaussian RBF kernel (always positive semi-definite).
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Polynomial kernel for finite-order feature maps.

    Examples
    --------
    Basic usage with a negative bias term:

    >>> from tuiml.algorithms.svm.kernels import SigmoidKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> kernel = SigmoidKernel(gamma=0.01, coef0=-1.0)
    >>> kernel.build(X_train)
    SigmoidKernel(...)
    >>> value = kernel.evaluate(X_train[0], X_train[1])
    """

    _libsvm_kernel_type = 3  # SIGMOID

    def __init__(self, gamma: float = 0.01,
                 coef0: float = 0.0,
                 cache_size: int = 250007):
        """Initialize sigmoid kernel.

        Parameters
        ----------
        gamma : float, default=0.01
            Coefficient for the dot product.
        coef0 : float, default=0.0
            Independent (bias) term.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.gamma = gamma
        self.coef0 = coef0

    def _libsvm_params(self) -> str:
        """Return libsvm parameter string."""
        return f"-g {self.gamma} -r {self.coef0}"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "gamma": {
                "type": "number",
                "default": 0.01,
                "description": "Coefficient for dot product"
            },
            "coef0": {
                "type": "number",
                "default": 0.0,
                "description": "Independent term"
            }
        }

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate sigmoid kernel: K(x, y) = tanh(gamma * <x, y> + coef0).

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            Sigmoid kernel value in (-1, 1).
        """
        dot = np.dot(x1, x2)
        return np.tanh(self.gamma * dot + self.coef0)

    def compute_matrix(self) -> np.ndarray:
        """Compute the sigmoid kernel matrix using vectorized operations.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix.
        """
        self._check_is_built()
        X = np.asarray(self.X_, dtype=float)
        K = np.tanh(self.gamma * (X @ X.T) + self.coef0)
        return K

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the sigmoid kernel matrix between two sample sets.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of samples.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of samples.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel matrix.
        """
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        K = np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        return K

    def __repr__(self) -> str:
        """String representation."""
        return f"SigmoidKernel(gamma={self.gamma}, coef0={self.coef0})"

# Aliases
TanhKernel = SigmoidKernel
HyperbolicTangentKernel = SigmoidKernel
