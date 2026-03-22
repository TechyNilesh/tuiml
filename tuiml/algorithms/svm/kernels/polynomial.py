"""Polynomial Kernel implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import CachedKernel, kernel

@kernel(tags=["polynomial", "nonlinear"], version="1.0.0")
class PolynomialKernel(CachedKernel):
    """Polynomial Kernel computing a **polynomial of the dot product**.

    The Polynomial Kernel maps input vectors into a feature space spanned by
    all **monomials up to degree** :math:`d`, enabling SVMs to learn
    polynomial decision boundaries without explicitly computing the
    high-dimensional feature map.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Compute the **dot product** :math:`\\langle x, y \\rangle`
    2. Scale by ``gamma`` and add the independent term ``coef0``
    3. Raise the result to the power ``degree``

    When ``coef0 > 0`` (``lower_order=True``), the implicit feature map includes
    all monomials of degree :math:`\\leq d`; when ``coef0 = 0``, only degree-:math:`d`
    monomials are included.

    Theory
    ------
    The polynomial kernel function is defined as:

    .. math::
        K(x, y) = (\\gamma \\, \\langle x, y \\rangle + c_0)^d

    where:

    - :math:`\\gamma` --- Scaling coefficient for the dot product
    - :math:`c_0` --- Independent (bias) term controlling lower-order contributions
    - :math:`d` --- Degree of the polynomial

    For :math:`c_0 = 0` this is a **homogeneous** polynomial kernel; for
    :math:`c_0 > 0` it is **inhomogeneous**, including cross-terms of all
    orders up to :math:`d`.

    Parameters
    ----------
    degree : int, default=3
        Degree of the polynomial.

    gamma : float, default=1.0
        Scaling coefficient for the dot product.

    coef0 : float or None, default=None
        Independent term. Defaults to ``1.0`` if ``lower_order`` is ``True``,
        else ``0.0``.

    lower_order : bool, default=True
        Whether to include lower-order polynomial terms (sets ``coef0``
        default).

    cache_size : int, default=250007
        Maximum number of cached kernel evaluations.

    Attributes
    ----------
    X\\_ : np.ndarray
        Training data stored after ``build()``.

    n_samples\\_ : int
        Number of training samples.

    n_features\\_ : int
        Number of features observed during ``build()``.

    Notes
    -----
    **Complexity:**

    - Single evaluation: :math:`O(p)` where :math:`p` = number of features
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples

    **When to use PolynomialKernel:**

    - When polynomial interactions among features are known or suspected
    - Image recognition tasks where higher-order feature interactions matter
    - When the RBF kernel overfits and a more constrained feature space is preferred
    - Degree 1 reduces to the linear kernel; degree 2-3 is common in practice

    References
    ----------
    .. [Scholkopf2002] Schoelkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.**
           *MIT Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.NormalizedPolynomialKernel` : Normalized variant of the polynomial kernel.
    :class:`~tuiml.algorithms.svm.kernels.LinearKernel` : Linear kernel (polynomial of degree 1).
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : RBF kernel for infinite-dimensional feature maps.

    Examples
    --------
    Basic usage with a cubic polynomial kernel:

    >>> from tuiml.algorithms.svm.kernels import PolynomialKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> kernel = PolynomialKernel(degree=3, lower_order=True)
    >>> kernel.build(X_train)
    PolynomialKernel(...)
    >>> value = kernel.evaluate(X_train[0], X_train[1])
    """

    _libsvm_kernel_type = 1  # POLY

    def __init__(self, degree: int = 3,
                 gamma: float = 1.0,
                 coef0: float = None,
                 lower_order: bool = True,
                 cache_size: int = 250007):
        """Initialize polynomial kernel.

        Parameters
        ----------
        degree : int, default=3
            Polynomial degree.
        gamma : float, default=1.0
            Scaling coefficient for the dot product.
        coef0 : float or None, default=None
            Independent term (defaults to 1.0 if ``lower_order`` else 0.0).
        lower_order : bool, default=True
            Include lower-order polynomial terms.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.degree = degree
        self.gamma = gamma
        self.lower_order = lower_order
        self.coef0 = coef0 if coef0 is not None else (1.0 if lower_order else 0.0)

    def _libsvm_params(self) -> str:
        """Return libsvm parameter string."""
        return f"-g {self.gamma} -d {self.degree} -r {self.coef0}"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "degree": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Degree of the polynomial"
            },
            "gamma": {
                "type": "number",
                "default": 1.0,
                "description": "Coefficient for dot product"
            },
            "coef0": {
                "type": "number",
                "default": 1.0,
                "description": "Independent term in polynomial"
            },
            "lower_order": {
                "type": "boolean",
                "default": True,
                "description": "Use lower-order terms"
            }
        }

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate polynomial kernel.

        K(x, y) = (gamma * <x, y> + coef0)^degree

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            Polynomial kernel value.
        """
        dot = np.dot(x1, x2)
        return (self.gamma * dot + self.coef0) ** self.degree

    def compute_matrix(self) -> np.ndarray:
        """Compute the polynomial kernel matrix using vectorized operations.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix.
        """
        self._check_is_built()
        X = np.asarray(self.X_, dtype=float)
        K = (self.gamma * (X @ X.T) + self.coef0) ** self.degree
        return K

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the polynomial kernel matrix between two sample sets.

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
        K = (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        return K

    def __repr__(self) -> str:
        """String representation."""
        return (f"PolynomialKernel(degree={self.degree}, "
               f"gamma={self.gamma}, coef0={self.coef0})")

# Alias for WEKA compatibility
PolyKernel = PolynomialKernel

@kernel(tags=["polynomial", "normalized"], version="1.0.0")
class NormalizedPolynomialKernel(CachedKernel):
    """Normalized Polynomial Kernel producing values in **[0, 1]**.

    A polynomial kernel **normalized** by the geometric mean of the
    self-similarities, ensuring that :math:`K(x, x) = 1` for all :math:`x`.
    This removes the influence of vector magnitude and focuses purely on
    angular relationships.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Compute the unnormalized polynomial value :math:`K_{raw}(x, y)`
    2. Compute self-similarities :math:`K_{raw}(x, x)` and :math:`K_{raw}(y, y)`
    3. Divide by :math:`\\sqrt{K_{raw}(x, x) \\cdot K_{raw}(y, y)}`

    Theory
    ------
    The normalized polynomial kernel is defined as:

    .. math::
        K_{norm}(x, y) = \\frac{(\\langle x, y \\rangle + c_0)^d}{\\sqrt{(\\langle x, x \\rangle + c_0)^d \\cdot (\\langle y, y \\rangle + c_0)^d}}

    This ensures :math:`K_{norm}(x, x) = 1` and :math:`|K_{norm}(x, y)| \\leq 1`.

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial.

    lower_order : bool, default=True
        Whether to include lower-order terms (controls ``coef0``).

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

    - Single evaluation: :math:`O(p)` where :math:`p` = number of features (three dot products)
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples

    **When to use NormalizedPolynomialKernel:**

    - When input vectors have varying magnitudes and normalization is desired
    - As a drop-in replacement for the standard polynomial kernel with better numerical stability
    - When kernel values should be bounded in :math:`[0, 1]`

    References
    ----------
    .. [Scholkopf2002] Schoelkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.**
           *MIT Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Standard (unnormalized) polynomial kernel.
    :class:`~tuiml.algorithms.svm.kernels.LinearKernel` : Linear kernel (degree-1 special case).

    Examples
    --------
    Basic usage with a quadratic normalized kernel:

    >>> from tuiml.algorithms.svm.kernels import NormalizedPolynomialKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> kernel = NormalizedPolynomialKernel(degree=2)
    >>> kernel.build(X_train)
    NormalizedPolynomialKernel(...)
    >>> value = kernel.evaluate(X_train[0], X_train[1])
    """

    def __init__(self, degree: int = 2,
                 lower_order: bool = True,
                 cache_size: int = 250007):
        """Initialize normalized polynomial kernel.

        Parameters
        ----------
        degree : int, default=2
            Polynomial degree.
        lower_order : bool, default=True
            Include lower-order polynomial terms.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.degree = degree
        self.lower_order = lower_order
        self.coef0 = 1.0 if lower_order else 0.0

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "degree": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Degree of the polynomial"
            },
            "lower_order": {
                "type": "boolean",
                "default": True,
                "description": "Use lower-order terms"
            }
        }

    def _poly_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the unnormalized polynomial kernel value.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First input vector.
        x2 : np.ndarray of shape (n_features,)
            Second input vector.

        Returns
        -------
        value : float
            :math:`(\\langle x_1, x_2 \\rangle + c_0)^d`.
        """
        dot = np.dot(x1, x2)
        return (dot + self.coef0) ** self.degree

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate normalized polynomial kernel.

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            Normalized polynomial kernel value.
        """
        k_xy = self._poly_value(x1, x2)
        k_xx = self._poly_value(x1, x1)
        k_yy = self._poly_value(x2, x2)

        normalizer = np.sqrt(k_xx * k_yy)
        if normalizer == 0:
            return 0.0
        return k_xy / normalizer

    def compute_matrix(self) -> np.ndarray:
        """Compute the normalized polynomial kernel matrix.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix.
        """
        self._check_is_built()
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X = xnp.asarray(self.X_, dtype=float)
        raw = (X @ X.T + self.coef0) ** self.degree
        diag = xnp.diag(raw)
        normalizer = xnp.sqrt(diag[:, None] * diag[None, :])
        K = xnp.where(normalizer == 0, 0.0, raw / normalizer)
        return np.asarray(K)

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the normalized polynomial kernel matrix between two sample sets.

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
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X1 = xnp.asarray(X1, dtype=float)
        X2 = xnp.asarray(X2, dtype=float)
        raw = (X1 @ X2.T + self.coef0) ** self.degree
        diag_1 = (xnp.sum(X1 ** 2, axis=1) + self.coef0) ** self.degree
        diag_2 = (xnp.sum(X2 ** 2, axis=1) + self.coef0) ** self.degree
        normalizer = xnp.sqrt(diag_1[:, None] * diag_2[None, :])
        K = xnp.where(normalizer == 0, 0.0, raw / normalizer)
        return np.asarray(K)

    def __repr__(self) -> str:
        """String representation."""
        return f"NormalizedPolynomialKernel(degree={self.degree})"

# Alias for WEKA compatibility
NormalizedPolyKernel = NormalizedPolynomialKernel
