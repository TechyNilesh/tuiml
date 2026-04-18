"""Expectation-Maximization clustering for Gaussian Mixture Models (GMM)."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import DensityBasedClusterer, clusterer

@clusterer(tags=["probabilistic", "gaussian-mixture"], version="1.0.0")
class GaussianMixtureClusterer(DensityBasedClusterer):
    r"""
    Expectation-Maximization clustering for Gaussian Mixture Models.

    Models the data as a mixture of :math:`K` **Gaussian distributions**. Each
    component is characterized by its mean :math:`\mu_k`, covariance
    :math:`\Sigma_k`, and mixing weight :math:`\pi_k`. Unlike hard-assignment
    methods like K-Means, EM provides **soft probabilistic** cluster assignments.

    Overview
    --------
    The algorithm iteratively refines component parameters:

    1. **Initialize** means, covariances, and mixing weights
    2. **E-step**: Compute responsibilities (posterior probability that each
       component generated each data point)
    3. **M-step**: Update parameters to maximize expected log-likelihood
    4. Repeat steps 2--3 until convergence or ``max_iter`` is reached
    5. Select the best result across ``n_init`` independent runs

    Theory
    ------
    The probability of an observation :math:`x` under the mixture model is:

    .. math::
        P(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)

    **E-step** computes the responsibility of component :math:`k` for point :math:`x_i`:

    .. math::
        \gamma_{ik} = \\frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}

    **M-step** updates the parameters:

    .. math::
        \pi_k = \\frac{N_k}{N}, \quad \mu_k = \\frac{1}{N_k} \sum_{i} \gamma_{ik} x_i, \quad \Sigma_k = \\frac{1}{N_k} \sum_{i} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T

    where :math:`N_k = \sum_i \gamma_{ik}`.

    Parameters
    ----------
    n_components : int, default=2
        The number of mixture components.
    max_iter : int, default=200
        The maximum number of EM iterations to perform.
    tol : float, default=1e-6
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    covariance_type : {"full", "diag", "spherical"}, default="full"
        String describing the type of covariance parameters to use:

        - ``"full"``: Each component has its own general covariance matrix.
        - ``"diag"``: Each component has its own diagonal covariance matrix.
        - ``"spherical"``: Each component has its own single variance.
    n_init : int, default=10
        The number of initializations to perform. The best results are kept.
    random_state : int, optional, default=None
        Controls the random seed given to the method chosen to
        initialize the parameters.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_components,)
        The weights of each mixture component.
    means_ : np.ndarray of shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : np.ndarray
        The covariance of each mixture component.
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of iterations used by the best fit of EM to reach the
        convergence.
    log_likelihood_ : float
        Log-likelihood of the best fit of EM.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k \cdot m^2 \cdot i)` for "full" covariance,
      where :math:`n` is samples, :math:`k` is components, :math:`m` is features,
      and :math:`i` is iterations.
    - For "diag": :math:`O(n \cdot k \cdot m \cdot i)`.
    - For "spherical": :math:`O(n \cdot k \cdot i)`.

    **When to use GaussianMixtureClusterer:**

    - When clusters have elliptical shapes of varying sizes and orientations
    - When soft (probabilistic) cluster assignments are needed
    - Density estimation and generative modeling
    - When the data is well-described by Gaussian distributions

    References
    ----------
    .. [Dempster1977] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).
           **Maximum likelihood from incomplete data via the EM algorithm.**
           *Journal of the Royal Statistical Society: Series B (Methodological)*,
           39(1), pp. 1-22.

    .. [McLachlan2000] McLachlan, G. J. & Peel, D. (2000).
           **Finite Mixture Models.**
           *Wiley Series in Probability and Statistics*.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : Hard-assignment centroid-based clustering.
    :class:`~tuiml.algorithms.clustering.DBSCANClusterer` : Density-based clustering with noise detection.

    Examples
    --------
    Gaussian mixture model clustering:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import GaussianMixtureClusterer
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> gmm = GaussianMixtureClusterer(n_components=2)
    >>> gmm.fit(X)
    >>> gmm.predict([[0, 0], [12, 3]])
    array([0, 1])
    """

    def __init__(self, n_components: int = 2,
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 covariance_type: str = 'full',
                 n_init: int = 10,
                 random_state: Optional[int] = None):
        """Initialize GaussianMixtureClusterer clusterer.

        Parameters
        ----------
        n_components : int, default=2
            Number of Gaussian components.
        max_iter : int, default=200
            Maximum iterations.
        tol : float, default=1e-6
            Convergence tolerance.
        covariance_type : str, default='full'
            Type of covariance.
        n_init : int, default=10
            Number of initializations.
        random_state : int, optional
            Random seed.
        """
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of mixture components"
            },
            "max_iter": {
                "type": "integer",
                "default": 200,
                "minimum": 1,
                "description": "Maximum EM iterations"
            },
            "tol": {
                "type": "number",
                "default": 1e-6,
                "minimum": 0,
                "description": "Convergence tolerance"
            },
            "covariance_type": {
                "type": "string",
                "default": "full",
                "enum": ["full", "diag", "spherical"],
                "description": "Covariance matrix type"
            },
            "n_init": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of initializations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "probabilistic"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k * d^2 * i) time per iteration"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). "
            "Maximum likelihood from incomplete data via the EM algorithm. "
            "J. Royal Statistical Society, 39(1), 1-38."
        ]

    def _initialize_parameters(self, X: np.ndarray,
                               rng: np.random.Generator) -> None:
        """Initialize GMM parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to initialize means, covariances, and weights.
        rng : np.random.Generator
            Random number generator for selecting initial means.
        """
        n_samples, n_features = X.shape

        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Initialize means using k-means++ style
        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()

        # Initialize covariances
        if self.covariance_type == 'full':
            # Full covariance matrices
            cov = np.cov(X.T) + 1e-6 * np.eye(n_features)
            self.covariances_ = np.array([cov.copy()
                                          for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            # Diagonal covariances
            var = np.var(X, axis=0) + 1e-6
            self.covariances_ = np.array([var.copy()
                                          for _ in range(self.n_components)])
        else:  # spherical
            # Single variance per component
            var = np.mean(np.var(X, axis=0)) + 1e-6
            self.covariances_ = np.array([var for _ in range(self.n_components)])

    def _compute_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Compute log probability of each sample under each component.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        log_prob : np.ndarray of shape (n_samples, n_components)
            Log probability of each sample under each Gaussian component.
        """
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
                try:
                    # Cholesky decomposition for numerical stability
                    L = np.linalg.cholesky(cov)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    diff = X - self.means_[k]
                    solved = np.linalg.solve(L, diff.T).T
                    mahal = np.sum(solved ** 2, axis=1)
                except np.linalg.LinAlgError:
                    # Fallback if not positive definite
                    cov = cov + 1e-6 * np.eye(n_features)
                    L = np.linalg.cholesky(cov)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    diff = X - self.means_[k]
                    solved = np.linalg.solve(L, diff.T).T
                    mahal = np.sum(solved ** 2, axis=1)

            elif self.covariance_type == 'diag':
                var = self.covariances_[k]
                log_det = np.sum(np.log(var))
                diff = X - self.means_[k]
                mahal = np.sum(diff ** 2 / var, axis=1)

            else:  # spherical
                var = self.covariances_[k]
                log_det = n_features * np.log(var)
                diff = X - self.means_[k]
                mahal = np.sum(diff ** 2, axis=1) / var

            log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) +
                                     log_det + mahal)

        return log_prob

    def _e_step(self, X: np.ndarray) -> tuple:
        """E-step: Compute responsibilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        resp : np.ndarray of shape (n_samples, n_components)
            Normalized responsibilities (posterior probabilities).
        log_likelihood : float
            Total log-likelihood of the data under the current model.
        """
        log_prob = self._compute_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-10)

        # Log responsibility (unnormalized)
        log_resp = log_prob + log_weights

        # Normalize using log-sum-exp trick
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        log_resp_sum = log_resp_max + np.log(
            np.sum(np.exp(log_resp - log_resp_max), axis=1, keepdims=True)
        )
        log_resp = log_resp - log_resp_sum

        # Compute log-likelihood
        log_likelihood = np.sum(log_resp_sum)

        return np.exp(log_resp), log_likelihood

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """M-step: Update parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data samples.
        resp : np.ndarray of shape (n_samples, n_components)
            Current responsibilities from the E-step.
        """
        n_samples, n_features = X.shape

        # Update weights
        nk = resp.sum(axis=0) + 1e-10
        self.weights_ = nk / n_samples

        # Update means
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k:k+1] * diff

            if self.covariance_type == 'full':
                cov = (weighted_diff.T @ diff) / nk[k]
                # Add regularization
                cov += 1e-6 * np.eye(n_features)
                self.covariances_[k] = cov

            elif self.covariance_type == 'diag':
                var = np.sum(weighted_diff * diff, axis=0) / nk[k]
                self.covariances_[k] = var + 1e-6

            else:  # spherical
                var = np.sum(weighted_diff * diff) / (nk[k] * n_features)
                self.covariances_[k] = var + 1e-6

    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple:
        """Run a single EM optimization from one random initialization.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        weights : np.ndarray of shape (n_components,)
            Fitted mixing weights.
        means : np.ndarray of shape (n_components, n_features)
            Fitted component means.
        covariances : np.ndarray
            Fitted component covariances.
        log_likelihood : float
            Final log-likelihood.
        n_iter : int
            Number of iterations performed.
        converged : bool
            Whether the algorithm converged.
        """
        self._initialize_parameters(X, rng)

        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            resp, log_likelihood = self._e_step(X)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                return (self.weights_.copy(), self.means_.copy(),
                        self.covariances_.copy(), log_likelihood,
                        iteration + 1, True)

            prev_log_likelihood = log_likelihood

            # M-step
            self._m_step(X, resp)

        return (self.weights_.copy(), self.means_.copy(),
                self.covariances_.copy(), log_likelihood,
                self.max_iter, False)

    def fit(self, X: np.ndarray) -> "GaussianMixtureClusterer":
        """Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : GaussianMixtureClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)

        best_log_likelihood = -np.inf
        best_params = None

        for _ in range(self.n_init):
            weights, means, covs, ll, n_iter, converged = self._single_run(X, rng)

            if ll > best_log_likelihood:
                best_log_likelihood = ll
                best_params = (weights, means, covs, n_iter, converged)

        self.weights_, self.means_, self.covariances_, self.n_iter_, self.converged_ = best_params
        self.log_likelihood_ = best_log_likelihood

        # Assign labels
        resp, _ = self._e_step(X)
        self.labels_ = np.argmax(resp, axis=1)
        self.n_clusters_ = self.n_components
        self.cluster_centers_ = self.means_
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        resp : np.ndarray of shape (n_samples, n_components)
            Posterior probability of each Gaussian component for each 
            sample in X.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        resp, _ = self._e_step(X)
        return resp

    def score(self, X: np.ndarray) -> float:
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to score.

        Returns
        -------
        score : float
            Log-likelihood of X under the Gaussian mixture model.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _, log_likelihood = self._e_step(X)
        return log_likelihood / X.shape[0]

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"GaussianMixtureClusterer(n_components={self.n_components}, "
                   f"converged={self.converged_}, n_iter={self.n_iter_})")
        return f"GaussianMixtureClusterer(n_components={self.n_components}, covariance_type='{self.covariance_type}')"
