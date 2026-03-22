"""KD-Tree Nearest Neighbor Search."""

import numpy as np
from typing import Tuple, Optional, List
import heapq

from tuiml.base.neighbors import NearestNeighborSearch
from tuiml._cpp_ext import neighbors as _cpp_nn

class KDTree(NearestNeighborSearch):
    """KD-Tree for **nearest neighbor search** in low-dimensional spaces.

    A KD-Tree is a binary tree that recursively partitions the search space
    along **axis-aligned splits**. It is highly efficient for k-nearest
    neighbor queries in **low to moderate dimensions** (typically
    :math:`d < 20`), but performance degrades toward brute-force speed as
    dimensionality increases.

    This implementation delegates tree construction and querying to an
    optimized **C++ backend** with optional OpenMP parallelism, providing
    orders-of-magnitude speedup over pure-Python recursive traversal.

    Overview
    --------
    The KD-Tree is constructed and queried as follows:

    1. Choose the splitting dimension with the **largest spread**.
    2. Find the **median** of the data along that dimension.
    3. Partition the data: points :math:`\\leq` median go left, others go
       right.
    4. Recursively build subtrees until the number of points in a node is
       at or below ``leaf_size``.
    5. During a query, traverse the tree starting from the closer subtree
       and prune the far subtree when the **splitting plane distance**
       exceeds the current k-th nearest distance.

    Theory
    ------
    The pruning criterion for a query point :math:`q` at an internal node
    splitting on dimension :math:`j` with value :math:`v` is:

    .. math::
        |q_j - v|^2 \\geq d_k^2

    where :math:`d_k` is the squared distance to the current k-th nearest
    neighbor. If this holds, the far subtree cannot contain a closer point
    and is skipped entirely.

    The distance metric is **Euclidean distance**:

    .. math::
        d(q, x_i) = \\sqrt{\\sum_{j=1}^{m} (q_j - x_{i,j})^2}

    Parameters
    ----------
    leaf_size : int, default=10
        Maximum number of points in a leaf node. Smaller values lead to
        deeper trees and potentially faster queries but slower construction.

    Attributes
    ----------
    X_ : np.ndarray
        The training data used to build the tree.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features in the training data.

    Notes
    -----
    **Complexity:**

    - Construction: :math:`O(n \\log n)` where :math:`n` = number of data points
    - Query (average case): :math:`O(\\log n)` per query point
    - Query (worst case): :math:`O(n)` per query point (high dimensions)
    - Space: :math:`O(n)` for the tree structure

    **When to use KDTree:**

    - Low-dimensional data (:math:`d < 20`)
    - Repeated nearest-neighbor queries on a fixed dataset
    - When axis-aligned splits naturally separate the data
    - When construction time can be amortized over many queries

    References
    ----------
    .. [Bentley1975] Bentley, J.L. (1975).
           **Multidimensional Binary Search Trees Used for Associative Searching.**
           *Communications of the ACM*, 18(9), pp. 509-517.
           DOI: `10.1145/361002.361007 <https://doi.org/10.1145/361002.361007>`_

    .. [Friedman1977] Friedman, J.H., Bentley, J.L. and Finkel, R.A. (1977).
           **An Algorithm for Finding Best Matches in Logarithmic Expected Time.**
           *ACM Transactions on Mathematical Software*, 3(3), pp. 209-226.
           DOI: `10.1145/355744.355745 <https://doi.org/10.1145/355744.355745>`_

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.search.BallTree` : Spherical partitions, better in higher dimensions.
    :class:`~tuiml.algorithms.neighbors.search.LinearNNSearch` : Brute-force search as a baseline.

    Examples
    --------
    Build a KD-Tree and query for the nearest neighbor:

    >>> from tuiml.algorithms.neighbors.search import KDTree
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> tree = KDTree(leaf_size=2)
    >>> tree.build(X)
    KDTree(n_samples=4, leaf_size=2)
    >>> dists, indices = tree.query([3.1, 4.1], k=1)
    """

    def __init__(self, leaf_size: int = 10):
        """Initialize KDTree.

        Parameters
        ----------
        leaf_size : int, default=10
            Maximum number of points stored in each leaf node.
        """
        super().__init__()
        self.leaf_size = leaf_size
        self._cpp_tree = None
        self._root = None

    def build(self, X: np.ndarray) -> "KDTree":
        """Build the KD-Tree structure from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : KDTree
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape

        self._cpp_tree = _cpp_nn.KDTree(self.leaf_size)
        self._cpp_tree.build(np.ascontiguousarray(X, dtype=np.float64))
        # The C++ backend owns the actual tree structure. Expose a non-None
        # root sentinel for API compatibility with earlier Python-tree tests.
        self._root = self._cpp_tree

        self._is_built = True
        return self

    def query(self, x: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors for a query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (k,)
            Distances to the k nearest neighbors.
        indices : np.ndarray of shape (k,)
            Indices of the k nearest neighbors in the training data.
        """
        self._check_is_built()
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(1, -1))
        dists, idxs = self._cpp_tree.query(x, min(k, self.n_samples_))
        return dists[0], idxs[0]

    def query_batch(self, X: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors for multiple query points.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            The query points.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (n_queries, k)
            Distances to the k nearest neighbors.
        indices : np.ndarray of shape (n_queries, k)
            Indices of the k nearest neighbors in the training data.
        """
        self._check_is_built()
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._cpp_tree.query(X, min(k, self.n_samples_))

    def query_radius(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find all neighbors within a specified radius.

        Falls back to a full k-query and filters by distance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.
        radius : float
            The maximum distance to search within.

        Returns
        -------
        distances : np.ndarray
            Distances to all neighbors within the radius.
        indices : np.ndarray
            Indices of neighbors within the radius.
        """
        self._check_is_built()
        dists, idxs = self.query(x, k=self.n_samples_)
        mask = dists <= radius
        return dists[mask], idxs[mask]

    def __repr__(self) -> str:
        """String representation."""
        if self._is_built:
            return f"KDTree(n_samples={self.n_samples_}, leaf_size={self.leaf_size})"
        return f"KDTree(leaf_size={self.leaf_size}, not built)"
