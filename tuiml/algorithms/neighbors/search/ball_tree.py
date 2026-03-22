"""Ball Tree Nearest Neighbor Search."""

import numpy as np
from typing import Tuple, Optional, List
import heapq

from tuiml.base.neighbors import NearestNeighborSearch
from tuiml._cpp_ext import neighbors as _cpp_nn

class BallTree(NearestNeighborSearch):
    """Ball Tree for **nearest neighbor search** in high-dimensional spaces.

    A Ball Tree is a binary tree where each node defines a **D-dimensional
    hypersphere** (ball) containing a subset of the data points. Compared to
    KD-trees, Ball Trees tend to be more efficient when the dimensionality
    is **moderate to high** because they use ball-shaped (spherical) partitions
    instead of axis-aligned splits.

    This implementation delegates tree construction and querying to an
    optimized **C++ backend** with optional OpenMP parallelism, providing
    orders-of-magnitude speedup over pure-Python recursive traversal.

    Overview
    --------
    The Ball Tree is constructed and queried as follows:

    1. Compute the **centroid** and **radius** of the bounding ball for all
       points in the current subset.
    2. If the number of points is at or below ``leaf_size``, create a leaf
       node storing the point indices.
    3. Otherwise, find the dimension with the greatest spread and split the
       points along the median of that dimension.
    4. Recursively build left and right child subtrees.
    5. During a query, prune subtrees whose bounding ball cannot contain
       a point closer than the current k-th nearest neighbor.

    Theory
    ------
    The pruning criterion for a query point :math:`q` and a ball node with
    center :math:`c` and radius :math:`r` is:

    .. math::
        d(q, c) - r \\geq d_k

    where :math:`d_k` is the distance to the current k-th nearest neighbor.
    If the inequality holds, the entire subtree is skipped because no point
    inside the ball can be closer than :math:`d_k`.

    The distance metric used is **Euclidean distance**:

    .. math::
        d(q, x_i) = \\sqrt{\\sum_{j=1}^{m} (q_j - x_{i,j})^2}

    Parameters
    ----------
    leaf_size : int, default=10
        Maximum number of points in a leaf node. Smaller values create
        deeper trees with more pruning opportunities but slower construction.

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
    - Query (worst case): :math:`O(n)` per query point (high-dimensional or adversarial data)
    - Space: :math:`O(n)` for the tree structure

    **When to use BallTree:**

    - Moderate to high-dimensional data where KD-trees degrade
    - When the data has intrinsic low-dimensional structure (e.g., manifolds)
    - Repeated nearest-neighbor queries against a fixed dataset
    - When non-axis-aligned clusters are present in the data

    References
    ----------
    .. [Omohundro1989] Omohundro, S.M. (1989).
           **Five Balltree Construction Algorithms.**
           *International Computer Science Institute Technical Report*.

    .. [Liu2006] Liu, T., Moore, A.W. and Gray, A.G. (2006).
           **New Algorithms for Efficient High-Dimensional Nonparametric Classification.**
           *Journal of Machine Learning Research*, 7, pp. 1135-1158.

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.search.KDTree` : Axis-aligned tree, faster in low dimensions.
    :class:`~tuiml.algorithms.neighbors.search.LinearNNSearch` : Brute-force search as a baseline.

    Examples
    --------
    Build a Ball Tree and query for the nearest neighbor:

    >>> from tuiml.algorithms.neighbors.search import BallTree
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> tree = BallTree(leaf_size=2)
    >>> tree.build(X)
    BallTree(n_samples=4, leaf_size=2)
    >>> dists, indices = tree.query([3.1, 4.1], k=1)
    """

    def __init__(self, leaf_size: int = 10):
        """Initialize BallTree.

        Parameters
        ----------
        leaf_size : int, default=10
            Maximum number of points stored in each leaf node.
        """
        super().__init__()
        self.leaf_size = leaf_size
        self._cpp_tree = None
        self._root = None

    def build(self, X: np.ndarray) -> "BallTree":
        """Build the Ball Tree structure from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : BallTree
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape

        self._cpp_tree = _cpp_nn.BallTree(self.leaf_size)
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
            return f"BallTree(n_samples={self.n_samples_}, leaf_size={self.leaf_size})"
        return f"BallTree(leaf_size={self.leaf_size}, not built)"
