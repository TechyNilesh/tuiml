"""Hierarchical Agglomerative Clustering (HAC) algorithm."""

import heapq
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from tuiml.base.algorithms import Clusterer, clusterer
from tuiml.algorithms.clustering.distance import pairwise_distances

@dataclass
class ClusterNode:
    """Node in the hierarchical clustering tree (dendrogram)."""
    id: int
    left: Optional["ClusterNode"] = None
    right: Optional["ClusterNode"] = None
    distance: float = 0.0
    count: int = 1
    indices: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

@clusterer(tags=["hierarchical", "agglomerative"], version="1.0.0")
class AgglomerativeClusterer(Clusterer):
    r"""
    Hierarchical Agglomerative Clustering.

    Builds a **hierarchy of clusters** by progressively merging the most similar
    clusters. This implementation uses a **bottom-up (agglomerative)** approach,
    starting with each point as its own cluster and merging them based on
    a linkage criterion.

    Overview
    --------
    The algorithm constructs a dendrogram (tree of merges):

    1. Initialize each data point as a singleton cluster
    2. Compute the pairwise distance matrix between all clusters
    3. Find the two closest clusters according to the linkage criterion
    4. Merge them into a single cluster and record the merge distance
    5. Repeat steps 2--4 until only one cluster remains
    6. Cut the dendrogram at the level that yields ``n_clusters`` clusters

    Theory
    ------
    The linkage criterion determines the distance between sets of observations
    as a function of the pairwise distances between observations:

    - **Single Linkage**:
      :math:`d(A, B) = \min \{ \\text{dist}(a, b) : a \in A, b \in B \}`
    - **Complete Linkage**:
      :math:`d(A, B) = \max \{ \\text{dist}(a, b) : a \in A, b \in B \}`
    - **Average Linkage (UPGMA)**:
      :math:`d(A, B) = \\frac{1}{|A| |B|} \sum_{a \in A} \sum_{b \in B} \\text{dist}(a, b)`
    - **Ward Linkage** minimizes the increase in total within-cluster variance:

    .. math::
        \Delta(A, B) = \sqrt{\\frac{|A| \cdot |B|}{|A| + |B|}} \| \\bar{A} - \\bar{B} \|_2

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Which linkage criterion to use.
    distance : {"euclidean", "manhattan"}, default="euclidean"
        Metric used to compute the linkage. If linkage is "ward", only
        "euclidean" is accepted.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point.
    children_ : np.ndarray of shape (n_samples-1, 2)
        The children of each non-leaf node. Each row represents a merge.
    distances_ : np.ndarray of shape (n_samples-1,)
        The distances between nodes which were merged.
    n_clusters_ : int
        The number of clusters found by the algorithm.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^3)` time complexity and :math:`O(n^2)` memory.
      This makes it unsuitable for very large datasets.

    **When to use AgglomerativeClusterer:**

    - When you need a full hierarchy or dendrogram of clusters
    - Exploratory analysis to understand data structure at multiple scales
    - Small to medium datasets (quadratic memory limits scalability)
    - When cluster shapes are non-globular and linkage choice matters

    References
    ----------
    .. [Mullner2011] Mullner, D. (2011).
           **Modern hierarchical, agglomerative clustering algorithms.**
           *arXiv preprint arXiv:1109.2378*.

    .. [Ward1963] Ward, J. H. (1963).
           **Hierarchical grouping to optimize an objective function.**
           *Journal of the American Statistical Association*, 58(301), pp. 236-244.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.CobwebClusterer` : Incremental hierarchical conceptual clustering.
    :class:`~tuiml.algorithms.clustering.DBSCANClusterer` : Density-based clustering with noise detection.

    Examples
    --------
    Agglomerative clustering with Ward linkage:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import AgglomerativeClusterer
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> hc = AgglomerativeClusterer(n_clusters=2, linkage='ward')
    >>> hc.fit(X)
    >>> hc.labels_
    array([0, 0, 0, 1, 1, 1])
    """

    def __init__(self, n_clusters: int = 2,
                 linkage: str = 'ward',
                 distance: str = 'euclidean'):
        """Initialize AgglomerativeClusterer.

        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters.
        linkage : str, default='ward'
            Linkage criterion.
        distance : str, default='euclidean'
            Distance metric.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance = distance
        self.children_ = None
        self.distances_ = None
        self.dendrogram_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_clusters": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of clusters"
            },
            "linkage": {
                "type": "string",
                "default": "ward",
                "enum": ["single", "complete", "average", "ward"],
                "description": "Linkage criterion"
            },
            "distance": {
                "type": "string",
                "default": "euclidean",
                "enum": ["euclidean", "manhattan"],
                "description": "Distance metric"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^3) time, O(n^2) space"

    def _compute_linkage_distance(self, dist_matrix: np.ndarray,
                                   cluster_i: List[int],
                                   cluster_j: List[int],
                                   X: np.ndarray = None) -> float:
        """Compute distance between two clusters based on linkage method.

        Parameters
        ----------
        dist_matrix : np.ndarray of shape (n_samples, n_samples)
            Precomputed pairwise distance matrix.
        cluster_i : list of int
            Indices of points in the first cluster.
        cluster_j : list of int
            Indices of points in the second cluster.
        X : np.ndarray of shape (n_samples, n_features), optional
            Original data, required for Ward linkage.

        Returns
        -------
        distance : float
            Inter-cluster distance according to the chosen linkage.
        """
        sub = dist_matrix[np.ix_(cluster_i, cluster_j)]

        if self.linkage == 'single':
            return float(sub.min())
        elif self.linkage == 'complete':
            return float(sub.max())
        elif self.linkage == 'average':
            return float(sub.mean())
        elif self.linkage == 'ward':
            if X is None:
                raise ValueError("Ward linkage requires original data")
            n_i = len(cluster_i)
            n_j = len(cluster_j)
            center_i = np.mean(X[cluster_i], axis=0)
            center_j = np.mean(X[cluster_j], axis=0)
            center_diff = center_i - center_j
            return np.sqrt((n_i * n_j) / (n_i + n_j) * np.sum(center_diff ** 2))
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def fit(self, X: np.ndarray) -> "AgglomerativeClusterer":
        """Fit the hierarchical clustering model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : AgglomerativeClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Compute pairwise distance matrix (used as mutable cluster distance matrix)
        n = n_samples
        cdist = pairwise_distances(X, metric=self.distance)
        sizes = np.ones(n, dtype=int)

        # Track merge history
        children = np.empty((n - 1, 2), dtype=np.intp)
        distances_arr = np.empty(n - 1, dtype=np.float64)

        np.fill_diagonal(cdist, np.inf)

        # Pre-select linkage function to avoid per-step branching
        linkage = self.linkage

        for step in range(n - 1):
            flat_idx = np.argmin(cdist)
            mi, mj = divmod(int(flat_idx), n)
            min_dist = cdist[mi, mj]

            if mi > mj:
                mi, mj = mj, mi

            children[step, 0] = mi
            children[step, 1] = mj
            distances_arr[step] = min_dist

            new_size = sizes[mi] + sizes[mj]

            # Update distances from merged cluster to all others
            d_mi = cdist[mi]
            d_mj = cdist[mj]

            if linkage == 'single':
                new_dists = np.minimum(d_mi, d_mj)
            elif linkage == 'complete':
                new_dists = np.maximum(d_mi, d_mj)
            elif linkage == 'average':
                new_dists = (sizes[mi] * d_mi + sizes[mj] * d_mj) / new_size
            elif linkage == 'ward':
                sk = sizes.astype(np.float64)
                si = float(sizes[mi])
                sj = float(sizes[mj])
                total = si + sj + sk
                new_dists = np.sqrt(
                    np.maximum(
                        ((si + sk) * d_mi ** 2
                         + (sj + sk) * d_mj ** 2
                         - sk * min_dist ** 2) / total,
                        0.0,
                    )
                )
            else:
                raise ValueError(f"Unknown linkage: {linkage}")

            cdist[mi, :] = new_dists
            cdist[:, mi] = new_dists
            cdist[mi, mi] = np.inf

            sizes[mi] = new_size
            cdist[mj, :] = np.inf
            cdist[:, mj] = np.inf

        self.children_ = children
        self.distances_ = distances_arr

        # Assign labels by cutting the dendrogram
        self.labels_ = self._cut_tree(n_samples)
        self.n_clusters_ = self.n_clusters
        self._is_fitted = True

        return self

    def _cut_tree(self, n_samples: int) -> np.ndarray:
        """Cut the dendrogram to get n_clusters clusters.

        Parameters
        ----------
        n_samples : int
            Number of original data points.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Sequential cluster labels from 0 to ``n_clusters - 1``.
        """
        # Union-Find approach: apply merges until n_clusters remain
        parent = list(range(n_samples))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_merges = len(self.children_)
        merges_to_apply = n_merges - (self.n_clusters - 1)

        for merge_idx in range(merges_to_apply):
            left, right = self.children_[merge_idx]
            # Map left/right through find to handle chained merges
            rl = find(left) if left < n_samples else left
            rr = find(right) if right < n_samples else right
            # Point right's root to left's root
            if rr < n_samples:
                parent[rr] = rl if rl < n_samples else rr
            if rl < n_samples and rr < n_samples:
                parent[rr] = rl

        # Assign labels
        roots = set()
        for i in range(n_samples):
            roots.add(find(i))
        root_to_label = {r: idx for idx, r in enumerate(sorted(roots))}
        labels = np.array([root_to_label[find(i)] for i in range(n_samples)])

        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        For hierarchical clustering, this assigns each new point to the
        cluster whose centroid is nearest.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute cluster centroids from training data
        if not hasattr(self, '_X_train'):
            raise RuntimeError("Training data not stored. Refit with store_data=True")

        # This is a limitation - hierarchical clustering doesn't naturally
        # support predicting new points. We use nearest centroid.
        raise NotImplementedError(
            "AgglomerativeClusterer does not support predicting new data. "
            "Use fit_predict() instead."
        )

    def get_dendrogram_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get data for plotting a dendrogram.

        Returns
        -------
        children : np.ndarray of shape (n_samples-1, 2)
            Merge history.
        distances : np.ndarray of shape (n_samples-1,)
            Distances at each merge.
        """
        self._check_is_fitted()
        return self.children_, self.distances_

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"AgglomerativeClusterer(n_clusters={self.n_clusters_}, "
                   f"linkage='{self.linkage}')")
        return f"AgglomerativeClusterer(n_clusters={self.n_clusters}, linkage='{self.linkage}')"
