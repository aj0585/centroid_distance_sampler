"""Centroid-distance based numerosity reduction.

Points far from their cluster centroid (i.e. near the cluster boundary) are
given a higher probability of being kept, so that a reduced training set
preserves the extent of each cluster better than a uniform random subset.
"""
from numbers import Integral, Real
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    column_or_1d,
)
from sklearn.utils.extmath import row_norms

__all__ = [
    "WEIGHT_FUNCTIONS",
    "compute_centroids",
    "centroid_distances",
    "sampling_probabilities",
    "centroid_distance_sample",
]

WeightFunc = Union[str, Callable[[np.ndarray], np.ndarray]]

#: Built-in weight functions mapping distance -> unnormalised weight.
WEIGHT_FUNCTIONS = {
    "linear": lambda d: d,
    "squared": np.square,
    "uniform": np.ones_like,
}

_METRICS = ("euclidean", "mahalanobis")


def _validate_inputs(X, labels) -> Tuple[np.ndarray, np.ndarray]:
    X = check_array(X, dtype=np.float64)
    labels = column_or_1d(labels, warn=True)
    check_consistent_length(X, labels)
    return X, labels


def compute_centroids(X, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the centroid (mean) of every cluster.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)
        Hard cluster assignment of each sample. Any hashable, sortable label
        type is accepted (ints, strings, ...).

    Returns
    -------
    classes : ndarray of shape (n_clusters,)
        Unique cluster labels, sorted.
    inverse : ndarray of shape (n_samples,)
        Index into ``classes`` (and ``centroids``) for each sample.
    centroids : ndarray of shape (n_clusters, n_features)
    """
    X, labels = _validate_inputs(X, labels)
    return _compute_centroids(X, labels)


def _compute_centroids(X, labels):
    classes, inverse = np.unique(labels, return_inverse=True)
    n = X.shape[0]
    # Sparse (n_clusters x n_samples) membership matrix: one matmul sums every
    # cluster in O(n_samples * n_features) without a Python loop.
    membership = sparse.csr_matrix(
        (np.ones(n), (inverse, np.arange(n))), shape=(classes.size, n)
    )
    counts = np.bincount(inverse, minlength=classes.size)
    centroids = np.asarray(membership @ X) / counts[:, np.newaxis]
    return classes, inverse, centroids


def centroid_distances(
    X,
    labels,
    metric: str = "euclidean",
    covariance_estimator=None,
) -> np.ndarray:
    """Distance of every sample to the centroid of its own cluster.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)
        Hard cluster assignment of each sample.
    metric : {"euclidean", "mahalanobis"}, default="euclidean"
        For ``"mahalanobis"`` a covariance matrix is estimated separately for
        every cluster, so distances are expressed relative to that cluster's
        own spread and shape.
    covariance_estimator : sklearn covariance estimator, default=None
        Only used with ``metric="mahalanobis"``. Unfitted estimator (e.g.
        ``LedoitWolf()``) that is cloned and fitted on each cluster to obtain
        its precision matrix. Defaults to ``EmpiricalCovariance()``, whose
        precision is a pseudo-inverse, so singular covariances are handled.
        Distances are always measured from the cluster centroid, not from
        the estimator's ``location_``.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Clusters with a single member get distance 0 under Mahalanobis, since
        their covariance is undefined.
    """
    X, labels = _validate_inputs(X, labels)
    return _centroid_distances(X, labels, metric, covariance_estimator)


def _centroid_distances(X, labels, metric, covariance_estimator):
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}.")

    classes, inverse, centroids = _compute_centroids(X, labels)
    diff = X - centroids[inverse]

    if metric == "euclidean":
        return row_norms(diff)

    if covariance_estimator is None:
        covariance_estimator = EmpiricalCovariance()
    distances = np.zeros(X.shape[0])
    for k in range(classes.size):
        idx = np.flatnonzero(inverse == k)
        if idx.size < 2:
            continue
        precision = clone(covariance_estimator).fit(X[idx]).get_precision()
        d_k = diff[idx]
        sq = np.sum((d_k @ precision) * d_k, axis=1)  # BLAS-backed quad form
        # Clip tiny negatives caused by round-off with near-singular matrices.
        distances[idx] = np.sqrt(np.maximum(sq, 0.0))
    return distances


def _resolve_weight_func(weight_func: WeightFunc):
    if callable(weight_func):
        return weight_func
    try:
        return WEIGHT_FUNCTIONS[weight_func]
    except (KeyError, TypeError):
        raise ValueError(
            f"weight_func must be callable or one of {sorted(WEIGHT_FUNCTIONS)}, "
            f"got {weight_func!r}."
        ) from None


def _distances_to_probabilities(distances, weight_func):
    weights = np.asarray(_resolve_weight_func(weight_func)(distances), dtype=float)
    if weights.shape != distances.shape:
        raise ValueError(
            f"weight_func returned shape {weights.shape}, expected {distances.shape}."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weight_func must return finite, non-negative weights.")
    total = weights.sum()
    if total == 0:
        # Every point is at its centroid (e.g. all singleton clusters): no
        # distance information, so fall back to uniform.
        return np.full_like(weights, 1.0 / weights.size)
    return weights / total


def sampling_probabilities(
    X,
    labels,
    metric: str = "euclidean",
    weight_func: WeightFunc = "linear",
    covariance_estimator=None,
) -> np.ndarray:
    """Probability of each sample under the centroid-distance distribution.

    Distances to cluster centroids are mapped through ``weight_func`` and
    normalised to sum to one across the whole dataset.

    Parameters
    ----------
    X, labels, metric, covariance_estimator
        See :func:`centroid_distances`.
    weight_func : {"linear", "squared", "uniform"} or callable, default="linear"
        Maps the distance array to non-negative weights. ``"linear"`` uses
        ``d``, ``"squared"`` uses ``d ** 2`` (stronger emphasis on boundary
        points), ``"uniform"`` ignores distance (plain random sampling). A
        callable receives the full ``(n_samples,)`` distance array and must
        return an array of the same shape.

    Returns
    -------
    probabilities : ndarray of shape (n_samples,)
        Non-negative, sums to 1.
    """
    X, labels = _validate_inputs(X, labels)
    distances = _centroid_distances(X, labels, metric, covariance_estimator)
    return _distances_to_probabilities(distances, weight_func)


def _resolve_n_samples(n_samples, n_total):
    if isinstance(n_samples, (bool, np.bool_)):
        raise TypeError("n_samples must be an int or a float, not bool.")
    if isinstance(n_samples, Integral):
        if not 1 <= n_samples <= n_total:
            raise ValueError(
                f"n_samples={n_samples} must be in [1, {n_total}] when an int."
            )
        return int(n_samples)
    if isinstance(n_samples, Real):
        if not 0.0 < n_samples <= 1.0:
            raise ValueError(
                f"n_samples={n_samples} must be in (0, 1] when a float."
            )
        n = int(n_samples * n_total)
        if n < 1:
            raise ValueError(
                f"n_samples={n_samples} of {n_total} samples rounds down to 0."
            )
        return n
    raise TypeError(f"n_samples must be an int or a float, got {type(n_samples)}.")


def _weighted_sample_without_replacement(p, n, random_state):
    """Draw ``n`` distinct indices, successively proportional to ``p``.

    Uses the Efraimidis-Spirakis exponential-race method: each item gets key
    ``E_i / p_i`` with ``E_i ~ Exp(1)`` and the ``n`` smallest keys win. This
    has exactly the distribution of drawing one item at a time proportionally
    to the remaining weights, but runs in O(n_total) via ``argpartition``.

    If fewer than ``n`` items have non-zero probability, all of them are
    taken and the remainder is filled uniformly from the zero-probability
    items.
    """
    positive = np.flatnonzero(p > 0)
    if n <= positive.size:
        keys = random_state.standard_exponential(positive.size) / p[positive]
        chosen = positive[np.argpartition(keys, n - 1)[:n]]
    else:
        zero = np.flatnonzero(p == 0)
        fill = random_state.choice(zero, n - positive.size, replace=False)
        chosen = np.concatenate([positive, fill])
    return np.sort(chosen)


def centroid_distance_sample(
    X,
    labels,
    n_samples: Union[int, float],
    *,
    metric: str = "euclidean",
    weight_func: WeightFunc = "linear",
    covariance_estimator=None,
    random_state=None,
) -> np.ndarray:
    """Select a subset of samples, favouring points far from their centroid.

    1. Compute the centroid of each cluster given by ``labels``.
    2. Compute each sample's distance to its cluster centroid.
    3. Map distances to weights with ``weight_func``.
    4. Normalise weights over the whole dataset into probabilities.
    5. Draw ``n_samples`` distinct samples without replacement.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix used to compute centroids and distances (numpy array
        or pandas DataFrame).
    labels : array-like of shape (n_samples,)
        Hard cluster labels, e.g. ``KMeans().fit_predict(X)``. Every distinct
        value is treated as one cluster, including noise labels such as
        DBSCAN's ``-1``.
    n_samples : int or float
        If int, the number of samples to keep. If float in (0, 1], the
        fraction of the dataset to keep (rounded down).
    metric : {"euclidean", "mahalanobis"}, default="euclidean"
        See :func:`centroid_distances`.
    weight_func : {"linear", "squared", "uniform"} or callable, default="linear"
        See :func:`sampling_probabilities`.
    covariance_estimator : sklearn covariance estimator, default=None
        See :func:`centroid_distances`.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the draw. Pass an int for reproducible
        output.

    Returns
    -------
    indices : ndarray of shape (n_kept,)
        Sorted positional indices of the kept samples. Use ``X[indices]``
        (or ``df.iloc[indices]``) to build the reduced dataset.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=1000, centers=3, random_state=0)
    >>> labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    >>> idx = centroid_distance_sample(X, labels, 0.1, random_state=0)
    >>> X_small, y_small = X[idx], y[idx]
    >>> X_small.shape
    (100, 2)
    """
    X, labels = _validate_inputs(X, labels)
    n = _resolve_n_samples(n_samples, X.shape[0])
    random_state = check_random_state(random_state)

    distances = _centroid_distances(X, labels, metric, covariance_estimator)
    p = _distances_to_probabilities(distances, weight_func)
    return _weighted_sample_without_replacement(p, n, random_state)
