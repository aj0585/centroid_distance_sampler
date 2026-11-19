import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_blobs

from centroid_distance_sampler import (
    centroid_distance_sample,
    centroid_distances,
    compute_centroids,
    sampling_probabilities,
)


@pytest.fixture
def blobs():
    X, labels = make_blobs(n_samples=600, centers=3, n_features=4, random_state=0)
    return X, labels


def test_compute_centroids_matches_groupwise_mean(blobs):
    X, labels = blobs
    classes, inverse, centroids = compute_centroids(X, labels)
    np.testing.assert_array_equal(classes, [0, 1, 2])
    np.testing.assert_array_equal(classes[inverse], labels)
    for k, c in enumerate(classes):
        np.testing.assert_allclose(centroids[k], X[labels == c].mean(axis=0))


def test_euclidean_distances(blobs):
    X, labels = blobs
    d = centroid_distances(X, labels)
    expected = np.empty(len(X))
    for c in np.unique(labels):
        m = labels == c
        expected[m] = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
    np.testing.assert_allclose(d, expected)


def test_mahalanobis_distances_match_scipy(blobs):
    X, labels = blobs
    d = centroid_distances(X, labels, metric="mahalanobis")
    for c in np.unique(labels):
        m = labels == c
        Xc = X[m]
        # EmpiricalCovariance is the maximum-likelihood (biased) estimate.
        VI = np.linalg.inv(np.cov(Xc, rowvar=False, bias=True))
        mu = Xc.mean(axis=0)
        expected = [mahalanobis(x, mu, VI) for x in Xc]
        np.testing.assert_allclose(d[m], expected, rtol=1e-8)


def test_mahalanobis_custom_estimator_and_singular_covariance():
    rng = np.random.RandomState(0)
    # Cluster 0 lies on a line (singular covariance); cluster 1 is a singleton.
    t = rng.randn(50)
    X = np.vstack([np.c_[t, 2 * t], [[10.0, 10.0]]])
    labels = np.r_[np.zeros(50), 1]
    d = centroid_distances(X, labels, metric="mahalanobis")
    assert np.all(np.isfinite(d))
    assert d[-1] == 0.0
    d_lw = centroid_distances(X, labels, metric="mahalanobis",
                              covariance_estimator=LedoitWolf())
    assert np.all(np.isfinite(d_lw))


def test_probabilities_sum_to_one_and_proportional(blobs):
    X, labels = blobs
    d = centroid_distances(X, labels)
    p = sampling_probabilities(X, labels)
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, d / d.sum())
    p2 = sampling_probabilities(X, labels, weight_func="squared")
    np.testing.assert_allclose(p2, d ** 2 / (d ** 2).sum())
    pu = sampling_probabilities(X, labels, weight_func="uniform")
    np.testing.assert_allclose(pu, 1 / len(X))


def test_callable_weight_func(blobs):
    X, labels = blobs
    p = sampling_probabilities(X, labels, weight_func=lambda d: np.exp(d))
    d = centroid_distances(X, labels)
    np.testing.assert_allclose(p, np.exp(d) / np.exp(d).sum())


@pytest.mark.parametrize("bad", [lambda d: -d, lambda d: d[:-1],
                                 lambda d: np.full_like(d, np.nan)])
def test_invalid_weight_func_output(blobs, bad):
    X, labels = blobs
    with pytest.raises(ValueError):
        sampling_probabilities(X, labels, weight_func=bad)


def test_unknown_options(blobs):
    X, labels = blobs
    with pytest.raises(ValueError, match="weight_func"):
        centroid_distance_sample(X, labels, 10, weight_func="cubic")
    with pytest.raises(ValueError, match="metric"):
        centroid_distance_sample(X, labels, 10, metric="cosine")


@pytest.mark.parametrize("n_samples, expected", [(1, 1), (100, 100), (600, 600),
                                                 (0.1, 60), (1.0, 600), (0.0025, 1)])
def test_output_size_unique_sorted(blobs, n_samples, expected):
    X, labels = blobs
    idx = centroid_distance_sample(X, labels, n_samples, random_state=0)
    assert idx.shape == (expected,)
    assert np.unique(idx).size == expected
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < len(X)


@pytest.mark.parametrize("n_samples", [0, 601, -5, 0.0, 1.5, 0.001])
def test_invalid_n_samples_value(blobs, n_samples):
    X, labels = blobs
    with pytest.raises(ValueError):
        centroid_distance_sample(X, labels, n_samples)


@pytest.mark.parametrize("n_samples", [True, "10", None])
def test_invalid_n_samples_type(blobs, n_samples):
    X, labels = blobs
    with pytest.raises(TypeError):
        centroid_distance_sample(X, labels, n_samples)


def test_reproducible(blobs):
    X, labels = blobs
    a = centroid_distance_sample(X, labels, 50, random_state=42)
    b = centroid_distance_sample(X, labels, 50, random_state=42)
    c = centroid_distance_sample(X, labels, 50, random_state=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_length_mismatch(blobs):
    X, labels = blobs
    with pytest.raises(ValueError):
        centroid_distance_sample(X, labels[:-1], 10)


def test_zero_weight_points_fill_uniformly():
    # Three singleton clusters (distance 0) plus one 2-point cluster whose
    # members both have positive distance.
    X = np.array([[0.0], [5.0], [9.0], [20.0], [22.0]])
    labels = np.array([0, 1, 2, 3, 3])
    idx = centroid_distance_sample(X, labels, 4, random_state=0)
    assert {3, 4} <= set(idx)
    assert idx.size == 4
    # n_samples within the positive-weight set never picks a zero-weight point.
    for seed in range(20):
        assert set(centroid_distance_sample(X, labels, 2, random_state=seed)) == {3, 4}


def test_all_singletons_falls_back_to_uniform():
    X = np.arange(10.0).reshape(-1, 1)
    p = sampling_probabilities(X, np.arange(10))
    np.testing.assert_allclose(p, 0.1)
    assert centroid_distance_sample(X, np.arange(10), 5, random_state=0).size == 5


def test_single_draw_frequency_matches_probabilities():
    # With n_samples=1, inclusion frequency must equal p (checks the
    # exponential-race sampler against the target distribution).
    X = np.array([[1.0], [2.0], [3.0], [-6.0]])
    labels = np.zeros(4)
    p = sampling_probabilities(X, labels)
    rng = np.random.RandomState(0)
    trials = 20000
    counts = np.bincount(
        [centroid_distance_sample(X, labels, 1, random_state=rng)[0]
         for _ in range(trials)], minlength=4)
    np.testing.assert_allclose(counts / trials, p, atol=0.015)


def test_successive_sampling_inclusion_probabilities():
    # Exact inclusion probabilities for drawing 2 of 3 items successively
    # proportional to weight: P(i in S) = 1 - P(i is the one left out).
    w = np.array([1.0, 2.0, 3.0])
    X = np.zeros((3, 1))  # distances are irrelevant: weights come from callable
    labels = np.zeros(3)
    tot = w.sum()
    left_out = np.array([
        sum(w[j] / tot * w[k] / (tot - w[j]) for j in range(3) for k in range(3)
            if len({i, j, k}) == 3)
        for i in range(3)
    ])
    expected = 1 - left_out
    rng = np.random.RandomState(1)
    trials = 20000
    hits = np.zeros(3)
    for _ in range(trials):
        hits[centroid_distance_sample(X, labels, 2, weight_func=lambda d: w,
                                      random_state=rng)] += 1
    np.testing.assert_allclose(hits / trials, expected, atol=0.015)


def test_boundary_points_favoured(blobs):
    X, labels = blobs
    d = centroid_distances(X, labels)
    far = d > np.quantile(d, 0.9)
    near = d < np.quantile(d, 0.1)
    rng = np.random.RandomState(0)
    hits = np.zeros(len(X))
    for _ in range(200):
        hits[centroid_distance_sample(X, labels, 60, random_state=rng)] += 1
    assert hits[far].mean() > 3 * hits[near].mean()


def test_pandas_and_string_labels(blobs):
    pd = pytest.importorskip("pandas")
    X, labels = blobs
    df = pd.DataFrame(X, columns=list("abcd"))
    str_labels = pd.Series(np.array(["x", "y", "z"])[labels])
    idx = centroid_distance_sample(df, str_labels, 30, random_state=0)
    np.testing.assert_array_equal(
        idx, centroid_distance_sample(X, labels, 30, random_state=0))
    assert df.iloc[idx].shape == (30, 4)
