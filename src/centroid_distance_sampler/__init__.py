"""Centroid-distance based numerosity reduction for clustered datasets."""
from ._sampler import (
    WEIGHT_FUNCTIONS,
    centroid_distance_sample,
    centroid_distances,
    compute_centroids,
    sampling_probabilities,
)

__all__ = [
    "WEIGHT_FUNCTIONS",
    "centroid_distance_sample",
    "centroid_distances",
    "compute_centroids",
    "sampling_probabilities",
]

__version__ = "0.1.0"
