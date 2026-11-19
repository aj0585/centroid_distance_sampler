"""Centroid-distance sampling on scikit-learn's synthetic clustering datasets.

Compares a uniform random subset with centroid-distance sampling on five
2-D datasets (the same family used in scikit-learn's clustering comparison).
Each dataset is clustered with a realistic algorithm, and the predicted
labels are passed to the sampler.

Outputs, written to ``examples/figures/``:

* ``subsets.png``: one 5% subset per dataset and method, drawn over the full data.
* ``metrics.png``: how the subsets cover each cluster as the kept fraction
  varies, averaged over many random seeds.
* ``metrics.csv``: the numbers behind ``metrics.png``.

Run from the repository root::

    python examples/synthetic_datasets.py

Requires matplotlib and pandas in addition to the package dependencies.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.datasets import make_blobs, make_circles, make_moons  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from centroid_distance_sampler import centroid_distance_sample  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
N = 3000
SHOW_FRACTION = 0.05
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 30

# Colour roles (reference data-viz palette, light mode). Scatter plots use at
# most three categorical slots, the limit validated for all-pairs contrast.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# The random baseline is a neutral reference line; the three centroid-distance
# variants take the first three categorical slots.
METHODS = {
    "Random": dict(weight_func="uniform", metric="euclidean", color=MUTED),
    "Euclidean, linear": dict(weight_func="linear", metric="euclidean",
                              color=CLUSTER_COLORS[0]),
    "Euclidean, squared": dict(weight_func="squared", metric="euclidean",
                               color=CLUSTER_COLORS[1]),
    "Mahalanobis, linear": dict(weight_func="linear", metric="mahalanobis",
                                color=CLUSTER_COLORS[2]),
}
SHOWN_METHODS = ["Random", "Euclidean, linear", "Mahalanobis, linear"]


def make_datasets():
    """Return ``{name: (X, labels, clusterer description, ARI)}``."""
    out = {}

    X, y = make_blobs(N, centers=3, random_state=170)
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(X)
    out["Isotropic blobs"] = (X, lab, "KMeans", adjusted_rand_score(y, lab))

    X, y = make_blobs(N, centers=3, random_state=170)
    X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])
    lab = GaussianMixture(3, covariance_type="full", random_state=0).fit_predict(X)
    out["Anisotropic blobs"] = (X, lab, "Gaussian mixture",
                                adjusted_rand_score(y, lab))

    X, y = make_blobs(N, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    lab = GaussianMixture(3, covariance_type="full", random_state=0).fit_predict(X)
    out["Unequal variance blobs"] = (X, lab, "Gaussian mixture",
                                     adjusted_rand_score(y, lab))

    X, y = make_moons(N, noise=0.05, random_state=0)
    lab = AgglomerativeClustering(2, linkage="single").fit_predict(X)
    out["Moons"] = (X, lab, "Single-linkage agglomerative",
                    adjusted_rand_score(y, lab))

    X, y = make_circles(N, factor=0.5, noise=0.04, random_state=0)
    lab = AgglomerativeClustering(2, linkage="single").fit_predict(X)
    out["Concentric circles"] = (X, lab, "Single-linkage agglomerative",
                                 adjusted_rand_score(y, lab))
    return out


def _sample(X, labels, method, fraction, seed):
    kw = {k: v for k, v in METHODS[method].items() if k != "color"}
    return centroid_distance_sample(X, labels, fraction, random_state=seed, **kw)


def subset_metrics(X, labels, idx):
    """Per-cluster coverage of the full data by the kept subset, averaged.

    * extent: convex-hull area of kept points / hull area of all points (%).
    * mean_gap, max_gap: distance from each original point to the nearest
      kept point of the same cluster (lower is better). ``max_gap`` measures
      how well the outer edge is covered and ``mean_gap`` how well the core is.
    """
    keep = np.zeros(len(X), bool)
    keep[idx] = True
    extent, mean_gap, max_gap = [], [], []
    for c in np.unique(labels):
        m = labels == c
        full, kept = X[m], X[m & keep]
        if len(kept) >= 3:
            extent.append(100 * ConvexHull(kept).volume / ConvexHull(full).volume)
        else:
            extent.append(0.0)
        if len(kept) == 0:
            continue
        gaps, _ = cKDTree(kept).query(full)
        mean_gap.append(gaps.mean())
        max_gap.append(gaps.max())
    return np.mean(extent), np.mean(mean_gap), np.mean(max_gap)


def evaluate(datasets):
    rows = []
    for name, (X, labels, _, _) in datasets.items():
        for method in METHODS:
            for frac in FRACTIONS:
                for seed in range(N_SEEDS):
                    idx = _sample(X, labels, method, frac, seed)
                    ext, mg, xg = subset_metrics(X, labels, idx)
                    rows.append(dict(dataset=name, method=method, fraction=frac,
                                     seed=seed, extent=ext, mean_gap=mg,
                                     max_gap=xg))
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def plot_subsets(datasets, summary):
    names = list(datasets)
    ncols = 1 + len(SHOWN_METHODS)
    fig, axes = plt.subplots(len(names), ncols, figsize=(3.1 * ncols, 3.0 * len(names)),
                             facecolor=SURFACE)
    pct = f"{SHOW_FRACTION:.0%}"
    col_titles = ["All data, coloured by cluster label"] + [
        f"{m} ({pct} kept)" if m == "Random" else f"Centroid distance: {m} ({pct})"
        for m in SHOWN_METHODS
    ]
    for r, name in enumerate(names):
        X, labels, clusterer, ari = datasets[name]
        colors = np.array(CLUSTER_COLORS)[labels]
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for s in ax.spines.values():
                s.set_color(GRID)
            if c == 0:
                ax.scatter(X[:, 0], X[:, 1], s=3, c=colors, linewidths=0, alpha=0.6)
                ax.set_ylabel(f"{name}\n{clusterer}, ARI {ari:.2f}", color=INK,
                              fontsize=9)
            else:
                method = SHOWN_METHODS[c - 1]
                idx = _sample(X, labels, method, SHOW_FRACTION, seed=0)
                ax.scatter(X[:, 0], X[:, 1], s=3, c=GRID, linewidths=0)
                ax.scatter(X[idx, 0], X[idx, 1], s=12, c=colors[idx],
                           edgecolors=SURFACE, linewidths=0.4)
                s = summary.loc[(name, method)]
                ax.text(0.02, 0.02,
                        f"extent kept {s.extent:.0f}%\n"
                        f"max gap {s.max_gap:.2f}  mean gap {s.mean_gap:.2f}",
                        transform=ax.transAxes, fontsize=7.5, color=INK_2,
                        va="bottom", ha="left",
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5))
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9, color=INK)
    fig.suptitle(
        f"{pct} subsets of {N:,} points. Grey points are discarded. Panel metrics "
        f"are averages over {N_SEEDS} seeds.",
        fontsize=10, color=INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "subsets.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_metrics(df):
    agg = df.groupby(["dataset", "method", "fraction"], sort=False)[
        ["extent", "mean_gap", "max_gap"]].mean()
    datasets = list(dict.fromkeys(df.dataset))
    metrics = [
        ("extent", "Extent kept, % of hull area\n(higher is better)"),
        ("max_gap", "Max gap to nearest kept point\n(edge coverage, lower is better)"),
        ("mean_gap", "Mean gap to nearest kept point\n(core coverage, lower is better)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(datasets),
                             figsize=(3.2 * len(datasets), 3.0 * len(metrics)),
                             facecolor=SURFACE, sharex=True)
    for c, name in enumerate(datasets):
        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            _style(ax)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for method, spec in METHODS.items():
                series = agg.loc[(name, method)][key]
                ax.plot(series.index, series.values, color=spec["color"], lw=2,
                        marker="o", ms=4, mec=SURFACE, mew=1,
                        zorder=3 if method != "Random" else 2)
            ax.set_xscale("log")
            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(name, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
            if r == len(metrics) - 1:
                ax.set_xlabel("Fraction of data kept", fontsize=8.5, color=INK_2)
    # Legend once, above the grid, so every panel shares it.
    handles = [plt.Line2D([], [], color=s["color"], lw=2, marker="o", ms=4,
                          mec=SURFACE) for s in METHODS.values()]
    fig.legend(handles, list(METHODS), loc="upper center", ncol=len(METHODS),
               frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "metrics.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets()
    df = evaluate(datasets)
    agg = plot_metrics(df)
    agg.round(4).to_csv(OUT / "metrics.csv")
    summary = df[df.fraction == SHOW_FRACTION].groupby(["dataset", "method"])[
        ["extent", "mean_gap", "max_gap"]].mean()
    plot_subsets(datasets, summary)

    pd.set_option("display.width", 120)
    print("Clustering quality (adjusted Rand index vs. generating labels):")
    for name, (_, _, clusterer, ari) in datasets.items():
        print(f"  {name:<24} {clusterer:<30} ARI={ari:.3f}")
    print(f"\nMetrics at {SHOW_FRACTION:.0%} kept, mean of {N_SEEDS} seeds:")
    print(summary.round(3).to_string())
    print(f"\nWrote figures to {OUT}")


if __name__ == "__main__":
    main()
