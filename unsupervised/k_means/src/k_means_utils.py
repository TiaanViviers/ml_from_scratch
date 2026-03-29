import matplotlib.pyplot as plt
import numpy as np

try:
    from .k_means import KMeans
except ImportError:
    from k_means import KMeans

################################################################################
# Input validation
################################################################################

def _validate_input(X):
    """Validate a data matrix in the repo's (d, N) convention.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.

    Returns
    -------
    ndarray of shape (d, N)
        The validated input converted to a NumPy array of dtype float.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (d, N), got {X.ndim}D.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must be non-empty, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or inf).")

    return X


def _validate_labels(labels, n_samples):
    """Validate a label vector against a given number of observations.

    Parameters
    ----------
    labels : array-like of shape (N,)
        One cluster label per observation.
    n_samples : int
        Expected number of observations.

    Returns
    -------
    ndarray of shape (N,)
        The validated label vector.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D with shape (N,), got {labels.ndim}D.")
    if labels.shape[0] != n_samples:
        raise ValueError(
            "labels must have one entry per observation: "
            f"got {labels.shape[0]} labels and N={n_samples}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer cluster indices.")

    return labels


################################################################################
# Scoring / Evaluation Metrics
################################################################################

def compute_distortion(X, labels, centroids):
    """Compute the k-means distortion for a hard clustering.

    Distortion is the within-cluster sum of squared Euclidean distances. Lower
    values are better because they mean observations lie closer to their
    assigned cluster centroids. Distortion almost always decreases as the number
    of clusters increases, so it should not be used by itself to choose `k`.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    labels : array-like of shape (N,)
        Hard cluster assignments for each observation.
    centroids : array-like of shape (d, k), optional
        Centroids corresponding to the labels. If omitted, they are recomputed
        from `X` and `labels`.

    Returns
    -------
    float
        Total within-cluster sum of squared distances.
    """
    X = _validate_input(X)
    labels = _validate_labels(labels, X.shape[1])

    centroids = np.asarray(centroids, dtype=float)
    unique_labels = np.arange(centroids.shape[1])
    distortion = 0.0

    for centroid_index, label in enumerate(unique_labels):
        cluster_points = X[:, labels == label]
        if cluster_points.shape[1] == 0:
            continue

        squared_distances = np.sum(
            (cluster_points - centroids[:, centroid_index][:, np.newaxis]) ** 2,
            axis=0
        )
        distortion += np.sum(squared_distances)

    return float(distortion)


def silhouette_score(X, labels):
    """Compute the mean silhouette score for a clustering.

    The silhouette score compares how close each observation is to points in
    its own cluster versus points in the nearest other cluster.

    Interpretation
    --------------
    - Values near 1 indicate well-separated, compact clusters.
    - Values near 0 indicate overlapping or weakly separated clusters.
    - Negative values suggest many points may be assigned to the wrong cluster.

    In general, higher silhouette scores are better.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    labels : array-like of shape (N,)
        Hard cluster assignments for each observation.

    Returns
    -------
    float
        Mean silhouette score over all observations.
    """
    X = _validate_input(X)
    labels = _validate_labels(labels, X.shape[1])

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("silhouette_score requires at least 2 clusters.")
    if len(unique_labels) == X.shape[1]:
        raise ValueError("silhouette_score is undefined when every point is its own cluster.")

    pairwise_distances = np.sqrt(
        np.sum((X[:, :, np.newaxis] - X[:, np.newaxis, :]) ** 2, axis=0)
    )

    silhouette_values = np.zeros(X.shape[1])

    for observation_index in range(X.shape[1]):
        label = labels[observation_index]
        same_cluster_mask = labels == label
        same_cluster_mask[observation_index] = False

        if np.any(same_cluster_mask):
            a_i = np.mean(pairwise_distances[observation_index, same_cluster_mask])
        else:
            a_i = 0.0

        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue

            other_cluster_mask = labels == other_label
            mean_distance = np.mean(pairwise_distances[observation_index, other_cluster_mask])
            b_i = min(b_i, mean_distance)

        silhouette_values[observation_index] = (b_i - a_i) / max(a_i, b_i)

    return float(np.mean(silhouette_values))


################################################################################
# Finding the optimal k
################################################################################

def cluster_sizes(labels):
    """Count how many observations belong to each cluster.

    Parameters
    ----------
    labels : array-like of shape (N,)
        Hard cluster assignments for each observation.

    Returns
    -------
    dict
        Dictionary mapping each cluster label to the number of observations
        assigned to it.
    """
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    return {int(label): int(count) for label, count in zip(unique_labels, counts)}


def evaluate_k_grid(X, k_values, **kmeans_kwargs):
    """Fit k-means over several candidate values of `k`.

    This is a simple helper for model selection experiments. For each candidate
    number of clusters it fits a `KMeans` model, then records useful summary
    information such as distortion and silhouette score.

    Interpretation
    --------------
    - Lower distortion is better, but it will usually keep decreasing as `k`
      increases.
    - Higher silhouette scores are generally better for cluster separation.
    - The best `k` is usually chosen by comparing these values together, not by
      relying on only one metric.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    k_values : iterable of int
        Candidate numbers of clusters to evaluate.
    **kmeans_kwargs
        Additional keyword arguments passed directly to `KMeans`.

    Returns
    -------
    list of dict
        One result dictionary per candidate `k`. Each dictionary contains the
        fitted model, labels, centroids, distortion, silhouette score, number
        of iterations, and convergence flag.
    """
    X = _validate_input(X)
    k_values = list(k_values)

    results = []

    for k in k_values:
        model = KMeans(n_clusters=k, **kmeans_kwargs)
        labels = model.fit_predict(X)

        result = {
            "k": k,
            "distortion": model.distortion,
            "silhouette_score": silhouette_score(X, labels),
            "n_iter": model.n_iter,
            "converged": model.converged,
            "labels": labels,
            "centroids": model.centroids.copy(),
            "model": model,
        }
        results.append(result)

    return results


def best_k_by_silhouette(results):
    """Select the candidate `k` with the highest silhouette score.

    Parameters
    ----------
    results : list of dict
        Output from `evaluate_k_grid`.

    Returns
    -------
    dict
        The result dictionary with the largest silhouette score.
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one evaluated k.")

    return max(results, key=lambda result: result["silhouette_score"])


################################################################################
# Plotting and Visualization
################################################################################

def plot_clusters_2d(X, labels, centroids=None, feature_indices=(0, 1), 
                     ax=None, title="K-Means Clusters"):
    """Plot clustered observations and optional centroids in two dimensions.

    This visualization is useful when you want to inspect how k-means separated
    the data in a chosen pair of features. Different clusters are shown in
    different colors, and the centroids can be overlaid as large `X` markers.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    labels : array-like of shape (N,)
        Hard cluster assignments for each observation.
    centroids : array-like of shape (d, k), optional
        Cluster centroids to overlay on the same feature axes.
    feature_indices : tuple of int, default=(0, 1)
        Pair of feature indices used for the x- and y-axes.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="K-Means Clusters"
        Title for the plot.

    Returns
    -------
    matplotlib axes
        The axes containing the cluster plot.
    """
    X = _validate_input(X)
    labels = _validate_labels(labels, X.shape[1])

    if len(feature_indices) != 2:
        raise ValueError("feature_indices must contain exactly 2 feature indices.")

    feature_x, feature_y = feature_indices
    if feature_x < 0 or feature_x >= X.shape[0] or feature_y < 0 or feature_y >= X.shape[0]:
        raise ValueError("feature_indices must be valid row indices into X.")

    if ax is None:
        _, ax = plt.subplots()

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for color_index, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X[feature_x, mask],
            X[feature_y, mask],
            color=cmap(color_index % 10),
            s=30,
            alpha=0.7,
            label=f"Cluster {label}",
        )

    if centroids is not None:
        centroids = np.asarray(centroids, dtype=float)
        ax.scatter(
            centroids[feature_x, :],
            centroids[feature_y, :],
            color="black",
            marker="X",
            s=180,
            linewidths=1.0,
            edgecolors="white",
            label="Centroids",
        )

    ax.set_xlabel(f"Feature {feature_x}")
    ax.set_ylabel(f"Feature {feature_y}")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_elbow_curve(results, ax=None, title="Elbow Curve"):
    """Plot distortion against the number of clusters.

    The elbow method looks for a bend in the distortion curve. Lower
    distortions are better, but they almost always decrease as `k` increases,
    so the useful signal is usually where the rate of improvement starts to
    flatten out.

    Parameters
    ----------
    results : list of dict
        Output from `evaluate_k_grid`.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="Elbow Curve"
        Title for the plot.

    Returns
    -------
    matplotlib axes
        The axes containing the elbow plot.
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one evaluated k.")

    if ax is None:
        _, ax = plt.subplots()

    k_values = [result["k"] for result in results]
    distortions = [result["distortion"] for result in results]

    ax.plot(k_values, distortions, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Distortion")
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)

    return ax


def plot_silhouette_curve(results, ax=None, title="Silhouette Score vs k"):
    """Plot silhouette score against the number of clusters.

    Higher silhouette scores are generally better. Peaks in this curve often
    indicate values of `k` that produce compact and well-separated clusters.

    Parameters
    ----------
    results : list of dict
        Output from `evaluate_k_grid`.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="Silhouette Score vs k"
        Title for the plot.

    Returns
    -------
    matplotlib axes
        The axes containing the silhouette-vs-k plot.
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one evaluated k.")

    if ax is None:
        _, ax = plt.subplots()

    k_values = [result["k"] for result in results]
    silhouette_scores = [result["silhouette_score"] for result in results]

    ax.plot(k_values, silhouette_scores, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)

    return ax
