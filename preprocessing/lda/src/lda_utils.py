"""Utility functions commonly used with LDA in CS315.

Convention:
    features are rows and observations are columns.
"""

import numpy as np
import matplotlib.pyplot as plt


def _validate_xy(X, y):
    """Validate X/y shape alignment for CS315 convention.

    Parameters:
    -------------------
    X : np.ndarray
        Data matrix with shape ``(n_features, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.

    Returns:
    --------------
    tuple[np.ndarray, np.ndarray]
        Sanitized arrays ``(X, y)``.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got {X.ndim}D.")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D, got {y.ndim}D.")
    if X.shape[1] != y.shape[0]:
        raise ValueError(
            f"X/y mismatch: X has {X.shape[1]} samples, y has {y.shape[0]} labels."
        )
    return X, y


def compute_class_means(X, y):
    """Compute per-class mean vectors.

    Parameters:
    -------------------
    X : np.ndarray
        Data matrix with shape ``(n_features, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.

    Returns:
    --------------
    tuple[np.ndarray, np.ndarray]
        ``(class_labels, class_means)`` where:
        - ``class_labels`` has shape ``(n_classes,)``
        - ``class_means`` has shape ``(n_features, n_classes)``
    """
    X, y = _validate_xy(X, y)
    class_labels = np.unique(y)
    means = []
    for c in class_labels:
        means.append(np.mean(X[:, y == c], axis=1))
    return class_labels, np.column_stack(means)


def compute_scatter_matrices(X, y):
    """Compute within-class and between-class scatter matrices.

    Parameters:
    -------------------
    X : np.ndarray
        Data matrix with shape ``(n_features, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.

    Returns:
    --------------
    tuple[np.ndarray, np.ndarray]
        ``(S_w, S_b)`` each with shape ``(n_features, n_features)``.
    """
    X, y = _validate_xy(X, y)
    class_labels, class_means = compute_class_means(X, y)
    global_mean = np.mean(X, axis=1)

    d = X.shape[0]
    S_w = np.zeros((d, d))
    S_b = np.zeros((d, d))

    for i, c in enumerate(class_labels):
        X_c = X[:, y == c]
        m_c = class_means[:, i]
        D_c = X_c - m_c[:, None]
        S_w += D_c @ D_c.T
        dm = m_c - global_mean
        S_b += X_c.shape[1] * np.outer(dm, dm)

    return S_w, S_b


def compute_normalized_scatter_matrices(X, y, mode="pooled_unbiased"):
    """Compute normalized within/between-class scatter matrices.

    Different libraries use different normalization conventions.
    This helper makes those conventions explicit and easy to compare.

    Parameters:
    -------------------
    X : np.ndarray
        Data matrix with shape ``(n_features, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.
    mode : str
        Normalization strategy for ``S_w``:
        - ``"pooled_unbiased"``: ``S_w / (n_samples - n_classes)``
        - ``"pooled_mle"``: ``S_w / n_samples``
        - ``"weighted_class_cov"``: ``sum_k pi_k * cov_k_unbiased``
          where ``pi_k = n_k / n_samples``.

    Returns:
    --------------
    tuple[np.ndarray, np.ndarray]
        ``(S_w_norm, S_b_norm)`` each with shape ``(n_features, n_features)``.
        ``S_b_norm`` is always returned as ``S_b / n_samples``.
    """
    X, y = _validate_xy(X, y)
    S_w, S_b = compute_scatter_matrices(X, y)

    n_samples = X.shape[1]
    class_labels = np.unique(y)
    n_classes = class_labels.size

    if mode == "pooled_unbiased":
        denom = n_samples - n_classes
        if denom <= 0:
            raise ValueError("Need n_samples > n_classes for pooled_unbiased mode.")
        S_w_norm = S_w / denom
    elif mode == "pooled_mle":
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        S_w_norm = S_w / n_samples
    elif mode == "weighted_class_cov":
        d = X.shape[0]
        S_w_norm = np.zeros((d, d))
        for c in class_labels:
            X_c = X[:, y == c]
            n_k = X_c.shape[1]
            if n_k < 2:
                raise ValueError(
                    "Each class must have at least 2 samples in weighted_class_cov mode."
                )
            D_c = X_c - np.mean(X_c, axis=1, keepdims=True)
            cov_k_unbiased = (D_c @ D_c.T) / (n_k - 1)
            S_w_norm += (n_k / n_samples) * cov_k_unbiased
    else:
        raise ValueError(
            "Unknown mode. Use one of: "
            "'pooled_unbiased', 'pooled_mle', 'weighted_class_cov'."
        )

    S_b_norm = S_b / n_samples
    return S_w_norm, S_b_norm


def explained_variance_summary(explained_variance_ratio):
    """Create explained-variance and cumulative-variance summary.

    Parameters:
    -------------------
    explained_variance_ratio : array-like
        Per-component explained-variance ratios.

    Returns:
    --------------
    dict[str, np.ndarray]
        Dictionary with keys:
        - ``component`` (1-based component indices)
        - ``explained_variance_ratio``
        - ``cumulative_explained_variance``
    """
    evr = np.asarray(explained_variance_ratio, dtype=float)
    if evr.ndim != 1:
        raise ValueError("explained_variance_ratio must be a 1D array-like object.")

    return {
        "component": np.arange(1, evr.size + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": np.cumsum(evr),
    }


def plot_class_scatter_before_lda(
    X,
    y,
    feature_indices=(0, 1),
    feature_names=None,
    ax=None,
    title=None,
    alpha=0.8,
    marker_size=36,
):
    """Plot class-separated scatter before LDA using two original features.

    Parameters:
    -------------------
    X : np.ndarray
        Data matrix with shape ``(n_features, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.
    feature_indices : tuple[int, int]
        Two feature indices to visualize on x/y axes.
    feature_names : list[str] | tuple[str, ...] | None
        Optional feature names for axis labels.
    ax : matplotlib.axes.Axes | None
        Existing axis to draw on. If None, a new figure+axis is created.
    title : str | None
        Plot title. If None, a default title is used.
    alpha : float
        Marker transparency.
    marker_size : float
        Marker size for scatter points.

    Returns:
    --------------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis used for plotting.
    """
    X, y = _validate_xy(X, y)

    if len(feature_indices) != 2:
        raise ValueError("feature_indices must contain exactly 2 indices.")
    i, j = int(feature_indices[0]), int(feature_indices[1])
    if i < 0 or j < 0 or i >= X.shape[0] or j >= X.shape[0]:
        raise ValueError(
            f"feature_indices {feature_indices} out of bounds for {X.shape[0]} features."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    class_labels = np.unique(y)
    cmap = plt.get_cmap("tab10")

    for idx, c in enumerate(class_labels):
        mask = (y == c)
        ax.scatter(
            X[i, mask],
            X[j, mask],
            s=marker_size,
            alpha=alpha,
            color=cmap(idx % 10),
            label=str(c),
        )

    if feature_names is not None and len(feature_names) > max(i, j):
        xlabel = feature_names[i]
        ylabel = feature_names[j]
    else:
        xlabel = f"Feature {i}"
        ylabel = f"Feature {j}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else "Before LDA: Class Scatter")
    ax.legend(title="Class")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_lda_projection(
    Z,
    y,
    component_indices=(0, 1),
    ax=None,
    title=None,
    alpha=0.8,
    marker_size=36,
):
    """Plot class-separated scatter in the LDA projected space.

    If `Z` has only one component, the y-axis is set to zeros so samples
    can still be visualized in a 2D scatter.

    Parameters:
    -------------------
    Z : np.ndarray
        Projected matrix with shape ``(n_components, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.
    component_indices : tuple[int, int]
        Two component indices to visualize on x/y axes.
    ax : matplotlib.axes.Axes | None
        Existing axis to draw on. If None, a new figure+axis is created.
    title : str | None
        Plot title. If None, a default title is used.
    alpha : float
        Marker transparency.
    marker_size : float
        Marker size for scatter points.

    Returns:
    --------------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis used for plotting.
    """
    Z, y = _validate_xy(Z, y)

    if len(component_indices) != 2:
        raise ValueError("component_indices must contain exactly 2 indices.")
    i, j = int(component_indices[0]), int(component_indices[1])
    if i < 0 or i >= Z.shape[0]:
        raise ValueError(
            f"First component index {i} out of bounds for {Z.shape[0]} components."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    # Allow 1D projected data by plotting against zeros.
    use_dummy_y = (j >= Z.shape[0]) or (Z.shape[0] == 1)
    y_values = np.zeros(Z.shape[1]) if use_dummy_y else Z[j, :]

    class_labels = np.unique(y)
    cmap = plt.get_cmap("tab10")

    for idx, c in enumerate(class_labels):
        mask = (y == c)
        ax.scatter(
            Z[i, mask],
            y_values[mask],
            s=marker_size,
            alpha=alpha,
            color=cmap(idx % 10),
            label=str(c),
        )

    ax.set_xlabel(f"LD{i + 1}")
    ax.set_ylabel("0" if use_dummy_y else f"LD{j + 1}")
    ax.set_title(title if title is not None else "After LDA: Projected Scatter")
    ax.legend(title="Class")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_before_after_lda(
    X,
    Z,
    y,
    feature_indices=(0, 1),
    component_indices=(0, 1),
    feature_names=None,
    figsize=(13, 5),
):
    """Create side-by-side before/after LDA scatter plots.

    Parameters:
    -------------------
    X : np.ndarray
        Original matrix with shape ``(n_features, n_samples)``.
    Z : np.ndarray
        Projected matrix with shape ``(n_components, n_samples)``.
    y : np.ndarray
        Label vector with shape ``(n_samples,)``.
    feature_indices : tuple[int, int]
        Feature indices for the pre-LDA scatter plot.
    component_indices : tuple[int, int]
        Component indices for the post-LDA scatter plot.
    feature_names : list[str] | tuple[str, ...] | None
        Optional feature names for original-feature axis labels.
    figsize : tuple[float, float]
        Matplotlib figure size.

    Returns:
    --------------
    tuple[matplotlib.figure.Figure, np.ndarray]
        The figure and the two subplot axes.
    """
    X, y = _validate_xy(X, y)
    Z, _ = _validate_xy(Z, y)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_class_scatter_before_lda(
        X,
        y,
        feature_indices=feature_indices,
        feature_names=feature_names,
        ax=axes[0],
        title="Before LDA",
    )
    plot_lda_projection(
        Z,
        y,
        component_indices=component_indices,
        ax=axes[1],
        title="After LDA",
    )
    fig.tight_layout()
    return fig, axes
