import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

try:
    from .gmm import GMM
except ImportError:
    from gmm import GMM

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


def _validate_responsibilities(responsibilities):
    """Validate a responsibility matrix for a GMM.

    Parameters
    ----------
    responsibilities : array-like of shape (k, N)
        Soft assignment probabilities for each observation.

    Returns
    -------
    ndarray of shape (k, N)
        The validated responsibility matrix.
    """
    responsibilities = np.asarray(responsibilities, dtype=float)

    if responsibilities.ndim != 2:
        raise ValueError(
            "responsibilities must be 2D with shape (k, N), "
            f"got {responsibilities.ndim}D."
        )
    if responsibilities.shape[0] == 0 or responsibilities.shape[1] == 0:
        raise ValueError(
            "responsibilities must be non-empty, "
            f"got shape {responsibilities.shape}."
        )
    if not np.all(np.isfinite(responsibilities)):
        raise ValueError("responsibilities contains non-finite values (NaN or inf).")

    column_sums = np.sum(responsibilities, axis=0)
    if not np.allclose(column_sums, 1.0):
        raise ValueError("Each observation's responsibilities must sum to 1.")

    return responsibilities


################################################################################
# Component summaries
################################################################################

def component_sizes(responsibilities):
    """Compute the effective size of each GMM component.

    For soft clustering, component size is the sum of responsibilities assigned
    to that component. These values are often denoted by `N_k` in EM
    derivations.

    Parameters
    ----------
    responsibilities : array-like of shape (k, N)
        Soft assignment probabilities for each observation.

    Returns
    -------
    ndarray of shape (k,)
        Effective number of observations assigned to each component.
    """
    responsibilities = _validate_responsibilities(responsibilities)
    return np.sum(responsibilities, axis=1)


def component_proportions(responsibilities):
    """Compute the effective proportion of observations in each component.

    These are the normalized component sizes. Larger values indicate components
    that explain a larger fraction of the dataset.

    Parameters
    ----------
    responsibilities : array-like of shape (k, N)
        Soft assignment probabilities for each observation.

    Returns
    -------
    ndarray of shape (k,)
        Effective component proportions, which sum to 1.
    """
    responsibilities = _validate_responsibilities(responsibilities)
    return component_sizes(responsibilities) / responsibilities.shape[1]


def hard_assignments(responsibilities):
    """Convert soft responsibilities into hard component labels.

    Each observation is assigned to the component with the largest
    responsibility.

    Parameters
    ----------
    responsibilities : array-like of shape (k, N)
        Soft assignment probabilities for each observation.

    Returns
    -------
    ndarray of shape (N,)
        Hard component labels obtained by taking the argmax over components.
    """
    responsibilities = _validate_responsibilities(responsibilities)
    return np.argmax(responsibilities, axis=0)


################################################################################
# Model selection metrics
################################################################################

def count_parameters(model, X):
    """Count the number of free parameters in a full-covariance GMM.

    For a model with `k` components and `d` features, the parameter count is:
    - means: `k * d`
    - covariances: `k * d * (d + 1) / 2`
    - mixture weights: `k - 1`

    Parameters
    ----------
    model : GMM
        Fitted Gaussian mixture model.
    X : array-like of shape (d, N)
        Data matrix used to infer the number of features.

    Returns
    -------
    int
        Number of free model parameters.
    """
    X = _validate_input(X)
    d = X.shape[0]
    k = model.n_components

    n_mean_params = k * d
    n_covariance_params = k * d * (d + 1) // 2
    n_weight_params = k - 1

    return int(n_mean_params + n_covariance_params + n_weight_params)


def compute_aic(model, X):
    """Compute the Akaike Information Criterion for a fitted GMM.

    Lower AIC values are better. AIC rewards better fit through higher
    log-likelihood, while penalizing models with more parameters.

    Parameters
    ----------
    model : GMM
        Fitted Gaussian mixture model.
    X : array-like of shape (d, N)
        Data matrix used to validate the sample size and feature count.

    Returns
    -------
    float
        AIC score for the fitted model.
    """
    X = _validate_input(X)
    n_parameters = count_parameters(model, X)
    return float(2 * n_parameters - 2 * model.log_likelihood_)


def compute_bic(model, X):
    """Compute the Bayesian Information Criterion for a fitted GMM.

    Lower BIC values are better. BIC applies a stronger complexity penalty than
    AIC and is commonly used to choose the number of mixture components.

    Parameters
    ----------
    model : GMM
        Fitted Gaussian mixture model.
    X : array-like of shape (d, N)
        Data matrix used to validate the sample size and feature count.

    Returns
    -------
    float
        BIC score for the fitted model.
    """
    X = _validate_input(X)
    n_parameters = count_parameters(model, X)
    n_samples = X.shape[1]
    return float(np.log(n_samples) * n_parameters - 2 * model.log_likelihood_)


################################################################################
# Finding the optimal number of components
################################################################################

def evaluate_component_grid(X, component_values, **gmm_kwargs):
    """Fit GMMs over several candidate values of `n_components`.

    This helper is the GMM analogue of a `k`-grid search for k-means. For each
    candidate number of mixture components it fits a GMM and records useful
    summary statistics such as log-likelihood, AIC, and BIC.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    component_values : iterable of int
        Candidate numbers of mixture components to evaluate.
    **gmm_kwargs
        Additional keyword arguments passed directly to `GMM`.

    Returns
    -------
    list of dict
        One result dictionary per candidate number of components.
    """
    X = _validate_input(X)
    component_values = list(component_values)

    results = []

    for n_components in component_values:
        model = GMM(n_components=n_components, **gmm_kwargs)
        model.fit(X)

        result = {
            "n_components": n_components,
            "log_likelihood": model.log_likelihood_,
            "aic": compute_aic(model, X),
            "bic": compute_bic(model, X),
            "n_iter": model.n_iter_,
            "converged": model.converged_,
            "weights": model.weights_.copy(),
            "means": model.means_.copy(),
            "covariances": model.covariances_.copy(),
            "responsibilities": model.responsibilities_.copy(),
            "labels": model.labels_.copy(),
            "model": model,
        }
        results.append(result)

    return results


def best_n_components_by_bic(results):
    """Select the candidate model with the lowest BIC.

    Parameters
    ----------
    results : list of dict
        Output from `evaluate_component_grid`.

    Returns
    -------
    dict
        The result dictionary with the smallest BIC score.
    """
    if len(results) == 0:
        raise ValueError("results must contain at least one evaluated model.")

    return min(results, key=lambda result: result["bic"])


################################################################################
# Plotting and visualization
################################################################################

def _validate_feature_indices(X, feature_indices):
    """Validate a pair of feature indices for 2D plotting."""
    if len(feature_indices) != 2:
        raise ValueError("feature_indices must contain exactly 2 feature indices.")

    feature_x, feature_y = feature_indices
    if feature_x < 0 or feature_x >= X.shape[0] or feature_y < 0 or feature_y >= X.shape[0]:
        raise ValueError("feature_indices must be valid row indices into X.")

    return feature_x, feature_y


def _plot_covariance_ellipse(
    ax,
    mean,
    covariance,
    feature_indices=(0, 1),
    n_std=2.0,
    edgecolor="black",
    linewidth=2.0,
    alpha=0.25,
):
    """Plot a covariance ellipse for a selected pair of features."""
    feature_x, feature_y = feature_indices
    mean_2d = mean[[feature_x, feature_y]]
    covariance_2d = covariance[np.ix_([feature_x, feature_y], [feature_x, feature_y])]

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_2d)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2.0 * n_std * np.sqrt(max(eigenvalues[0], 0.0))
    height = 2.0 * n_std * np.sqrt(max(eigenvalues[1], 0.0))

    ellipse = Ellipse(
        xy=mean_2d,
        width=width,
        height=height,
        angle=angle,
        facecolor=edgecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(ellipse)


def plot_gmm_clusters_2d(
    X,
    responsibilities=None,
    labels=None,
    means=None,
    covariances=None,
    feature_indices=(0, 1),
    ax=None,
    title="GMM Hard Assignments",
):
    """Plot GMM hard cluster assignments in two selected feature dimensions.

    This is the GMM analogue of a k-means cluster plot. Observations are colored
    by hard component label, while component means and covariance ellipses are
    overlaid to show the Gaussian geometry.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    responsibilities : array-like of shape (k, N), optional
        Soft assignment probabilities. If `labels` is not provided, hard labels
        are obtained by taking the argmax over responsibilities.
    labels : array-like of shape (N,), optional
        Hard component labels. Provide either `labels` or `responsibilities`.
    means : array-like of shape (d, k), optional
        Component means to overlay.
    covariances : array-like of shape (k, d, d), optional
        Component covariance matrices used to draw ellipses.
    feature_indices : tuple of int, default=(0, 1)
        Pair of feature indices used for the x- and y-axes.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="GMM Hard Assignments"
        Title for the plot.

    Returns
    -------
    matplotlib axes
        The axes containing the plot.
    """
    X = _validate_input(X)
    feature_x, feature_y = _validate_feature_indices(X, feature_indices)

    if labels is None:
        if responsibilities is None:
            raise ValueError("Provide either labels or responsibilities.")
        responsibilities = _validate_responsibilities(responsibilities)
        labels = hard_assignments(responsibilities)
    else:
        labels = np.asarray(labels)

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
            label=f"Component {label}",
        )

    if means is not None:
        means = np.asarray(means, dtype=float)
        ax.scatter(
            means[feature_x, :],
            means[feature_y, :],
            color="black",
            marker="X",
            s=180,
            linewidths=1.0,
            edgecolors="white",
            label="Means",
        )

    if means is not None and covariances is not None:
        covariances = np.asarray(covariances, dtype=float)
        for component_index in range(means.shape[1]):
            color = cmap(component_index % 10)
            _plot_covariance_ellipse(
                ax,
                means[:, component_index],
                covariances[component_index],
                feature_indices=feature_indices,
                edgecolor=color,
            )

    ax.set_xlabel(f"Feature {feature_x}")
    ax.set_ylabel(f"Feature {feature_y}")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_gmm_responsibilities_2d(
    X,
    responsibilities,
    feature_indices=(0, 1),
    axes=None,
    title_prefix="Responsibilities",
):
    """Plot one responsibility-colored scatter plot per GMM component.

    Each subplot shows the same observations in a 2D feature plane, but colors
    them according to how strongly one particular component explains each point.
    This gives a direct visualisation of the soft assignments in a GMM.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    responsibilities : array-like of shape (k, N)
        Soft assignment probabilities for each observation.
    feature_indices : tuple of int, default=(0, 1)
        Pair of feature indices used for the x- and y-axes.
    axes : array-like of matplotlib axes, optional
        Existing axes to draw on. If omitted, a row of subplots is created.
    title_prefix : str, default="Responsibilities"
        Prefix used in subplot titles.

    Returns
    -------
    ndarray of matplotlib axes
        The axes containing the responsibility plots.
    """
    X = _validate_input(X)
    responsibilities = _validate_responsibilities(responsibilities)
    feature_x, feature_y = _validate_feature_indices(X, feature_indices)

    n_components = responsibilities.shape[0]

    if axes is None:
        _, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4))

    axes = np.atleast_1d(axes)
    if len(axes) != n_components:
        raise ValueError(f"axes must contain exactly {n_components} subplots.")

    for component_index, ax in enumerate(axes):
        scatter = ax.scatter(
            X[feature_x, :],
            X[feature_y, :],
            c=responsibilities[component_index, :],
            cmap="viridis",
            s=30,
            alpha=0.85,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel(f"Feature {feature_x}")
        ax.set_ylabel(f"Feature {feature_y}")
        ax.set_title(f"{title_prefix}: component {component_index}")
        plt.colorbar(scatter, ax=ax)

    return axes


def plot_gmm_density_2d(
    X,
    means,
    covariances,
    weights=None,
    feature_indices=(0, 1),
    ax=None,
    title="GMM Density Contours",
    grid_size=200,
):
    """Plot 2D contour lines for the fitted component and mixture densities.

    This view emphasizes the Gaussian distributions themselves rather than just
    the hard cluster assignments. It is useful for seeing how wide, elongated,
    or overlapping the learned components are.

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix used to define a sensible plotting region.
    means : array-like of shape (d, k)
        Component means.
    covariances : array-like of shape (k, d, d)
        Component covariance matrices.
    weights : array-like of shape (k,), optional
        Mixture weights. If omitted, all components are weighted equally.
    feature_indices : tuple of int, default=(0, 1)
        Pair of feature indices used for the x- and y-axes.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="GMM Density Contours"
        Title for the plot.
    grid_size : int, default=200
        Number of grid points per axis for contour evaluation.

    Returns
    -------
    matplotlib axes
        The axes containing the density contour plot.
    """
    X = _validate_input(X)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    feature_x, feature_y = _validate_feature_indices(X, feature_indices)

    if weights is None:
        weights = np.full(means.shape[1], 1.0 / means.shape[1])
    else:
        weights = np.asarray(weights, dtype=float)

    if ax is None:
        _, ax = plt.subplots()

    x_min, x_max = X[feature_x, :].min(), X[feature_x, :].max()
    y_min, y_max = X[feature_y, :].min(), X[feature_y, :].max()

    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)

    xs = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
    ys = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    mixture_density = np.zeros(grid.shape[1])
    cmap = plt.get_cmap("tab10")

    for component_index in range(means.shape[1]):
        mean = means[[feature_x, feature_y], component_index]
        covariance = covariances[component_index][np.ix_([feature_x, feature_y], [feature_x, feature_y])]

        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")

        diff = grid - mean[:, np.newaxis]
        solved = np.linalg.solve(covariance, diff)
        mahalanobis = np.sum(diff * solved, axis=0)

        component_density = np.exp(
            -0.5 * (2 * np.log(2.0 * np.pi) + log_det + mahalanobis)
        )
        mixture_density += weights[component_index] * component_density

        ax.contour(
            xx,
            yy,
            component_density.reshape(xx.shape),
            levels=4,
            colors=[cmap(component_index % 10)],
            alpha=0.7,
        )

    ax.contour(
        xx,
        yy,
        mixture_density.reshape(xx.shape),
        levels=6,
        colors="black",
        linewidths=1.2,
    )
    ax.scatter(X[feature_x, :], X[feature_y, :], color="gray", s=10, alpha=0.35)
    ax.set_xlabel(f"Feature {feature_x}")
    ax.set_ylabel(f"Feature {feature_y}")
    ax.set_title(title)

    return ax


def plot_gmm_feature_density(
    X,
    means,
    covariances,
    weights,
    feature_index=0,
    ax=None,
    title="GMM Feature Density",
    bins=30,
    grid_size=500,
):
    """Plot a 1D feature histogram with GMM component and mixture densities.

    This is the standard "distribution plot" view for a GMM on a single
    feature. It overlays:
    - the empirical data histogram
    - each weighted Gaussian component density
    - the total mixture density

    Parameters
    ----------
    X : array-like of shape (d, N)
        Data matrix with features on rows and observations on columns.
    means : array-like of shape (d, k)
        Component means.
    covariances : array-like of shape (k, d, d)
        Component covariance matrices.
    weights : array-like of shape (k,)
        Mixture weights.
    feature_index : int, default=0
        Feature index whose 1D marginal density is plotted.
    ax : matplotlib axes, optional
        Existing axes object to draw on. If omitted, a new one is created.
    title : str, default="GMM Feature Density"
        Title for the plot.
    bins : int, default=30
        Number of histogram bins.
    grid_size : int, default=500
        Number of x-grid points used for the density curves.

    Returns
    -------
    matplotlib axes
        The axes containing the density plot.
    """
    X = _validate_input(X)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if feature_index < 0 or feature_index >= X.shape[0]:
        raise ValueError("feature_index must be a valid row index into X.")

    if ax is None:
        _, ax = plt.subplots()

    feature_values = X[feature_index, :]
    x_min, x_max = feature_values.min(), feature_values.max()
    x_margin = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)

    ax.hist(
        feature_values,
        bins=bins,
        density=True,
        alpha=0.35,
        color="gray",
        edgecolor="white",
        label="Observed data",
    )

    mixture_density = np.zeros_like(x_grid)
    cmap = plt.get_cmap("tab10")

    for component_index in range(means.shape[1]):
        mu = means[feature_index, component_index]
        variance = covariances[component_index, feature_index, feature_index]
        sigma = np.sqrt(max(variance, 0.0))

        if sigma == 0.0:
            raise ValueError("Feature variance must be positive for density plotting.")

        component_density = (
            1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        ) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)

        weighted_density = weights[component_index] * component_density
        mixture_density += weighted_density

        ax.plot(
            x_grid,
            weighted_density,
            color=cmap(component_index % 10),
            linewidth=1.8,
            label=(
                f"Component {component_index}: "
                f"mu={mu:.2f}, sigma={sigma:.2f}, w={weights[component_index]:.2f}"
            ),
        )

    ax.plot(
        x_grid,
        mixture_density,
        color="red",
        linewidth=2.2,
        label="Mixture density",
    )

    ax.set_xlabel(f"Feature {feature_index}")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return ax
