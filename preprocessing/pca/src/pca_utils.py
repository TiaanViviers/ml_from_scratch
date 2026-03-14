import numpy as np
import matplotlib.pyplot as plt


def frobenius_norm(A, B):
    """
    Calculate Frobenius-norm reconstruction error between two matrices.
    
    This is the absolute error magnitude:
        ||A - B||_F
    A value of 0 means perfect reconstruction/no error.
    
    Parameters:
    -----------
    A : ndarray
        Approximation/predicted matrix.
    B : ndarray
        Reference/ground-truth matrix.

    Returns:
    --------
    err_fro : float
        Frobenius norm of the difference matrix.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A{A.shape} and B{B.shape} must match.")

    E = A - B
    err_fro = np.linalg.norm(E, ord="fro")
    return err_fro
        
        
def relative_frobenius_norm(A, B):
    """
    Calculate relative Frobenius-norm error between two matrices.
    
    Relative error is defined as:
        ||A - B||_F / ||B||_F
    where `B` is the reference matrix.

    Parameters:
    -----------
    A : ndarray
        Approximation/predicted matrix.
    B : ndarray
        Reference/ground-truth matrix.

    Returns:
    --------
    rel_err_fro : float
        Relative Frobenius error. Returns:
        - 0.0 if both A and B are zero matrices
        - inf if B is a zero matrix but A is not
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A{A.shape} and B{B.shape} must match.")

    diff = A - B
    norm_diff = np.linalg.norm(diff, ord="fro")
    norm_ref = np.linalg.norm(B, ord="fro")
    
    if norm_ref == 0:
        return float('inf') if norm_diff != 0 else 0.0
    
    rel_err_fro = norm_diff / norm_ref
    return rel_err_fro


def mse(A, B):
    """
    Calculate mean squared error (MSE) between two matrices.
    
    MSE is computed over all entries:
        mean((A - B)^2)
    
    Parameters:
    -----------
    A : ndarray
        Approximation/predicted matrix.
    B : ndarray
        Reference/ground-truth matrix.

    Returns:
    --------
    mse_val : float
        Mean squared error over all matrix entries.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A{A.shape} and B{B.shape} must match.")

    E = A - B
    E_squared = np.square(E)
    mse_val = np.mean(E_squared)
    return mse_val


def scree_plot(n_components, explained_variance_ratio, kind=None, threshold=None, color=None):
    """
    Plot explained-variance ratios of retained principal components.
    
    The scree plot can be rendered as bars or a line and optionally includes
    a horizontal threshold reference.
    
    Parameters:
    -----------
    n_components : int
        Number of components to display.
    explained_variance_ratio : ndarray of shape (k,)
        Explained variance ratio values for available components.
    kind : str or None, default=None
        Plot style. Supported values: "bar", "line". If None, uses "bar".
    threshold : float or None, default=None
        Optional horizontal reference value to draw on the plot.
    color : str or None, default=None
        Matplotlib color for the series. If None, defaults to "blue".
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Axes object containing the generated scree plot.
    """
    if explained_variance_ratio is None:
        raise ValueError("explained_variance_ratio cannot be None.")
    if not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be an integer.")
    if n_components <= 0:
        raise ValueError("n_components must be >= 1.")
    if n_components > len(explained_variance_ratio):
        raise ValueError(
            f"n_components={n_components} exceeds available explained variance "
            f"entries ({len(explained_variance_ratio)})."
        )
    
    ax = plt.subplot()
    color = "blue" if color is None else color
    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    
    if kind is None or kind.lower() == "bar":
        ax.bar(pc_names, explained_variance_ratio[:n_components], color=color)
    elif kind.lower() == "line":
        ax.plot(pc_names, explained_variance_ratio[:n_components], color=color)
    else:
        raise ValueError(f"kind={kind}, not supported, only bar|line")
    
    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Proportion of explained variance")
    ax.set_title("SCREE Plot")
    
    return ax
