import itertools
import numpy as np


################################################################################
# Input validation
################################################################################

def _validate_label_vectors(y_true, y_pred):
    """Validate two label vectors for clustering evaluation.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class or cluster labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    tuple of ndarray
        The validated `(y_true, y_pred)` vectors.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D with shape (N,), got {y_true.ndim}D.")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1D with shape (N,), got {y_pred.ndim}D.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of observations: "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    return y_true, y_pred


def _comb2(n):
    """Compute n choose 2 for a scalar or array."""
    n = np.asarray(n)
    return n * (n - 1) / 2.0


################################################################################
# Contingency matrix and label alignment
################################################################################

def contingency_matrix(y_true, y_pred):
    """Compute the contingency matrix between true and predicted labels.

    Rows correspond to the unique labels in `y_true`, and columns correspond to
    the unique labels in `y_pred`. Entry `(i, j)` counts how many observations
    belong to true class `i` and predicted cluster `j`.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class or cluster labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    matrix : ndarray of shape (n_true_classes, n_pred_clusters)
        Contingency matrix of joint label counts.
    true_labels : ndarray
        Sorted unique labels from `y_true`.
    pred_labels : ndarray
        Sorted unique labels from `y_pred`.
    """
    y_true, y_pred = _validate_label_vectors(y_true, y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)

    for true_index, true_label in enumerate(true_labels):
        for pred_index, pred_label in enumerate(pred_labels):
            matrix[true_index, pred_index] = np.sum(
                (y_true == true_label) & (y_pred == pred_label)
            )

    return matrix, true_labels, pred_labels


def align_cluster_labels(y_true, y_pred):
    """Align predicted cluster labels to true labels by maximizing agreement.

    This is useful when a clustering algorithm recovers the correct grouping but
    uses arbitrary label names. The alignment is done as a post-processing step
    and does not affect the fitted unsupervised model itself.

    This implementation uses a simple brute-force search over permutations,
    which is suitable for small numbers of clusters and keeps the logic easy to
    understand.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    y_aligned : ndarray of shape (N,)
        Predicted labels remapped to best match `y_true`.
    mapping : dict
        Dictionary mapping original predicted labels to aligned labels.
    best_accuracy : float
        Fraction of labels matching after alignment.
    """
    y_true, y_pred = _validate_label_vectors(y_true, y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    if len(pred_labels) > len(true_labels):
        raise ValueError(
            "align_cluster_labels requires the number of predicted clusters to "
            "be less than or equal to the number of true labels."
        )

    best_mapping = None
    best_accuracy = -np.inf
    best_aligned = None

    for true_perm in itertools.permutations(true_labels, len(pred_labels)):
        mapping = {
            pred_label: aligned_label
            for pred_label, aligned_label in zip(pred_labels, true_perm)
        }
        y_aligned = np.array([mapping[label] for label in y_pred])
        accuracy = np.mean(y_aligned == y_true)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
            best_aligned = y_aligned

    return best_aligned, best_mapping, float(best_accuracy)


def clustering_accuracy(y_true, y_pred, align=True):
    """Compute clustering accuracy with optional label alignment.

    If `align=True`, predicted cluster labels are first permuted to best match
    the true labels. This makes accuracy meaningful for clustering outputs,
    whose label identities are otherwise arbitrary.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.
    align : bool, default=True
        Whether to align predicted labels before computing accuracy.

    Returns
    -------
    float
        Fraction of correctly labelled observations.
    """
    y_true, y_pred = _validate_label_vectors(y_true, y_pred)

    if align:
        y_pred, _, _ = align_cluster_labels(y_true, y_pred)

    return float(np.mean(y_true == y_pred))


################################################################################
# Pair-counting metrics
################################################################################

def rand_index(y_true, y_pred):
    """Compute the Rand Index between two clusterings.

    The Rand Index measures agreement between two partitions by considering all
    pairs of observations. It counts how often the two labelings agree on
    whether a pair belongs to the same cluster or to different clusters.

    Interpretation
    --------------
    - 1.0 indicates perfect agreement.
    - Values closer to 0 indicate weaker agreement.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class or cluster labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    float
        Rand Index score.
    """
    matrix, _, _ = contingency_matrix(y_true, y_pred)
    n_samples = np.sum(matrix)
    total_pairs = _comb2(n_samples)

    same_same = np.sum(_comb2(matrix))
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    same_diff = np.sum(_comb2(row_sums)) - same_same
    diff_same = np.sum(_comb2(col_sums)) - same_same
    diff_diff = total_pairs - same_same - same_diff - diff_same

    return float((same_same + diff_diff) / total_pairs)


def adjusted_rand_index(y_true, y_pred):
    """Compute the Adjusted Rand Index (ARI) between two clusterings.

    ARI corrects the Rand Index for chance agreement. Unlike plain accuracy, it
    does not require any label alignment, because it depends only on the
    grouping structure.

    Interpretation
    --------------
    - 1.0 indicates perfect agreement.
    - 0.0 indicates agreement no better than random chance.
    - Negative values indicate worse-than-chance agreement.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth class or cluster labels.
    y_pred : array-like of shape (N,)
        Predicted cluster labels.

    Returns
    -------
    float
        Adjusted Rand Index.
    """
    matrix, _, _ = contingency_matrix(y_true, y_pred)

    sum_comb_cells = np.sum(_comb2(matrix))
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    sum_comb_rows = np.sum(_comb2(row_sums))
    sum_comb_cols = np.sum(_comb2(col_sums))
    total_pairs = _comb2(np.sum(matrix))

    expected_index = (sum_comb_rows * sum_comb_cols) / total_pairs
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)

    if max_index == expected_index:
        return 0.0

    ari = (sum_comb_cells - expected_index) / (max_index - expected_index)
    return float(ari)
