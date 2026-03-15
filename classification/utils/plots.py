"""Common plotting utilities for classification tasks.

All functions are designed to be simple to call with NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from .metrics import confusion_matrix as _confusion_matrix
except ImportError:
    from metrics import confusion_matrix as _confusion_matrix


def _as_1d(x, name: str):
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _binary_targets(y_true, pos_label):
    yt = _as_1d(y_true, "y_true")
    y01 = (yt == pos_label).astype(int)
    n_pos = int(np.sum(y01 == 1))
    n_neg = int(np.sum(y01 == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative classes must be present.")
    return yt, y01, n_pos, n_neg


def _ensure_axis(ax=None, figsize=(6, 5)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize=None,
    cmap="Blues",
    ax=None,
    title="Confusion Matrix",
    annotate=True,
):
    """Plot confusion matrix heatmap.

    Args:
        normalize: None, "true", "pred", or "all".
    """
    cm, used_labels = _confusion_matrix(y_true, y_pred, labels=labels)
    mat = cm.astype(float)
    if normalize is not None:
        if normalize == "true":
            den = np.sum(mat, axis=1, keepdims=True)
            den[den == 0] = 1.0
            mat = mat / den
        elif normalize == "pred":
            den = np.sum(mat, axis=0, keepdims=True)
            den[den == 0] = 1.0
            mat = mat / den
        elif normalize == "all":
            den = np.sum(mat)
            den = den if den != 0 else 1.0
            mat = mat / den
        else:
            raise ValueError("normalize must be one of: None, 'true', 'pred', 'all'.")

    fig, ax = _ensure_axis(ax=ax)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(used_labels)))
    ax.set_yticks(np.arange(len(used_labels)))
    ax.set_xticklabels(used_labels)
    ax.set_yticklabels(used_labels)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                text = f"{val:.2f}" if normalize is not None else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center")

    return fig, ax


def plot_roc_curve_binary(
    y_true,
    y_score,
    pos_label=1,
    ax=None,
    title="ROC Curve",
):
    """Plot binary ROC curve from positive-class scores/probabilities."""
    _, y01, n_pos, n_neg = _binary_targets(y_true, pos_label=pos_label)
    ys = np.asarray(y_score, dtype=float).reshape(-1)
    if ys.shape[0] != y01.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    order = np.argsort(-ys)
    y_sorted = y01[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    auc = float(np.trapz(tpr, fpr))

    fig, ax = _ensure_axis(ax=ax)
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return fig, ax, auc


def plot_precision_recall_curve_binary(
    y_true,
    y_score,
    pos_label=1,
    ax=None,
    title="Precision-Recall Curve",
):
    """Plot binary precision-recall curve from positive-class scores/probabilities."""
    _, y01, n_pos, _ = _binary_targets(y_true, pos_label=pos_label)
    ys = np.asarray(y_score, dtype=float).reshape(-1)
    if ys.shape[0] != y01.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    order = np.argsort(-ys)
    y_sorted = y01[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    fig, ax = _ensure_axis(ax=ax)
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    return fig, ax


def plot_threshold_metrics_binary(
    y_true,
    y_score,
    pos_label=1,
    thresholds=None,
    ax=None,
    title="Metrics vs Threshold",
):
    """Plot precision/recall/F1/accuracy as threshold changes."""
    yt = _as_1d(y_true, "y_true")
    ys = np.asarray(y_score, dtype=float).reshape(-1)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    y01 = (yt == pos_label).astype(int)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    thresholds = np.asarray(thresholds, dtype=float)

    precisions = []
    recalls = []
    f1s = []
    accs = []

    for t in thresholds:
        pred = (ys >= t).astype(int)
        tp = np.sum((pred == 1) & (y01 == 1))
        tn = np.sum((pred == 0) & (y01 == 0))
        fp = np.sum((pred == 1) & (y01 == 0))
        fn = np.sum((pred == 0) & (y01 == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        acc = (tp + tn) / len(y01)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accs.append(acc)

    fig, ax = _ensure_axis(ax=ax, figsize=(7, 5))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, accs, label="Accuracy")
    ax.set_xlim(float(np.min(thresholds)), float(np.max(thresholds)))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return fig, ax


def _to_nx2(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D.")
    if X.shape[0] == 2:
        return X.T
    if X.shape[1] == 2:
        return X
    raise ValueError("X must have exactly 2 features (shape (2,N) or (N,2)).")


def plot_decision_regions_2d(
    X,
    y,
    predict_fn,
    grid_step=0.02,
    padding=0.5,
    ax=None,
    title="Decision Regions",
    xlabel="x1",
    ylabel="x2",
):
    """Plot 2D decision regions and points.

    Input contract:
    - X: shape (2, N) or (N, 2)
    - y: shape (N,)
    - predict_fn: callable that accepts points in shape (2, M) and returns labels (M,)
    """
    Xn2 = _to_nx2(X)
    y = _as_1d(y, "y")
    if Xn2.shape[0] != y.shape[0]:
        raise ValueError("X and y sample counts do not match.")

    x_min, x_max = Xn2[:, 0].min() - padding, Xn2[:, 0].max() + padding
    y_min, y_max = Xn2[:, 1].min() - padding, Xn2[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]  # (M,2)

    # predict_fn follows repo convention: features x observations => (2, M)
    pred = np.asarray(predict_fn(grid.T)).reshape(-1)
    if pred.shape[0] != grid.shape[0]:
        raise ValueError("predict_fn returned unexpected number of predictions.")

    classes = np.unique(np.concatenate([y, pred]))
    class_to_int = {c: i for i, c in enumerate(classes)}
    zz = np.array([class_to_int[v] for v in pred], dtype=int).reshape(xx.shape)

    fig, ax = _ensure_axis(ax=ax, figsize=(7, 5))
    ax.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(len(classes) + 1) - 0.5, cmap="tab10")

    for cls in classes:
        m = y == cls
        ax.scatter(Xn2[m, 0], Xn2[m, 1], s=20, label=str(cls))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Class")
    return fig, ax

