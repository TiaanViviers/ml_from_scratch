"""Simple classification metrics utilities (NumPy only).

Conventions used here:
- confusion matrix rows = true labels
- confusion matrix cols = predicted labels
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def _as_1d(x, name: str):
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _safe_divide(num, den):
    return float(num / den) if den != 0 else 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix.

    Also called: contingency table, error matrix.
    Rows are true labels, columns are predicted labels.
    """
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    labels = np.asarray(labels)

    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(yt, yp):
        cm[index[t], index[p]] += 1
    return cm, labels


def format_confusion_matrix(cm, labels, true_axis_name="True", pred_axis_name="Predicted"):
    """Pretty string formatter for confusion matrix with axis names."""
    cm = np.asarray(cm)
    labels = [str(v) for v in labels]
    width = max(10, max(len(x) for x in labels) + 2)

    lines = [f"{pred_axis_name} ->"]
    lines.append(f"{true_axis_name:<10}" + "".join(f"{lab:>{width}}" for lab in labels))
    for i, lab in enumerate(labels):
        row = f"{lab:<10}" + "".join(f"{int(cm[i, j]):>{width}d}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def print_confusion_matrix(y_true, y_pred, labels=None, true_axis_name="True", pred_axis_name="Predicted"):
    """Print formatted confusion matrix and return (cm, labels)."""
    cm, used_labels = confusion_matrix(y_true, y_pred, labels=labels)
    print(format_confusion_matrix(cm, used_labels, true_axis_name=true_axis_name, pred_axis_name=pred_axis_name))
    return cm, used_labels


def _binary_counts(y_true, y_pred, pos_label=1):
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    pos = yt == pos_label
    neg = ~pos
    pred_pos = yp == pos_label
    pred_neg = ~pred_pos

    tp = int(np.sum(pos & pred_pos))
    tn = int(np.sum(neg & pred_neg))
    fp = int(np.sum(neg & pred_pos))
    fn = int(np.sum(pos & pred_neg))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    """Accuracy.

    Also called: ACC, hit rate.
    """
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(yt == yp))


def error_rate(y_true, y_pred):
    """Error rate.

    Also called: misclassification rate, classification error.
    """
    return 1.0 - accuracy(y_true, y_pred)


def precision(y_true, y_pred, pos_label=1):
    """Precision.

    Also called: PPV (positive predictive value).
    """
    tp, _, fp, _ = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(tp, tp + fp)


def recall(y_true, y_pred, pos_label=1):
    """Recall.

    Also called: sensitivity, TPR (true positive rate), hit rate.
    """
    tp, _, _, fn = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(tp, tp + fn)


def specificity(y_true, y_pred, pos_label=1):
    """Specificity.

    Also called: TNR (true negative rate).
    """
    _, tn, fp, _ = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(tn, tn + fp)


def false_positive_rate(y_true, y_pred, pos_label=1):
    """False positive rate.

    Also called: FPR, fall-out, Type I error rate (alpha).
    """
    _, tn, fp, _ = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(fp, fp + tn)


def false_negative_rate(y_true, y_pred, pos_label=1):
    """False negative rate.

    Also called: FNR, miss rate, Type II error rate (beta).
    """
    tp, _, _, fn = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(fn, fn + tp)


def negative_predictive_value(y_true, y_pred, pos_label=1):
    """Negative predictive value.

    Also called: NPV.
    """
    _, tn, _, fn = _binary_counts(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(tn, tn + fn)


def f1_score(y_true, y_pred, pos_label=1):
    """F1 score.

    Also called: F-measure (beta=1), Dice score (binary context).
    """
    p = precision(y_true, y_pred, pos_label=pos_label)
    r = recall(y_true, y_pred, pos_label=pos_label)
    return _safe_divide(2.0 * p * r, p + r)


def balanced_accuracy(y_true, y_pred, pos_label=1):
    """Balanced accuracy.

    Also called: average of sensitivity and specificity.
    """
    return 0.5 * (recall(y_true, y_pred, pos_label=pos_label) + specificity(y_true, y_pred, pos_label=pos_label))


def matthews_corrcoef(y_true, y_pred, pos_label=1):
    """Matthews correlation coefficient.

    Also called: MCC.
    """
    tp, tn, fp, fn = _binary_counts(y_true, y_pred, pos_label=pos_label)
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return _safe_divide(tp * tn - fp * fn, np.sqrt(den))


def log_loss(y_true, y_proba, labels=None, eps=EPS):
    """Multiclass log-loss.

    Also called: cross-entropy loss, negative log-likelihood (NLL).
    Expects y_proba shape (n_samples, n_classes).
    """
    yt = _as_1d(y_true, "y_true")
    p = np.asarray(y_proba, dtype=float)
    if p.ndim != 2:
        raise ValueError("y_proba must be 2D with shape (n_samples, n_classes).")
    if p.shape[0] != yt.shape[0]:
        raise ValueError("y_proba rows must match number of samples in y_true.")

    if labels is None:
        labels = np.unique(yt)
    labels = np.asarray(labels)
    if p.shape[1] != len(labels):
        raise ValueError("y_proba columns must match number of labels.")

    idx = {lab: i for i, lab in enumerate(labels)}
    p = np.clip(p, eps, 1.0 - eps)
    p = p / np.sum(p, axis=1, keepdims=True)

    losses = [-np.log(p[i, idx[yi]]) for i, yi in enumerate(yt)]
    return float(np.mean(losses))


def brier_score(y_true, y_prob_pos, pos_label=1):
    """Binary Brier score.

    Also called: Brier loss.
    """
    yt = _as_1d(y_true, "y_true")
    pp = np.asarray(y_prob_pos, dtype=float).reshape(-1)
    if yt.shape[0] != pp.shape[0]:
        raise ValueError("y_true and y_prob_pos must have the same length.")
    y01 = (yt == pos_label).astype(float)
    return float(np.mean((pp - y01) ** 2))


def roc_auc_binary(y_true, y_score, pos_label=1):
    """Binary ROC AUC (rank-based).

    Also called: AUC, AUROC.
    """
    yt = _as_1d(y_true, "y_true")
    ys = np.asarray(y_score, dtype=float).reshape(-1)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    y01 = (yt == pos_label).astype(int)
    n_pos = int(np.sum(y01 == 1))
    n_neg = int(np.sum(y01 == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC undefined when only one class is present.")

    order = np.argsort(ys)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ys) + 1, dtype=float)

    rank_sum_pos = np.sum(ranks[y01 == 1])
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

