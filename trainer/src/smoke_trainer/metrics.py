from __future__ import annotations

import numpy as np


def _binary_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("labels and scores must be equal-length vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("metrics require finite scores and binary labels")
    return y, s


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y, s = _binary_inputs(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    distinct = np.r_[np.flatnonzero(np.diff(s[order])), len(y) - 1]
    true_positive = np.cumsum(sorted_y, dtype=np.int64)[distinct]
    predicted_positive = distinct + 1
    precision = true_positive / predicted_positive
    recall = true_positive / positives
    recall_change = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_change * precision))


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, s = _binary_inputs(labels, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return np.asarray([]), np.asarray([])
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    sorted_s = s[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_s)), len(sorted_s) - 1]
    true_positive = np.cumsum(sorted_y, dtype=np.int64)[distinct]
    false_positive = (distinct + 1) - true_positive
    tpr = np.r_[0.0, true_positive / positives]
    fpr = np.r_[0.0, false_positive / negatives]
    return fpr, tpr


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    fpr, tpr = roc_curve(labels, scores)
    if not len(fpr):
        return None
    widths = np.diff(fpr)
    heights = (tpr[:-1] + tpr[1:]) * 0.5
    return float(np.sum(widths * heights))


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _binary_inputs(labels, probabilities)
    if not int(y.sum()):
        raise ValueError("cannot choose a threshold without positive labels")
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    sorted_p = p[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_p)), len(sorted_p) - 1]
    true_positive = np.cumsum(sorted_y, dtype=np.int64)[distinct]
    predicted_positive = distinct + 1
    false_positive = predicted_positive - true_positive
    false_negative = int(y.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.zeros_like(denominator, dtype=np.float64)
    np.divide(2 * true_positive, denominator, out=f1, where=denominator != 0)
    return float(sorted_p[distinct[int(np.argmax(f1))]])


def metric_summary(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float
) -> dict[str, float | int | None]:
    y, p = _binary_inputs(labels, probabilities)
    predicted = p >= threshold
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "points": len(y),
        "positives": int(positive.sum()),
        "prevalence": float(positive.mean()) if len(y) else None,
        "average_precision": average_precision(y, p),
        "roc_auc": roc_auc(y, p),
        "brier_score": float(np.mean(np.square(p - y))) if len(y) else None,
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }
