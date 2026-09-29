"""Classification metrics written from their definitions (no sklearn.metrics)."""
import numpy as np


def roc_curve(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    idx = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (idx + 1) - tps
    return np.r_[0, fps / max(fps[-1], 1)], np.r_[0, tps / max(tps[-1], 1)]


def pr_curve(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    idx = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (idx + 1) - tps
    return tps / (tps + fps), tps / max(tps[-1], 1)


def auc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def roc_points(y, s):
    fpr, tpr = roc_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}


def pr_points(y, s):
    p, r = pr_curve(y, s)
    return {"precision": p, "recall": r}


def calibration_bins(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    chunks = np.array_split(np.argsort(p), n_bins)
    return {"mean_predicted": [float(p[c].mean()) for c in chunks], "observed": [float(y[c].mean()) for c in chunks]}


def classification_summary(y, score, prob=None, threshold=0.0):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
           "specificity": tn / (tn + fp) if tn + fp else 0.0,
           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "auc": roc_points(y, score)["auc"], "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    if prob is not None:
        p = np.clip(np.asarray(prob, dtype=float), 1e-12, 1 - 1e-12)
        out["log_loss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out["brier"] = float(np.mean((p - y) ** 2))
    return out
