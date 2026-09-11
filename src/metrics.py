"""Threshold selection + F1/accuracy helpers for MIL evaluation.

Per-fold protocol (used by `run_mil_stenosis_stridor.py`):
  1. inner 80/20 split inside the training fold
  2. fit model on the 80% portion, score the 20% portion (inner val)
  3. pick the threshold that maximises F1 on inner val (no leakage to test)
  4. refit model on the full training fold, score the test fold
  5. apply the picked threshold to test scores -> F1, accuracy

This mirrors what the deep MIL pipeline already does for early stopping.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              cohen_kappa_score, confusion_matrix, f1_score,
                              roc_auc_score)


def pick_threshold_f1(y_true: np.ndarray, scores: np.ndarray,
                      n_candidates: int = 200) -> float:
    """Return the threshold (on `scores`) that maximises F1 on (`y_true`, `scores`).

    Uses up to `n_candidates` quantile-spaced candidate cuts; falls back to a
    sorted unique scan if there are fewer distinct values. Returns 0.5 (or the
    score median) if no positive threshold is found (degenerate case).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5

    uniq = np.unique(s)
    if len(uniq) <= n_candidates:
        candidates = uniq
    else:
        candidates = np.quantile(s, np.linspace(0.01, 0.99, n_candidates))

    best_thr, best_f1 = float(np.median(s)), -1.0
    for t in candidates:
        y_pred = (s >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        f = f1_score(y, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(t)
    return best_thr


def fold_f1_acc(y_test: np.ndarray, score_test: np.ndarray, thr: float
                ) -> tuple[float, float]:
    """F1 and accuracy on a test fold given a precomputed threshold."""
    y_pred = (np.asarray(score_test) >= thr).astype(int)
    return (float(f1_score(y_test, y_pred, zero_division=0)),
            float(accuracy_score(y_test, y_pred)))


def stats_dict(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()),   max=float(a.max()))


def pooled_f1_acc(y_list: list[np.ndarray], score_list: list[np.ndarray],
                  thr_list: list[float]) -> tuple[float, float]:
    """Concatenate per-fold (y, score, thr-applied) and compute pooled F1 + acc."""
    y_all  = np.concatenate(y_list)
    yp_all = np.concatenate([(np.asarray(s) >= t).astype(int)
                              for s, t in zip(score_list, thr_list)])
    return (float(f1_score(y_all, yp_all, zero_division=0)),
            float(accuracy_score(y_all, yp_all)))


# ---------------------------------------------------------------------------
# Multi-class metrics (sub-task figures : localization, severity, stridor)
# ---------------------------------------------------------------------------
def multiclass_metrics(y_true: np.ndarray, scores: np.ndarray,
                       n_classes: int, ordinal: bool = False,
                       offsets: np.ndarray | None = None) -> dict:
    """Macro-AUC (OvR), macro-F1, balanced accuracy ; quadratic kappa if ordinal.

    `scores`  : (N, n_classes), class probabilities or logits.
    `y_true`  : (N,), int class indices in [0, n_classes).
    `offsets` : optional (n_classes,), additive per-class offsets applied
                BEFORE argmax (does not affect AUC because it is ranking-based).

    Returns per-class AUCs + a confusion matrix.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    s_pred = s if offsets is None else s + np.asarray(offsets).reshape(1, -1)
    y_pred = s_pred.argmax(axis=1)
    # macro AUC uses the raw scores (offsets are class-wise additive)
    per_class_auc = {}
    aucs = []
    for k in range(n_classes):
        y_bin = (y == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auc[k] = float('nan')
            continue
        per_class_auc[k] = float(roc_auc_score(y_bin, s[:, k]))
        aucs.append(per_class_auc[k])
    out = {
        'macro_auc':    float(np.mean(aucs)) if aucs else float('nan'),
        'per_class_auc': per_class_auc,
        'macro_f1':     float(f1_score(y, y_pred, average='macro', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y, y_pred)),
        'accuracy':     float(accuracy_score(y, y_pred)),
        'confusion':    confusion_matrix(y, y_pred, labels=list(range(n_classes))).tolist(),
    }
    if ordinal:
        out['kappa'] = float(cohen_kappa_score(y, y_pred, weights='quadratic',
                                                 labels=list(range(n_classes))))
    return out


def optimize_multiclass_thresholds(y_true: np.ndarray, scores: np.ndarray,
                                    n_classes: int, n_passes: int = 3,
                                    grid_size: int = 41) -> np.ndarray:
    """Find per-class additive offsets `b` (shape (n_classes,)) that maximise the
    macro-F1 of `argmax_k (scores_k + b_k)`. Coordinate-descent grid search on a
    symmetric grid in [-3, 3] ; `n_passes` outer sweeps over the classes.

    Returns the optimal offset vector (kept = 0 for classes that bring no
    improvement).
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    grid = np.linspace(-3.0, 3.0, grid_size)
    b = np.zeros(n_classes, dtype=np.float64)
    best = f1_score(y, s.argmax(axis=1), average='macro', zero_division=0)
    for _ in range(n_passes):
        for k in range(n_classes):
            best_v = b[k]
            for v in grid:
                b_try = b.copy(); b_try[k] = v
                pred = (s + b_try).argmax(axis=1)
                f = f1_score(y, pred, average='macro', zero_division=0)
                if f > best:
                    best = f; best_v = v
            b[k] = best_v
    return b
