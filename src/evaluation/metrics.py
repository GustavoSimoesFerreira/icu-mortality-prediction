"""
Evaluation metrics and a subgroup fairness audit.

We deliberately report more than accuracy: medical outcomes are imbalanced, so
AUPRC and calibration matter, and performance should be checked per subgroup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return the headline metrics for a set of predicted probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0        # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "positive_rate": round(float(y_pred.mean()), 4),
    }


def fairness_by_group(y_true, y_prob, groups: pd.DataFrame, group_columns) -> pd.DataFrame:
    """Compute AUROC and positive rate for each value of each sensitive column.

    A large AUROC gap between subgroups is a signal of unequal model quality.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []

    for col in group_columns:
        if col not in groups.columns:
            continue
        values = groups[col].astype("object").fillna("Unknown")
        for val, mask in values.groupby(values).groups.items():
            idx = groups.index.get_indexer(mask)
            yt, yp = y_true[idx], y_prob[idx]
            # AUROC is undefined if a subgroup has only one class present.
            auroc = round(float(roc_auc_score(yt, yp)), 4) if len(np.unique(yt)) > 1 else None
            rows.append({
                "attribute": col,
                "group": str(val),
                "n": len(idx),
                "positive_rate": round(float(yt.mean()), 4),
                "auroc": auroc,
            })

    return pd.DataFrame(rows)
