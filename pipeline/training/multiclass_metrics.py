from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


@dataclass(frozen=True)
class MulticlassEvaluation:
    metrics: dict[str, float | int]
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame


def evaluate_multiclass_probabilities(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
    class_labels: list[str],
) -> MulticlassEvaluation:
    """Evaluate multiclass probability outputs.

    This intentionally does not create threshold sweeps or binary confidence
    buckets. Multiclass models produce one conserved probability distribution
    per fight, so their first-pass artifact contract is class metrics plus a
    confusion matrix.
    """

    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D multiclass probabilities, received shape {probabilities.shape}.")
    if probabilities.shape[0] != len(y_true):
        raise ValueError(f"Probability rows do not match labels: {probabilities.shape[0]} != {len(y_true)}")
    if probabilities.shape[1] != len(class_labels):
        raise ValueError(
            "Probability columns do not match class labels: "
            f"{probabilities.shape[1]} != {len(class_labels)}"
        )

    y_numeric = pd.to_numeric(y_true, errors="raise").astype(int)
    expected_classes = list(range(len(class_labels)))
    observed_classes = sorted(y_numeric.unique().tolist())
    unexpected = sorted(set(observed_classes) - set(expected_classes))
    if unexpected:
        raise ValueError(f"Target contains class values not represented by class_labels: {unexpected}")

    y_pred = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_numeric, y_pred)),
        "log_loss": float(log_loss(y_numeric, probabilities, labels=expected_classes)),
        "macro_f1": float(f1_score(y_numeric, y_pred, labels=expected_classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_numeric, y_pred, labels=expected_classes, average="weighted", zero_division=0)),
        "row_count": int(len(y_numeric)),
        "class_count": int(len(class_labels)),
    }

    class_rows = []
    for class_index, class_label in enumerate(class_labels):
        true_mask = y_numeric.eq(class_index)
        pred_mask = pd.Series(y_pred, index=y_numeric.index).eq(class_index)
        class_rows.append(
            {
                "class_index": class_index,
                "class_label": class_label,
                "support": int(true_mask.sum()),
                "predicted_count": int(pred_mask.sum()),
                "actual_rate": float(true_mask.mean()) if len(true_mask) else 0.0,
                "predicted_rate": float(pred_mask.mean()) if len(pred_mask) else 0.0,
                "mean_probability": float(probabilities[:, class_index].mean()),
                "one_vs_rest_f1": float(
                    f1_score(true_mask.astype(int), pred_mask.astype(int), zero_division=0)
                ),
            }
        )

    cm = confusion_matrix(y_numeric, y_pred, labels=expected_classes)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels).reset_index(names="actual_class")

    return MulticlassEvaluation(
        metrics=metrics,
        class_metrics=pd.DataFrame(class_rows),
        confusion_matrix=cm_df,
    )
