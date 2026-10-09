"""Metrics for multi-class and binary intrusion detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)


def attack_score(proba: np.ndarray, normal_idx: int = 0) -> np.ndarray:
    """Probability that a connection is an attack (1 - P(normal))."""
    return 1.0 - proba[:, normal_idx]


def compute_metrics(y_true, y_pred, proba, class_names, is_novel=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # Attack-vs-normal view (meaningful for both tasks; class 0 is always 'normal')
    is_attack = y_true != 0
    pred_attack = y_pred != 0
    m["detection_rate"] = pred_attack[is_attack].mean()            # attack recall
    m["false_positive_rate"] = pred_attack[~is_attack].mean()      # normals flagged
    m["attack_auc"] = roc_auc_score(is_attack, attack_score(proba))
    if is_novel is not None and is_novel.any():
        # Share of attacks never seen in training that are still flagged as an attack
        m["novel_attack_detection_rate"] = pred_attack[is_novel].mean()
        m["known_attack_detection_rate"] = pred_attack[is_attack & ~is_novel].mean()

    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    for c in class_names:
        m[f"recall_{c}"] = report[c]["recall"]
        m[f"precision_{c}"] = report[c]["precision"]
        m[f"f1_{c}"] = report[c]["f1-score"]
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
