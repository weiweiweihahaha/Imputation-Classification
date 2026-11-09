import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from code.core.classifiers import predict_with_scores


def _binary_flip_rates(y_true, y_pred):
    # Class-conditional flip rates are only well-defined in binary problems.
    labels = np.unique(y_true)
    if len(labels) != 2:
        return np.nan, np.nan

    negative_label, positive_label = labels[0], labels[1]
    cm = confusion_matrix(y_true, y_pred, labels=[negative_label, positive_label])
    negatives = cm[0].sum()
    positives = cm[1].sum()
    flip_neg = float(cm[0, 1] / negatives) if negatives else np.nan
    flip_pos = float(cm[1, 0] / positives) if positives else np.nan
    return flip_pos, flip_neg


def evaluate_decision_consequences(
    ref_model,
    imputed_model,
    X_ground_test,
    X_imputed_test,
    y_true,
    sample_ids,
    hard_mask,
    metadata,
):
    # Same model + different inputs: isolates the effect of structural
    # perturbation introduced by imputation.
    ref_on_ground = predict_with_scores(ref_model, X_ground_test)
    ref_on_imputed = predict_with_scores(ref_model, X_imputed_test)
    # Retrained model: captures the actual end-to-end impute-then-classify
    # behavior users care about.
    inference_start = time.perf_counter()
    retrained_on_imputed = predict_with_scores(imputed_model, X_imputed_test)
    inference_runtime_seconds = time.perf_counter() - inference_start

    ref_input_flip = ref_on_ground["y_pred"] != ref_on_imputed["y_pred"]
    retrained_flip = ref_on_ground["y_pred"] != retrained_on_imputed["y_pred"]
    hard_mask = np.asarray(hard_mask, dtype=bool)
    # Positive means the reference model is more confident on imputed input
    # than on clean input.
    margin_shift = ref_on_imputed["margin"] - ref_on_ground["margin"]

    flip_pos, flip_neg = _binary_flip_rates(y_true, retrained_on_imputed["y_pred"])

    summary = {
        "m2_inconsistency_ref_input": float(np.mean(ref_input_flip)),
        "m2_inconsistency_retrained": float(np.mean(retrained_flip)),
        "m2_hard_inconsistency_ref_input": float(np.mean(ref_input_flip[hard_mask])) if hard_mask.any() else np.nan,
        "m2_hard_inconsistency_retrained": float(np.mean(retrained_flip[hard_mask])) if hard_mask.any() else np.nan,
        "m2_flip_pos": flip_pos,
        "m2_flip_neg": flip_neg,
        "m2_margin_shift": float(np.mean(margin_shift)),
        "m2_hard_margin_shift": float(np.mean(margin_shift[hard_mask])) if hard_mask.any() else np.nan,
    }

    sample_level = pd.DataFrame(
        {
            **metadata,
            # Keep one row per test sample so hard-case studies can be done
            # after aggregation without rerunning the experiment.
            "sample_id": np.asarray(sample_ids),
            "y_true": np.asarray(y_true),
            "y_pred_ref_on_ground": ref_on_ground["y_pred"],
            "y_pred_ref_on_imputed": ref_on_imputed["y_pred"],
            "y_pred_retrained_on_imputed": retrained_on_imputed["y_pred"],
            "margin_ref_ground": ref_on_ground["margin"],
            "margin_ref_imputed": ref_on_imputed["margin"],
            "margin_shift": margin_shift,
            "is_hard": hard_mask.astype(int),
            "is_flip_ref_input": ref_input_flip.astype(int),
            "is_flip_retrained": retrained_flip.astype(int),
        }
    )

    timing_summary = {
        "inference_runtime_seconds": float(inference_runtime_seconds),
    }

    return summary, sample_level, ref_on_ground, retrained_on_imputed, timing_summary
