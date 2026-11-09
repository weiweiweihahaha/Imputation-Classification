import numpy as np
import pandas as pd


RELATIVE_W1_METRIC_VERSION = "ground_iqr_scaled_v2"


def _safe_series(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return series.astype(float)


def _is_continuous_numeric(series):
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not isinstance(series.dtype, pd.CategoricalDtype)
    )


def _wasserstein_1_approx(left, right):
    # A lightweight W1 approximation avoids adding a scipy dependency here.
    left_series = _safe_series(left)
    right_series = _safe_series(right)
    if left_series.empty or right_series.empty:
        return np.nan

    grid_size = max(len(left_series), len(right_series), 16)
    quantiles = np.linspace(0.0, 1.0, grid_size)
    left_q = np.quantile(left_series.to_numpy(), quantiles)
    right_q = np.quantile(right_series.to_numpy(), quantiles)
    return float(np.mean(np.abs(left_q - right_q)))


def _relative_wasserstein_1(wasserstein_1, ground_iqr):
    """Return a unit-free, bounded W1 while preserving zero versus positive."""
    if not np.isfinite(wasserstein_1) or not np.isfinite(ground_iqr):
        return np.nan
    wasserstein_1 = max(float(wasserstein_1), 0.0)
    ground_iqr = max(float(ground_iqr), 0.0)
    if wasserstein_1 == 0.0:
        return 0.0
    if ground_iqr == 0.0:
        return 1.0
    return wasserstein_1 / (wasserstein_1 + ground_iqr)


def _iqr_scaled_wasserstein_1(wasserstein_1, ground_iqr):
    """Measure W1 after scaling a feature by its ground-truth IQR."""
    if not np.isfinite(wasserstein_1) or not np.isfinite(ground_iqr):
        return np.nan
    wasserstein_1 = max(float(wasserstein_1), 0.0)
    ground_iqr = max(float(ground_iqr), 0.0)
    if wasserstein_1 == 0.0:
        return 0.0
    if ground_iqr == 0.0:
        return np.nan
    return wasserstein_1 / ground_iqr


def _fisher_ratio(frame, label_col, feature_cols):
    # This is a simple per-feature separability proxy, not a full LDA criterion.
    ratios = []
    labels = frame[label_col]
    valid_labels = labels.dropna().unique()
    if len(valid_labels) <= 1:
        return {}

    for col in feature_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        valid_mask = values.notna() & labels.notna()
        if valid_mask.sum() == 0:
            continue

        grouped = []
        for label in valid_labels:
            class_values = values[valid_mask & (labels == label)].to_numpy(dtype=float)
            if class_values.size:
                grouped.append(class_values)

        if len(grouped) <= 1:
            continue

        class_means = np.array([group.mean() for group in grouped], dtype=float)
        between = float(np.var(class_means))
        within = float(np.mean([np.var(group) for group in grouped]))
        ratios.append((col, between / (within + 1e-12)))

    return dict(ratios)


def compute_structure_metrics(ground_df, imputed_df, label_col=0):
    # M1 combines marginal distortion, dependency distortion, and class-
    # conditional separability change into one summary table.
    feature_cols = [col for col in ground_df.columns if col != label_col]
    continuous_feature_cols = [
        col
        for col in feature_cols
        if _is_continuous_numeric(ground_df[col]) and _is_continuous_numeric(imputed_df[col])
    ]
    per_feature_rows = []
    mean_diffs = []
    std_diffs = []
    wasserstein_values = []
    scaled_wasserstein_values = []
    relative_wasserstein_values = []
    zero_iqr_count = 0

    for col in continuous_feature_cols:
        ground_feature = _safe_series(ground_df[col])
        imputed_feature = _safe_series(imputed_df[col])
        if ground_feature.empty or imputed_feature.empty:
            continue

        mean_diff = float(abs(ground_feature.mean() - imputed_feature.mean()))
        std_diff = float(abs(ground_feature.std(ddof=0) - imputed_feature.std(ddof=0)))
        w1 = _wasserstein_1_approx(ground_feature, imputed_feature)
        ground_q25, ground_q75 = np.quantile(
            ground_feature.to_numpy(),
            [0.25, 0.75],
        )
        ground_iqr = float(max(ground_q75 - ground_q25, 0.0))
        scaled_w1 = _iqr_scaled_wasserstein_1(w1, ground_iqr)
        relative_w1 = _relative_wasserstein_1(w1, ground_iqr)

        mean_diffs.append(mean_diff)
        std_diffs.append(std_diff)
        wasserstein_values.append(w1)
        if np.isfinite(scaled_w1):
            scaled_wasserstein_values.append(scaled_w1)
        relative_wasserstein_values.append(relative_w1)
        zero_iqr_count += int(ground_iqr == 0.0)
        per_feature_rows.append(
            {
                "feature": str(col),
                "mean_abs_diff": mean_diff,
                "std_abs_diff": std_diff,
                "wasserstein_1": w1,
                "ground_q25": float(ground_q25),
                "ground_q75": float(ground_q75),
                "ground_iqr": ground_iqr,
                "wasserstein_1_iqr_scaled": scaled_w1,
                "wasserstein_1_relative_iqr": relative_w1,
                "w1_scale_degenerate": int(ground_iqr == 0.0),
                "ground_finite_count": int(len(ground_feature)),
            }
        )

    if continuous_feature_cols:
        ground_features = ground_df[continuous_feature_cols].apply(pd.to_numeric, errors="coerce")
        imputed_features = imputed_df[continuous_feature_cols].apply(pd.to_numeric, errors="coerce")
        # Frobenius norm summarizes how much the numeric feature correlation graph moved.
        ground_corr = ground_features.corr().fillna(0.0)
        imputed_corr = imputed_features.corr().fillna(0.0)
        corr_frobenius = float(np.linalg.norm((ground_corr - imputed_corr).to_numpy(), ord="fro"))
    else:
        corr_frobenius = np.nan

    ground_fisher = _fisher_ratio(ground_df, label_col=label_col, feature_cols=continuous_feature_cols)
    imputed_fisher = _fisher_ratio(imputed_df, label_col=label_col, feature_cols=continuous_feature_cols)
    fisher_deltas = []
    for col in continuous_feature_cols:
        if col in ground_fisher and col in imputed_fisher:
            fisher_deltas.append(abs(ground_fisher[col] - imputed_fisher[col]))

    summary = {
        "m1_mean_shift": float(np.nanmean(mean_diffs)) if mean_diffs else np.nan,
        "m1_std_shift": float(np.nanmean(std_diffs)) if std_diffs else np.nan,
        # Keep raw W1 unchanged for compatibility with existing summaries.
        "m1_w1_mean": float(np.nanmean(wasserstein_values)) if wasserstein_values else np.nan,
        # Average W1 after scaling each feature by its ground-truth IQR.
        # Constant ground-truth features do not have a defined scale and are
        # excluded from this average.
        "m1_w1_iqr_scaled_mean": (
            float(np.nanmean(scaled_wasserstein_values))
            if scaled_wasserstein_values
            else np.nan
        ),
        "m1_w1_relative_iqr_mean": (
            float(np.nanmean(relative_wasserstein_values))
            if relative_wasserstein_values
            else np.nan
        ),
        "m1_w1_feature_count": len(wasserstein_values),
        "m1_w1_iqr_scaled_feature_count": len(scaled_wasserstein_values),
        "m1_w1_zero_iqr_count": zero_iqr_count,
        "m1_w1_metric_version": RELATIVE_W1_METRIC_VERSION,
        "m1_corr_frob": corr_frobenius,
        "m1_fisher_delta": float(np.nanmean(fisher_deltas)) if fisher_deltas else np.nan,
    }

    return summary, pd.DataFrame(per_feature_rows)
