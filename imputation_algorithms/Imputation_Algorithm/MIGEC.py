import numpy as np
import pandas as pd
import skfuzzy as fuzz


def _safe_column_means(frame):
    means = frame.mean(axis=0, skipna=True)
    return means.fillna(0.0)


def _standardize(values, mean, std):
    safe_std = std.copy()
    safe_std[safe_std == 0] = 1.0
    return (values - mean) / safe_std


def _grey_relation_score(row_std, centroid, observed_mask, zeta=0.5):
    if not np.any(observed_mask):
        return np.nan

    diffs = np.abs(row_std[observed_mask] - centroid[observed_mask])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan

    min_diff = np.min(diffs)
    max_diff = np.max(diffs)
    if max_diff == 0:
        return 1.0

    coeffs = (min_diff + zeta * max_diff) / (diffs + zeta * max_diff)
    return float(np.mean(coeffs))


def _select_fill_values(scores, cluster_means, global_means):
    score_array = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(score_array)
    if not finite_mask.any():
        return global_means
    best_cluster = int(np.argmax(score_array[finite_mask]))
    best_cluster_index = int(np.flatnonzero(finite_mask)[best_cluster])
    return cluster_means[best_cluster_index]


def MIGEC(df, random_seed=42):
    np.random.seed(random_seed)

    result = df.copy()
    missing_mask = result.isnull()
    if not missing_mask.to_numpy().any():
        return result

    numeric = result.apply(pd.to_numeric, errors="coerce")
    global_means = _safe_column_means(numeric)

    complete_mask = ~numeric.isnull().any(axis=1)
    complete_df = numeric.loc[complete_mask].copy()

    if complete_df.empty:
        filled = numeric.fillna(global_means)
        return filled

    train_values = complete_df.to_numpy(dtype=float)
    train_mean = np.mean(train_values, axis=0)
    train_std = np.std(train_values, axis=0)
    train_std[~np.isfinite(train_std)] = 1.0
    train_std[train_std == 0] = 1.0
    train_standardized = _standardize(train_values, train_mean, train_std)

    n_clusters = min(2, train_standardized.shape[0])
    if n_clusters < 1:
        return numeric.fillna(global_means)

    try:
        centroids, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
            train_standardized.T,
            c=n_clusters,
            m=2,
            error=0.005,
            maxiter=1000,
            seed=random_seed,
        )
    except Exception:
        return numeric.fillna(global_means)

    cluster_assignments = np.argmax(memberships, axis=0)
    cluster_means = []
    for cluster_idx in range(n_clusters):
        members = complete_df.loc[cluster_assignments == cluster_idx]
        if members.empty:
            cluster_means.append(global_means)
        else:
            cluster_means.append(_safe_column_means(members))

    incomplete_indices = numeric.index[~complete_mask]
    for index in incomplete_indices:
        row = numeric.loc[index].to_numpy(dtype=float)
        observed_mask = np.isfinite(row)

        if not np.any(observed_mask):
            numeric.loc[index] = global_means
            continue

        row_standardized = _standardize(row, train_mean, train_std)
        scores = [
            _grey_relation_score(row_standardized, centroids[cluster_idx], observed_mask)
            for cluster_idx in range(n_clusters)
        ]

        fill_values = _select_fill_values(scores, cluster_means, global_means)

        filled_row = numeric.loc[index].copy()
        filled_row = filled_row.fillna(fill_values).fillna(global_means)
        numeric.loc[index] = filled_row

    return numeric
