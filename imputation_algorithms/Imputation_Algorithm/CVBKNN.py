import numpy as np
import pandas as pd
import math
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold


def _column_fallback(imputed, xj):
    column = imputed.iloc[:, xj]
    observed = column.dropna()
    if observed.empty:
        return 0.0
    return float(observed.mean())


def _column_fallback_array(imputed, xj):
    column = imputed[:, xj]
    observed = column[~np.isnan(column)]
    if observed.size == 0:
        return 0.0
    return float(np.mean(observed))


def _compute_weights(X, y):
    if X.shape[0] <= 1 or X.shape[1] == 0:
        return None
    n_neighbors = min(3, X.shape[0] - 1)
    if n_neighbors < 1:
        return None
    try:
        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors)
    except ValueError:
        return None
    mi_sum = np.sum(mi)
    if mi_sum == 0:
        return None
    return mi / mi_sum

def mean(neighbors):
    return np.mean(neighbors, axis=0)

def median(neighbors):
    return np.median(neighbors, axis=0)

def idwm(y, distances, delta=1e-6):
    weights = [1 / (delta + dist) for dist in distances]
    weighted_sum = sum(w * y_val for w, y_val in zip(weights, y))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def irwm(y, distances):
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
    weights = [(len(sorted_indices) - k + 1) for k in sorted_indices]
    weighted_sum = sum(w * y[k] for w, k in zip(weights, sorted_indices))
    total_weight = sum(range(1, len(sorted_indices) + 1))
    return weighted_sum / total_weight

def dudani(y, distances):
    max_dist = max(distances)
    min_dist = min(distances)
    weights = [(max_dist - dist) / (max_dist - min_dist) if max_dist != min_dist else 1 for dist in distances]
    weighted_sum = sum(w * y_val for w, y_val in zip(weights, y))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def _weighted_distance_matrix(X_test, X_train, weights, kind, rho=0.5):
    diff = np.abs(X_test[:, None, :] - X_train[None, :, :])
    if kind == "Euclidean":
        return np.sqrt(np.sum((diff ** 2) * weights, axis=2))
    if kind == "Manhattan":
        return np.sum(diff * weights, axis=2)
    if kind == "GRA":
        delta_min = np.min(diff, axis=2, keepdims=True)
        delta_max = np.max(diff, axis=2, keepdims=True)
        denominator = diff + rho * delta_max
        zero_rows = np.all(denominator == 0, axis=2)
        safe_denominator = np.where(denominator == 0, 1.0, denominator)
        grc = (delta_min + rho * delta_max) / safe_denominator
        grg = np.sum(weights * grc, axis=2)
        distances = 1 - grg
        distances[zero_rows] = 0.0
        return distances
    raise ValueError(f"Unsupported distance kind: {kind}")


def _predict_from_sorted_neighbors(y_train, sorted_indices, sorted_distances, k, adaptation):
    neighbor_indices = sorted_indices[:, :k]
    neighbor_distances = sorted_distances[:, :k]
    neighbor_values = y_train[neighbor_indices]
    if adaptation == "mean":
        return np.mean(neighbor_values, axis=1)
    if adaptation == "median":
        return np.median(neighbor_values, axis=1)
    if adaptation == "IDWM":
        weights = 1.0 / (1e-6 + neighbor_distances)
        return np.sum(weights * neighbor_values, axis=1) / np.sum(weights, axis=1)
    if adaptation == "IRWM":
        rank_weights = np.arange(k, 0, -1, dtype=float)
        return np.sum(rank_weights * neighbor_values, axis=1) / np.sum(rank_weights)
    if adaptation == "Dudani":
        max_dist = np.max(neighbor_distances, axis=1, keepdims=True)
        min_dist = np.min(neighbor_distances, axis=1, keepdims=True)
        denom = max_dist - min_dist
        weights = np.ones_like(neighbor_distances, dtype=float)
        np.divide(
            max_dist - neighbor_distances,
            denom,
            out=weights,
            where=denom != 0,
        )
        return np.sum(weights * neighbor_values, axis=1) / np.sum(weights, axis=1)
    raise ValueError(f"Unsupported adaptation kind: {adaptation}")


def _build_training_arrays(imputed, xi, xj):
    row = imputed[xi]
    observed_mask = ~np.isnan(row)
    observed_mask[xj] = False
    feature_indices = np.flatnonzero(observed_mask)
    if feature_indices.size == 0:
        return feature_indices, None, None, None, None

    candidate_mask = ~np.isnan(imputed[:, xj])
    candidate_mask[xi] = False
    candidate_mask &= ~np.isnan(imputed[:, feature_indices]).any(axis=1)
    train_indices = np.flatnonzero(candidate_mask)
    if train_indices.size == 0:
        return feature_indices, train_indices, None, None, None

    X_train = imputed[train_indices][:, feature_indices]
    Y_train = imputed[train_indices, xj]
    X_test = row[feature_indices][None, :]
    return feature_indices, train_indices, X_train, Y_train, X_test


def _select_best_configuration(X_train, Y_train, candidate_ks):
    if X_train is None or Y_train is None or X_train.size == 0 or Y_train.size == 0:
        return None
    if X_train.ndim != 2 or X_train.shape[1] == 0 or X_train.shape[0] <= 1:
        return None

    scores = {}
    counts = {}
    n_splits = min(10, X_train.shape[0])
    if n_splits < 2:
        return None

    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        Y_train_fold, Y_test_fold = Y_train[train_index], Y_train[test_index]
        if X_train_fold.shape[0] == 0 or X_train_fold.shape[1] == 0:
            continue

        weights = _compute_weights(X_train_fold, Y_train_fold)
        if weights is None:
            continue

        for distance_name in ("Euclidean", "Manhattan", "GRA"):
            distance_matrix = _weighted_distance_matrix(X_test_fold, X_train_fold, weights, distance_name)
            sorted_indices = np.argsort(distance_matrix, axis=1)
            sorted_distances = np.take_along_axis(distance_matrix, sorted_indices, axis=1)

            for k in candidate_ks:
                effective_k = min(k, X_train_fold.shape[0])
                if effective_k < 1:
                    continue
                for adaptation_name in ("mean", "median", "IDWM", "IRWM", "Dudani"):
                    predictions = _predict_from_sorted_neighbors(
                        Y_train_fold,
                        sorted_indices,
                        sorted_distances,
                        effective_k,
                        adaptation_name,
                    )
                    key = (distance_name, adaptation_name, k)
                    scores[key] = scores.get(key, 0.0) + float(np.sum((Y_test_fold - predictions) ** 2))
                    counts[key] = counts.get(key, 0) + len(Y_test_fold)

    best = None
    best_score = math.inf
    for distance_name in ("Euclidean", "Manhattan", "GRA"):
        for adaptation_name in ("mean", "median", "IDWM", "IRWM", "Dudani"):
            for k in candidate_ks:
                key = (distance_name, adaptation_name, k)
                if counts.get(key, 0) == 0:
                    continue
                mse = scores[key] / counts[key]
                if mse < best_score:
                    best = key
                    best_score = mse
    return best

def CVBKNN_impute(xi, xj, imputed):
    return CVBKNN_impute_cached(xi, xj, imputed, {})


def CVBKNN_impute_cached(xi, xj, imputed, parameter_cache):
    feature_indices, train_indices, X_train, Y_train, X_test = _build_training_arrays(imputed, xi, xj)
    if X_train is None or Y_train is None or X_test is None or X_train.size == 0 or Y_train.size == 0:
        return _column_fallback_array(imputed, xj)
    if X_train.ndim != 2 or X_train.shape[1] == 0:
        return _column_fallback_array(imputed, xj)

    candidate_ks = [2 * idx + 1 for idx in range(0, int((np.sqrt(imputed.shape[0]) - 1) / 2) + 1)]
    cache_key = (xj, tuple(feature_indices.tolist()), tuple(train_indices.tolist()))
    config = parameter_cache.get(cache_key)
    if config is None:
        config = _select_best_configuration(X_train, Y_train, candidate_ks)
        parameter_cache[cache_key] = config
    if config is None:
        return _column_fallback_array(imputed, xj)

    dis_result, ada_result, k_result = config
    weights = _compute_weights(X_train, Y_train)
    if weights is None:
        return _column_fallback_array(imputed, xj)
    distance_matrix = _weighted_distance_matrix(X_test, X_train, weights, dis_result)
    sorted_indices = np.argsort(distance_matrix, axis=1)
    sorted_distances = np.take_along_axis(distance_matrix, sorted_indices, axis=1)
    effective_k = min(k_result, X_train.shape[0])
    if effective_k < 1:
        return _column_fallback_array(imputed, xj)
    prediction = _predict_from_sorted_neighbors(Y_train, sorted_indices, sorted_distances, effective_k, ada_result)
    return float(prediction[0])

def normalize(df):
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized

def denormalize(df_normalized, df_original):
    df_denormalized = df_normalized * (df_original.max() - df_original.min()) + df_original.min()
    return df_denormalized

def CVBKNN(df_original):
    df = normalize(df_original)
    imputed = df.to_numpy(dtype=float, copy=True)
    row_missing_ratio = np.mean(np.isnan(imputed), axis=1)
    col_missing_ratio = np.mean(np.isnan(imputed), axis=0)
    rowsTrain_index = np.argsort(row_missing_ratio)
    colsTrain_index = np.argsort(col_missing_ratio)
    parameter_cache = {}

    for col_index in colsTrain_index:
        if col_missing_ratio[col_index] == 0:
            continue
        for row_index in rowsTrain_index:
            if row_missing_ratio[row_index] == 0:
                continue
            if np.isnan(imputed[row_index, col_index]):
                imputed[row_index, col_index] = CVBKNN_impute_cached(row_index, col_index, imputed, parameter_cache)

    result = pd.DataFrame(imputed, index=df.index, columns=df.columns)
    return denormalize(result, df_original)
