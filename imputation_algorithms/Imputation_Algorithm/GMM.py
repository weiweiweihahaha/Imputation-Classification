import numpy as np
from sklearn.mixture import GaussianMixture


def _columnwise_mean_fill(values):
    filled = np.asarray(values, dtype=float).copy()
    valid_counts = np.sum(~np.isnan(filled), axis=0)
    column_sums = np.nansum(filled, axis=0)
    column_means = np.divide(
        column_sums,
        valid_counts,
        out=np.zeros_like(column_sums, dtype=float),
        where=valid_counts > 0,
    )
    missing_mask = np.isnan(filled)
    if missing_mask.any():
        filled[missing_mask] = np.take(column_means, np.where(missing_mask)[1])
    return filled


def _observed_log_prob(row, mean, covariance, weight, reg_covar):
    observed_idx = np.flatnonzero(~np.isnan(row))
    if observed_idx.size == 0:
        return float(np.log(max(weight, 1e-12)))

    diff = row[observed_idx] - mean[observed_idx]
    covariance_oo = covariance[np.ix_(observed_idx, observed_idx)]
    covariance_oo = covariance_oo + np.eye(observed_idx.size) * reg_covar

    sign, logdet = np.linalg.slogdet(covariance_oo)
    if sign <= 0:
        covariance_oo = covariance_oo + np.eye(observed_idx.size) * max(reg_covar, 1e-6)
        sign, logdet = np.linalg.slogdet(covariance_oo)

    precision_oo = np.linalg.pinv(covariance_oo)
    mahalanobis = diff.T @ precision_oo @ diff
    normalizer = observed_idx.size * np.log(2.0 * np.pi) + logdet
    return float(np.log(max(weight, 1e-12)) - 0.5 * (normalizer + mahalanobis))


def _conditional_component_mean(row, mean, covariance, missing_idx, reg_covar):
    observed_idx = np.flatnonzero(~np.isnan(row))
    if missing_idx.size == 0:
        return np.array([], dtype=float)
    if observed_idx.size == 0:
        return mean[missing_idx]

    covariance_oo = covariance[np.ix_(observed_idx, observed_idx)]
    covariance_mo = covariance[np.ix_(missing_idx, observed_idx)]
    covariance_oo = covariance_oo + np.eye(observed_idx.size) * reg_covar
    precision_oo = np.linalg.pinv(covariance_oo)
    centered_observed = row[observed_idx] - mean[observed_idx]
    return mean[missing_idx] + covariance_mo @ precision_oo @ centered_observed


def _softmax(log_probs):
    finite_mask = np.isfinite(log_probs)
    if not finite_mask.any():
        return np.full(log_probs.shape, 1.0 / len(log_probs), dtype=float)
    shifted = log_probs[finite_mask] - np.max(log_probs[finite_mask])
    probs = np.exp(shifted)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return np.full(log_probs.shape, 1.0 / len(log_probs), dtype=float)
    result = np.zeros_like(log_probs, dtype=float)
    result[finite_mask] = probs / probs_sum
    if result.sum() == 0:
        return np.full(log_probs.shape, 1.0 / len(log_probs), dtype=float)
    return result / result.sum()


def _impute_row_with_gmm(row, gmm):
    missing_idx = np.flatnonzero(np.isnan(row))
    if missing_idx.size == 0:
        return row.copy()

    log_probs = np.array(
        [
            _observed_log_prob(row, mean, covariance, weight, gmm.reg_covar)
            for weight, mean, covariance in zip(gmm.weights_, gmm.means_, gmm.covariances_)
        ],
        dtype=float,
    )
    responsibilities = _softmax(log_probs)

    conditional_means = np.array(
        [
            _conditional_component_mean(row, mean, covariance, missing_idx, gmm.reg_covar)
            for mean, covariance in zip(gmm.means_, gmm.covariances_)
        ],
        dtype=float,
    )

    row_imputed = row.copy()
    row_imputed[missing_idx] = responsibilities @ conditional_means
    return row_imputed


def GMM(missing_data, max_iter=20, tol=1e-4, random_state=42):
    X = missing_data.to_numpy(dtype=float)
    missing_mask = np.isnan(X)

    if not missing_mask.any():
        return X.copy()

    filled = _columnwise_mean_fill(X)
    n_samples = filled.shape[0]
    if n_samples < 2:
        return filled

    n_components = min(5, n_samples)
    if n_components < 1:
        return filled

    for _ in range(max_iter):
        previous = filled.copy()
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=random_state,
            )
            gmm.fit(filled)
        except Exception:
            return previous

        for row_idx in np.flatnonzero(missing_mask.any(axis=1)):
            updated_row = _impute_row_with_gmm(X[row_idx], gmm)
            filled[row_idx, missing_mask[row_idx]] = updated_row[missing_mask[row_idx]]

        max_change = np.max(np.abs(filled[missing_mask] - previous[missing_mask]))
        if max_change < tol:
            break

    return filled
