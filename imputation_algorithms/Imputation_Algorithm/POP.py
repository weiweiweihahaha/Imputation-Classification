import numpy as np

def POP_impute(sample_i, sample_j, data):
    row = data[sample_i]
    observed_mask = ~np.isnan(row)
    observed_mask[sample_j] = False
    observed_values = row[observed_mask]
    if observed_values.size == 0:
        return 0.0

    candidate_mask = ~np.isnan(data[:, sample_j])
    candidate_mask[sample_i] = False
    if observed_values.size:
        candidate_mask &= ~np.isnan(data[:, observed_mask]).any(axis=1)

    candidate_rows = np.flatnonzero(candidate_mask)
    if candidate_rows.size == 0:
        return 0.0

    vectors = data[candidate_rows][:, observed_mask]
    vector_j = data[candidate_rows, sample_j]
    h = 20.0
    diff = (observed_values - vectors) / h
    sum_t = np.sum(diff * diff, axis=1) / 2.0
    kernels = np.exp(-sum_t) / np.sqrt(2 * np.pi)
    sumx = float(np.dot(vector_j, kernels))
    sumy = float(np.sum(kernels)) + (1 / (data.shape[0] ** 2))
    return sumx / sumy

def POP(df):
    data = df.to_numpy(dtype=float, copy=True)
    missing_positions = np.argwhere(np.isnan(data))
    for i, j in missing_positions:
        data[i, j] = POP_impute(i, j, data)
    return data
