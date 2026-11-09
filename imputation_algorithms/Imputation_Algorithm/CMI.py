import numpy as np
from sklearn.impute import KNNImputer
import math
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


def _mean_fallback(data):
    column_means = np.nanmean(data, axis=0)
    safe_means = np.where(np.isnan(column_means), 0.0, column_means)
    return np.where(np.isnan(data), safe_means, data)


def CMI(missing_data):
    X = missing_data.to_numpy()
    if not np.isnan(X).any():
        return X

    complete = X[~np.isnan(X).any(axis=1)]
    incomplete = X[np.isnan(X).any(axis=1), :]

    complete_indices = np.where(~np.isnan(X).any(axis=1))[0]
    incomplete_indices = np.where(np.isnan(X).any(axis=1))[0]

    if incomplete.shape[0] == 0:
        return X
    if complete.shape[0] == 0:
        return KNNImputer(n_neighbors=min(4, max(1, len(X) - 1))).fit_transform(X)

    k_clusters = min(5, len(complete))
    KMean_cluster = KMeans(n_clusters=k_clusters, n_init="auto", random_state=0)
    KMean_cluster.fit(complete)

    cluster_labels = KMean_cluster.predict(complete)
    cluster_centers = KMean_cluster.cluster_centers_

    dist_mat = pairwise_distances(incomplete, cluster_centers, metric='nan_euclidean')
    incomplete_cluster_labels = [[] for _ in range(len(cluster_centers))]
    complete_cluster_labels = [[] for _ in range(len(cluster_centers))]

    complete_mean = np.zeros((len(cluster_centers), complete.shape[1]), dtype=float)
    for label in range(len(cluster_centers)):
        mask = cluster_labels == label
        if np.any(mask):
            complete_mean[label] = np.mean(complete[mask], axis=0)

    for i in range(complete.shape[0]):
        complete_cluster_labels[cluster_labels[i]].append(i)

    incomplete_mean = np.copy(incomplete)

    for i in range(incomplete.shape[0]):
        closest_cluster = np.nanargmin(dist_mat[i, :])
        incomplete_cluster_labels[closest_cluster].append(i)

        idx = np.where(np.isnan(incomplete[i]))[0]
        incomplete_mean[i][idx] = complete_mean[closest_cluster][idx]

    def K(x):
        return math.exp(-(x**2 / 2)) / math.sqrt(math.pi*2)
    h = 5
    def f(X1, X2, indice):
        res = 1
        for s in range(X1.shape[0]):
            if indice == s:
                continue
            res *= K((X1[s] - X2[s]) / h)
        return res

    def kernel_function(X, indice):
        m = 0 
        m_ = 0
        Xi, i, cluster = X
        for j in range(complete.shape[0]):
            if j in complete_cluster_labels[cluster]:
                mul_j = f(Xi, complete[j], indice)
                m += mul_j * complete[j][indice]
                m_ += mul_j
        for j in range(incomplete_mean.shape[0]):
            if j == i :
                continue
            if j in incomplete_cluster_labels[cluster]:
                if indice in np.argwhere(np.isnan(incomplete[j])).flatten():
                    continue
                mul_j = f(Xi, incomplete_mean[j], indice)
                m += mul_j * incomplete_mean[j][indice]
                m_ += mul_j

        return m, m_

    for cluster in range(len(cluster_centers)):
        for i in incomplete_cluster_labels[cluster]:
            incomplete_instance = incomplete_mean[i]
            nan_indices = np.argwhere(np.isnan(incomplete[i])).flatten()
            for indice in nan_indices:
                m, m_ = kernel_function((incomplete_instance, i, cluster), indice)
                if m_ <= 0 or not np.isfinite(m_):
                    incomplete[i][indice] = complete_mean[cluster][indice]
                else:
                    incomplete[i][indice] = m / m_
                incomplete_mean[i][indice] = incomplete[i][indice]

    merged_data = np.concatenate((complete, incomplete), axis=0)
    sorted_indices = np.concatenate((complete_indices, incomplete_indices))
    sorted_indices_argsort = np.argsort(sorted_indices)
    merged_data_sorted = merged_data[sorted_indices_argsort]
    return _mean_fallback(merged_data_sorted)
