import numpy as np
import pandas as pd
from minisom import MiniSom


def SOM(missing_data):
    X = missing_data.to_numpy(dtype=float)
    complete = X[~np.isnan(X).any(axis=1)]
    incomplete = X[np.isnan(X).any(axis=1), :]

    complete_indices = np.where(~np.isnan(X).any(axis=1))[0]
    incomplete_indices = np.where(np.isnan(X).any(axis=1))[0]

    _complete = np.copy(complete)
    std = complete.std(axis=0)
    mean = complete.mean(axis=0)
    usable_columns = np.isfinite(std) & (std > 0)
    if not np.any(usable_columns):
        raise ValueError("SOM requires at least one non-zero-variance column.")

    complete_train = (complete[:, usable_columns] - mean[usable_columns]) / std[usable_columns]
    complete_train = np.nan_to_num(complete_train, nan=0.0, posinf=0.0, neginf=0.0)

    som = MiniSom(15, 15, complete_train.shape[1], sigma=1.0, learning_rate=0.5)
    som.pca_weights_init(complete_train)
    som.train_random(complete_train, 1000)

    incomplete_normalized = (
        incomplete[:, usable_columns] - mean[usable_columns]
    ) / std[usable_columns]
    incomplete_normalized = np.nan_to_num(
        incomplete_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    for i, incomplete_data in enumerate(incomplete_normalized):
        win_position = som.winner(incomplete_data)
        complete_data = som._weights[win_position]
        denormalized = np.real(complete_data) * std[usable_columns] + mean[usable_columns]
        row = incomplete[i]
        missing_train_columns = np.isnan(row[usable_columns])
        row[usable_columns] = np.where(missing_train_columns, denormalized, row[usable_columns])
        skipped_columns = ~usable_columns
        row[skipped_columns] = np.where(np.isnan(row[skipped_columns]), mean[skipped_columns], row[skipped_columns])
        incomplete[i] = row

    merged_data = np.concatenate((_complete, incomplete), axis=0)
    sorted_indices = np.concatenate((complete_indices, incomplete_indices))
    sorted_indices_argsort = np.argsort(sorted_indices)
    merged_data_sorted = merged_data[sorted_indices_argsort]
    return merged_data_sorted
