import numpy as np  
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
import sys
import os
from types import SimpleNamespace

from code.core.data_io import (
    build_clean_data_path,
    load_real_missing_data,
    parse_run_stem,
    resolve_missing_data_path,
    uses_real_missing,
)


def _fallback_fill(df):
    result = df.copy()
    for column in result.columns:
        series = result[column]
        if not series.isnull().any():
            continue
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            fill_value = numeric.mean()
        else:
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else 0
        result[column] = series.fillna(fill_value)
    return result


def _load_missing_frame(args):
    if uses_real_missing(args):
        frame = load_real_missing_data(args)
    else:
        frame = pd.read_csv(resolve_missing_data_path(args), delimiter=",", header=None)
    frame = frame.copy().reset_index(drop=True)
    frame.columns = list(range(frame.shape[1]))
    return frame


def _mode_fill(series):
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return 0


def _prepare_mlp_frame(frame):
    result = frame.copy()
    if result.shape[1] <= 1:
        raise ValueError("MLPI requires at least one feature column in addition to the label column.")

    feature_columns = list(result.columns[1:])
    numeric_columns = []
    for column in feature_columns:
        numeric = pd.to_numeric(result[column], errors="coerce")
        non_null_count = int(result[column].notna().sum())
        if non_null_count > 0 and int(numeric.notna().sum()) == non_null_count:
            result[column] = numeric
            numeric_columns.append(column)
        else:
            result[column] = result[column].fillna(_mode_fill(result[column]))

    if not numeric_columns:
        raise ValueError("MLPI found no numeric feature columns to impute.")

    numeric_data = result.loc[:, numeric_columns].astype(float)
    return result, numeric_columns, numeric_data

dataset_name = sys.argv[1]
name = sys.argv[2]
run_parts = parse_run_stem(dataset_name)
missing_args = SimpleNamespace(
    dataset=run_parts["dataset"],
    dataset_category=os.getenv("FRAMEWORK_DATASET_CATEGORY", "no_real_missing"),
    missing_percent=run_parts["missing_percent"],
    max_missing_lenth=run_parts["max_missing_lenth"],
    missing_mechanism=run_parts["mechanism"],
    iter=run_parts["iteration"],
    seed=run_parts["seed"],
    real_missing_data_dir=os.getenv("FRAMEWORK_REAL_MISSING_DATA_DIR", "data/Real_missing_data"),
    real_missing_truth_dir=os.getenv("FRAMEWORK_REAL_MISSING_TRUTH_DIR", "data/Real_missing_truth"),
    real_missing_label_col=int(os.getenv("FRAMEWORK_REAL_MISSING_LABEL_COL", "0") or 0),
    missing_data_dir=os.getenv("FRAMEWORK_MISSING_DATA_DIR", "data/Missing_data"),
    clean_data_dir=os.getenv("FRAMEWORK_CLEAN_DATA_DIR", "data/Clean_data"),
)
original_data = _load_missing_frame(missing_args)
output_data, numeric_columns, df = _prepare_mlp_frame(original_data)

features = df.columns.tolist()
train_df = df.dropna()
test_df = df[df.isnull().any(axis=1)]
df_array = test_df.to_numpy(dtype=float)

if test_df.empty:
    output_path = build_clean_data_path(missing_args, "MLPI", ensure_parent=True)
    output_data.to_csv(output_path, sep=",", header=False, index=False)
    sys.exit(0)

if train_df.empty:
    fallback_numeric = _fallback_fill(df)
    output_data.loc[:, numeric_columns] = fallback_numeric
    output_path = build_clean_data_path(missing_args, "MLPI", ensure_parent=True)
    output_data.to_csv(output_path, sep=",", header=False, index=False)
    sys.exit(0)

scaler = StandardScaler()  
train_features = scaler.fit_transform(train_df[features])  
test_features = scaler.transform(test_df[features])  
test_features = np.nan_to_num(test_features)

model = MLPRegressor(
    hidden_layer_sizes=(64,32,16),
    activation='relu',
    solver='adam',
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=5000,
    random_state=run_parts["seed"] if run_parts["seed"] is not None else 40,
    verbose=False,
)

model.fit(train_features, train_df[features])

predictions = model.predict(test_features)
imputed = np.empty((predictions.shape[0], predictions.shape[1]))
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        if np.isnan(df_array[i, j]):
            imputed[i, j] = predictions[i, j]
        else:
            imputed[i, j] = df_array[i, j]
output_data.loc[test_df.index, numeric_columns] = imputed
output_path = build_clean_data_path(missing_args, "MLPI", ensure_parent=True)
output_data.to_csv(output_path, sep=",", header=False, index=False)
