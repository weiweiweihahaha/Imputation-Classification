import numpy as np
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imputation_algorithms.Imputation_Algorithm._autoencoders import TFVariationalAutoencoder
import pandas as pd
import random
from types import SimpleNamespace
from code.core.data_io import (
    build_clean_data_path,
    load_real_missing_data,
    parse_run_stem,
    resolve_missing_data_path,
    uses_real_missing,
)
'''
==============================================================================
'''
# Load data with missing values from a csv for analysis:
import sys
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


def _prepare_vae_frame(frame):
    result = frame.copy()
    if result.shape[1] <= 1:
        raise ValueError("VAEI requires at least one feature column in addition to the label column.")

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
        raise ValueError("VAEI found no numeric feature columns to impute.")

    numeric_data = result.loc[:, numeric_columns].astype(float)
    return result, numeric_columns, numeric_data


original_data = _load_missing_frame(missing_args)
output_data, numeric_columns, Xdata_df = _prepare_vae_frame(original_data)
# VAE network size:
Decoder_hidden1 = 50
Decoder_hidden2 = 50
Encoder_hidden1 = 50
Encoder_hidden2 = 50

# dimensionality of latent space:
latent_size = 10

# Keep VAEI practical inside the end-to-end benchmark pipeline.
training_epochs = 30
batch_size = 64
learning_rate = 0.001

# specify number of imputation iterations:
ImputeIter = 25

'''
==============================================================================
'''
Xdata_Missing = Xdata_df.to_numpy(dtype=float, copy=True)
if Xdata_Missing.size == 0:
    raise ValueError("VAEI received an empty numeric feature matrix.")
# Properties of data:
Xdata_length = Xdata_Missing.shape[0] # number of data points to use
n_x = Xdata_Missing.shape[1] # dimensionality of data space
ObsRowInd = np.where(np.isfinite(np.sum(Xdata_Missing,axis=1)))
NanRowInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=1)))
NanIndex = np.where(np.isnan(Xdata_Missing))
Xdata_Missing_Rows = NanRowInd[0] # number of rows with missing values
complete_row_count = ObsRowInd[0].size

if complete_row_count == 0:
    raise ValueError(
        "VAEI requires at least one fully observed row before training, "
        "but found 0 complete rows in the numeric feature matrix."
    )

# Zscore of data produces much better results
sc = StandardScaler()
Xdata_Missing_complete = np.copy(Xdata_Missing[ObsRowInd[0],:])
# standardise using complete records:
sc.fit(Xdata_Missing_complete)
Xdata_Missing[NanIndex] = 0
Xdata_Missing = sc.transform(Xdata_Missing)
Xdata_Missing[NanIndex] = np.nan
del Xdata_Missing_complete
'''
==============================================================================
'''
# INITIALISE AND TRAIN VAE
# define dict for network structure:
network_architecture = \
    dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
         n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
         n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
         n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
         n_input=n_x, # data input size
         n_z=latent_size)  # dimensionality of latent space

# initialise VAE:
vae = TFVariationalAutoencoder(network_architecture, 
                             learning_rate=learning_rate, 
                             batch_size=batch_size)

# train VAE on corrupted data:
vae = vae.train(XData=Xdata_Missing,
                training_epochs=training_epochs)
'''
==============================================================================
'''
# IMPUTE MISSING VALUES
# impute missing values:
X_impute = vae.impute(X_corrupt = Xdata_Missing, max_iter = ImputeIter)
X_impute_original = sc.inverse_transform(X_impute)
df_array = Xdata_df.to_numpy(dtype=float)
imputed = np.empty((X_impute_original.shape[0], X_impute_original.shape[1]))
#print(df_array.shape)
#print(type(df_array))
for i in range(X_impute_original.shape[0]):
    for j in range(X_impute_original.shape[1]):
        if np.isnan(df_array[i, j]):
            imputed[i, j] = X_impute_original[i, j]
        else:
            imputed[i, j] = df_array[i, j]
output_data.loc[:, numeric_columns] = imputed
imputed_data = output_data
output_path = build_clean_data_path(missing_args, "VAEI", ensure_parent=True)
imputed_data.to_csv(output_path, sep=",", header=False, index=False)
