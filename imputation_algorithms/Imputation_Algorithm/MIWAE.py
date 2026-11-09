from imputation_algorithms.Imputation_Algorithm._tf_compat import quiet_tensorflow_logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import sys
import matplotlib.pyplot as plt
from types import SimpleNamespace

from code.core.data_io import (
    build_clean_data_path,
    load_real_missing_data,
    parse_run_stem,
    resolve_missing_data_path,
    uses_real_missing,
)

quiet_tensorflow_logging(tf)
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


def _prepare_miwae_frame(frame):
    result = frame.copy()
    if result.shape[1] <= 1:
        raise ValueError("MIWAE requires at least one feature column in addition to the label column.")

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
        raise ValueError("MIWAE found no numeric feature columns to impute.")

    numeric_data = result.loc[:, numeric_columns].astype(float)
    return result, numeric_columns, numeric_data


original_data = _load_missing_frame(missing_args)
output_data, numeric_columns, data = _prepare_miwae_frame(original_data)
data_values = data.to_numpy(dtype=float, copy=True)
x_0 = np.copy(data_values)
x_0[np.isnan(data_values)] = 0
scale = np.std(x_0, 0)
scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
center = np.mean(x_0, 0)
xmiss = (data_values - center) / scale
n = xmiss.shape[0] # number of observations
p = xmiss.shape[1] # number of features
mask = np.isfinite(xmiss).astype(np.float32) # binary mask that indicates which values are missing
xhat_0 = np.copy(xmiss)
xhat_0[np.isnan(xmiss)] = 0

d = 10 # 10D manifold
p_z = tfd.MultivariateNormalDiag(loc=tf.zeros(d, tf.float32))
K = 20
L = 10000
h = 128 # number of hidden units (same for all MLPs)

sigma = "relu"

decoder = tfk.Sequential([
tfkl.InputLayer(input_shape=[d,]),
tfkl.Dense(h, activation=sigma,kernel_initializer="he_normal"),
tfkl.Dense(h, activation=sigma,kernel_initializer="he_normal"),
tfkl.Dense(3*p,kernel_initializer="he_normal") # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
])

iota = tf.Variable(np.zeros([1,p]),dtype=tf.float32) # this will be updated during training

encoder = tfk.Sequential([
tfkl.InputLayer(input_shape=[p,]),
tfkl.Dense(h, activation=sigma,kernel_initializer="he_normal"),
tfkl.Dense(h, activation=sigma,kernel_initializer="he_normal"),
tfkl.Dense(h, activation=sigma,kernel_initializer="he_normal"),
tfkl.Dense(3*d,kernel_initializer="he_normal")
])

@tf.function
def miwae_bound(xhat_0_batch, mask_batch,K):

    tiledmask = tf.tile(mask_batch,[K,1])
    mask_complement_float = tf.abs(mask_batch-1)

    tilediota = tf.tile(iota,[xhat_0_batch.shape[0],1])
    iotax = xhat_0_batch + tf.multiply(tilediota,mask_complement_float)

    out_encoder = encoder(iotax)
    q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d], scale=tf.nn.softplus(out_encoder[..., d:(2*d)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d):(3*d)])))
    zgivenx = q_zgivenxobs.sample(K)
    zgivenx_flat = tf.reshape(zgivenx,[K*xhat_0_batch.shape[0],d])

    out_decoder = decoder(zgivenx_flat)
    data_flat = tf.reshape(tf.tile(xhat_0_batch,[K,1]),[-1,1])
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = tf.nn.softplus(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*p):(3*p)]) + 3
    all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
    all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[K*xhat_0_batch.shape[0],p])

    logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz,tiledmask),1),[K,xhat_0_batch.shape[0]])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    average_miwae_bound = tf.reduce_mean(tf.reduce_logsumexp(logpxobsgivenz + logpz - logq,0) -tf.math.log(tf.cast(K,tf.float32)) ) 
    return average_miwae_bound

@tf.function
def miwae_impute(xhat_0_batch, mask_batch,L=1000,kind="single",num_samples=20):

    # kind = "single" will return only the single imputation
    # kind = "multiple" will return num_samples multiple imputations
    # kind = "both" will both the single imputation and num_samples multiple imputations

    tiledmask = tf.tile(mask_batch,[L,1])
    mask_complement_float = tf.abs(mask_batch-1)

    tilediota = tf.tile(iota,[xhat_0_batch.shape[0],1])
    iotax = xhat_0_batch + tf.multiply(tilediota,mask_complement_float)

    out_encoder = encoder(iotax)
    q_zgivenxobs = tfd.Independent(distribution=tfd.StudentT(loc=out_encoder[..., :d], scale=tf.nn.softplus(out_encoder[..., d:(2*d)]), df=3 + tf.nn.softplus(out_encoder[..., (2*d):(3*d)])))
    zgivenx = q_zgivenxobs.sample(L)
    zgivenx_flat = tf.reshape(zgivenx,[L*xhat_0_batch.shape[0],d])

    out_decoder = decoder(zgivenx_flat)
    data_flat = tf.reshape(tf.tile(xhat_0_batch,[L,1]),[-1,1])
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = tf.nn.softplus(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = tf.nn.softplus(out_decoder[..., (2*p):(3*p)]) + 3
    xgivenz = tfd.Independent(distribution=tfd.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model))
    all_log_pxgivenz_flat = tfd.StudentT(loc=tf.reshape(all_means_obs_model,[-1,1]),scale=tf.reshape(all_scales_obs_model,[-1,1]),df=tf.reshape(all_degfreedom_obs_model,[-1,1])).log_prob(data_flat)
    all_log_pxgivenz = tf.reshape(all_log_pxgivenz_flat,[L*xhat_0_batch.shape[0],p])

    logpxobsgivenz = tf.reshape(tf.reduce_sum(tf.multiply(all_log_pxgivenz,tiledmask),1),[L,xhat_0_batch.shape[0]])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    log_imp_weights = logpxobsgivenz + logpz - logq # same importance wieghts used for single and multiple imputation

    if ((kind=="single") or (kind=="both")):

        imp_weights = tf.nn.softmax(log_imp_weights,0) # these are w_1,....,w_L for all observations in the batch
        xms = tf.reshape(xgivenz.mean(),[L,xhat_0_batch.shape[0],p])
        xm = xhat_0_batch + tf.multiply(tf.einsum('ki,kij->ij', imp_weights, xms),mask_complement_float)

    if ((kind=="multiple") or (kind=="both")):

        sir_logits = tf.transpose(log_imp_weights)
        sir = tfd.Categorical(logits = sir_logits).sample(num_samples)
        xmul = tf.reshape(xhat_0_batch + tf.multiply(tf.reshape(xgivenz.sample(),[L,xhat_0_batch.shape[0],p]),mask_complement_float),[L,xhat_0_batch.shape[0],p])

    if (kind=="single"):
        return xm

    if (kind=="multiple"):
        return tf.gather(tf.transpose(xmul,perm=[1,0,2]), tf.transpose(sir), axis = 1, batch_dims=1)

    if (kind=="both"):
        return xm, tf.gather(tf.transpose(xmul,perm=[1,0,2]), tf.transpose(sir), axis = 1, batch_dims=1)
    
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

all_parameters = encoder.trainable_variables + decoder.trainable_variables + [iota]

@tf.function
def gradient_step_miwae(xhat_0_batch, mask_batch,K):
    with tf.GradientTape() as tape: # the gradient tape saves all the step that needs to be saved fopr automatic differentiation
        loss = -  miwae_bound(xhat_0_batch, mask_batch,K)  # the loss is the negative ELBO
    gradients = tape.gradient(loss, all_parameters)  # here, the gradient is automatically computed
    optimizer.apply_gradients(zip(gradients, all_parameters))  # Adam iteration

train_dataset = tf.data.Dataset.from_tensor_slices((xhat_0,mask)).shuffle(n).batch(32) # TF creates the batches for us


if __name__ == "__main__":
    EPOCHS = 101
    mse_miwae=np.array([])
    xhat_miwae = np.copy(xhat_0)

    for epoch in range(1,EPOCHS+1):
        for xhat_0_batch, mask_batch in train_dataset:
            xhat_0_batch = tf.cast(xhat_0_batch, dtype=tf.float32)
            mask_batch = tf.cast(mask_batch, dtype=tf.float32)
            gradient_step_miwae(xhat_0_batch, mask_batch,K = K) # Adam iteration
        
        if (epoch % 10) == 1:
            xhat_0 = tf.cast(xhat_0, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            train_miwae_bound = miwae_bound(xhat_0, mask, K = K)
            print('Epoch  %g' %epoch)
            print('Train MIWAE bound  %g' %train_miwae_bound.numpy())
            print('-----------')

    print("Train Done")
    for i in range(n):
        xhat_0_reshaped = tf.reshape(xhat_0[i, :], [1, p])
        mask_reshaped = tf.reshape(mask[i, :], [1, p])
        xhat_mul = miwae_impute(xhat_0_reshaped, mask_reshaped, kind="multiple", L=L).numpy()
        #xhat_mul = miwae_impute(xhat_0[i,:].reshape([1,p]), mask[i,:].reshape([1,p]), kind = "multiple",L = L).numpy()
        xhat_mul = np.average(xhat_mul, axis=1)
        xhat_miwae[i,:] = xhat_mul

    xhat_miwae = xhat_miwae * scale + center
    miwae_imputed = np.copy(data_values)
    miwae_imputed[np.isnan(data_values)] = np.where(
        np.isnan(data_values),
        xhat_miwae,
        data_values,
    )[np.isnan(data_values)]
    output_data.loc[:, numeric_columns] = miwae_imputed
    df = output_data
    output_path = build_clean_data_path(missing_args, "MIWAE", ensure_parent=True)
    df.to_csv(output_path, sep=",", header=False, index=False)
