from __future__ import annotations

import os
import random
import sys
import tempfile
import types

import numpy as np
import pandas as pd


def _load_optional_dependencies():
    import hnswlib
    import jax
    import jax.core as core
    import jax.random as random_module

    try:
        from jax._src import core as private_core
    except Exception:
        private_core = None

    # Keep plotting/cache side effects inside a writable temp location.
    cache_root = os.path.join(tempfile.gettempdir(), "nomi_runtime_cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "xdg"))

    # neural_tangents 0.6.x still expects several legacy JAX aliases.
    try:
        from jax._src import linear_util as linear_util
    except Exception:
        linear_util = None
    if linear_util is not None and not hasattr(jax, "linear_util"):
        jax.linear_util = linear_util

    try:
        from jax._src.api_util import flatten_fun_nokwargs
    except Exception:
        flatten_fun_nokwargs = None
    if flatten_fun_nokwargs is not None and not hasattr(jax, "flatten_fun_nokwargs"):
        jax.flatten_fun_nokwargs = flatten_fun_nokwargs

    try:
        from jax.interpreters import ad as ad_module
    except Exception:
        ad_module = None
    if ad_module is not None and not hasattr(jax, "ad"):
        jax.ad = ad_module

    for name in [
        "ShapedArray",
        "Primitive",
        "AbstractValue",
        "Jaxpr",
        "JaxprEqn",
        "Literal",
        "Value",
        "Var",
    ]:
        source = None
        if hasattr(core, name):
            source = core
        elif private_core is not None and hasattr(private_core, name):
            source = private_core
        if source is not None and not hasattr(jax, name):
            setattr(jax, name, getattr(source, name))

    if not hasattr(random_module, "KeyArray") and hasattr(jax, "Array"):
        random_module.KeyArray = jax.Array

    if "jax.util" not in sys.modules:
        try:
            from jax._src import util as util_module
        except Exception:
            util_module = None
        if util_module is not None:
            shim = types.ModuleType("jax.util")
            shim.safe_map = getattr(util_module, "safe_map", None)
            shim.safe_zip = getattr(util_module, "safe_zip", None)
            sys.modules["jax.util"] = shim

    try:
        import jax.interpreters.pxla as pxla
    except Exception:
        pxla = None
    if pxla is not None and not hasattr(pxla, "ShardedDeviceArray") and hasattr(jax, "Array"):
        pxla.ShardedDeviceArray = jax.Array

    import neural_tangents as nt

    return hnswlib, nt


def _normalization(data, parameters=None):
    """Column-wise min-max normalization that preserves NaNs."""
    norm_data = data.copy().astype(float)
    _, dim = norm_data.shape

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for idx in range(dim):
            column = norm_data[:, idx]
            observed = column[~np.isnan(column)]
            if observed.size == 0:
                min_val[idx] = 0.0
                max_val[idx] = 1.0
                norm_data[:, idx] = 0.0
                continue

            min_val[idx] = np.nanmin(column)
            column = column - min_val[idx]
            span = np.nanmax(column)
            if not np.isfinite(span) or span < 1e-6:
                span = 1.0
            max_val[idx] = span
            norm_data[:, idx] = column / (span + 1e-6)

        parameters = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
        for idx in range(dim):
            span = max(float(max_val[idx]), 1.0)
            norm_data[:, idx] = (norm_data[:, idx] - min_val[idx]) / (span + 1e-6)

    return norm_data, parameters


def _renormalization(norm_data, norm_parameters):
    renorm_data = norm_data.copy()
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]

    _, dim = norm_data.shape
    for idx in range(dim):
        renorm_data[:, idx] = renorm_data[:, idx] * (max_val[idx] + 1e-6) + min_val[idx]

    return renorm_data


def _rounding(imputed_data, data_x):
    rounded_data = imputed_data.copy()
    _, dim = data_x.shape

    for idx in range(dim):
        observed = data_x[~np.isnan(data_x[:, idx]), idx]
        if observed.size == 0:
            continue
        if len(np.unique(observed)) < 20:
            rounded_data[:, idx] = np.round(rounded_data[:, idx])

    return rounded_data


def _dist2sim(neigh_dist):
    neigh_dist = np.asarray(neigh_dist, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = 1.0 / neigh_dist

    inf_mask = np.isinf(dist)
    if inf_mask.any():
        rows = np.any(inf_mask, axis=1)
        dist[rows] = inf_mask[rows].astype(float)

    dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
    denom = np.sum(dist, axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return dist / denom


def _normalize_uncertainty(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if not np.isfinite(max_val - min_val) or (max_val - min_val) < 1e-12:
        return np.ones_like(values)
    return (values - min_val) / (max_val - min_val) + 1.0


def _sample_batch_index(total, batch_size):
    return np.random.permutation(total)[:batch_size]


def _build_kernel(dims):
    from neural_tangents import stax

    # 改成更贴近官方 main_hnsw_fast.py 的结构
    return stax.serial(
        stax.Dense(2 * dims),
        stax.Relu(),
        stax.Dense(dims),
        stax.Relu(),
        stax.Dense(1),
        stax.Sigmoid_like(),
    )


def _predict_with_nt(nt, kernel_fn, x_train, y_train, x_test):
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn,
        x_train,
        y_train.reshape(-1, 1),
        diag_reg=1e-4,
    )
    mean, cov = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
    return np.asarray(mean).reshape(-1), np.asarray(cov)


def _fallback_fill(x_train, y_train, x_test):
    if y_train.size == 0:
        return np.zeros(x_test.shape[0], dtype=float), None
    fill_value = float(np.mean(y_train))
    return np.full(x_test.shape[0], fill_value, dtype=float), None


def NOMI_impute(
    missing_data: pd.DataFrame,
    *,
    random_state: int = 42,
    k_candidate: int = 10,
    beta: float = 1.0,
    tau: float = 1.0,
    max_iter: int = 3,
    metric: str = "l2",
    batch_cap: int = 300,
) -> pd.DataFrame:
    """Impute missing feature values with an official-like NOMI core loop."""
    try:
        hnswlib, nt = _load_optional_dependencies()
    except ImportError as exc:
        raise RuntimeError(
            "NOMI requires the optional packages `hnswlib`, `jax`, `jaxlib`, and "
            "`neural-tangents`. Install them before running the NOMI imputer."
        ) from exc

    if missing_data.shape[1] <= 1:
        return missing_data.copy()

    np.random.seed(random_state)
    random.seed(random_state)

    # 保持你原始版本的接口习惯：默认第一列视为 label，其余列做插补
    label_column = missing_data.iloc[:, [0]].copy()
    feature_frame = missing_data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    feature_values = feature_frame.to_numpy(dtype=float)

    if not np.isnan(feature_values).any():
        return missing_data.copy()

    _, dim = feature_values.shape
    norm_data, norm_parameters = _normalization(feature_values)
    data_m = (~np.isnan(feature_values)).astype(np.uint8)
    data_m_imputed = data_m.copy()
    imputed_x = np.nan_to_num(norm_data, nan=0.0)

    _, _, kernel_fn = _build_kernel(dim)

    for iteration in range(max_iter):
        for target_dim in range(dim):
            x_wo_target = np.delete(imputed_x, target_dim, axis=1)
            observed_mask = data_m_imputed[:, target_dim].astype(bool)

            x_train = x_wo_target[observed_mask]
            y_train = imputed_x[observed_mask, target_dim]
            x_test = x_wo_target[~observed_mask]
            true_indices = np.where(~observed_mask)[0]

            if x_test.shape[0] == 0:
                continue

            if x_train.shape[0] <= 1 or x_train.shape[1] == 0:
                prediction, pred_cov = _fallback_fill(x_train, y_train, x_test)
            else:
                neighbors = min(int(k_candidate), x_train.shape[0])
                if neighbors <= 1:
                    prediction, pred_cov = _fallback_fill(x_train, y_train, x_test)
                else:
                    index = hnswlib.Index(space=metric, dim=x_train.shape[1])
                    index.init_index(max_elements=x_train.shape[0], ef_construction=200, M=16)
                    index.add_items(x_train)
                    index.set_ef(max(neighbors + 1, 32))

                    batch_size = min(batch_cap, x_train.shape[0])
                    batch_index = _sample_batch_index(x_train.shape[0], batch_size)
                    x_batch = x_train[batch_index]
                    y_batch = y_train[batch_index]

                    train_ind, train_dist = index.knn_query(x_batch, k=neighbors)
                    test_ind, test_dist = index.knn_query(x_test, k=neighbors)

                    if neighbors == 1:
                        prediction, pred_cov = _fallback_fill(x_train, y_train, x_test)
                    else:
                        train_dist = np.sqrt(train_dist)
                        test_dist = np.sqrt(test_dist)

                        train_sim = _dist2sim(train_dist[:, 1:])
                        test_sim = _dist2sim(test_dist[:, :-1])

                        x_batch_fit = train_sim * y_train[train_ind[:, 1:]]
                        x_test_fit = test_sim * y_train[test_ind[:, :-1]]

                        prediction, pred_cov = _predict_with_nt(
                            nt,
                            kernel_fn,
                            x_batch_fit,
                            y_batch,
                            x_test_fit,
                        )

            if iteration == 0:
                imputed_x[true_indices, target_dim] = prediction
                continue

            if iteration <= 3:
                if pred_cov is None:
                    pred_std = np.ones_like(prediction, dtype=float)
                else:
                    diag = np.diag(pred_cov) if pred_cov.ndim == 2 else pred_cov
                    pred_std = np.sqrt(np.clip(np.asarray(diag), a_min=0.0, a_max=None))
                    pred_std = _normalize_uncertainty(pred_std)
                    pred_std = np.nan_to_num(pred_std, nan=1.0, posinf=1.0, neginf=1.0)

                confident = np.where(pred_std <= tau)[0]
                if confident.size:
                    # 按你的要求改成切片赋值，贴近你给出的另一个复现版本
                    for idx in confident:
                        data_m_imputed[true_indices[idx]:, target_dim] = 1

                blend = beta / pred_std
                imputed_x[true_indices, target_dim] = (
                    (1.0 - blend) * imputed_x[true_indices, target_dim]
                    + blend * prediction
                )
                continue

            imputed_x[true_indices, target_dim] = prediction

    imputed = _renormalization(imputed_x, norm_parameters)
    imputed = _rounding(imputed, feature_values)
    imputed_frame = pd.DataFrame(
        imputed,
        index=missing_data.index,
        columns=missing_data.columns[1:],
    )
    restored_features = feature_frame.combine_first(imputed_frame)
    return pd.concat([label_column, restored_features], axis=1)
