import numpy as np
import pandas as pd
from contextlib import contextmanager
from skfuzzy.cluster import cmeans
from sklearn.metrics import silhouette_samples
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter


def _fallback_fill(df):
    """Fallback imputation that keeps IFC callable when R-side Fclust fails."""
    filled = df.copy()
    for col in filled.columns:
        numeric = pd.to_numeric(filled[col], errors="coerce")
        if numeric.notna().any():
            filled[col] = numeric.fillna(numeric.mean())
        else:
            mode = filled[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else 0
            filled[col] = filled[col].fillna(fill_val)
    return filled.to_numpy()


@contextmanager
def _silence_r_console():
    try:
        from rpy2.rinterface_lib import callbacks
    except Exception:
        yield
        return

    def _discard_console_output(*args, **kwargs):
        return None

    original_print = callbacks.consolewrite_print
    original_warnerror = callbacks.consolewrite_warnerror
    original_showmessage = callbacks.showmessage
    callbacks.consolewrite_print = _discard_console_output
    callbacks.consolewrite_warnerror = _discard_console_output
    callbacks.showmessage = _discard_console_output
    try:
        yield
    finally:
        callbacks.consolewrite_print = original_print
        callbacks.consolewrite_warnerror = original_warnerror
        callbacks.showmessage = original_showmessage


def IFC(missing_data):
    source_df = missing_data.copy()
    source_array = source_df.to_numpy(dtype=float, copy=True)
    nan_mask = np.isnan(source_array)
    if not nan_mask.any():
        return source_array
    nan_rows, nan_cols = np.where(nan_mask)
    mean = np.nanmean(source_array, axis=0)
    imputed = np.where(nan_mask, mean, source_array)
    ST = 0.01
    AVT = ST + 1
    eta = 0
    etax = 100
    try:
        with _silence_r_console():
            ClustImpute = rpackages.importr('fclust')
            clust_impute_func = ClustImpute.Fclust
    except Exception:
        return _fallback_fill(source_df)

    while AVT > ST and eta < etax:
        AVT = 0
        FS = -1
        data_matrix = np.asarray(imputed, dtype=float)
        if data_matrix.shape[0] <= 2:
            return imputed
        max_clusters = min(10, max(2, data_matrix.shape[0] - 1))
        k_best = 2
        for k in range(2, max_clusters + 1):
            center, u, u0, d, jm, p, fpc = cmeans(data_matrix.T, m=2, c=k, error=0.5, maxiter=10000)
            labels = np.argmax(u, axis=0)
            silhouette_values = silhouette_samples(data_matrix, labels)
            sorted_u = np.sort(u, axis=0)
            gaps = sorted_u[-1] - sorted_u[-2]
            gap_sum = float(np.sum(gaps))
            if gap_sum <= 0:
                continue
            vs = float(np.dot(gaps, silhouette_values) / gap_sum)
            if vs > FS:
                FS = vs
                k_best = k
        try:
            with localconverter(default_converter + pandas2ri.converter):
                robjects.globalenv['dataframe'] = robjects.conversion.py2rpy(pd.DataFrame(imputed))
            with _silence_r_console():
                result = clust_impute_func(robjects.globalenv['dataframe'], k_best)
            u = np.array(result.rx2('U'))
            center = np.array(result.rx2('H'))
            estimates = u @ center
            previous_values = imputed[nan_rows, nan_cols]
            updated_values = estimates[nan_rows, nan_cols]
            denom = np.where(np.abs(previous_values) > 1e-12, np.abs(previous_values), 1.0)
            AVT = float(np.sum(np.abs(updated_values - previous_values) / denom))
            imputed[nan_rows, nan_cols] = updated_values
            eta += 1
        except Exception:
            return _fallback_fill(source_df)
    return imputed
