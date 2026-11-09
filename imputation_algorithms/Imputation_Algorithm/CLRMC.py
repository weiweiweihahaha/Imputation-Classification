import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter

_ECLRMC_PACKAGE = None


def _get_eclrmc_package():
    global _ECLRMC_PACKAGE
    if _ECLRMC_PACKAGE is None:
        _ECLRMC_PACKAGE = rpackages.importr("ECLRMC")
    return _ECLRMC_PACKAGE


def _to_dataframe(result):
    with localconverter(default_converter + pandas2ri.converter):
        converted = ro.conversion.rpy2py(result)
    if isinstance(converted, pd.DataFrame):
        return converted.copy()
    return pd.DataFrame(np.asarray(converted))


def _fill_remaining_missing(df):
    filled = df.copy()
    for col in filled.columns:
        series = filled[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > 0:
            mean_val = numeric.mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            filled[col] = numeric.fillna(mean_val)
        else:
            modes = series.mode(dropna=True)
            if not modes.empty:
                filled[col] = series.fillna(modes.iloc[0])
            else:
                filled[col] = series.fillna(0)
    return filled


def CLRMC(missing_data, dataset_name="iris_0.1_2_MCAR_0", beta=0.2, workdir=None):
    with localconverter(default_converter + pandas2ri.converter):
        missing_data_r = ro.conversion.py2rpy(missing_data)

    if workdir:
        ro.r["setwd"](workdir)

    package = _get_eclrmc_package()
    result_df = _to_dataframe(package.CLRMC(ro.r["as.matrix"](missing_data_r), beta=beta))

    if result_df.isna().any().any():
        result_df = _fill_remaining_missing(result_df)
    return result_df
