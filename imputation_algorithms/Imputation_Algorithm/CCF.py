import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter

_CCF_R_FUNCTION_LOADED = False


def _fallback_fill(df):
    filled = df.copy()
    for col in filled.columns:
        numeric = pd.to_numeric(filled[col], errors="coerce")
        if numeric.notna().any():
            filled[col] = numeric.fillna(numeric.mean())
        else:
            mode = filled[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else 0
            filled[col] = filled[col].fillna(fill_val)
    return filled


def _nominal_indices(df):
    indices = []
    for idx, dtype in enumerate(df.dtypes, start=1):
        if not pd.api.types.is_numeric_dtype(dtype):
            indices.append(idx)
    return indices


def _ensure_ccf_function():
    global _CCF_R_FUNCTION_LOADED
    if _CCF_R_FUNCTION_LOADED:
        return
    ro.r(
        """
        run_ccf <- function(missing_data, nominal_indices) {
            library(RoughSets)
            decision_table <- SF.asDecisionTable(
                dataset = missing_data,
                decision.attr = 1,
                indx.nominal = nominal_indices
            )
            indx <- MV.conceptClosestFit(decision_table)
            indx$val.NA
        }
        """
    )
    _CCF_R_FUNCTION_LOADED = True


def _to_dataframe(result):
    with localconverter(default_converter + pandas2ri.converter):
        converted = ro.conversion.rpy2py(result)
    if isinstance(converted, pd.DataFrame):
        return converted.copy()
    return pd.DataFrame(converted)


def CCF(missdata):
    source_df = missdata.copy()
    if not source_df.isna().any().any():
        return source_df

    nominal_indices = ro.IntVector(_nominal_indices(source_df))

    try:
        _ensure_ccf_function()
        with localconverter(default_converter + pandas2ri.converter):
            missing_data_r = ro.conversion.py2rpy(source_df)
        val_na = ro.r["run_ccf"](missing_data_r, nominal_indices)
        result_df = source_df.copy()
        val_na_df = _to_dataframe(val_na)
        if not val_na_df.empty and {"Row", "Col", "Value"}.issubset(val_na_df.columns):
            valid_rows = val_na_df["Value"].notna()
            if valid_rows.any():
                row_idx = pd.to_numeric(val_na_df.loc[valid_rows, "Row"], errors="coerce").astype("Int64") - 1
                col_idx = pd.to_numeric(val_na_df.loc[valid_rows, "Col"], errors="coerce").astype("Int64") - 1
                values = val_na_df.loc[valid_rows, "Value"].to_numpy()
                valid_index = row_idx.notna() & col_idx.notna()
                if valid_index.any():
                    row_values = row_idx[valid_index].astype(int).to_numpy()
                    col_values = col_idx[valid_index].astype(int).to_numpy()
                    result_array = result_df.to_numpy(copy=True)
                    result_array[row_values, col_values] = values[valid_index.to_numpy()]
                    result_df = pd.DataFrame(result_array, index=result_df.index, columns=result_df.columns)

        if result_df.isna().any().any():
            result_df = _fallback_fill(result_df)
        return result_df
    except Exception:
        return _fallback_fill(source_df)
