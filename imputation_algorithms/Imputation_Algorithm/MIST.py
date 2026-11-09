import os
import tempfile

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter


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


def _fallback_value(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        mean_val = numeric.mean()
        return 0.0 if pd.isna(mean_val) else mean_val
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return 0


def _is_numeric_like(series):
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    return int(numeric.notna().sum()) == int(non_null.shape[0])


def _prepare_mixed_mist_input(missing_data, *, label_col=0, max_factor_levels=50):
    original = pd.DataFrame(missing_data).copy().reset_index(drop=True)
    original.columns = list(range(original.shape[1]))
    prepared = original.copy()
    kept_columns = []
    factor_columns = []
    numeric_columns = []
    removed_columns = []
    fallback_values = {}

    for column in prepared.columns:
        series = prepared[column]
        if column == label_col:
            removed_columns.append(column)
            fallback_values[column] = None
            continue

        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        if non_null.empty or unique_count <= 1:
            removed_columns.append(column)
            fallback_values[column] = _fallback_value(series)
            continue

        if _is_numeric_like(series):
            numeric = pd.to_numeric(series, errors="coerce")
            std = numeric.dropna().std(ddof=0)
            if not np.isfinite(std) or std <= 0:
                removed_columns.append(column)
                fallback_values[column] = _fallback_value(series)
                continue
            prepared[column] = numeric
            kept_columns.append(column)
            numeric_columns.append(column)
            continue

        if unique_count > max_factor_levels:
            removed_columns.append(column)
            fallback_values[column] = _fallback_value(series)
            continue

        prepared[column] = series.astype("category")
        kept_columns.append(column)
        factor_columns.append(column)

    if not kept_columns:
        restored = original.copy()
        for column in restored.columns:
            if column == label_col:
                continue
            restored[column] = restored[column].fillna(_fallback_value(restored[column]))
        return restored, None

    metadata = {
        "original": original,
        "kept_columns": kept_columns,
        "r_columns": [f"V{idx}" for idx in range(len(kept_columns))],
        "removed_columns": removed_columns,
        "factor_columns": factor_columns,
        "numeric_columns": numeric_columns,
        "fallback_values": fallback_values,
        "label_col": label_col,
    }
    return prepared.loc[:, kept_columns].copy(), metadata


def _restore_mixed_mist_output(imputed, metadata):
    original = metadata["original"]
    restored = original.copy()
    imputed = pd.DataFrame(imputed).copy().reset_index(drop=True)
    kept_columns = metadata["kept_columns"]

    if imputed.shape != (original.shape[0], len(kept_columns)):
        raise ValueError(
            "MIST returned unexpected shape: "
            f"expected {(original.shape[0], len(kept_columns))}, got {imputed.shape}"
        )

    imputed.columns = kept_columns
    for column in kept_columns:
        restored[column] = imputed[column]

    for column in metadata["removed_columns"]:
        if column == metadata["label_col"]:
            restored[column] = original[column]
        else:
            restored[column] = original[column].fillna(metadata["fallback_values"][column])

    return _fill_remaining_missing(restored)


def MIST(missing_data, dataset_name="wine_0.1_2_MCAR_0", method="cart", m=5, workdir=None):
    mist_input, metadata = _prepare_mixed_mist_input(missing_data)
    if metadata is None:
        return mist_input

    r_input = mist_input.copy()
    r_input.columns = metadata["r_columns"]
    factor_r_columns = [
        metadata["r_columns"][metadata["kept_columns"].index(column)]
        for column in metadata["factor_columns"]
    ]

    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["missing_data"] = ro.conversion.py2rpy(r_input)

    fd, out_path = tempfile.mkstemp(prefix="mist_imputed_", suffix=".csv")
    os.close(fd)
    try:
        ro.globalenv["out_path"] = out_path
        ro.globalenv["factor_columns"] = ro.StrVector(factor_r_columns)
        ro.r(
            f"""
            library(mice)
            {f'setwd("{workdir}")' if workdir else ""}

            missing_data <- as.data.frame(missing_data)
            for (col in factor_columns) {{
                if (col %in% names(missing_data)) {{
                    missing_data[[col]] <- as.factor(missing_data[[col]])
                }}
            }}
            method_vec <- mice::make.method(missing_data)
            method_vec[colSums(is.na(missing_data)) == 0] <- ""
            method_vec[method_vec != ""] <- "{method}"
            pred <- mice::make.predictorMatrix(missing_data)
            diag(pred) <- 0
            cObs <- mice(
                missing_data,
                m={m},
                method=method_vec,
                predictorMatrix=pred,
                remove.collinear=TRUE,
                printFlag=FALSE
            )
            cObs_complete <- mice::complete(cObs, 1)
            write.table(cObs_complete, file=out_path, sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE, na="")
            """
        )
        result_df = pd.read_csv(out_path, header=None)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    if result_df.isna().any().any():
        result_df = _fill_remaining_missing(result_df)
    return _restore_mixed_mist_output(result_df, metadata)
