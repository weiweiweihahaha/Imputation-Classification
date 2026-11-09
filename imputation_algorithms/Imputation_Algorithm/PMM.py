import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter


_RIDGE_SEQUENCE = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def _column_fallback_value(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        mean_val = numeric.mean()
        return 0.0 if pd.isna(mean_val) else mean_val

    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return 0


def _is_constant_for_mice(series):
    return series.dropna().nunique(dropna=True) <= 1


def _is_high_cardinality_or_mixed_for_mice(series, *, max_categories=20):
    non_null = series.dropna()
    if non_null.empty:
        return True

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        return False

    return True


def _numeric_collinearity_drop_columns(frame, *, rank_tol=1e-8, corr_threshold=0.999999):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    candidate_columns = []
    candidate_series = []
    standardized_columns = []

    for column in numeric.columns:
        series = numeric[column]
        if series.notna().sum() < 2:
            continue
        mean_value = series.mean()
        filled = series.fillna(mean_value)
        std_value = filled.std(ddof=0)
        if not np.isfinite(std_value) or std_value <= rank_tol:
            continue
        candidate_columns.append(column)
        candidate_series.append(series)
        standardized_columns.append(((filled - mean_value) / std_value).to_numpy(dtype=float))

    kept_vectors = []
    kept_series = []
    dropped = []
    current_rank = 0
    for column, series, vector in zip(candidate_columns, candidate_series, standardized_columns):
        if kept_vectors:
            pairwise_collinear = False
            for existing in kept_series:
                mask = series.notna() & existing.notna()
                if mask.sum() < 2:
                    continue
                left = series.loc[mask].to_numpy(dtype=float)
                right = existing.loc[mask].to_numpy(dtype=float)
                if np.std(left) <= rank_tol or np.std(right) <= rank_tol:
                    continue
                corr = np.corrcoef(left, right)[0, 1]
                if np.isfinite(corr) and abs(corr) >= corr_threshold:
                    pairwise_collinear = True
                    break
            if pairwise_collinear:
                dropped.append(column)
                continue

            kept_matrix = np.column_stack(kept_vectors)
            correlations = np.abs(np.corrcoef(kept_matrix, vector, rowvar=False)[-1, :-1])
            if np.nanmax(correlations) >= corr_threshold:
                dropped.append(column)
                continue
            candidate_matrix = np.column_stack(kept_vectors + [vector])
        else:
            candidate_matrix = vector.reshape(-1, 1)

        next_rank = np.linalg.matrix_rank(candidate_matrix, tol=rank_tol)
        if next_rank <= current_rank:
            dropped.append(column)
            continue
        kept_vectors.append(vector)
        kept_series.append(series)
        current_rank = next_rank

    return dropped


def _prepare_mice_family_input(frame):
    original = pd.DataFrame(frame).copy().reset_index(drop=True)
    original.columns = list(frame.columns)
    protected_columns = [0] if 0 in original.columns else []
    constant_columns = [
        column for column in original.columns
        if column not in protected_columns and _is_constant_for_mice(original[column])
    ]
    mixed_columns = [
        column for column in original.columns
        if column not in protected_columns
        and column not in constant_columns
        and _is_high_cardinality_or_mixed_for_mice(original[column])
    ]
    reduced = original.drop(columns=protected_columns + constant_columns + mixed_columns)
    collinear_columns = _numeric_collinearity_drop_columns(reduced)
    kept_columns = [
        column for column in original.columns
        if column not in set(protected_columns).union(constant_columns, mixed_columns, collinear_columns)
    ]
    removed_columns = [column for column in original.columns if column not in kept_columns]
    return original.loc[:, kept_columns].copy(), {
        "original_columns": list(original.columns),
        "kept_columns": kept_columns,
        "removed_columns": removed_columns,
        "fallback_values": {
            column: _column_fallback_value(original[column])
            for column in removed_columns
        },
    }


def _restore_mice_family_output(original, imputed, metadata, *, keep_label=True):
    original = pd.DataFrame(original).copy().reset_index(drop=True)
    original.columns = metadata["original_columns"]
    kept_columns = metadata["kept_columns"]
    restored = pd.DataFrame(index=original.index, columns=metadata["original_columns"])

    if kept_columns:
        imputed_frame = pd.DataFrame(imputed).copy()
        if imputed_frame.shape == (len(kept_columns), len(original)):
            imputed_frame = imputed_frame.T
        if imputed_frame.shape != (len(original), len(kept_columns)):
            raise ValueError(
                "PMM imputed shape mismatch after pruning: "
                f"expected {(len(original), len(kept_columns))}, got {imputed_frame.shape}"
            )
        imputed_frame = imputed_frame.reset_index(drop=True)
        imputed_frame.columns = kept_columns
        for column in kept_columns:
            restored[column] = imputed_frame[column]

    for column in metadata["removed_columns"]:
        restored[column] = original[column].fillna(metadata["fallback_values"][column])

    result = original.combine_first(restored)
    if keep_label and 0 in result.columns:
        result[0] = original[0]
    return _fill_remaining_missing(result)


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


def PMM(missdata):
    original_missing_data = pd.DataFrame(missdata).copy()
    missing_data, mice_metadata = _prepare_mice_family_input(original_missing_data)
    if missing_data.shape[1] == 0 or not missing_data.isnull().any().any():
        return _restore_mice_family_output(
            original_missing_data,
            missing_data,
            mice_metadata,
            keep_label=True,
        )

    r_input = missing_data.copy()
    r_input.columns = [f"V{i}" for i in range(r_input.shape[1])]
    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["missing_data"] = ro.conversion.py2rpy(r_input)

    fd, out_path = tempfile.mkstemp(prefix="pmm_imputed_", suffix=".csv")
    os.close(fd)
    ro.globalenv["out_path"] = out_path

    def _run_mice_with_ridge(ridge_value):
        ro.r(
            f"""
            library(mice)
            missing_data <- as.data.frame(missing_data)
            cObs <- mice(
                missing_data,
                m = 5,
                method = "pmm",
                ridge = {ridge_value},
                remove.collinear = TRUE,
                printFlag = FALSE
            )
            result_data <- mice::complete(cObs, 1)
            write.table(
                result_data,
                file=out_path,
                sep=",",
                row.names=FALSE,
                col.names=FALSE,
                quote=FALSE,
                na=""
            )
            """
        )

    try:
        last_error = None
        for ridge in _RIDGE_SEQUENCE:
            try:
                _run_mice_with_ridge(ridge)
                break
            except RRuntimeError as err:
                last_error = err
                warnings.warn(
                    f"mice PMM failed with ridge={ridge}: {err}",
                    RuntimeWarning,
                )
        else:
            warnings.warn(
                "PMM imputation failed after ridge escalation; falling back to "
                "single-column mean/mode fills for the pruned input.",
                RuntimeWarning,
            )
            result_df = _fill_remaining_missing(missing_data)
            return _restore_mice_family_output(
                original_missing_data,
                result_df,
                mice_metadata,
                keep_label=True,
            )

        result_df = pd.read_csv(out_path, header=None)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    if result_df.isna().any().any():
        result_df = _fill_remaining_missing(result_df)
    return _restore_mice_family_output(
        original_missing_data,
        result_df,
        mice_metadata,
        keep_label=True,
    )
