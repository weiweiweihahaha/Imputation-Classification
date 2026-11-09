import pandas as pd


def normalize_frame(df):
    normalized = df.copy()
    normalized.columns = list(range(normalized.shape[1]))
    normalized = normalized.reset_index(drop=True)
    return normalized


def fill_missing_values(df):
    filled = df.copy()
    for col in filled.columns:
        if col == 0:
            continue
        if (
            pd.api.types.is_object_dtype(filled[col].dtype)
            or pd.api.types.is_string_dtype(filled[col].dtype)
            or isinstance(filled[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(filled[col].dtype)
        ):
            modes = filled[col].mode(dropna=True)
            if not modes.empty:
                filled[col] = filled[col].fillna(modes.iloc[0])
            continue
        numeric = pd.to_numeric(filled[col], errors="coerce")
        if numeric.notna().any():
            filled[col] = numeric.fillna(numeric.mean())
        else:
            modes = filled[col].mode(dropna=True)
            if not modes.empty:
                filled[col] = filled[col].fillna(modes.iloc[0])
    return filled
