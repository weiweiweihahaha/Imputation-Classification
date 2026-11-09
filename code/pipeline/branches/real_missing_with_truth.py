import numpy as np
import pandas as pd

from code.config.dataset_profiles import DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH
from code.core.data_io import load_real_missing_data, load_real_truth_data

from .common import fill_missing_values, normalize_frame


def _finite_label_mask(labels):
    numeric = pd.to_numeric(labels, errors="coerce")
    return numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float))


def _present_label_mask(labels):
    as_text = labels.astype("string")
    return labels.notna() & as_text.str.strip().ne("")


def _normalize_labels(ground_df, missing_df):
    numeric_ground = pd.to_numeric(ground_df[0], errors="coerce")
    present_ground_mask = _present_label_mask(ground_df[0])
    numeric_valid_mask = numeric_ground.notna() & np.isfinite(numeric_ground.to_numpy(dtype=float))

    if bool(present_ground_mask.any()) and bool((numeric_valid_mask == present_ground_mask).all()):
        normalized_ground = ground_df.copy()
        normalized_missing = missing_df.copy()
        normalized_ground[0] = numeric_ground
        normalized_missing[0] = pd.to_numeric(missing_df[0], errors="coerce")
        return normalized_ground, normalized_missing, _finite_label_mask(normalized_ground[0]), _finite_label_mask(
            normalized_missing[0]
        )

    valid_label_mask = present_ground_mask
    categories = sorted(ground_df.loc[valid_label_mask, 0].astype(str).unique().tolist())
    label_mapping = {label: idx for idx, label in enumerate(categories)}

    normalized_ground = ground_df.copy()
    normalized_missing = missing_df.copy()
    normalized_ground[0] = ground_df[0].astype(str).map(label_mapping)
    normalized_missing[0] = missing_df[0].astype(str).map(label_mapping)
    missing_label_valid_mask = normalized_missing[0].notna()
    return normalized_ground, normalized_missing, valid_label_mask, missing_label_valid_mask


def prepare_real_missing_with_truth_data(args, dataset_name):
    ground_df = normalize_frame(load_real_truth_data(args))
    missing_df = normalize_frame(load_real_missing_data(args))
    if ground_df.shape[0] != missing_df.shape[0]:
        raise ValueError(
            f"Observed/truth row mismatch for dataset={args.dataset}: "
            f"observed_rows={missing_df.shape[0]} truth_rows={ground_df.shape[0]}"
        )

    ground_df, missing_df, valid_label_mask, missing_label_valid_mask = _normalize_labels(ground_df, missing_df)
    label_fill_mask = valid_label_mask & ~missing_label_valid_mask
    if bool(label_fill_mask.any()):
        missing_df.loc[label_fill_mask, 0] = ground_df.loc[label_fill_mask, 0]

    args.valid_real_missing_truth_rows = valid_label_mask[valid_label_mask].index.to_list()
    if not bool(valid_label_mask.all()):
        ground_df = ground_df.loc[valid_label_mask].reset_index(drop=True)
        missing_df = missing_df.loc[valid_label_mask].reset_index(drop=True)

    # Some oracle bundles still contain sparse feature-level gaps. Keep the
    # raw truth table for masked-error/structure metrics, but hand classifiers
    # a feature-complete reference table so sklearn validation can proceed.
    reference_clean_df = fill_missing_values(ground_df)
    missing_filled_df = fill_missing_values(missing_df)
    return {
        "dataset_category": DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH,
        "has_ground_truth": True,
        "ground_df": ground_df,
        "missing_df": missing_df,
        "missing_filled_df": missing_filled_df,
        "reference_clean_df": reference_clean_df,
        "reference_clean_source": "data/Real_missing_truth/<dataset>.csv",
    }
