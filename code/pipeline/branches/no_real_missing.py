from code.config.dataset_profiles import DATASET_CATEGORY_NO_REAL_MISSING
from code.core.data_io import load_data

from .common import fill_missing_values, normalize_frame


def prepare_no_real_missing_data(args):
    ground_df = normalize_frame(load_data(args, args.missing_mechanism, "Ground"))
    missing_df = normalize_frame(load_data(args, args.missing_mechanism, imputation=None))
    missing_filled_df = fill_missing_values(missing_df)
    return {
        "dataset_category": DATASET_CATEGORY_NO_REAL_MISSING,
        "has_ground_truth": True,
        "ground_df": ground_df,
        "missing_df": missing_df,
        "missing_filled_df": missing_filled_df,
        "reference_clean_df": ground_df,
        "reference_clean_source": "data/Ground_truth_data/<dataset>.csv",
    }
