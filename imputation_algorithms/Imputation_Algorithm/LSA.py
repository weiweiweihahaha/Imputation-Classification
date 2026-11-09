import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter


def LSA(missdata):
    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["missing_data"] = missdata

    ro.r(
        """
        library(missMethods)
        missing_data <- as.data.frame(missing_data)
        missing_data[is.nan(as.matrix(missing_data))] <- NA
        result <- impute_LS_adaptive(missing_data)
        """
    )

    result_r = ro.globalenv["result"]
    try:
        with localconverter(default_converter + numpy2ri.converter):
            result_np = ro.conversion.rpy2py(result_r)
        result_df = pd.DataFrame(np.asarray(result_np))
    except Exception:
        with localconverter(default_converter + pandas2ri.converter):
            result_df = ro.conversion.rpy2py(result_r)
        if not isinstance(result_df, pd.DataFrame):
            result_df = pd.DataFrame(result_df)

    # `missMethods::impute_LS_adaptive` already returns rows x columns.
    return result_df
