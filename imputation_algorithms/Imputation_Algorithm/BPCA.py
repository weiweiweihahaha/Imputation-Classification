import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter


def BPCA(missdata):
    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["missing_data"] = missdata

    ro.r(
        """
        library(pcaMethods)
        missing_data <- as.data.frame(missing_data)
        missing_data[is.nan(as.matrix(missing_data))] <- NA
        pc <- pca(t(missing_data), method = "bpca")
        cObs <- completeObs(pc)
        result <- t(cObs)
        """
    )

    result_r = ro.globalenv["result"]
    with localconverter(default_converter + numpy2ri.converter):
        result_np = ro.conversion.rpy2py(result_r)
    return pd.DataFrame(np.asarray(result_np))
