import pandas as pd
import numpy as np
import statsmodels.api as sm
def LOESS(df):
    def fill_na_loess(series, frac=0.1):
        x = series.dropna().index
        y = series.dropna().values
        lowess = sm.nonparametric.lowess(y, x, frac=frac)
        series.interpolate(method='linear', inplace=True)
        series.fillna(pd.Series(lowess[:, 1], index=lowess[:, 0]), inplace=True)
        return series
    for col in df.columns:
        df[col] = fill_na_loess(df[col])
    return df