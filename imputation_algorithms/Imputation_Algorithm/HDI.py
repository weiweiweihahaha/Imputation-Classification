import scipy
import pandas as pd
import numpy as np 


def _fallback_fill(df):
    result = df.copy()
    for column in result.columns:
        series = result[column]
        if not series.isnull().any():
            continue
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            fill_value = numeric.mean()
        else:
            modes = series.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else 0
        result[column] = series.fillna(fill_value)
    return result


def HDI(dataset):
    datos_Hot_Deck = dataset.copy()
    colums_names = list(dataset.columns)
    fallback_full = _fallback_fill(datos_Hot_Deck)

    rows_with_nan = datos_Hot_Deck[datos_Hot_Deck.isnull().any(axis=1)].copy()
    
    rows_without_nan = datos_Hot_Deck.dropna()

    if rows_with_nan.empty:
        return datos_Hot_Deck

    if rows_without_nan.empty:
        return _fallback_fill(datos_Hot_Deck)

    for i in range(len(rows_with_nan)): 

        row_null = rows_with_nan.iloc[i,:].isnull()
        row_aux = rows_with_nan.iloc[i,:]

        row_clean = row_aux.loc[~row_null].values.reshape(1,-1)
        if row_clean.shape[1] == 0:
            fallback_row = _fallback_fill(pd.DataFrame([row_aux], columns=dataset.columns)).iloc[0]
            rows_with_nan.iloc[i, :] = fallback_row.values
            continue

        matrix_without_nan = rows_without_nan.loc[:,~row_null].values
        if matrix_without_nan.shape[0] == 0 or matrix_without_nan.shape[1] == 0:
            fallback_values = fallback_full.loc[rows_with_nan.index[i]]
            for x_loop in range(len(colums_names)):
                if row_null.iloc[x_loop]:
                    rows_with_nan.iloc[i, x_loop] = fallback_values.iloc[x_loop]
            continue

        x = scipy.spatial.distance.cdist(row_clean, matrix_without_nan, 'euclidean')
        result = np.where(x == np.amin(x))

        for x_loop in range(len(colums_names)):
            if row_null.iloc[x_loop] == True:
                rows_with_nan.iloc[i,x_loop] = rows_without_nan.iloc[result[1][0],x_loop]    
    res = pd.concat([rows_with_nan, rows_without_nan], axis=0).sort_index()
    return res
