import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from imputation_algorithms.Imputation_Algorithm._core import GINN
from imputation_algorithms.Imputation_Algorithm._utils import data2onehot
def GINN_impute(reader):
    cx_tr = reader.to_numpy()
    
    tr_indices = np.where(~np.isnan(cx_tr).any(axis=1))[0]
    te_indices = np.where(np.isnan(cx_tr).any(axis=1))[0]

    cat_cols = []
    num_cols = []

    for col_index in range(cx_tr.shape[1]):
        num_cols.append(col_index)
    mask_tr = np.zeros_like(cx_tr)
    #mask_te = np.zeros_like(cx_te)

    for index in range(cx_tr.shape[0]):
        for cols in range(cx_tr.shape[1]):
            if np.isnan(cx_tr[index,cols]):
                mask_tr[index,cols] = 0
            else:
                mask_tr[index,cols] = 1
    
    [oh_x, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols] = data2onehot(cx_tr, mask_tr, num_cols, cat_cols)

    oh_x_tr = oh_x[:cx_tr.shape[0],:]
    #oh_x_te = oh_x[cx_tr.shape[0]:,:]

    oh_mask_tr = oh_mask[:cx_tr.shape[0],:]
    oh_num_mask_tr = oh_mask[:cx_tr.shape[0],:]
    oh_cat_mask_tr = oh_mask[:cx_tr.shape[0],:]

    #oh_mask_te = oh_mask[cx_tr.shape[0]:,:]
    #oh_num_mask_te = oh_mask[cx_tr.shape[0]:,:]
    #oh_cat_mask_te = oh_mask[cx_tr.shape[0]:,:]

    scaler_tr = preprocessing.MinMaxScaler()
    oh_x_tr = scaler_tr.fit_transform(oh_x_tr)

    #scaler_te = preprocessing.MinMaxScaler()
    #oh_x_te = scaler_te.fit_transform(oh_x_te)

    imputer = GINN(oh_x_tr,oh_mask_tr,oh_num_mask_tr,oh_cat_mask_tr,oh_cat_cols,num_cols,cat_cols)

    imputer.fit()
    imputed_tr = scaler_tr.inverse_transform(imputer.transform())

    #imputer.add_data(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te)

    #imputed_te = imputer.transform()
    #imputed_te = scaler_te.inverse_transform(imputed_te[cx_te.shape[0]:])
    
    #imputer.add_data(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te)
    
    #imputer = GINN(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te,oh_cat_cols,num_cols,cat_cols)
    #imputer.fit()
    
    #imputed_te = imputer.transform()
    #imputed_te = scaler_te.inverse_transform(imputed_te[cx_tr.shape[0]:])

    array = np.zeros_like(cx_tr)
    num = 0
    for index in range(cx_tr.shape[0]):
        for cols in range(cx_tr.shape[1]):
            if np.isnan(cx_tr[index,cols]):
                array[index,cols] = imputed_tr[num,cols]
            else:
                array[index,cols] = cx_tr[index,cols]
        if index in te_indices:
            num = num + 1
    return array