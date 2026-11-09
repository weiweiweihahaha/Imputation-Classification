from curses import noecho
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import as_float_array
from sklearn.preprocessing import OrdinalEncoder

def is_float(value):
    '''
    Verify if value's dtype is float.
    '''
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True

is_float_vectorized = np.vectorize(is_float)

def is_nan(value):
    '''
    Verify if a value is NaN.
    '''
    return value != value

is_nan_vectorized = np.vectorize(is_nan)

def is_string(value):
    '''
    Verify if value's dtype is string.
    '''
    return isinstance(value, str)

is_string_vectorized = np.vectorize(is_string)

def is_not_string(value):
    '''
    Verify if value's dtype isn't string.
    '''
    if not isinstance(value, str) and value == value:
        return True
    else:
        return False
    
is_not_string_vectorized = np.vectorize(is_not_string)


def encode_categories(X, categorical_features_index):
    '''
    Given an X and an array of categorical features for X,
    the function iterates over all categorical features to
    get original and new values of OrdinalEncoded features.
    It also gets categorical features that needed to be casted
    as string for containing both float and string dtypes.
    '''
    encoded_categories = {}
    casted_as_string_categories = []
    for column_index in categorical_features_index:
        ordinal_encoder = OrdinalEncoder()
        if np.any(is_string_vectorized(X[:,column_index])) and np.any(is_not_string_vectorized(X[:,column_index])):
            X[:,column_index] = ordinal_encoder.fit_transform(X[np.invert(is_nan_vectorized(X[:,column_index])),column_index].reshape(-1,1).astype(str)).reshape(1,-1)
            casted_as_string_categories.append(column_index)
        else:
            X[:,column_index] = ordinal_encoder.fit_transform(X[:,column_index].reshape(-1,1)).reshape(1,-1)
        context_categories = [value for value in ordinal_encoder.categories_[0] if not is_nan(value)]
        encoded_categories[column_index] = dict(zip(range(len(context_categories)), context_categories))
    return X, encoded_categories, casted_as_string_categories


def get_inferred_categories_index(self):
    '''
    Get an array containing the index of the categorical features
    that were inferred by XGBImputer.
    '''
    inferred_features_index = np.hstack([np.nonzero(self.isnan_array.sum(axis=0))[0].reshape(-1,1), self.isnan_array.sum(axis=0)[self.isnan_array.sum(axis=0) > 0].reshape(-1,1)])
    inferred_features_index = inferred_features_index[inferred_features_index[:, 1].argsort()]
    return inferred_features_index[:,0]

class XGBImputer():
    def __init__(self, categorical_index=[], **kwargs):
        self.categorical_index = np.array(categorical_index)
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}
    
    def transform(self, X):

        if type(X) != np.ndarray:
            X = np.array(X)

        self.columns_index = np.arange(X.shape[1])
        X, self.encoded_categories, self.casted_as_string_categories = encode_categories(X, self.categorical_index)

        X = as_float_array(X)
        self.isnan_array = np.isnan(X)
        
        self.inferred_features_index = get_inferred_categories_index(self)
        self.inferred_categorical_features_index = np.intersect1d(self.inferred_features_index, self.categorical_index)
        self.numerical_features_index = np.setdiff1d(self.columns_index, self.categorical_index)
        self.inferred_numerical_features_index = np.intersect1d(self.inferred_features_index, self.numerical_features_index)
        
        mean_simple_imputer = SimpleImputer(strategy='mean')
        X[:,self.numerical_features_index] = mean_simple_imputer.fit_transform(X[:,self.numerical_features_index])
        
        mode_simple_imputer = SimpleImputer(strategy='most_frequent')
        if self.categorical_index.size > 0:
            X[:,self.categorical_index] = mode_simple_imputer.fit_transform(X[:,self.categorical_index])
        
        Ximp = X.copy()
        
        iterations_counter = 1
        gamma_inferred_categorical_features_old = np.inf
        gamma_inferred_categorical_features_new = 0
        gamma_inferred_numerical_features_old = np.inf
        gamma_inferred_numerical_features_new = 0

        while (gamma_inferred_categorical_features_new < gamma_inferred_categorical_features_old or gamma_inferred_numerical_features_new < gamma_inferred_numerical_features_old):

            Ximp_old = Ximp.copy()

            if iterations_counter > 1:
                gamma_inferred_categorical_features_old = gamma_inferred_categorical_features_new
                gamma_inferred_numerical_features_old = gamma_inferred_numerical_features_new

            for column_index in self.inferred_features_index:

                if column_index in self.categorical_index:
                    xgb = XGBClassifier(subsample=0.7, use_label_encoder=False, verbosity=0)
                else:
                    xgb = XGBRegressor(subsample=0.7, verbosity=0)
                
                if self.kwargs:
                    xgb.set_params(**self.kwargs)

                X_obs = np.delete(Ximp, column_index, axis=1)[np.invert(self.isnan_array[:,column_index]),:]
                y_obs = Ximp[np.invert(self.isnan_array[:,column_index]),column_index]
                X_mis = np.delete(Ximp, column_index, axis=1)[self.isnan_array[:,column_index],:]
                
                one_hot_encoded_features_index = np.hstack([self.categorical_index[self.categorical_index < column_index],self.categorical_index[self.categorical_index > column_index]-1])
                ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), one_hot_encoded_features_index)])

                X_obs = ct.fit_transform(X_obs)
                X_mis = ct.transform(X_mis)

                xgb.fit(X_obs, y_obs)

                y_mis = xgb.predict(X_mis)

                Ximp[self.isnan_array[:,column_index],column_index] = y_mis

            if self.inferred_categorical_features_index.size != 0:
                gamma_inferred_categorical_features_new = np.sum(Ximp[:,self.inferred_categorical_features_index] != Ximp_old[:,self.inferred_categorical_features_index])/self.inferred_categorical_features_index.size
            gamma_inferred_numerical_features_new = np.sum((Ximp[:,self.inferred_numerical_features_index] - Ximp_old[:,self.inferred_numerical_features_index])**2)/np.sum((Ximp[:, self.inferred_numerical_features_index]) ** 2)
            
            iterations_counter += 1
            
        return Ximp
