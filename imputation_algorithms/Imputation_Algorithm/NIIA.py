import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


class NIIAImputer:
    def __init__(self, epsilon=1e-3, max_iter=100):
        self.epsilon = epsilon  # 收敛阈值
        self.max_iter = max_iter  # 最大迭代次数

    def _initial_imputation(self, data):
        """
        初始插补：离散变量使用众数，连续变量使用均值
        """
        for column in data.columns:
            col = data[column]
            if col.dtype == "object" or col.nunique(dropna=True) < 10:  # 离散变量
                modes = col.mode(dropna=True)
                if not modes.empty:
                    data[column] = col.fillna(modes.iloc[0])
            else:  # 连续变量
                numeric_col = pd.to_numeric(col, errors="coerce")
                data[column] = numeric_col.fillna(numeric_col.mean())
        return data

    def _impute_iteration(self, data):
        """
        对数据进行一次迭代插补
        """
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if data[column].dtype == "object" or data[column].nunique(dropna=True) < 10:  # 离散变量
                    modes = data[column].mode(dropna=True)
                    if not modes.empty:
                        data[column] = data[column].fillna(modes.iloc[0])
                else:  # 连续变量
                    # 使用最近邻插补
                    known_data = data[data[column].notnull()].copy()
                    missing_data = data[data[column].isnull()].copy()

                    if len(known_data) > 0 and len(missing_data) > 0:
                        X_train = known_data.drop(columns=[column]).apply(pd.to_numeric, errors="coerce")
                        y_train = pd.to_numeric(known_data[column], errors="coerce")
                        X_missing = missing_data.drop(columns=[column]).apply(pd.to_numeric, errors="coerce")

                        # KNNRegressor cannot accept NaN in X; use column means for temporary features.
                        fill_values = X_train.mean()
                        X_train = X_train.fillna(fill_values)
                        X_missing = X_missing.fillna(fill_values)

                        valid_y = ~y_train.isna()
                        if valid_y.sum() >= 2:
                            knn = KNeighborsRegressor(n_neighbors=min(3, int(valid_y.sum())), weights="distance")
                            knn.fit(X_train.loc[valid_y], y_train.loc[valid_y])
                            imputed_values = knn.predict(X_missing)
                            data.loc[missing_data.index, column] = imputed_values

        return data

    def fit_transform(self, data):
        """
        对缺失数据进行插补，直到收敛
        """
        data = data.copy()
        prev_data = data.copy()
        data = self._initial_imputation(data)

        t = 0
        while t < self.max_iter:
            t += 1
            data = self._impute_iteration(data)

            # 收敛检查，只对数值列进行检查
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_prev_data = prev_data.select_dtypes(include=[np.number])

            diff = np.sum((numeric_data.fillna(0) - numeric_prev_data.fillna(0)).abs().values.ravel())
            if diff < self.epsilon:
                break
            prev_data = data.copy()

        return data

# 示例使用
# if __name__ == "__main__":
#     data = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 5],
#         'B': [np.nan, 'yes', 'no', np.nan, 'yes'],
#         'C': [1.5, 2.5, np.nan, 4.5, 5.5]
#     })

#     imputer = NIIAImputer()
#     imputed_data = imputer.fit_transform(data)
#     print("插补后的数据:", imputed_data)
