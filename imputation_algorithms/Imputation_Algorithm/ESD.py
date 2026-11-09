import numpy as np


class ESDImputer:
    def __init__(self, epsilon=1e-3, max_iter=100):
        self.epsilon = epsilon  # 收敛阈值
        self.max_iter = max_iter  # 最大迭代次数
        self.mean = None
        self.cov_matrix = None

    def _fallback_mean(self, data):
        mean = np.nanmean(data, axis=0)
        if np.isnan(mean).any():
            global_mean = np.nanmean(data)
            if np.isnan(global_mean):
                global_mean = 0.0
            mean = np.where(np.isnan(mean), global_mean, mean)
        return mean

    def estimate_covariance(self, data):
        """
        使用 ECM 算法（期望条件最大化）估计协方差矩阵。
        """
        complete_data = data[~np.isnan(data).any(axis=1)]
        n_features = data.shape[1]

        # If there are not enough fully observed rows, fall back to column-wise
        # statistics computed from all available values.
        if complete_data.shape[0] >= 2:
            mean = np.mean(complete_data, axis=0)
            cov_matrix = np.cov(complete_data, rowvar=False)
        else:
            mean = self._fallback_mean(data)
            filled = np.where(np.isnan(data), mean, data)
            if filled.shape[0] >= 2:
                cov_matrix = np.cov(filled, rowvar=False)
            else:
                cov_matrix = np.eye(n_features, dtype=float)

        if np.isscalar(cov_matrix):
            cov_matrix = np.array([[float(cov_matrix)]], dtype=float)

        cov_matrix = np.atleast_2d(cov_matrix).astype(float)
        if cov_matrix.shape != (n_features, n_features):
            safe_diag = np.nan_to_num(np.diag(cov_matrix), nan=1.0, posinf=1.0, neginf=1.0)
            if safe_diag.size == 0:
                safe_diag = np.ones(n_features, dtype=float)
            cov_matrix = np.diag(np.resize(safe_diag, n_features))

        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        cov_matrix += np.eye(n_features, dtype=float) * 1e-8
        return mean, cov_matrix

    def expected_squared_distance(self, xi, xj, mean, cov_matrix):
        """
        计算两个含有缺失值的数据样本 xi 和 xj 之间的期望平方距离。
        """
        missing_indices_i = np.isnan(xi)
        missing_indices_j = np.isnan(xj)

        # 使用条件均值填充缺失值
        xi_imputed = np.copy(xi)
        xj_imputed = np.copy(xj)
        xi_imputed[missing_indices_i] = mean[missing_indices_i]
        xj_imputed[missing_indices_j] = mean[missing_indices_j]

        # 计算填充值之间的平方距离
        dist_squared = np.sum((xi_imputed - xj_imputed) ** 2)

        # 为缺失的组件添加方差项
        # 提取与缺失值对应的协方差矩阵对角线元素（方差项）
        s2_i = np.sum(np.diag(cov_matrix)[missing_indices_i])
        s2_j = np.sum(np.diag(cov_matrix)[missing_indices_j])

        return dist_squared + s2_i + s2_j

    def calculate_distance_matrix(self, data):
        """
        计算给定数据集的成对期望平方距离。
        """
        # 估计均值和协方差
        mean, cov_matrix = self.estimate_covariance(data)

        # 初始化距离矩阵
        n_samples = data.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        # 计算成对距离
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self.expected_squared_distance(data[i], data[j], mean, cov_matrix)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def fill_missing_values(self, data):
        """
        使用条件均值和协方差对数据中的缺失值进行填补。
        """
        # 估计均值和协方差
        mean, cov_matrix = self.estimate_covariance(data)

        # 填补缺失值
        filled_data = np.copy(data)
        for i in range(data.shape[0]):
            missing_indices = np.isnan(filled_data[i])
            if np.any(missing_indices):
                # 使用条件均值填补缺失值
                filled_data[i, missing_indices] = mean[missing_indices]

        return filled_data

    def fit_transform(self, data):
        """
        对缺失数据进行插补，直到收敛
        """
        data = np.asarray(data, dtype=float).copy()
        prev_data = np.copy(data)

        if data.ndim != 2:
            raise ValueError("ESDImputer expects a 2D array-like input.")

        # 初始插补
        data = self.fill_missing_values(data)

        t = 0
        while t < self.max_iter:
            t += 1
            # 进行一次插补迭代
            data = self.fill_missing_values(data)

            # 收敛检查
            diff = np.nansum(np.abs(data - prev_data))
            if diff < self.epsilon:
                break
            prev_data = np.copy(data)

        return data

if __name__ == "__main__":
    data = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 5.0],
        [np.nan, 3.0, 6.0],
        [4.0, 5.0, 6.0]
    ])

    imputer = ESDImputer(epsilon=1e-3, max_iter=100)
    imputer.fit_transform(data)
