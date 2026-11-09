import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from code.core.compute_backend import normalize_compute_backend

_FLOAT32_MAX = float(np.finfo(np.float32).max)


class DenseFloat32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(128, 128),
        activation="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=300,
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=40,
        device="cuda",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.device = device

    def _lazy_imports(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "TorchMLPClassifier requires the `torch` package. "
                "Install PyTorch to use the GPU MLP backend."
            ) from exc
        return torch, nn, DataLoader, TensorDataset

    def _normalize_hidden_sizes(self):
        if isinstance(self.hidden_layer_sizes, int):
            return (int(self.hidden_layer_sizes),)
        return tuple(int(width) for width in self.hidden_layer_sizes)

    def _build_network(self, nn_module, input_dim, output_dim):
        layers = []
        in_dim = int(input_dim)
        activation_cls = nn_module.ReLU if self.activation == "relu" else nn_module.Tanh
        for hidden_dim in self._normalize_hidden_sizes():
            layers.append(nn_module.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            in_dim = hidden_dim
        layers.append(nn_module.Linear(in_dim, int(output_dim)))
        return nn_module.Sequential(*layers)

    def _resolve_device(self, torch_module):
        requested = str(self.device or "cpu").lower()
        if requested == "cuda":
            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "TorchMLPClassifier was asked to run on CUDA, but "
                    "torch.cuda.is_available() is False."
                )
            return torch_module.device("cuda")
        return torch_module.device("cpu")

    def _prepare_validation_split(self, X_np, y_np):
        if not self.early_stopping or self.validation_fraction <= 0:
            return X_np, y_np, None, None
        if len(X_np) < 10:
            return X_np, y_np, None, None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_np,
                y_np,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_np,
            )
        except ValueError:
            return X_np, y_np, None, None
        return X_train, y_train, X_val, y_val

    def fit(self, X, y):
        torch, nn_module, DataLoader, TensorDataset = self._lazy_imports()
        rng_seed = int(self.random_state)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)

        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError("TorchMLPClassifier expects a 2D feature matrix.")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(np.asarray(y))
        classes = label_encoder.classes_

        train_X_np, train_y_np, val_X_np, val_y_np = self._prepare_validation_split(X_np, y_encoded)

        device = self._resolve_device(torch)
        model = self._build_network(nn_module, X_np.shape[1], len(classes)).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.learning_rate_init),
            weight_decay=float(self.alpha),
        )
        criterion = nn_module.CrossEntropyLoss()

        train_dataset = TensorDataset(
            torch.from_numpy(train_X_np),
            torch.from_numpy(train_y_np.astype(np.int64, copy=False)),
        )
        batch_size = max(1, min(int(self.batch_size), len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        best_state = None
        best_val_loss = None
        epochs_without_improve = 0

        for _ in range(int(self.max_iter)):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if val_X_np is None:
                continue

            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(val_X_np).to(device))
                val_targets = torch.from_numpy(val_y_np.astype(np.int64, copy=False)).to(device)
                val_loss = float(criterion(val_logits, val_targets).detach().cpu().item())

            if best_val_loss is None or val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= int(self.n_iter_no_change):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model.eval()
        self.device_ = device
        self.label_encoder_ = label_encoder
        self.classes_ = classes
        self.n_features_in_ = int(X_np.shape[1])
        return self

    def _predict_logits(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPClassifier must be fitted before prediction.")
        torch, _, _, _ = self._lazy_imports()
        X_np = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_np).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
        return logits.detach().cpu().numpy()

    def predict_proba(self, X):
        torch, _, _, _ = self._lazy_imports()
        logits = self._predict_logits(X)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        encoded = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def decision_function(self, X):
        return self._predict_logits(X)


def _resolve_training_backend(compute_backend):
    normalized = normalize_compute_backend(compute_backend)
    return "cpu" if normalized == "auto" else normalized


def _resolve_classifier_backend(classify, compute_backend):
    backend = _resolve_training_backend(compute_backend)
    # MLP is intentionally kept on the sklearn CPU path so methodology,
    # tuning space, and runtime behavior stay consistent across runs.
    if classify == "MLP":
        return "cpu"
    return backend


def _infer_column_groups(X_train):
    categorical_cols = []
    numeric_cols = []

    for position, col in enumerate(X_train.columns):
        series = X_train[col]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            categorical_cols.append(position)
        else:
            numeric_cols.append(position)

    return numeric_cols, categorical_cols


def _dataset_has_missing_values(X_train):
    if hasattr(X_train, "isnull"):
        return bool(X_train.isnull().to_numpy().any())
    try:
        return bool(np.isnan(np.asarray(X_train)).any())
    except TypeError:
        return False


def _format_max_abs(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "nan"
    return f"{float(np.max(np.abs(finite))):.6g}"


def _validate_target_labels(y, *, context):
    if hasattr(y, "isnull") and bool(y.isnull().any()):
        raise ValueError(f"{context} contains missing labels.")

    values = np.asarray(y)
    if not np.issubdtype(values.dtype, np.number):
        return

    numeric = values.astype(float, copy=False)
    if np.isfinite(numeric).all():
        return

    bad_count = int((~np.isfinite(numeric)).sum())
    raise ValueError(f"{context} contains {bad_count} non-finite labels.")


def validate_feature_frame(X, *, context="feature matrix"):
    if isinstance(X, pd.DataFrame):
        numeric_cols, _ = _infer_column_groups(X)
        issues = []
        for position in numeric_cols:
            column = X.columns[position]
            numeric = pd.to_numeric(X.iloc[:, position], errors="coerce").to_numpy(dtype=float)
            non_finite_count = int((~np.isfinite(numeric)).sum())
            too_large_count = int((np.isfinite(numeric) & (np.abs(numeric) > _FLOAT32_MAX)).sum())
            if non_finite_count == 0 and too_large_count == 0:
                continue
            issues.append(
                f"col={column} non_finite={non_finite_count} "
                f"float32_overflow={too_large_count} max_abs={_format_max_abs(numeric)}"
            )
        if not issues:
            return
        detail = "; ".join(issues[:5])
        if len(issues) > 5:
            detail += f"; +{len(issues) - 5} more columns"
        raise ValueError(
            f"{context} contains values that are not safe for sklearn validation "
            f"(float32 limit={_FLOAT32_MAX:.3e}): {detail}"
        )

    values = np.asarray(X)
    if not np.issubdtype(values.dtype, np.number):
        return

    numeric = values.astype(float, copy=False)
    non_finite_count = int((~np.isfinite(numeric)).sum())
    too_large_count = int((np.isfinite(numeric) & (np.abs(numeric) > _FLOAT32_MAX)).sum())
    if non_finite_count == 0 and too_large_count == 0:
        return
    raise ValueError(
        f"{context} contains values that are not safe for sklearn validation "
        f"(float32 limit={_FLOAT32_MAX:.3e}, non_finite={non_finite_count}, "
        f"float32_overflow={too_large_count}, max_abs={_format_max_abs(numeric)})"
    )


def _build_preprocessor(X_train, *, compute_backend="cpu"):
    compute_backend = _resolve_training_backend(compute_backend)
    numeric_cols, categorical_cols = _infer_column_groups(X_train)
    transformers = []
    has_missing_values = _dataset_has_missing_values(X_train)
    use_fast_clean_path = compute_backend == "gpu" and not has_missing_values

    if numeric_cols:
        numeric_steps = []
        if not use_fast_clean_path:
            numeric_steps.append(("imputer", SimpleImputer(strategy="mean")))
        numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "num",
                Pipeline(steps=numeric_steps),
                numeric_cols,
            )
        )
    if categorical_cols:
        categorical_steps = []
        if not use_fast_clean_path:
            categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        categorical_steps.append(
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            )
        )
        transformers.append(
            (
                "cat",
                Pipeline(steps=categorical_steps),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns are available for classifier training.")

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def _build_classifier_spec(classify, *, compute_backend="cpu", gpu_device=0):
    compute_backend = _resolve_classifier_backend(classify, compute_backend)
    if classify == "DecisionTree":
        estimator = DecisionTreeClassifier(random_state=40)
        grid = {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": [None, "sqrt", "log2"],
        }
        return estimator, grid
    if classify == "LogisticRegression":
        estimator = LogisticRegression(max_iter=5000, random_state=40)
        grid = {
            "model__solver": ["lbfgs", "saga"],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        }
        return estimator, grid
    if classify == "RandomForest":
        estimator = RandomForestClassifier(random_state=40, n_jobs=1)
        grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
        }
        return estimator, grid
    if classify == "NaiveBayes":
        estimator = GaussianNB()
        grid = {
            "model__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        }
        return estimator, grid
    if classify == "KNN":
        estimator = KNeighborsClassifier()
        grid = {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["minkowski", "manhattan"],
        }
        return estimator, grid
    if classify == "Adaboost":
        estimator = AdaBoostClassifier(random_state=40)
        grid = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0],
        }
        return estimator, grid
    if classify == "SVM":
        estimator = SVC(probability=True, random_state=40)
        grid = [
            {
                "model__kernel": ["rbf"],
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            {
                "model__kernel": ["linear"],
                "model__C": [0.1, 1.0, 10.0],
            },
        ]
        return estimator, grid
    if classify == "MLP":
        estimator = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=40,
        )
        grid = {
            "model__activation": ["relu", "tanh"],
            "model__solver": ["adam"],
            "model__learning_rate_init": [1e-4, 1e-3, 1e-2],
            "model__alpha": [1e-5, 1e-4, 1e-3],
        }
        return estimator, grid
    if classify == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("LightGBM classifier requires `lightgbm` package. Please install it first.") from exc

        class _NumpyLGBMClassifier(LGBMClassifier):
            def _to_numpy(self, X):
                return X.to_numpy() if hasattr(X, "to_numpy") else X

            def fit(self, X, y, **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    return super().fit(self._to_numpy(X), y, **kwargs)

            def predict(self, X, **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    return super().predict(self._to_numpy(X), **kwargs)

            def predict_proba(self, X, **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    return super().predict_proba(self._to_numpy(X), **kwargs)

            def decision_function(self, X, **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    return super().decision_function(self._to_numpy(X), **kwargs)

        lightgbm_kwargs = {
            "random_state": 40,
            "verbosity": -1,
            "n_jobs": 1,
        }
        if compute_backend == "gpu":
            lightgbm_kwargs.update(
                {
                    "device_type": "gpu",
                    "gpu_device_id": int(gpu_device),
                    "max_bin": 255,
                }
            )
        estimator = _NumpyLGBMClassifier(**lightgbm_kwargs)
        grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [-1, 5, 10],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [10, 20, 50],
            "model__learning_rate": [0.03, 0.1],
        }
        return estimator, grid

    raise ValueError(f"Unsupported classifier: {classify}")


def _build_pipeline(classify, X_train, *, compute_backend="cpu", gpu_device=0):
    compute_backend = _resolve_classifier_backend(classify, compute_backend)
    preprocessor = _build_preprocessor(X_train, compute_backend=compute_backend)
    estimator, param_grid = _build_classifier_spec(
        classify,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
    )
    steps = [("preprocess", preprocessor)]
    if compute_backend == "gpu":
        # Keep feature matrices dense and float32 to reduce host/device transfer cost.
        steps.append(("to_float32", DenseFloat32Transformer()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps=steps)
    return pipeline, param_grid


def _normalize_search_params(params):
    normalized = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _count_param_candidates(param_space):
    if isinstance(param_space, list):
        return sum(_count_param_candidates(item) for item in param_space)
    if isinstance(param_space, dict):
        total = 1
        for values in param_space.values():
            total *= len(values)
        return total
    return 0


def fit_model(
    classify,
    X_train,
    y_train,
    *,
    tuning_cv_folds=5,
    tuning_scoring="f1_macro",
    tuning_n_jobs=-1,
    compute_backend="cpu",
    gpu_device=0,
):
    validate_feature_frame(X_train, context=f"{classify} training features")
    _validate_target_labels(y_train, context=f"{classify} target labels")
    compute_backend = _resolve_classifier_backend(classify, compute_backend)
    pipeline, param_grid = _build_pipeline(
        classify,
        X_train,
        compute_backend=compute_backend,
        gpu_device=gpu_device,
    )
    search_n_jobs = 1 if compute_backend == "gpu" else tuning_n_jobs
    total_candidates = max(1, _count_param_candidates(param_grid))
    search_iterations = min(20, total_candidates)
    cv = StratifiedKFold(n_splits=tuning_cv_folds, shuffle=True, random_state=40)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=search_iterations,
        scoring=tuning_scoring,
        cv=cv,
        n_jobs=search_n_jobs,
        refit=True,
        random_state=40,
    )
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    best_estimator.model_selection_metadata = {
        "classifier": classify,
        "best_params": _normalize_search_params(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "tuning_scoring": tuning_scoring,
        "tuning_cv_folds": int(tuning_cv_folds),
        "compute_backend": compute_backend,
        "gpu_device": int(gpu_device),
        "search_n_jobs": int(search_n_jobs),
        "search_strategy": "randomized",
        "search_iterations": int(search_iterations),
        "search_space_candidates": int(total_candidates),
    }
    return best_estimator


def get_model_metadata(model):
    metadata = getattr(model, "model_selection_metadata", None)
    if not metadata:
        return {
            "classifier": "",
            "best_params": {},
            "best_cv_score": np.nan,
            "tuning_scoring": "",
            "tuning_cv_folds": np.nan,
        }

    payload = dict(metadata)
    payload["best_params"] = dict(payload.get("best_params", {}))
    return payload


def get_model_metadata_json(model):
    metadata = get_model_metadata(model)
    return json.dumps(metadata.get("best_params", {}), ensure_ascii=False, sort_keys=True)


def _scores_to_margin(score_matrix):
    scores = np.asarray(score_matrix)
    if scores.ndim == 1:
        return np.abs(scores)
    if scores.shape[1] == 1:
        return np.abs(scores[:, 0])
    if scores.shape[1] == 2:
        return np.abs(scores[:, 1] - scores[:, 0])
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def predict_with_scores(model, X_test):
    validate_feature_frame(X_test, context="prediction features")
    y_pred = model.predict(X_test)
    probabilities = None
    raw_scores = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        margin = _scores_to_margin(probabilities)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        margin = _scores_to_margin(raw_scores)
    else:
        margin = np.zeros(len(y_pred), dtype=float)

    return {
        "y_pred": np.asarray(y_pred),
        "probabilities": probabilities,
        "scores": raw_scores,
        "margin": np.asarray(margin, dtype=float),
    }


def make_classify(classify, data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=40,
        stratify=y,
    )
    model = fit_model(classify, X_train, y_train)
    outputs = predict_with_scores(model, X_test)
    return outputs["y_pred"], np.asarray(y_test)
