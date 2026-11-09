from dataclasses import asdict, dataclass, field
from typing import List

from code.config.dataset_profiles import (
    DATASET_CATEGORY_NO_REAL_MISSING,
    DATASET_CATEGORIES,
)


@dataclass(frozen=True)
class ExperimentDefaults:
    workspace_dir: str = "."
    compute_backend: str = "cpu"
    gpu_device: int = 0
    iter: int = 10
    base_seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 52, 62, 72, 82])
    dataset: str = "mobile"
    datasets: List[str] = field(
        default_factory=lambda: [
            "glass",
            "hcv",
            "iris",
            "mobile",
            "obesity",
            "rice",
            "seed",
            "wine",
        ]
    )
    real_missing_main_datasets: List[str] = field(
        default_factory=lambda: [
            "adult",
            "chronic_kidney_disease",
            "cervical_cancer",
            "heart_disease",
            "horse_colic",
        ]
    )
    real_missing_supplementary_datasets: List[str] = field(
        default_factory=lambda: [
            "dermatology",
            "hcc_survival",
        ]
    )
    real_missing_optional_datasets: List[str] = field(
        default_factory=lambda: [
            "hepatitis",
        ]
    )
    class_method: List[str] = field(
        default_factory=lambda: [
            "DecisionTree",
            "LogisticRegression",
            "RandomForest",
            "NaiveBayes",
            "KNN",
            "Adaboost",
            "SVM",
            "MLP",
            "LightGBM",
        ]
    )
    class_metrics: List[str] = field(default_factory=lambda: ["f1_score", "Accuracy"])
    dataset_category: str = DATASET_CATEGORY_NO_REAL_MISSING
    dataset_categories: List[str] = field(default_factory=lambda: list(DATASET_CATEGORIES))
    ground_truth_data_dir: str = "data/Ground_truth_data"
    real_missing_data_dir: str = "data/Real_missing_data"
    real_missing_truth_dir: str = "data/Real_missing_truth"
    real_missing_label_col: int = 0
    missing_data_dir: str = "data/Missing_data"
    clean_data_dir: str = "result_data"
    result_dir: str = "result"
    missing_percent: float = 0.4
    max_missing_lenth: int = 4
    missing_mechanism: str = "MCAR"
    missing_mechanisms: List[str] = field(default_factory=lambda: ["MCAR", "MAR", "MNAR"])
    mar_logistic_beta: float = 1.0
    mar_logistic_clip: float = 3.0
    hard_sample_percent: float = 20.0
    tuning_cv_folds: int = 5
    tuning_n_jobs: int = -1
    tuning_scoring: str = "f1_macro"
    tuning_refit: bool = True
    imputation_method: List[str] = field(
        default_factory=lambda: [
            "Mean",
            "Mice",
            "Mode",
            "KNNI",
            "CMI",
            "CI",
            "XGBI",
            "GMM",
            "MIDA",
            "SOM",
            "Holoclean",
            "MISSF",
            "MIWAE",
            "Datawig",
            "RRSI",
            "LOESS",
            "HDI",
            "MLPI",
            "VAEI",
            "CDI",
            "IIM",
            "DLM",
            "GINN",
            "NOMI",
            "KNNE",
            "GCID",
            "IFC",
            "MVNI",
            "SMC-FCS",
            "MIGEC",
            "Sice",
            "TWDA",
            "MIPCA",
            "NMF",
            "POP",
            "RIBG",
            "CVBKNN",
            "BPCA",
            "LSA",
            "PMM",
            "CCF",
            "ESD",
            "MIST",
            "CLRMC",
            "ECLRMC",
            "NIIA",
        ]
    )


DEFAULTS = ExperimentDefaults()


def get_default_config():
    # Return a fresh copy so callers can safely mutate list values.
    return asdict(DEFAULTS)


# Backward-compatible alias used by older callers.
cfg = get_default_config()
