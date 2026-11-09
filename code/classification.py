import argparse

from code.config.args import add_compute_backend_arguments
from code.core.runtime import configure_runtime_for_stability
from code.core.compute_backend import enforce_safe_gpu_parallelism, initialize_compute_backend
from code.core.path_config import add_workspace_path_arguments, apply_workspace_paths

configure_runtime_for_stability()

from code.config.dataset_profiles import DATASET_CATEGORIES
from code.config.settings import cfg


parser = argparse.ArgumentParser(description="staged imputation-classification experiment")
add_workspace_path_arguments(parser)
add_compute_backend_arguments(parser)
parser.add_argument("--dataset", dest="dataset", type=str, help="dataset name", default=cfg["dataset"])
parser.add_argument(
    "--dataset_category",
    dest="dataset_category",
    type=str,
    choices=DATASET_CATEGORIES,
    help="dataset type profile",
    default=cfg["dataset_category"],
)
parser.add_argument("--iter", dest="iter", type=int, help="iter for test", default=cfg["iter"])
parser.add_argument("--seed", dest="seed", type=int, help="base seed", default=cfg["base_seed"])
parser.add_argument(
    "--class_method",
    dest="class_method",
    type=str,
    nargs="+",
    help="class method",
    default=cfg["class_method"],
)
parser.add_argument(
    "--class_metrics",
    dest="class_metrics",
    type=str,
    nargs="+",
    help="class metrics",
    default=cfg["class_metrics"],
)
parser.add_argument(
    "--imputation_method",
    dest="imputation_method",
    type=str,
    nargs="+",
    help="your imputation method",
    default=cfg["imputation_method"],
)
parser.add_argument(
    "--missing_percent",
    dest="missing_percent",
    type=float,
    help="generate missing percent",
    default=cfg["missing_percent"],
)
parser.add_argument(
    "--max_missing_lenth",
    dest="max_missing_lenth",
    type=int,
    help="generate continous max missing lenth",
    default=cfg["max_missing_lenth"],
)
parser.add_argument(
    "--missing_mechanism",
    dest="missing_mechanism",
    type=str,
    help="generate missing mechanism",
    choices=["MCAR", "MAR", "MNAR"],
    default=cfg["missing_mechanism"],
)
parser.add_argument(
    "--mar_logistic_beta",
    dest="mar_logistic_beta",
    type=float,
    help="logistic slope used by score-driven MAR/MNAR generation",
    default=cfg["mar_logistic_beta"],
)
parser.add_argument(
    "--mar_logistic_clip",
    dest="mar_logistic_clip",
    type=float,
    help="clip threshold applied to standardized anchor scores before logistic masking",
    default=cfg["mar_logistic_clip"],
)
parser.add_argument(
    "--hard_sample_percent",
    dest="hard_sample_percent",
    type=float,
    help="percent of smallest-margin test samples treated as hard",
    default=cfg["hard_sample_percent"],
)
parser.add_argument(
    "--tuning_cv_folds",
    dest="tuning_cv_folds",
    type=int,
    help="number of stratified CV folds used for classifier tuning",
    default=cfg["tuning_cv_folds"],
)
parser.add_argument(
    "--tuning_n_jobs",
    dest="tuning_n_jobs",
    type=int,
    help="parallel jobs used by RandomizedSearchCV",
    default=cfg["tuning_n_jobs"],
)
parser.add_argument(
    "--tuning_scoring",
    dest="tuning_scoring",
    type=str,
    help="scoring metric used for classifier tuning",
    default=cfg["tuning_scoring"],
)


if __name__ == "__main__":
    args = parser.parse_args()
    apply_workspace_paths(args)
    initialize_compute_backend(args)
    for note in enforce_safe_gpu_parallelism(args):
        print(f"[compute-backend] {note}")
    from code.pipeline.runner import run_experiment

    outputs = run_experiment(args)
    # print(outputs["summary"])
    print("finish!")
