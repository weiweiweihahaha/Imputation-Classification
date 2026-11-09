from argparse import Namespace
from types import SimpleNamespace

from code.config.dataset_profiles import DATASET_CATEGORIES
from code.config.settings import get_default_config
from code.core.path_config import add_workspace_path_arguments


def add_compute_backend_arguments(parser):
    defaults = get_default_config()
    parser.add_argument(
        "--compute_backend",
        choices=["cpu", "gpu", "auto"],
        default=defaults["compute_backend"],
        help="classification compute backend; gpu/auto enables RAPIDS acceleration when available",
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=defaults["gpu_device"],
        help="CUDA device index used when --compute_backend requests GPU execution",
    )
    return parser


def add_single_experiment_arguments(parser, include_class_metrics=False):
    defaults = get_default_config()
    add_workspace_path_arguments(parser)
    add_compute_backend_arguments(parser)
    parser.add_argument("--dataset", dest="dataset", type=str, default=defaults["dataset"])
    parser.add_argument(
        "--dataset_category",
        dest="dataset_category",
        choices=DATASET_CATEGORIES,
        default=defaults["dataset_category"],
    )
    parser.add_argument("--iter", dest="iter", type=int, default=defaults["iter"])
    parser.add_argument("--seed", dest="seed", type=int, default=defaults["base_seed"])
    parser.add_argument("--seeds", dest="seeds", nargs="+", type=int, default=defaults["seeds"])
    parser.add_argument("--class_method", dest="class_method", nargs="+", default=defaults["class_method"])
    if include_class_metrics:
        parser.add_argument(
            "--class_metrics",
            dest="class_metrics",
            nargs="+",
            default=defaults["class_metrics"],
        )
    parser.add_argument(
        "--imputation_method",
        dest="imputation_method",
        nargs="+",
        default=defaults["imputation_method"],
    )
    parser.add_argument("--missing_percent", dest="missing_percent", type=float, default=defaults["missing_percent"])
    parser.add_argument(
        "--max_missing_lenth",
        dest="max_missing_lenth",
        type=int,
        default=defaults["max_missing_lenth"],
    )
    parser.add_argument(
        "--missing_mechanism",
        dest="missing_mechanism",
        choices=["MCAR", "MAR", "MNAR"],
        default=defaults["missing_mechanism"],
    )
    parser.add_argument("--mar_logistic_beta", dest="mar_logistic_beta", type=float, default=defaults["mar_logistic_beta"])
    parser.add_argument("--mar_logistic_clip", dest="mar_logistic_clip", type=float, default=defaults["mar_logistic_clip"])
    parser.add_argument(
        "--hard_sample_percent",
        dest="hard_sample_percent",
        type=float,
        default=defaults["hard_sample_percent"],
    )
    parser.add_argument("--tuning_cv_folds", type=int, default=defaults["tuning_cv_folds"])
    parser.add_argument("--tuning_n_jobs", type=int, default=defaults["tuning_n_jobs"])
    parser.add_argument("--tuning_scoring", type=str, default=defaults["tuning_scoring"])
    return parser


def add_missing_generation_arguments(parser):
    defaults = get_default_config()
    add_workspace_path_arguments(parser)
    parser.add_argument("--dataset", dest="dataset", type=str, default=defaults["dataset"])
    parser.add_argument(
        "--dataset_category",
        dest="dataset_category",
        choices=DATASET_CATEGORIES,
        default=defaults["dataset_category"],
    )
    parser.add_argument("--iter", dest="iter", type=int, default=defaults["iter"])
    parser.add_argument("--seed", dest="seed", type=int, default=defaults["base_seed"])
    parser.add_argument("--missing_percent", dest="missing_percent", type=float, default=defaults["missing_percent"])
    parser.add_argument(
        "--max_missing_lenth",
        dest="max_missing_lenth",
        type=int,
        default=defaults["max_missing_lenth"],
    )
    parser.add_argument(
        "--missing_mechanism",
        dest="missing_mechanism",
        choices=["MCAR", "MAR", "MNAR"],
        default=defaults["missing_mechanism"],
    )
    parser.add_argument("--mar_logistic_beta", dest="mar_logistic_beta", type=float, default=defaults["mar_logistic_beta"])
    parser.add_argument("--mar_logistic_clip", dest="mar_logistic_clip", type=float, default=defaults["mar_logistic_clip"])
    return parser


def add_batch_experiment_arguments(parser):
    defaults = get_default_config()
    add_workspace_path_arguments(parser)
    add_compute_backend_arguments(parser)
    parser.add_argument("--datasets", nargs="+", default=defaults["datasets"])
    parser.add_argument(
        "--dataset_category",
        choices=DATASET_CATEGORIES,
        default=defaults["dataset_category"],
    )
    parser.add_argument("--missing_percents", nargs="+", type=float, default=[defaults["missing_percent"]])
    parser.add_argument(
        "--mechanisms",
        "--missing_mechanisms",
        dest="mechanisms",
        nargs="+",
        choices=["MCAR", "MAR", "MNAR"],
        default=defaults["missing_mechanisms"],
    )
    parser.add_argument("--classifiers", nargs="+", default=defaults["class_method"])
    parser.add_argument("--imputers", nargs="+", default=defaults["imputation_method"])
    parser.add_argument(
        "--iter",
        type=int,
        default=defaults["iter"],
        help="number of repeated runs to execute when --iters is not provided",
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=None,
        help="explicit iteration ids to run; overrides --iter repeat count",
    )
    parser.add_argument("--seed", type=int, default=defaults["base_seed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults["seeds"])
    parser.add_argument("--max_missing_lenth", type=int, default=defaults["max_missing_lenth"])
    parser.add_argument("--hard_sample_percent", type=float, default=defaults["hard_sample_percent"])
    parser.add_argument("--tuning_cv_folds", type=int, default=defaults["tuning_cv_folds"])
    parser.add_argument("--tuning_n_jobs", type=int, default=defaults["tuning_n_jobs"])
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=12,
        help="outer parallel workers for batch tasks; use 0 to auto-pick a safer value based on CPU cores",
    )
    parser.add_argument("--tuning_scoring", type=str, default=defaults["tuning_scoring"])
    return parser


def resolve_batch_iterations(args):
    defaults = get_default_config()
    if getattr(args, "iters", None):
        return list(args.iters)
    repeat_count = int(getattr(args, "iter", defaults["iter"]))
    if repeat_count < 1:
        raise ValueError("--iter must be >= 1 for batch runs.")
    return list(range(1, repeat_count + 1))


def resolve_batch_seeds(args):
    defaults = get_default_config()
    explicit_seeds = getattr(args, "seeds", None)
    if explicit_seeds:
        return list(explicit_seeds)
    return [getattr(args, "seed", defaults["base_seed"])]


def normalize_single_run_args(args):
    defaults = get_default_config()
    payload = defaults.copy()
    payload["seed"] = payload.pop("base_seed")

    source = vars(args) if isinstance(args, Namespace) else vars(args)
    payload.update(source)

    payload["class_method"] = list(payload.get("class_method", defaults["class_method"]))
    payload["class_metrics"] = list(payload.get("class_metrics", defaults["class_metrics"]))
    payload["imputation_method"] = list(payload.get("imputation_method", defaults["imputation_method"]))
    return SimpleNamespace(**payload)


def build_single_run_namespace(
    *,
    dataset,
    dataset_category,
    iteration,
    seed,
    class_method,
    class_metrics=None,
    imputation_method,
    missing_percent,
    max_missing_lenth,
    missing_mechanism,
    hard_sample_percent,
    mar_logistic_beta=None,
    mar_logistic_clip=None,
    tuning_cv_folds=None,
    tuning_n_jobs=None,
    tuning_scoring=None,
    result_tag=None,
    compute_backend=None,
    gpu_device=None,
    selected_compute_backend=None,
    workspace_dir=None,
    ground_truth_data_dir=None,
    real_missing_data_dir=None,
    real_missing_truth_dir=None,
    real_missing_label_col=None,
    missing_data_dir=None,
    clean_data_dir=None,
    result_dir=None,
):
    defaults = get_default_config()
    return SimpleNamespace(
        workspace_dir=defaults["workspace_dir"] if workspace_dir is None else workspace_dir,
        dataset=dataset,
        dataset_category=dataset_category,
        iter=iteration,
        seed=seed,
        class_method=list(class_method),
        class_metrics=list(class_metrics or defaults["class_metrics"]),
        imputation_method=list(imputation_method),
        missing_percent=missing_percent,
        max_missing_lenth=max_missing_lenth,
        missing_mechanism=missing_mechanism,
        hard_sample_percent=hard_sample_percent,
        mar_logistic_beta=defaults["mar_logistic_beta"] if mar_logistic_beta is None else mar_logistic_beta,
        mar_logistic_clip=defaults["mar_logistic_clip"] if mar_logistic_clip is None else mar_logistic_clip,
        tuning_cv_folds=defaults["tuning_cv_folds"] if tuning_cv_folds is None else tuning_cv_folds,
        tuning_n_jobs=defaults["tuning_n_jobs"] if tuning_n_jobs is None else tuning_n_jobs,
        tuning_scoring=defaults["tuning_scoring"] if tuning_scoring is None else tuning_scoring,
        result_tag=result_tag,
        compute_backend=defaults["compute_backend"] if compute_backend is None else compute_backend,
        gpu_device=defaults["gpu_device"] if gpu_device is None else gpu_device,
        selected_compute_backend=selected_compute_backend,
        ground_truth_data_dir=defaults["ground_truth_data_dir"] if ground_truth_data_dir is None else ground_truth_data_dir,
        real_missing_data_dir=defaults["real_missing_data_dir"] if real_missing_data_dir is None else real_missing_data_dir,
        real_missing_truth_dir=defaults["real_missing_truth_dir"] if real_missing_truth_dir is None else real_missing_truth_dir,
        real_missing_label_col=defaults["real_missing_label_col"] if real_missing_label_col is None else real_missing_label_col,
        missing_data_dir=defaults["missing_data_dir"] if missing_data_dir is None else missing_data_dir,
        clean_data_dir=defaults["clean_data_dir"] if clean_data_dir is None else clean_data_dir,
        result_dir=defaults["result_dir"] if result_dir is None else result_dir,
    )
