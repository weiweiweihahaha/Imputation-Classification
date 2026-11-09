import os

from code.config.settings import get_default_config


INPUT_DIR_ATTRS = (
    "ground_truth_data_dir",
    "real_missing_data_dir",
    "real_missing_truth_dir",
)

WORKSPACE_DIR_ATTRS = (
    "missing_data_dir",
    "clean_data_dir",
    "result_dir",
)


def _resolve_path(base_dir, value):
    expanded = os.path.expanduser(str(value))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def add_workspace_path_arguments(parser):
    defaults = get_default_config()
    parser.add_argument(
        "--workspace_dir",
        default=defaults["workspace_dir"],
        help="base directory used to resolve relative data/result paths for an isolated machine workspace",
    )
    parser.add_argument("--ground_truth_data_dir", default=defaults["ground_truth_data_dir"])
    parser.add_argument("--real_missing_data_dir", default=defaults["real_missing_data_dir"])
    parser.add_argument("--real_missing_truth_dir", default=defaults["real_missing_truth_dir"])
    parser.add_argument(
        "--real_missing_label_col",
        type=int,
        default=defaults["real_missing_label_col"],
        help="column index of the label in real-missing CSVs before reordering it to column 0",
    )
    parser.add_argument("--missing_data_dir", default=defaults["missing_data_dir"])
    parser.add_argument("--clean_data_dir", default=defaults["clean_data_dir"])
    parser.add_argument("--result_dir", default=defaults["result_dir"])
    return parser


def apply_workspace_paths(args):
    defaults = get_default_config()
    repo_root = os.getcwd()
    workspace_dir = _resolve_path(repo_root, getattr(args, "workspace_dir", defaults["workspace_dir"]))
    setattr(args, "workspace_dir", workspace_dir)

    for attr in INPUT_DIR_ATTRS:
        current_value = getattr(args, attr, defaults[attr])
        setattr(args, attr, _resolve_path(repo_root, current_value))

    for attr in WORKSPACE_DIR_ATTRS:
        current_value = getattr(args, attr, defaults[attr])
        setattr(args, attr, _resolve_path(workspace_dir, current_value))

    return args


def build_workspace_subprocess_env(args):
    env = {
        "FRAMEWORK_WORKSPACE_DIR": getattr(args, "workspace_dir", ""),
        "FRAMEWORK_DATASET_CATEGORY": getattr(args, "dataset_category", ""),
        "FRAMEWORK_GROUND_TRUTH_DATA_DIR": getattr(args, "ground_truth_data_dir", ""),
        "FRAMEWORK_REAL_MISSING_DATA_DIR": getattr(args, "real_missing_data_dir", ""),
        "FRAMEWORK_REAL_MISSING_TRUTH_DIR": getattr(args, "real_missing_truth_dir", ""),
        "FRAMEWORK_REAL_MISSING_LABEL_COL": str(getattr(args, "real_missing_label_col", "")),
        "FRAMEWORK_MISSING_DATA_DIR": getattr(args, "missing_data_dir", ""),
        "FRAMEWORK_CLEAN_DATA_DIR": getattr(args, "clean_data_dir", ""),
        "FRAMEWORK_RESULT_DIR": getattr(args, "result_dir", ""),
        "FRAMEWORK_SEED": str(getattr(args, "seed", "")),
    }
    selected_backend = str(
        getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu"))
    ).strip().lower()
    if selected_backend != "gpu":
        # TensorFlow-based imputers are more stable in our batch jobs when we
        # explicitly keep them on CPU and avoid repeated GPU probing.
        env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        env.setdefault("JAX_PLATFORM_NAME", "cpu")
    return env
