import json
import os
import sys
from datetime import datetime
from contextlib import contextmanager

from code.config.dataset_profiles import (
    DATASET_CATEGORY_NO_REAL_MISSING,
    DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH,
)
from code.core.data_io import (
    build_imputation_group_label,
    build_run_stem,
    get_dataset_category,
)
from code.core.path_config import apply_workspace_paths
from code.pipeline.branches import (
    prepare_no_real_missing_data,
    prepare_real_missing_with_truth_data,
)
from code.pipeline.unified_training import run_training_pipeline


def _ensure_workspace(args):
    os.makedirs(getattr(args, "clean_data_dir", "result_data"), exist_ok=True)
    if get_dataset_category(args) == DATASET_CATEGORY_NO_REAL_MISSING:
        os.makedirs(getattr(args, "missing_data_dir", "data/Missing_data"), exist_ok=True)
    os.makedirs(getattr(args, "result_dir", "result"), exist_ok=True)

def _stringify_path_value(value):
    return str(value).replace(os.sep, "_")


def _build_result_root(args, dataset_category):
    base_parts = [getattr(args, "result_dir", "result"), dataset_category, _stringify_path_value(args.dataset)]
    if dataset_category == DATASET_CATEGORY_NO_REAL_MISSING:
        base_parts.extend(
            [
                f"mp_{args.missing_percent}",
                f"ml_{args.max_missing_lenth}",
                _stringify_path_value(args.missing_mechanism),
            ]
        )
    imputation_label = build_imputation_group_label(getattr(args, "imputation_method", None))
    seed_leaf = f"seed_{args.seed}"
    if imputation_label:
        seed_leaf = f"{seed_leaf}_{imputation_label}"
    base_parts.extend([f"iter_{args.iter}", seed_leaf])
    result_root = os.path.join(*base_parts)
    os.makedirs(result_root, exist_ok=True)
    return result_root


def _build_stage_cache_root(args, result_root):
    run_name = (
        f"{args.dataset}"
        f"_iter{args.iter}"
        f"_seed{args.seed}"
        f"_mr{args.missing_percent}"
        f"_ml{args.max_missing_lenth}"
        f"_{args.missing_mechanism}"
    )
    cache_root = os.path.join(result_root, "stage_cache", run_name)
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def _save_stage_frame(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def _save_stage_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


@contextmanager
def _capture_run_log(result_root):
    log_dir = os.path.join(result_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "classify.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "a", encoding="utf-8") as handle:
        started_at = datetime.now().isoformat(timespec="seconds")
        handle.write(f"\n=== classify start {started_at} ===\n")
        sys.stdout = _TeeStream(original_stdout, handle)
        sys.stderr = _TeeStream(original_stderr, handle)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            finished_at = datetime.now().isoformat(timespec="seconds")
            handle.write(f"=== classify end {finished_at} ===\n")
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _load_existing_manifest(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _prepare_branch_data(args, dataset_name, dataset_category):
    if dataset_category == DATASET_CATEGORY_NO_REAL_MISSING:
        return prepare_no_real_missing_data(args)
    if dataset_category == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
        return prepare_real_missing_with_truth_data(args, dataset_name)
    raise ValueError(f"Unsupported dataset_category: {dataset_category}")


def run_experiment(args):
    apply_workspace_paths(args)
    _ensure_workspace(args)
    dataset_category = get_dataset_category(args)
    result_root = _build_result_root(args, dataset_category)
    with _capture_run_log(result_root) as log_path:
        dataset_result_root = result_root
        cache_root = _build_stage_cache_root(args, result_root)
        dataset_name = build_run_stem(
            args.dataset,
            args.missing_percent,
            args.max_missing_lenth,
            args.missing_mechanism,
            args.iter,
            seed=getattr(args, "seed", None),
        )

        print(
            f"[run] dataset={args.dataset} seed={args.seed} iter={args.iter} "
            f"mechanism={args.missing_mechanism} missing={args.missing_percent} "
            f"backend={getattr(args, 'selected_compute_backend', getattr(args, 'compute_backend', 'cpu'))}"
        )

        branch_payload = _prepare_branch_data(args, dataset_name, dataset_category)
        ground_df = branch_payload["ground_df"]
        missing_df = branch_payload["missing_df"]
        missing_filled_df = branch_payload["missing_filled_df"]
        reference_clean_df = branch_payload["reference_clean_df"]
        reference_clean_source = branch_payload["reference_clean_source"]
        has_ground_truth = branch_payload["has_ground_truth"]

        if reference_clean_df is None:
            raise ValueError("Unable to build training data: neither truth data nor missing data is available.")

        if ground_df is not None:
            _save_stage_frame(os.path.join(cache_root, "01_ground.csv"), ground_df)
        else:
            _save_stage_frame(os.path.join(cache_root, "01_ground_proxy.csv"), reference_clean_df)
        _save_stage_frame(os.path.join(cache_root, "02_missing.csv"), missing_df)
        _save_stage_frame(os.path.join(cache_root, "03_missing_filled_baseline.csv"), missing_filled_df)

        if dataset_category == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
            imputer_input_path = os.path.join(cache_root, "05_imputer_input.csv")
            # Subprocess-backed imputers load their own input from disk. Point
            # them at the same filtered, label-complete table used here.
            missing_df.to_csv(imputer_input_path, header=False, index=False)
            args.real_missing_observed_path = imputer_input_path

        train_outputs = run_training_pipeline(
            args,
            dataset_name=dataset_name,
            dataset_category=dataset_category,
            has_ground_truth=has_ground_truth,
            ground_df=ground_df,
            missing_df=missing_df,
            missing_filled_df=missing_filled_df,
            reference_clean_df=reference_clean_df,
            reference_clean_source=reference_clean_source,
            cache_root=cache_root,
            dataset_result_root=dataset_result_root,
            save_stage_frame=_save_stage_frame,
        )
        manifest_path = os.path.join(result_root, "manifest.json")
        existing_manifest = _load_existing_manifest(manifest_path)
        _save_stage_json(
            manifest_path,
            {
                "created_at": existing_manifest.get("created_at", datetime.now().isoformat(timespec="seconds")),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": args.dataset,
                "dataset_category": dataset_category,
                "reference_clean_source": reference_clean_source,
                "iter": args.iter,
                "seed": args.seed,
                "missing_percent": args.missing_percent,
                "max_missing_lenth": args.max_missing_lenth,
                "missing_mechanism": args.missing_mechanism,
                "class_method": list(args.class_method),
                "imputation_method": list(args.imputation_method),
                "hard_sample_percent": getattr(args, "hard_sample_percent", 20.0),
                "tuning_cv_folds": getattr(args, "tuning_cv_folds", 5),
                "tuning_n_jobs": getattr(args, "tuning_n_jobs", -1),
                "tuning_scoring": getattr(args, "tuning_scoring", "f1_macro"),
                "compute_backend": getattr(args, "compute_backend", "cpu"),
                "selected_compute_backend": getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu")),
                "gpu_device": getattr(args, "gpu_device", 0),
                "last_stage": "classify",
                "run_log": log_path,
            },
        )
        _save_stage_json(
            os.path.join(cache_root, "manifest.json"),
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": args.dataset,
                "dataset_category": dataset_category,
                "iter": args.iter,
                "seed": args.seed,
                "missing_percent": args.missing_percent,
                "max_missing_lenth": args.max_missing_lenth,
                "missing_mechanism": args.missing_mechanism,
                "class_method": list(args.class_method),
                "imputation_method": list(args.imputation_method),
                "compute_backend": getattr(args, "compute_backend", "cpu"),
                "selected_compute_backend": getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu")),
                "gpu_device": getattr(args, "gpu_device", 0),
                "run_log": log_path,
            },
        )

        print(f"[run] completed result_dir={result_root}")

        return {
            "result_dir": result_root,
            "stage_cache_dir": cache_root,
            "run_log": log_path,
            "summary": train_outputs["summary"],
            "sample_level": train_outputs["sample_level"],
            "structure_features": train_outputs["structure_features"],
        }
