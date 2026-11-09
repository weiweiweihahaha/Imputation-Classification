import os
import traceback
from contextlib import contextmanager

import fcntl

import pandas as pd
import numpy as np

from code.config.imputer_families import get_imputer_family
from code.core.classifiers import fit_model, get_model_metadata_json
from code.pipeline.evaluate_imputation import evaluate_imputation_method
from code.pipeline.model_reference import fit_reference_model, score_reference_model
from code.pipeline.split import build_fixed_split


def _merge_stage_frame(path, new_frame, key_cols):
    if new_frame is None or new_frame.empty:
        return new_frame if new_frame is not None else pd.DataFrame()

    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    else:
        combined = new_frame.copy()

    merge_keys = [col for col in key_cols if col in combined.columns]
    if merge_keys:
        combined = combined.drop_duplicates(subset=merge_keys, keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")
    return combined


@contextmanager
def _locked_output_merge(dataset_result_root):
    lock_path = os.path.join(dataset_result_root, ".run_outputs.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_training_pipeline(
    args,
    *,
    dataset_name,
    dataset_category,
    has_ground_truth,
    ground_df,
    missing_df,
    missing_filled_df,
    reference_clean_df,
    reference_clean_source,
    cache_root,
    dataset_result_root,
    save_stage_frame,
):
    args.expected_imputed_rows = int(reference_clean_df.shape[0])
    train_idx, test_idx = build_fixed_split(reference_clean_df, label_col=0, random_state=args.seed)
    split_df = pd.DataFrame(
        {
            "sample_id": list(train_idx) + list(test_idx),
            "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
        }
    )
    save_stage_frame(os.path.join(cache_root, "04_split.csv"), split_df)

    summary_rows = []
    sample_frames = []
    structure_frames = []
    hard_percent = getattr(args, "hard_sample_percent", 20.0)

    for classifier_name in args.class_method:
        classifier_cache_dir = os.path.join(cache_root, "classifiers", classifier_name)
        os.makedirs(classifier_cache_dir, exist_ok=True)

        ref_model, ref_model_meta = fit_reference_model(
            classifier_name,
            reference_clean_df,
            train_idx,
            label_col=0,
            tuning_cv_folds=getattr(args, "tuning_cv_folds", 5),
            tuning_scoring=getattr(args, "tuning_scoring", "f1_macro"),
            tuning_n_jobs=getattr(args, "tuning_n_jobs", -1),
            compute_backend=getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu")),
            gpu_device=getattr(args, "gpu_device", 0),
        )
        X_ground_test, y_true, ground_outputs = score_reference_model(
            ref_model,
            reference_clean_df,
            test_idx,
            label_col=0,
        )
        reference_test_df = X_ground_test.copy().reset_index(drop=True)
        reference_test_df.insert(0, "sample_id", test_idx)
        reference_test_df.insert(1, "y_true", y_true.to_numpy())
        reference_test_df.insert(2, "y_pred_ref", ground_outputs["y_pred"])
        reference_test_df["margin_ref"] = ground_outputs["margin"]
        save_stage_frame(
            os.path.join(classifier_cache_dir, "01_reference_test_predictions.csv"),
            reference_test_df,
        )

        X_missing_train = missing_filled_df.loc[train_idx, missing_filled_df.columns != 0]
        y_missing_train = missing_filled_df.loc[train_idx, 0]
        missing_model = fit_model(
            classifier_name,
            X_missing_train,
            y_missing_train,
            tuning_cv_folds=getattr(args, "tuning_cv_folds", 5),
            tuning_scoring=getattr(args, "tuning_scoring", "f1_macro"),
            tuning_n_jobs=getattr(args, "tuning_n_jobs", -1),
            compute_backend=getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu")),
            gpu_device=getattr(args, "gpu_device", 0),
        )
        missing_model_meta = getattr(missing_model, "model_selection_metadata", {})
        missing_train_df = X_missing_train.copy().reset_index(drop=True)
        missing_train_df.insert(0, "sample_id", train_idx)
        missing_train_df.insert(1, "y_train_missing", y_missing_train.to_numpy())
        save_stage_frame(
            os.path.join(classifier_cache_dir, "02_missing_train_baseline.csv"),
            missing_train_df,
        )

        reference_bundle = {
            "model": ref_model,
            "X_ground_test": X_ground_test,
            "y_true": y_true,
            "ground_outputs": ground_outputs,
            "ground_df": ground_df,
            "dataset_category": dataset_category,
            "has_ground_truth": has_ground_truth,
            "reference_clean_source": reference_clean_source,
            "reference_model_meta": ref_model_meta,
            "missing_df": missing_df,
            "missing_model": missing_model,
            "missing_model_meta": missing_model_meta,
            "missing_filled_df": missing_filled_df,
        }

        for imputation_name in args.imputation_method:
            stage_cache_dir = os.path.join(classifier_cache_dir, "imputers", imputation_name)
            try:
                summary, sample_level, structure_feature = evaluate_imputation_method(
                    args=args,
                    classifier_name=classifier_name,
                    imputation_name=imputation_name,
                    dataset_name=dataset_name,
                    ground_df=ground_df,
                    reference_bundle=reference_bundle,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    hard_percent=hard_percent,
                    stage_cache_dir=stage_cache_dir,
                )
            except Exception as exc:
                os.makedirs(stage_cache_dir, exist_ok=True)
                error_message = f"{type(exc).__name__}: {exc}"
                with open(os.path.join(stage_cache_dir, "99_error.log"), "w", encoding="utf-8") as f:
                    f.write(error_message + "\n\n")
                    f.write(traceback.format_exc())

                summary = {
                    "dataset": args.dataset,
                    "dataset_category": dataset_category,
                    "reference_clean_source": reference_clean_source,
                    "has_ground_truth": int(has_ground_truth),
                    "iter": getattr(args, "iter", np.nan),
                    "seed": getattr(args, "seed", np.nan),
                    "missing_percent": args.missing_percent,
                    "max_missing_lenth": args.max_missing_lenth,
                    "missing_mechanism": args.missing_mechanism,
                    "imputer": imputation_name,
                    "imputer_family": get_imputer_family(imputation_name),
                    "classifier": classifier_name,
                    "status": "failed",
                    "error": error_message,
                    "f1_ground_ref": np.nan,
                    "f1_missing": np.nan,
                    "f1_imputed": np.nan,
                    "accuracy_imputed": np.nan,
                    "delta_f1": np.nan,
                    "imputation_runtime_seconds": np.nan,
                    "train_runtime_seconds": np.nan,
                    "inference_runtime_seconds": np.nan,
                    "total_runtime_seconds": np.nan,
                    "runtime_seconds": np.nan,
                    "reference_best_params": get_model_metadata_json(ref_model),
                    "reference_best_cv_score": ref_model_meta.get("best_cv_score", np.nan),
                    "missing_best_params": get_model_metadata_json(missing_model),
                    "missing_best_cv_score": missing_model_meta.get("best_cv_score", np.nan),
                    "imputed_best_params": "",
                    "imputed_best_cv_score": np.nan,
                    "tuning_scoring": getattr(args, "tuning_scoring", "f1_macro"),
                    "tuning_cv_folds": getattr(args, "tuning_cv_folds", 5),
                }
                sample_level = pd.DataFrame()
                structure_feature = pd.DataFrame()
            else:
                summary["status"] = "ok"
                summary["error"] = ""

            summary_rows.append(summary)
            if not sample_level.empty:
                sample_frames.append(sample_level)
            if not structure_feature.empty:
                structure_frames.append(structure_feature)

    summary_path = os.path.join(dataset_result_root, "run_summary.csv")
    sample_path = os.path.join(dataset_result_root, "sample_level.csv")
    structure_path = os.path.join(dataset_result_root, "structure_features.csv")

    with _locked_output_merge(dataset_result_root):
        summary_df = pd.DataFrame(summary_rows)
        summary_df = _merge_stage_frame(
            summary_path,
            summary_df,
            [
                "dataset",
                "iter",
                "seed",
                "missing_percent",
                "max_missing_lenth",
                "missing_mechanism",
                "imputer",
                "classifier",
            ],
        )
        sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
        sample_df = _merge_stage_frame(
            sample_path,
            sample_df,
            [
                "dataset",
                "iter",
                "seed",
                "missing_percent",
                "missing_mechanism",
                "imputer",
                "classifier",
                "sample_id",
            ],
        )
        structure_df = pd.concat(structure_frames, ignore_index=True) if structure_frames else pd.DataFrame()
        structure_df = _merge_stage_frame(
            structure_path,
            structure_df,
            [
                "dataset",
                "imputer",
                "classifier",
                "feature",
            ],
        )
        save_stage_frame(summary_path, summary_df)
        save_stage_frame(sample_path, sample_df)
        save_stage_frame(structure_path, structure_df)

        save_stage_frame(os.path.join(cache_root, "90_run_summary.csv"), summary_df)
        save_stage_frame(os.path.join(cache_root, "91_sample_level.csv"), sample_df)
        save_stage_frame(os.path.join(cache_root, "92_structure_features.csv"), structure_df)

    return {
        "summary": summary_df,
        "sample_level": sample_df,
        "structure_features": structure_df,
    }
