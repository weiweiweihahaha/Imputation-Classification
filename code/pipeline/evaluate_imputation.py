import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from code.Imputation import impute
from code.analysis.decision_metrics import evaluate_decision_consequences
from code.analysis.structure_metrics import compute_structure_metrics
from code.config.imputer_families import get_imputer_family
from code.core.classifiers import validate_feature_frame
from code.core.data_io import (
    build_clean_data_path,
    build_clean_timing_path,
    load_data,
    load_real_missing_data,
    load_real_truth_data,
)
from code.pipeline.model_retrained import fit_retrained_model


def _normalize_frame(df):
    # Reindex columns so all loaders expose the same label/features layout.
    normalized = df.copy()
    normalized.columns = list(range(normalized.shape[1]))
    normalized = normalized.reset_index(drop=True)
    return normalized


def _fill_missing_values(df):
    filled = df.copy()
    for col in filled.columns:
        if col == 0:
            continue
        if (
            filled[col].dtype == "object"
            or pd.api.types.is_string_dtype(filled[col].dtype)
            or isinstance(filled[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(filled[col].dtype)
        ):
            modes = filled[col].mode(dropna=True)
            if not modes.empty:
                filled[col] = filled[col].fillna(modes.iloc[0])
        else:
            filled[col] = filled[col].fillna(pd.to_numeric(filled[col], errors="coerce").mean())
    return filled


def _infer_valid_real_missing_truth_rows(args):
    try:
        ground_df = _normalize_frame(load_real_truth_data(args))
        missing_df = _normalize_frame(load_real_missing_data(args))
    except Exception:
        return None
    if ground_df.shape[0] != missing_df.shape[0]:
        return None
    ground_labels = pd.to_numeric(ground_df[0], errors="coerce")
    valid_mask = (
        ground_labels.notna()
        & np.isfinite(ground_labels.to_numpy(dtype=float))
    )
    return valid_mask[valid_mask].index.to_list()


def _write_imputation_timing_sidecar(timing_path, args, dataset_name, imputation_name, imputation_runtime_seconds):
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    with open(timing_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "imputation_runtime_seconds": float(imputation_runtime_seconds),
                "dataset": args.dataset,
                "dataset_name": dataset_name,
                "imputer": imputation_name,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _lookup_historical_imputation_runtime(args, imputation_name):
    result_root = Path(getattr(args, "result_dir", "result"))
    if not result_root.exists():
        return np.nan

    candidates = []
    for summary_path in sorted(result_root.rglob("run_summary.csv")):
        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue
        required_cols = {"dataset", "missing_percent", "max_missing_lenth", "missing_mechanism", "imputer"}
        if not required_cols.issubset(frame.columns):
            continue

        subset = frame[
            (frame["dataset"].astype(str) == str(args.dataset))
            & (pd.to_numeric(frame["missing_percent"], errors="coerce") == float(args.missing_percent))
            & (pd.to_numeric(frame["max_missing_lenth"], errors="coerce") == float(args.max_missing_lenth))
            & (frame["missing_mechanism"].astype(str) == str(args.missing_mechanism))
            & (frame["imputer"].astype(str) == str(imputation_name))
        ]
        if "seed" in frame.columns and getattr(args, "seed", None) is not None:
            subset = subset[pd.to_numeric(subset["seed"], errors="coerce") == float(args.seed)]
        if subset.empty:
            continue

        if "imputation_runtime_seconds" in subset.columns:
            values = pd.to_numeric(subset["imputation_runtime_seconds"], errors="coerce").dropna().tolist()
            candidates.extend(float(value) for value in values)
            if values:
                continue

        if "imputation_runtime_seconds" not in frame.columns and "runtime_seconds" in subset.columns:
            values = pd.to_numeric(subset["runtime_seconds"], errors="coerce").dropna().tolist()
            candidates.extend(float(value) for value in values)

    if not candidates:
        return np.nan
    return float(np.median(candidates))


def _prepare_imputed_data(args, dataset_name, imputation_name, ground_df=None):
    file_path = build_clean_data_path(args, imputation_name)
    timing_path = build_clean_timing_path(args, imputation_name)
    imputation_runtime_seconds = np.nan

    def _run_imputer():
        start = time.perf_counter()
        impute(args, imputation_name, dataset_name, args.dataset)
        runtime_seconds = time.perf_counter() - start
        _write_imputation_timing_sidecar(
            timing_path,
            args,
            dataset_name,
            imputation_name,
            runtime_seconds,
        )
        return runtime_seconds

    def _align_to_active_rows(frame):
        active_rows = getattr(args, "valid_real_missing_truth_rows", None)
        if active_rows is None:
            active_rows = _infer_valid_real_missing_truth_rows(args)
            if active_rows is not None:
                args.valid_real_missing_truth_rows = active_rows
        expected_rows = getattr(args, "expected_imputed_rows", None)
        if expected_rows is None and ground_df is not None:
            expected_rows = _normalize_frame(ground_df).shape[0]
        if expected_rows is None or frame.shape[0] == int(expected_rows):
            return frame
        if (
            active_rows is not None
            and len(active_rows) == int(expected_rows)
            and max(active_rows, default=-1) < frame.shape[0]
        ):
            return frame.iloc[active_rows].reset_index(drop=True)
        if 0 in frame.columns:
            labels = pd.to_numeric(frame[0], errors="coerce")
            valid_mask = labels.notna() & np.isfinite(labels.to_numpy(dtype=float))
            if int(valid_mask.sum()) == int(expected_rows):
                return frame.loc[valid_mask].reset_index(drop=True)
        return frame

    if not pd.io.common.file_exists(file_path):
        imputation_runtime_seconds = _run_imputer()
    elif os.path.exists(timing_path):
        with open(timing_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        imputation_runtime_seconds = float(payload.get("imputation_runtime_seconds", np.nan))
    else:
        historical_runtime = _lookup_historical_imputation_runtime(args, imputation_name)
        if not np.isnan(historical_runtime):
            imputation_runtime_seconds = historical_runtime
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation_name,
                imputation_runtime_seconds,
            )

    clean_df = _align_to_active_rows(_normalize_frame(load_data(args, args.missing_mechanism, imputation_name)))
    expected_rows = getattr(args, "expected_imputed_rows", None)
    if expected_rows is not None and clean_df.shape[0] != int(expected_rows):
        imputation_runtime_seconds = _run_imputer()
        clean_df = _align_to_active_rows(
            _normalize_frame(load_data(args, args.missing_mechanism, imputation_name))
        )
        if clean_df.shape[0] != int(expected_rows):
            raise ValueError(
                f"Imputer {imputation_name} produced {clean_df.shape[0]} rows, "
                f"but the active evaluation split expects {int(expected_rows)} rows."
            )
    if clean_df.isnull().any().any():
        clean_df = _fill_missing_values(clean_df)
    if ground_df is not None and 0 in clean_df.columns:
        ground_labels = _normalize_frame(ground_df)[0].to_numpy()
        if clean_df.shape[0] != len(ground_labels):
            clean_df = _align_to_active_rows(clean_df)
        if clean_df.shape[0] != len(ground_labels):
            raise ValueError(
                f"Cannot assign {len(ground_labels)} ground-truth labels to "
                f"{clean_df.shape[0]} imputed rows for {imputation_name}."
            )
        clean_df[0] = ground_labels
    if 0 in clean_df.columns and clean_df[0].isnull().any():
        raise ValueError(
            f"Imputer {imputation_name} produced NaN values in label column 0 after post-processing."
        )
    return clean_df, imputation_runtime_seconds


def _save_stage_frame(path, frame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def _compute_masked_error_metrics(ground_df, missing_df, imputed_df, label_col=0):
    if ground_df is None:
        return {
            "rmse_masked": np.nan,
            "nrmse": np.nan,
            "masked_cell_count": 0,
        }

    feature_cols = [col for col in ground_df.columns if col != label_col]
    rmse_terms = []
    nrmse_terms = []
    total_missing_cells = 0

    for col in feature_cols:
        missing_mask = missing_df[col].isna().to_numpy()
        if not missing_mask.any():
            continue

        truth = pd.to_numeric(ground_df.loc[missing_mask, col], errors="coerce").to_numpy(dtype=float)
        estimate = pd.to_numeric(imputed_df.loc[missing_mask, col], errors="coerce").to_numpy(dtype=float)
        valid_mask = ~(np.isnan(truth) | np.isnan(estimate))
        if not valid_mask.any():
            continue

        truth = truth[valid_mask]
        estimate = estimate[valid_mask]
        rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
        rmse_terms.append(rmse)
        total_missing_cells += int(valid_mask.sum())

        full_col = pd.to_numeric(ground_df[col], errors="coerce").to_numpy(dtype=float)
        full_col = full_col[~np.isnan(full_col)]
        data_range = float(full_col.max() - full_col.min()) if full_col.size else 0.0
        if data_range > 0:
            nrmse_terms.append(rmse / data_range)

    return {
        "rmse_masked": float(np.mean(rmse_terms)) if rmse_terms else np.nan,
        "nrmse": float(np.mean(nrmse_terms)) if nrmse_terms else np.nan,
        "masked_cell_count": int(total_missing_cells),
    }


def evaluate_imputation_method(
    args,
    classifier_name,
    imputation_name,
    dataset_name,
    ground_df,
    reference_bundle,
    train_idx,
    test_idx,
    hard_percent,
    stage_cache_dir=None,
):
    has_truth = bool(reference_bundle.get("has_ground_truth", True))
    dataset_category = reference_bundle.get("dataset_category", "no_real_missing")
    reference_clean_source = reference_bundle.get("reference_clean_source", "unknown")
    truth_labels = ground_df if has_truth and ground_df is not None else None
    imputed_df, imputation_runtime_seconds = _prepare_imputed_data(
        args,
        dataset_name,
        imputation_name,
        ground_df=truth_labels,
    )
    if has_truth and ground_df is not None:
        if imputed_df.shape[0] != ground_df.shape[0]:
            raise ValueError(
                f"Imputer {imputation_name} produced {imputed_df.shape[0]} rows, "
                f"but ground truth has {ground_df.shape[0]} rows after label filtering."
            )
        imputed_df[0] = ground_df[0].to_numpy()
    imputed_feature_df = imputed_df.loc[:, imputed_df.columns != 0]
    validate_feature_frame(
        imputed_feature_df,
        context=(
            f"imputed feature frame for dataset={args.dataset}, "
            f"imputer={imputation_name}, classifier={classifier_name}"
        ),
    )
    if stage_cache_dir is not None:
        _save_stage_frame(f"{stage_cache_dir}/01_imputed.csv", imputed_df)

    if has_truth and ground_df is not None:
        # M1: quantify how much the completed table departs from the full-data
        # structure before looking at any classifier output.
        structure_summary, structure_feature = compute_structure_metrics(ground_df, imputed_df, label_col=0)
    else:
        structure_summary = {
            "m1_mean_shift": np.nan,
            "m1_std_shift": np.nan,
            "m1_w1_mean": np.nan,
            "m1_w1_relative_iqr_mean": np.nan,
            "m1_w1_feature_count": 0,
            "m1_w1_zero_iqr_count": 0,
            "m1_w1_metric_version": np.nan,
            "m1_corr_frob": np.nan,
            "m1_fisher_delta": np.nan,
        }
        structure_feature = pd.DataFrame(
            columns=[
                "feature",
                "mean_abs_diff",
                "std_abs_diff",
                "wasserstein_1",
                "ground_q25",
                "ground_q75",
                "ground_iqr",
                "wasserstein_1_iqr_scaled",
                "wasserstein_1_relative_iqr",
                "w1_scale_degenerate",
                "ground_finite_count",
            ]
        )

    if stage_cache_dir is not None:
        structure_summary_df = pd.DataFrame([structure_summary])
        _save_stage_frame(f"{stage_cache_dir}/02_structure_summary.csv", structure_summary_df)
        _save_stage_frame(f"{stage_cache_dir}/03_structure_features.csv", structure_feature)
    error_summary = _compute_masked_error_metrics(
        ground_df=reference_bundle["ground_df"],
        missing_df=reference_bundle["missing_df"],
        imputed_df=imputed_df,
        label_col=0,
    )
    if stage_cache_dir is not None:
        error_summary_df = pd.DataFrame([error_summary])
        _save_stage_frame(f"{stage_cache_dir}/04_masked_error.csv", error_summary_df)

    ref_model = reference_bundle["model"]
    X_ground_test = reference_bundle["X_ground_test"]
    y_true = reference_bundle["y_true"]
    ground_outputs = reference_bundle["ground_outputs"]
    sample_ids = np.asarray(test_idx)

    quantile = max(0.0, min(1.0, hard_percent / 100.0))
    if quantile == 0:
        hard_mask = np.zeros_like(ground_outputs["margin"], dtype=bool)
    else:
        # Hard samples are defined once from the reference model's smallest
        # margins on the ground-truth test set.
        threshold = np.quantile(ground_outputs["margin"], quantile)
        hard_mask = ground_outputs["margin"] <= threshold
    if stage_cache_dir is not None:
        hard_df = pd.DataFrame(
            {
                "sample_id": np.asarray(test_idx),
                "margin_ref": ground_outputs["margin"],
                "is_hard": hard_mask,
            }
        )
        _save_stage_frame(f"{stage_cache_dir}/05_hard_samples.csv", hard_df)

    X_imputed_test = imputed_feature_df.loc[test_idx]
    train_start = time.perf_counter()
    imputed_model, imputed_model_meta = fit_retrained_model(
        classifier_name,
        imputed_df,
        train_idx,
        label_col=0,
        tuning_cv_folds=getattr(args, "tuning_cv_folds", 5),
        tuning_scoring=getattr(args, "tuning_scoring", "f1_macro"),
        tuning_n_jobs=getattr(args, "tuning_n_jobs", -1),
        compute_backend=getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu")),
        gpu_device=getattr(args, "gpu_device", 0),
    )
    train_runtime_seconds = time.perf_counter() - train_start
    if stage_cache_dir is not None:
        imputed_train_df = imputed_df.loc[train_idx].reset_index(drop=True)
        imputed_train_df.insert(0, "sample_id", train_idx)
        _save_stage_frame(f"{stage_cache_dir}/06_imputed_train.csv", imputed_train_df)
        imputed_test_df = X_imputed_test.copy().reset_index(drop=True)
        imputed_test_df.insert(0, "sample_id", test_idx)
        _save_stage_frame(f"{stage_cache_dir}/07_imputed_test_features.csv", imputed_test_df)

    # M2: compare predictions from the same reference model under changed
    # inputs, and from a retrained model under the full impute-then-classify
    # path.
    decision_summary, sample_level, _, retrained_outputs, decision_timing = evaluate_decision_consequences(
        ref_model=ref_model,
        imputed_model=imputed_model,
        X_ground_test=X_ground_test,
        X_imputed_test=X_imputed_test,
        y_true=y_true.to_numpy(),
        sample_ids=sample_ids,
        hard_mask=hard_mask,
        metadata={
            "dataset": args.dataset,
            "iter": getattr(args, "iter", np.nan),
            "seed": getattr(args, "seed", np.nan),
            "missing_percent": args.missing_percent,
            "missing_mechanism": args.missing_mechanism,
            "imputer": imputation_name,
            "classifier": classifier_name,
        },
    )
    if stage_cache_dir is not None:
        decision_summary_df = pd.DataFrame([decision_summary])
        _save_stage_frame(f"{stage_cache_dir}/08_decision_summary.csv", decision_summary_df)
        _save_stage_frame(f"{stage_cache_dir}/09_sample_level.csv", sample_level)

    inference_runtime_seconds = float(decision_timing["inference_runtime_seconds"])
    total_runtime_seconds = float(
        np.nansum([imputation_runtime_seconds, train_runtime_seconds, inference_runtime_seconds])
    )

    X_missing_test = reference_bundle["missing_filled_df"].loc[test_idx, reference_bundle["missing_filled_df"].columns != 0]
    missing_outputs = reference_bundle["missing_model"].predict(X_missing_test)
    # Delta-F1 is measured against the missing-data baseline to answer whether
    # the imputation was worth doing in the end-to-end pipeline.
    f1_missing = float(f1_score(y_true, missing_outputs, average="macro", zero_division=0))
    f1_imputed = float(f1_score(y_true, retrained_outputs["y_pred"], average="macro", zero_division=0))
    acc_imputed = float(accuracy_score(y_true, retrained_outputs["y_pred"]))
    f1_ground_ref = float(f1_score(y_true, ground_outputs["y_pred"], average="macro", zero_division=0))

    summary = {
        "dataset": args.dataset,
        "dataset_category": dataset_category,
        "reference_clean_source": reference_clean_source,
        "has_ground_truth": int(has_truth),
        "iter": getattr(args, "iter", np.nan),
        "seed": getattr(args, "seed", np.nan),
        "missing_percent": args.missing_percent,
        "max_missing_lenth": args.max_missing_lenth,
        "missing_mechanism": args.missing_mechanism,
        "imputer": imputation_name,
        "imputer_family": get_imputer_family(imputation_name),
        "classifier": classifier_name,
        "f1_ground_ref": f1_ground_ref,
        "f1_missing": f1_missing,
        "f1_imputed": f1_imputed,
        "accuracy_imputed": acc_imputed,
        "delta_f1": f1_imputed - f1_missing,
        "imputation_runtime_seconds": float(imputation_runtime_seconds) if not np.isnan(imputation_runtime_seconds) else np.nan,
        "train_runtime_seconds": float(train_runtime_seconds),
        "inference_runtime_seconds": inference_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "runtime_seconds": total_runtime_seconds,
        "reference_best_params": json.dumps(
            reference_bundle.get("reference_model_meta", {}).get("best_params", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "reference_best_cv_score": float(reference_bundle.get("reference_model_meta", {}).get("best_cv_score", np.nan)),
        "missing_best_params": json.dumps(
            reference_bundle.get("missing_model_meta", {}).get("best_params", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "missing_best_cv_score": float(reference_bundle.get("missing_model_meta", {}).get("best_cv_score", np.nan)),
        "imputed_best_params": json.dumps(
            imputed_model_meta.get("best_params", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "imputed_best_cv_score": float(imputed_model_meta.get("best_cv_score", np.nan)),
        "tuning_scoring": getattr(args, "tuning_scoring", "f1_macro"),
        "tuning_cv_folds": int(getattr(args, "tuning_cv_folds", 5)),
    }
    if stage_cache_dir is not None:
        timing_df = pd.DataFrame(
            [
                {
                    "imputation_runtime_seconds": summary["imputation_runtime_seconds"],
                    "train_runtime_seconds": summary["train_runtime_seconds"],
                    "inference_runtime_seconds": summary["inference_runtime_seconds"],
                    "total_runtime_seconds": summary["total_runtime_seconds"],
                }
            ]
        )
        _save_stage_frame(f"{stage_cache_dir}/04_runtime_breakdown.csv", timing_df)
    summary.update(error_summary)
    summary.update(structure_summary)
    summary.update(decision_summary)
    if stage_cache_dir is not None:
        summary_df = pd.DataFrame([summary])
        _save_stage_frame(f"{stage_cache_dir}/10_summary.csv", summary_df)

    structure_metadata = {
        "dataset": args.dataset,
        "dataset_category": dataset_category,
        "iter": getattr(args, "iter", 0),
        "seed": getattr(args, "seed", np.nan),
        "missing_percent": getattr(args, "missing_percent", np.nan),
        "max_missing_lenth": getattr(args, "max_missing_lenth", np.nan),
        "missing_mechanism": getattr(args, "missing_mechanism", ""),
        "imputer": imputation_name,
        "classifier": classifier_name,
    }
    for position, (column, value) in enumerate(structure_metadata.items()):
        structure_feature.insert(position, column, value)
    return summary, sample_level, structure_feature
