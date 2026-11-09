import argparse
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import holoclean
import pandas as pd
from detect import NullDetector, ViolationDetector
from repair.featurize import (
    ConstraintFeaturizer,
    FreqFeaturizer,
    InitAttrFeaturizer,
    OccurAttrFeaturizer,
)
from utils import NULL_REPR


REPO_ROOT = Path(__file__).resolve().parents[3]


def _normalize_constraint_column_name(name):
    return re.sub(r"[^0-9a-z]+", "", str(name).strip().lower())


def _resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_parser():
    parser = argparse.ArgumentParser(description="Run the vendored Holoclean repair pipeline")
    parser.add_argument("--dataset-name", required=True, help="run stem used for logging/debugging")
    parser.add_argument("--save-name", required=True, help="base filename used for the repaired CSV")
    parser.add_argument("--dataset", required=True, help="dataset identifier, e.g. iris")
    parser.add_argument("--input-path", required=True, help="CSV with missing values to repair")
    parser.add_argument("--input-has-header", action="store_true", help="read the input CSV with a header row")
    parser.add_argument("--constraint-path", help="optional denial constraint file")
    parser.add_argument("--output-path", required=True, help="destination CSV path for repaired output")
    parser.add_argument("--timing-path", help="destination JSON path for timing/coverage metadata")
    parser.add_argument("--db-user", default=os.environ.get("DB_USER", "holocleanuser"))
    parser.add_argument("--db-pwd", default=os.environ.get("DB_PWD", ""))
    parser.add_argument("--db-host", default=os.environ.get("DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DB_PORT", "5432")))
    parser.add_argument("--db-name", default=os.environ.get("DB_NAME", "holo"))
    parser.add_argument("--domain-thresh-1", type=float, default=None)
    parser.add_argument("--domain-thresh-2", type=float, default=None)
    parser.add_argument("--weak-label-thresh", type=float, default=None)
    parser.add_argument("--max-domain", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=None)
    return parser


def _resolve_holoclean_runtime_config(args):
    config = {
        "domain_thresh_1": 0.0,
        "domain_thresh_2": 0.0,
        "weak_label_thresh": 0.99,
        "max_domain": 10000,
        "threads": 1,
        "timeout": 3 * 60000,
    }
    if args.dataset == "obesity":
        # Obesity has dense denial constraints; keep candidate domains tight so
        # constraint SQL stays tractable.
        config.update(
            {
                "domain_thresh_1": 0.01,
                "domain_thresh_2": 0.05,
                "weak_label_thresh": 0.95,
                "max_domain": 64,
                "threads": 1,
                "timeout": 60000,
            }
        )

    if args.domain_thresh_1 is not None:
        config["domain_thresh_1"] = float(args.domain_thresh_1)
    if args.domain_thresh_2 is not None:
        config["domain_thresh_2"] = float(args.domain_thresh_2)
    if args.weak_label_thresh is not None:
        config["weak_label_thresh"] = float(args.weak_label_thresh)
    if args.max_domain is not None:
        config["max_domain"] = int(args.max_domain)
    if args.threads is not None:
        config["threads"] = int(args.threads)
    if args.timeout_ms is not None:
        config["timeout"] = int(args.timeout_ms)
    return config


def _infer_column_names(dataset, raw_input, constraint_path):
    if all(isinstance(column, str) for column in raw_input.columns):
        return [_normalize_constraint_column_name(column) for column in raw_input.columns]

    if constraint_path is None:
        return [f"c{idx}" for idx in range(raw_input.shape[1])]

    ground_truth_path = REPO_ROOT / "data" / "Ground_truth_data" / f"{dataset}.csv"
    if not ground_truth_path.exists():
        return [f"c{idx}" for idx in range(raw_input.shape[1])]

    try:
        header_frame = pd.read_csv(ground_truth_path, nrows=0)
    except Exception:
        return [f"c{idx}" for idx in range(raw_input.shape[1])]

    header_columns = [_normalize_constraint_column_name(col) for col in header_frame.columns]
    if len(header_columns) != raw_input.shape[1]:
        return [f"c{idx}" for idx in range(raw_input.shape[1])]
    return header_columns


def _apply_fallback(missing_data, holoclean_output, *, label_column=0):
    result = holoclean_output.copy()
    for idx, column in enumerate(result.columns):
        if idx == label_column:
            continue
        missing_mask = _is_missing_series(missing_data[column]) & _is_missing_series(result[column])
        if not missing_mask.any():
            continue
        observed_series = result[column].mask(_is_missing_series(result[column]))
        numeric_series = pd.to_numeric(observed_series, errors="coerce")
        if numeric_series.notna().any():
            fill_value = numeric_series.mean()
        else:
            modes = observed_series.mode(dropna=True)
            if modes.empty:
                continue
            fill_value = modes.iloc[0]
        result.loc[missing_mask, column] = fill_value
    return result


def _is_missing_series(series):
    normalized = series.astype(str).str.strip().str.lower()
    return series.isna() | normalized.eq(NULL_REPR.lower())


def _count_repairs(missing_data, repaired_data):
    missing_mask = missing_data.apply(_is_missing_series).to_numpy()
    repaired_mask = repaired_data.apply(_is_missing_series).to_numpy()
    total_missing = int(missing_mask.sum())
    repaired = int((missing_mask & ~repaired_mask).sum())
    remaining = int((missing_mask & repaired_mask).sum())
    coverage = float(repaired / total_missing) if total_missing else 1.0
    return total_missing, repaired, remaining, coverage


def _is_recoverable_holoclean_error(exc):
    message = str(exc)
    return (
        "No weak labels available" in message
        or "None of ['_vid_'] are in the columns" in message
    )


def main():
    args = build_parser().parse_args()
    runtime_config = _resolve_holoclean_runtime_config(args)

    input_path = _resolve_repo_path(args.input_path)
    output_path = _resolve_repo_path(args.output_path)
    timing_path = _resolve_repo_path(args.timing_path) if args.timing_path else output_path.with_name(
        output_path.name.replace("_clean.csv", "_timing.json")
    )
    constraint_path = _resolve_repo_path(args.constraint_path) if args.constraint_path else None

    if not input_path.exists():
        raise FileNotFoundError(f"Missing Holoclean input CSV: {input_path}")
    if constraint_path is not None and not constraint_path.exists():
        constraint_path = None

    raw_input = pd.read_csv(input_path, header=0 if args.input_has_header else None)

    hc = holoclean.HoloClean(
        db_user=args.db_user,
        db_pwd=args.db_pwd,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        domain_thresh_1=runtime_config["domain_thresh_1"],
        domain_thresh_2=runtime_config["domain_thresh_2"],
        weak_label_thresh=runtime_config["weak_label_thresh"],
        max_domain=runtime_config["max_domain"],
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=runtime_config["threads"],
        batch_size=1,
        verbose=True,
        timeout=runtime_config["timeout"],
        feature_norm=False,
        weight_norm=False,
        print_fw=True,
    ).session

    holoclean_start = time.perf_counter()
    (REPO_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="holoclean_legacy_", dir=str(REPO_ROOT / "tmp")) as tmpdir:
        normalized_input = raw_input.copy()
        normalized_input.columns = _infer_column_names(args.dataset, raw_input, constraint_path)
        normalized_input_path = Path(tmpdir) / "input.csv"
        normalized_input.to_csv(normalized_input_path, index=False)

        hc.load_data(args.dataset, str(normalized_input_path))

        detectors = [NullDetector()]
        featurizers = [
            InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
        ]

        if constraint_path is not None:
            hc.load_dcs(str(constraint_path))
            hc.ds.set_constraints(hc.get_dcs())
            if hc.get_dcs():
                detectors.append(ViolationDetector())
                featurizers.append(ConstraintFeaturizer())
        else:
            hc.ds.set_constraints([])

        hc.detect_errors(detectors)
        hc.setup_domain()
        try:
            holoclean_output = hc.repair_errors(
                featurizers,
                name=args.dataset,
                save_name=args.save_name,
                output_path=str(output_path),
            )
        except Exception as exc:
            if not _is_recoverable_holoclean_error(exc):
                raise
            logging.warning(
                "Holoclean produced no usable repairs for %s; falling back to the original input. Cause: %s",
                args.dataset_name,
                exc,
            )
            holoclean_output = normalized_input.copy()
    holoclean_runtime_seconds = time.perf_counter() - holoclean_start

    if not isinstance(holoclean_output, pd.DataFrame):
        holoclean_output = pd.read_csv(output_path, header=None)
    aligned_input = raw_input.copy()
    aligned_input.columns = list(holoclean_output.columns)

    total_missing, repaired_cells, remaining_after_holoclean, repair_coverage = _count_repairs(
        aligned_input,
        holoclean_output,
    )

    fallback_start = time.perf_counter()
    final_output = _apply_fallback(aligned_input, holoclean_output)
    fallback_runtime_seconds = time.perf_counter() - fallback_start
    final_output.to_csv(output_path, index=False, header=False)

    final_missing_cells = int(
        (
            aligned_input.apply(_is_missing_series).to_numpy()
            & final_output.apply(_is_missing_series).to_numpy()
        ).sum()
    )
    total_runtime_seconds = holoclean_runtime_seconds + fallback_runtime_seconds

    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "imputation_runtime_seconds": float(total_runtime_seconds),
                "holoclean_runtime_seconds": float(holoclean_runtime_seconds),
                "fallback_runtime_seconds": float(fallback_runtime_seconds),
                "dataset": args.dataset,
                "dataset_name": args.dataset_name,
                "imputer": "Holoclean",
                "fallback_applied": True,
                "input_missing_cells": total_missing,
                "holoclean_repaired_cells": repaired_cells,
                "holoclean_remaining_missing_cells": remaining_after_holoclean,
                "repair_coverage": repair_coverage,
                "final_remaining_missing_cells": final_missing_cells,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
