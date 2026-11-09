import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

from code.config.dataset_profiles import DATASET_CATEGORIES
from code.config.args import add_compute_backend_arguments, resolve_batch_iterations, resolve_batch_seeds
from code.config.dataset_profiles import DATASET_CATEGORY_NO_REAL_MISSING
from code.config.settings import get_default_config
from code.core.data_io import build_clean_data_path, build_clean_timing_path, build_run_stem
from code.core.path_config import add_workspace_path_arguments, apply_workspace_paths


DEFAULTS = get_default_config()
IMPUTE_METHOD_TIMEOUT_SECONDS = 600


def _run_command(command, *, log_path=None):
    if log_path is None:
        subprocess.run(command, check=True)
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        started_at = datetime.now().isoformat(timespec="seconds")
        handle.write(f"\n=== command start {started_at} ===\n")
        handle.write("$ " + " ".join(str(part) for part in command) + "\n")
        handle.flush()
        try:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            finished_at = datetime.now().isoformat(timespec="seconds")
            handle.write(f"=== command failed {finished_at} returncode={exc.returncode} ===\n")
            raise
        else:
            finished_at = datetime.now().isoformat(timespec="seconds")
            handle.write(f"=== command end {finished_at} ===\n")


def _resolve_parallel_jobs(value):
    requested = int(value)
    if requested < 0:
        raise ValueError("--parallel_jobs must be >= 0")
    if requested == 0:
        return max(1, min(4, (os.cpu_count() or 1) - 1))
    return requested


def _safe_log_stem(value):
    safe_chars = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "task"


def _append_workspace_args(command, args):
    command.extend(
        [
            "--workspace_dir",
            args.workspace_dir,
            "--ground_truth_data_dir",
            args.ground_truth_data_dir,
            "--real_missing_data_dir",
            args.real_missing_data_dir,
            "--real_missing_truth_dir",
            args.real_missing_truth_dir,
            "--real_missing_label_col",
            str(args.real_missing_label_col),
            "--missing_data_dir",
            args.missing_data_dir,
            "--clean_data_dir",
            args.clean_data_dir,
            "--result_dir",
            args.result_dir,
        ]
    )
    return command


def _write_stage_report(stage_name, records, result_dir):
    logs_root = Path(result_dir) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    total_count = len(records)
    failed_records = [record for record in records if record["status"] == "failed"]
    skipped_records = [record for record in records if record["status"] == "skipped"]
    ok_count = len([record for record in records if record["status"] == "ok"])
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage_name,
        "total": total_count,
        "ok": ok_count,
        "skipped": len(skipped_records),
        "failed": len(failed_records),
        "records": records,
    }
    summary_json_path = logs_root / f"{stage_name}_batch_report.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[{stage_name}] summary total={total_count} ok={ok_count} skipped={len(skipped_records)} "
        f"failed={len(failed_records)} "
        f"report={summary_json_path}"
    )
    return summary_json_path


def _build_imputation_output_paths(args, dataset, iteration, missing_percent, mechanism, method):
    run_args = SimpleNamespace(
        dataset=dataset,
        seed=getattr(args, "seed", DEFAULTS["base_seed"]),
        iter=iteration,
        missing_percent=missing_percent,
        max_missing_lenth=getattr(args, "max_missing_lenth", DEFAULTS["max_missing_lenth"]),
        missing_mechanism=mechanism,
        clean_data_dir=getattr(args, "clean_data_dir", DEFAULTS["clean_data_dir"]),
    )
    return Path(build_clean_data_path(run_args, method)), Path(build_clean_timing_path(run_args, method))


def _base_single_args(parser):
    add_workspace_path_arguments(parser)
    parser.add_argument("--dataset", type=str, default=DEFAULTS["dataset"])
    parser.add_argument("--dataset_category", choices=DATASET_CATEGORIES, default=DEFAULTS["dataset_category"])
    parser.add_argument("--iter", type=int, default=DEFAULTS["iter"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["base_seed"])
    parser.add_argument("--missing_percent", type=float, default=DEFAULTS["missing_percent"])
    parser.add_argument("--max_missing_lenth", type=int, default=DEFAULTS["max_missing_lenth"])
    parser.add_argument("--missing_mechanism", choices=["MCAR", "MAR", "MNAR"], default=DEFAULTS["missing_mechanism"])
    parser.add_argument("--mar_logistic_beta", type=float, default=DEFAULTS["mar_logistic_beta"])
    parser.add_argument("--mar_logistic_clip", type=float, default=DEFAULTS["mar_logistic_clip"])


def _append_single_experiment_args(parser):
    _base_single_args(parser)
    add_compute_backend_arguments(parser)
    parser.add_argument("--class_method", nargs="+", default=DEFAULTS["class_method"])
    parser.add_argument("--imputation_method", nargs="+", default=DEFAULTS["imputation_method"])
    parser.add_argument("--hard_sample_percent", type=float, default=DEFAULTS["hard_sample_percent"])
    parser.add_argument("--tuning_cv_folds", type=int, default=DEFAULTS["tuning_cv_folds"])
    parser.add_argument("--tuning_n_jobs", type=int, default=DEFAULTS["tuning_n_jobs"])
    parser.add_argument("--tuning_scoring", type=str, default=DEFAULTS["tuning_scoring"])


def _append_imputation_args(parser):
    _base_single_args(parser)
    parser.add_argument("--imputation_method", nargs="+", default=DEFAULTS["imputation_method"])
    parser.add_argument("--method_timeout_seconds", type=int, default=IMPUTE_METHOD_TIMEOUT_SECONDS)


def _append_stage_array_args(parser, *, include_seeds=False):
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--missing_percents", nargs="+", type=float, default=None)
    parser.add_argument("--mechanisms", nargs="+", choices=["MCAR", "MAR", "MNAR"], default=None)
    parser.add_argument("--iters", nargs="+", type=int, default=None)
    if include_seeds:
        parser.add_argument("--seeds", nargs="+", type=int, default=None)


def _build_generate_command(args):
    command = [
        sys.executable,
        "-m",
        "code.generate_missing_data",
        "--dataset",
        args.dataset,
        "--dataset_category",
        args.dataset_category,
        "--iter",
        str(args.iter),
        "--seed",
        str(args.seed),
        "--missing_percent",
        str(args.missing_percent),
        "--max_missing_lenth",
        str(args.max_missing_lenth),
        "--missing_mechanism",
        args.missing_mechanism,
        "--mar_logistic_beta",
        str(args.mar_logistic_beta),
        "--mar_logistic_clip",
        str(args.mar_logistic_clip),
    ]
    return _append_workspace_args(command, args)


def _clone_args(args, **overrides):
    payload = vars(args).copy()
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _resolve_stage_grid(args, *, include_seeds=False):
    datasets = list(args.datasets) if getattr(args, "datasets", None) else [args.dataset]
    missing_percents = (
        [float(value) for value in args.missing_percents]
        if getattr(args, "missing_percents", None)
        else [float(args.missing_percent)]
    )
    mechanisms = list(args.mechanisms) if getattr(args, "mechanisms", None) else [args.missing_mechanism]
    iterations = list(args.iters) if getattr(args, "iters", None) else [int(args.iter)]
    seeds = list(args.seeds) if include_seeds and getattr(args, "seeds", None) else [int(args.seed)]

    if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
        missing_percents = missing_percents[:1]
        mechanisms = mechanisms[:1]

    return {
        "datasets": datasets,
        "missing_percents": missing_percents,
        "mechanisms": mechanisms,
        "iterations": iterations,
        "seeds": seeds,
    }


def _build_run_command(args):
    command = [
        sys.executable,
        "-m",
        "code.classification",
        "--compute_backend",
        args.compute_backend,
        "--gpu_device",
        str(args.gpu_device),
        "--dataset",
        args.dataset,
        "--dataset_category",
        args.dataset_category,
        "--iter",
        str(args.iter),
        "--seed",
        str(args.seed),
        "--missing_percent",
        str(args.missing_percent),
        "--max_missing_lenth",
        str(args.max_missing_lenth),
        "--missing_mechanism",
        args.missing_mechanism,
        "--mar_logistic_beta",
        str(args.mar_logistic_beta),
        "--mar_logistic_clip",
        str(args.mar_logistic_clip),
        "--hard_sample_percent",
        str(args.hard_sample_percent),
        "--tuning_cv_folds",
        str(args.tuning_cv_folds),
        "--tuning_n_jobs",
        str(args.tuning_n_jobs),
        "--tuning_scoring",
        args.tuning_scoring,
    ]
    if args.class_method:
        command.extend(["--class_method", *args.class_method])
    if args.imputation_method:
        command.extend(["--imputation_method", *args.imputation_method])
    return _append_workspace_args(command, args)


def _build_classify_command(args):
    return _build_run_command(args)


def _build_batch_command(args, *, log_prefix=None, report_stage=None, report_filename=None):
    iterations = resolve_batch_iterations(args)
    seeds = resolve_batch_seeds(args)
    command = [
        sys.executable,
        "-m",
        "code.scripts.run_experiments",
        "--compute_backend",
        args.compute_backend,
        "--gpu_device",
        str(args.gpu_device),
        "--max_missing_lenth",
        str(args.max_missing_lenth),
        "--hard_sample_percent",
        str(args.hard_sample_percent),
        "--tuning_cv_folds",
        str(args.tuning_cv_folds),
        "--tuning_n_jobs",
        str(args.tuning_n_jobs),
        "--parallel_jobs",
        str(args.parallel_jobs),
        "--tuning_scoring",
        args.tuning_scoring,
    ]
    command.extend(["--iters", *[str(v) for v in iterations]])
    if seeds:
        command.extend(["--seeds", *[str(v) for v in seeds]])
    if args.datasets:
        command.extend(["--datasets", *args.datasets])
    command.extend(["--dataset_category", args.dataset_category])
    if args.missing_percents:
        command.extend(["--missing_percents", *[str(v) for v in args.missing_percents]])
    if args.mechanisms:
        command.extend(["--mechanisms", *args.mechanisms])
    if args.classifiers:
        command.extend(["--classifiers", *args.classifiers])
    if args.imputers:
        command.extend(["--imputers", *args.imputers])
    if log_prefix:
        command.extend(["--log_prefix", log_prefix])
    if report_stage:
        command.extend(["--report_stage", report_stage])
    if report_filename:
        command.extend(["--report_filename", report_filename])
    return _append_workspace_args(command, args)


def _build_analyze_command(args):
    return [
        sys.executable,
        "-m",
        "code.scripts.analyze_results",
        "--input",
        args.input,
        "--output_dir",
        args.output_dir,
    ]


def _resolve_eval_input_root(input_path):
    candidate = Path(str(input_path))
    if candidate.is_dir():
        return candidate
    if candidate.name in {"run_summary.csv", "sample_level.csv"}:
        return candidate.parent
    return candidate.parent


def _resolve_eval_output_dir(base_output_dir, input_path, index, default_leaf):
    if not base_output_dir or base_output_dir == f"result/{default_leaf}":
        return str(_resolve_eval_input_root(input_path) / default_leaf)
    candidate = Path(str(input_path))
    name = candidate.stem if candidate.is_file() else candidate.name
    if not name:
        name = f"run_{index + 1}"
    return str(Path(base_output_dir) / name)


def _resolve_default_result_path(args, value, default_leaf=None):
    if default_leaf is None:
        return args.result_dir if not value or value == "result" else value
    default_value = f"result/{default_leaf}"
    if not value or value == default_value:
        return str(Path(args.result_dir) / default_leaf)
    return value


def _resolve_default_summary_input(args, value):
    if value and value != "result":
        return value
    result_root = Path(args.result_dir)
    direct_summary = result_root / "run_summary.csv"
    if direct_summary.exists():
        return str(result_root)
    combined_dir = result_root / "combined_csv_cleaned"
    if combined_dir.exists():
        return str(combined_dir)
    return args.result_dir


def _resolve_default_sample_level_input(args, value):
    if value and value != "result":
        return value
    return args.result_dir


def _run_imputation_stage(args):
    grid = _resolve_stage_grid(args)
    records = []
    for dataset in grid["datasets"]:
        for iteration in grid["iterations"]:
            for missing_percent in grid["missing_percents"]:
                for mechanism in grid["mechanisms"]:
                    dataset_name = build_run_stem(
                        dataset,
                        missing_percent,
                        args.max_missing_lenth,
                        mechanism,
                        iteration,
                        seed=getattr(args, "seed", DEFAULTS["base_seed"]),
                    )
                    for method in args.imputation_method:
                        clean_path, timing_path = _build_imputation_output_paths(
                            args,
                            dataset,
                            iteration,
                            missing_percent,
                            mechanism,
                            method,
                        )
                        if clean_path.exists() and timing_path.exists():
                            print(f"[impute] skip method={method} dataset={dataset_name} reason=already_imputed")
                            records.append(
                                {
                                    "status": "skipped",
                                    "label": f"{dataset_name}::{method}",
                                    "dataset": dataset,
                                    "dataset_name": dataset_name,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "method": method,
                                    "started_at": None,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "error": "",
                                    "reason": "clean.csv and timing.json already exist",
                                }
                            )
                            continue
                        print(f"[impute] start method={method} dataset={dataset_name}")
                        started_at = datetime.now().isoformat(timespec="seconds")
                        command = [
                            sys.executable,
                            "-m",
                            "code.scripts.run_imputer_task",
                            "--dataset",
                            dataset,
                            "--dataset_category",
                            args.dataset_category,
                            "--iter",
                            str(iteration),
                            "--seed",
                            str(args.seed),
                            "--missing_percent",
                            str(missing_percent),
                            "--max_missing_lenth",
                            str(args.max_missing_lenth),
                            "--missing_mechanism",
                            mechanism,
                            "--mar_logistic_beta",
                            str(args.mar_logistic_beta),
                            "--mar_logistic_clip",
                            str(args.mar_logistic_clip),
                            "--method",
                            method,
                        ]
                        _append_workspace_args(command, args)
                        try:
                            subprocess.run(
                                command,
                                check=True,
                                timeout=getattr(args, "method_timeout_seconds", IMPUTE_METHOD_TIMEOUT_SECONDS),
                            )
                        except subprocess.TimeoutExpired as exc:
                            timeout_seconds = max(1, math.ceil(float(exc.timeout or 0)))
                            error_text = f"TimeoutExpired: Timed out after {timeout_seconds} seconds"
                            print(f"[impute] timeout method={method} dataset={dataset_name} error={error_text}")
                            records.append(
                                {
                                    "status": "failed",
                                    "label": f"{dataset_name}::{method}",
                                    "dataset": dataset,
                                    "dataset_name": dataset_name,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "method": method,
                                    "started_at": started_at,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "error": error_text,
                                }
                            )
                            continue
                        except subprocess.CalledProcessError as exc:
                            error_text = f"CalledProcessError: returncode={exc.returncode}"
                            print(f"[impute] failed method={method} dataset={dataset_name} error={error_text}")
                            records.append(
                                {
                                    "status": "failed",
                                    "label": f"{dataset_name}::{method}",
                                    "dataset": dataset,
                                    "dataset_name": dataset_name,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "method": method,
                                    "started_at": started_at,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "error": error_text,
                                }
                            )
                            continue
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            print(f"[impute] failed method={method} dataset={dataset_name} error={error_text}")
                            records.append(
                                {
                                    "status": "failed",
                                    "dataset": dataset,
                                    "dataset_name": dataset_name,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "method": method,
                                    "started_at": started_at,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "error": error_text,
                                }
                            )
                            continue

                        print(f"[impute] done method={method}")
                        records.append(
                            {
                                "status": "ok",
                                "dataset": dataset,
                                "dataset_name": dataset_name,
                                "iter": iteration,
                                "missing_percent": missing_percent,
                                "missing_mechanism": mechanism,
                                "method": method,
                                "started_at": started_at,
                                "finished_at": datetime.now().isoformat(timespec="seconds"),
                                "error": "",
                            }
                        )

    for record in records:
        record["label"] = f"{record['dataset_name']}::{record['method']}"
    _write_stage_report("impute", records, args.result_dir)


def _run_classify_grid_task(args, *, seed, dataset, iteration, missing_percent, mechanism, classifier, imputer):
    run_args = _clone_args(
        args,
        seed=seed,
        dataset=dataset,
        iter=iteration,
        missing_percent=missing_percent,
        missing_mechanism=mechanism,
        class_method=[classifier],
        imputation_method=[imputer],
    )
    label = (
        f"{dataset}_{missing_percent}_{args.max_missing_lenth}_{mechanism}_{iteration}"
        f"_seed{seed}_{classifier}_{imputer}"
    )
    log_path = Path(args.result_dir) / "logs" / "classify_tasks" / f"{_safe_log_stem(label)}.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[classify] start dataset={label} log={log_path}")
    try:
        _run_command(_build_classify_command(run_args), log_path=log_path)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"[classify] failed dataset={label} error={error_text} log={log_path}")
        return {
            "status": "failed",
            "label": label,
            "dataset": dataset,
            "iter": iteration,
            "seed": seed,
            "missing_percent": missing_percent,
            "missing_mechanism": mechanism,
            "classifier": classifier,
            "imputer": imputer,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": error_text,
            "log_path": str(log_path),
        }
    print(f"[classify] done dataset={label} log={log_path}")
    return {
        "status": "ok",
        "label": label,
        "dataset": dataset,
        "iter": iteration,
        "seed": seed,
        "missing_percent": missing_percent,
        "missing_mechanism": mechanism,
        "classifier": classifier,
        "imputer": imputer,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "error": "",
        "log_path": str(log_path),
    }


def _build_plot_command(args):
    return [
        sys.executable,
        "-m",
        "code.scripts.plotting.plot_experiment_results",
        "--summary",
        args.summary,
        "--analysis_dir",
        args.analysis_dir,
        "--output_dir",
        args.output_dir,
        "--top_k",
        str(args.top_k),
    ]


def _build_paper_command(args):
    return [
        sys.executable,
        "-m",
        "code.scripts.run_paper_experiments",
        "--summary",
        args.summary,
        "--sample_level",
        args.sample_level,
        "--output_dir",
        args.output_dir,
        "--num_bins",
        str(args.num_bins),
        "--bootstrap_rounds",
        str(args.bootstrap_rounds),
        "--seed",
        str(args.seed),
    ]


def _add_generate_parser(subparsers):
    parser = subparsers.add_parser("generate", help="generate one missing-data file")
    _base_single_args(parser)
    _append_stage_array_args(parser)
    return parser


def _add_impute_parser(subparsers):
    parser = subparsers.add_parser("impute", help="run imputation only and write clean csv files")
    _append_imputation_args(parser)
    _append_stage_array_args(parser)
    return parser


def _add_classify_parser(subparsers):
    parser = subparsers.add_parser("classify", help="run classification and metrics using existing clean data")
    _append_single_experiment_args(parser)
    _append_stage_array_args(parser, include_seeds=True)
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=1,
        help=(
            "outer parallel workers for classify tasks at "
            "seed x dataset x iter x missing_percent x mechanism x classifier x imputer granularity; "
            "1 keeps legacy serial behavior, 0 auto-picks a conservative value"
        ),
    )
    return parser


def _add_evaluate_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help="run result analysis and optional plots/paper summaries")
    add_workspace_path_arguments(parser)
    parser.add_argument("--input", default="result")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--output_dir", default="result/analysis")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--summary", default="result")
    parser.add_argument("--analysis_dir", default="result/analysis")
    parser.add_argument("--figures_dir", default="result/figures")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--sample_level", default="result")
    parser.add_argument("--paper_dir", default="result/paper_experiments")
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--bootstrap_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULTS["base_seed"])
    return parser


def _configure_batch_parser(parser):
    add_workspace_path_arguments(parser)
    add_compute_backend_arguments(parser)
    parser.add_argument("--datasets", nargs="+", default=DEFAULTS["datasets"])
    parser.add_argument("--dataset_category", choices=DATASET_CATEGORIES, default=DEFAULTS["dataset_category"])
    parser.add_argument("--missing_percents", nargs="+", type=float, default=[DEFAULTS["missing_percent"]])
    parser.add_argument(
        "--mechanisms",
        "--missing_mechanisms",
        dest="mechanisms",
        nargs="+",
        choices=["MCAR", "MAR", "MNAR"],
        default=DEFAULTS["missing_mechanisms"],
    )
    parser.add_argument("--classifiers", nargs="+", default=DEFAULTS["class_method"])
    parser.add_argument("--imputers", nargs="+", default=DEFAULTS["imputation_method"])
    parser.add_argument(
        "--iter",
        type=int,
        default=DEFAULTS["iter"],
        help="number of repeated runs to execute when --iters is not provided",
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=None,
        help="explicit iteration ids to run; overrides --iter repeat count",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["base_seed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULTS["seeds"])
    parser.add_argument("--max_missing_lenth", type=int, default=DEFAULTS["max_missing_lenth"])
    parser.add_argument("--hard_sample_percent", type=float, default=DEFAULTS["hard_sample_percent"])
    parser.add_argument("--tuning_cv_folds", type=int, default=DEFAULTS["tuning_cv_folds"])
    parser.add_argument("--tuning_n_jobs", type=int, default=DEFAULTS["tuning_n_jobs"])
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=12,
        help="outer parallel workers for batch tasks; use 0 to auto-pick a safer value based on CPU cores",
    )
    parser.add_argument("--tuning_scoring", type=str, default=DEFAULTS["tuning_scoring"])
    return parser


def _configure_analyze_parser(parser):
    parser.add_argument("--input", default="result")
    parser.add_argument("--output_dir", default="result/analysis")
    return parser


def _configure_plot_parser(parser):
    parser.add_argument("--summary", default="result")
    parser.add_argument("--analysis_dir", default="result/analysis")
    parser.add_argument("--output_dir", default="result/figures")
    parser.add_argument("--top_k", type=int, default=10)
    return parser


def _configure_paper_parser(parser):
    parser.add_argument("--summary", default="result")
    parser.add_argument("--sample_level", default="result")
    parser.add_argument("--output_dir", default="result/paper_experiments")
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--bootstrap_rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULTS["base_seed"])
    return parser


def _add_full_parser(subparsers):
    parser = subparsers.add_parser("full", help="run the full pipeline for batch experiments")
    _configure_batch_parser(parser)
    parser.add_argument("--summary", default="result")
    parser.add_argument("--analysis_dir", default="result/analysis")
    parser.add_argument("--figures_dir", default="result/figures")
    parser.add_argument("--paper_dir", default="result/paper_experiments")
    parser.add_argument("--sample_level", default="result")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--bootstrap_rounds", type=int, default=500)
    return parser


def _add_core_parser(subparsers):
    parser = subparsers.add_parser(
        "core",
        help="run impute + classify only using existing missing data, without analysis or plotting",
    )
    _configure_batch_parser(parser)
    return parser


def main():
    parser = argparse.ArgumentParser(description="Unified workflow interface for all experiment stages")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    _add_generate_parser(subparsers)
    _add_impute_parser(subparsers)
    _add_classify_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_core_parser(subparsers)
    _add_full_parser(subparsers)

    args = parser.parse_args()
    apply_workspace_paths(args)

    if args.stage == "generate":
        grid = _resolve_stage_grid(args)
        records = []
        for dataset in grid["datasets"]:
            for iteration in grid["iterations"]:
                for missing_percent in grid["missing_percents"]:
                    for mechanism in grid["mechanisms"]:
                        label = f"{dataset}_{missing_percent}_{args.max_missing_lenth}_{mechanism}_{iteration}"
                        started_at = datetime.now().isoformat(timespec="seconds")
                        if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
                            reason = (
                                "real-missing datasets skip synthetic missing-data generation and "
                                "read directly from Real_missing_data/Real_missing_truth"
                            )
                            print(f"[generate] skipped dataset={label} reason={reason}")
                            records.append(
                                {
                                    "status": "skipped",
                                    "label": label,
                                    "dataset": dataset,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "started_at": started_at,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "reason": reason,
                                    "error": "",
                                }
                            )
                            continue
                        run_args = _clone_args(
                            args,
                            dataset=dataset,
                            iter=iteration,
                            missing_percent=missing_percent,
                            missing_mechanism=mechanism,
                        )
                        try:
                            _run_command(_build_generate_command(run_args))
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            print(f"[generate] failed dataset={label} error={error_text}")
                            records.append(
                                {
                                    "status": "failed",
                                    "label": label,
                                    "dataset": dataset,
                                    "iter": iteration,
                                    "missing_percent": missing_percent,
                                    "missing_mechanism": mechanism,
                                    "started_at": started_at,
                                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                                    "error": error_text,
                                }
                            )
                            continue
                        records.append(
                            {
                                "status": "ok",
                                "label": label,
                                "dataset": dataset,
                                "iter": iteration,
                                "missing_percent": missing_percent,
                                "missing_mechanism": mechanism,
                                "started_at": started_at,
                                "finished_at": datetime.now().isoformat(timespec="seconds"),
                                "error": "",
                            }
                        )
        _write_stage_report("generate", records, args.result_dir)
        return

    if args.stage == "impute":
        _run_imputation_stage(args)
        return

    if args.stage == "classify":
        grid = _resolve_stage_grid(args, include_seeds=True)
        tasks = [
            {
                "seed": seed,
                "dataset": dataset,
                "iteration": iteration,
                "missing_percent": missing_percent,
                "mechanism": mechanism,
                "classifier": classifier,
                "imputer": imputer,
            }
            for seed in grid["seeds"]
            for dataset in grid["datasets"]
            for iteration in grid["iterations"]
            for missing_percent in grid["missing_percents"]
            for mechanism in grid["mechanisms"]
            for classifier in args.class_method
            for imputer in args.imputation_method
        ]
        parallel_jobs = _resolve_parallel_jobs(getattr(args, "parallel_jobs", 1))
        records = []
        print(f"[classify] dispatch total={len(tasks)} parallel_jobs={parallel_jobs}")
        if parallel_jobs == 1 or len(tasks) <= 1:
            for task in tasks:
                records.append(_run_classify_grid_task(args, **task))
        else:
            with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
                futures = [executor.submit(_run_classify_grid_task, args, **task) for task in tasks]
                for future in as_completed(futures):
                    records.append(future.result())
        _write_stage_report("classify", records, args.result_dir)
        return

    if args.stage == "evaluate":
        input_paths = list(args.inputs) if args.inputs else [_resolve_default_summary_input(args, args.input)]
        multi_input = len(input_paths) > 1
        records = []
        for index, input_path in enumerate(input_paths):
            label = str(input_path)
            started_at = datetime.now().isoformat(timespec="seconds")
            analyze_args = _clone_args(
                args,
                input=input_path,
                output_dir=_resolve_eval_output_dir(
                    args.output_dir,
                    input_path,
                    index,
                    "analysis",
                ) if multi_input or args.output_dir == "result/analysis" else args.output_dir,
            )
            try:
                _run_command(_build_analyze_command(analyze_args))

                summary_path = input_path if not args.summary or args.summary == "result" else args.summary
                analysis_dir = analyze_args.output_dir if not args.analysis_dir or args.analysis_dir == "result/analysis" else (
                    _resolve_eval_output_dir(args.analysis_dir, input_path, index, "analysis") if multi_input else args.analysis_dir
                )
                figures_dir = (
                    _resolve_eval_output_dir(args.figures_dir, input_path, index, "figures")
                    if multi_input or args.figures_dir == "result/figures"
                    else args.figures_dir
                )
                sample_level_path = (
                    _resolve_default_sample_level_input(args, args.sample_level)
                    if not args.sample_level or args.sample_level == "result"
                    else args.sample_level
                )
                paper_dir = (
                    _resolve_eval_output_dir(args.paper_dir, input_path, index, "paper_experiments")
                    if multi_input or args.paper_dir == "result/paper_experiments"
                    else args.paper_dir
                )

                if args.plot:
                    _run_command(
                        [
                            sys.executable,
                            "-m",
                            "code.scripts.plotting.plot_experiment_results",
                            "--summary",
                            summary_path,
                            "--analysis_dir",
                            analysis_dir,
                            "--output_dir",
                            figures_dir,
                            "--top_k",
                            str(args.top_k),
                        ]
                    )
                if args.paper:
                    _run_command(
                        [
                            sys.executable,
                            "-m",
                            "code.scripts.run_paper_experiments",
                            "--summary",
                            summary_path,
                            "--sample_level",
                            sample_level_path,
                            "--output_dir",
                            paper_dir,
                            "--num_bins",
                            str(args.num_bins),
                            "--bootstrap_rounds",
                            str(args.bootstrap_rounds),
                            "--seed",
                            str(args.seed),
                        ]
                    )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                print(f"[evaluate] failed input={label} error={error_text}")
                records.append(
                    {
                        "status": "failed",
                        "label": label,
                        "started_at": started_at,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": error_text,
                    }
                )
                continue
            records.append(
                {
                    "status": "ok",
                    "label": label,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": "",
                }
            )
        _write_stage_report("evaluate", records, args.result_dir)
        return

    if args.stage == "core":
        core_command = _build_batch_command(
            args,
            log_prefix="core-batch",
            report_stage="core_batch",
            report_filename="core_batch_report.json",
        )
        core_command.append("--skip_missing_generation")
        _run_command(core_command)
        return

    if args.stage == "full":
        summary_root = _resolve_default_summary_input(args, args.summary)
        analysis_dir = _resolve_default_result_path(args, args.analysis_dir, "analysis")
        figures_dir = _resolve_default_result_path(args, args.figures_dir, "figures")
        paper_dir = _resolve_default_result_path(args, args.paper_dir, "paper_experiments")
        sample_level_root = _resolve_default_sample_level_input(args, args.sample_level)
        batch_command = _build_batch_command(args)
        analyze_command = [
            sys.executable,
            "-m",
            "code.scripts.analyze_results",
            "--input",
            summary_root,
            "--output_dir",
            analysis_dir,
        ]
        plot_command = [
            sys.executable,
            "-m",
            "code.scripts.plotting.plot_experiment_results",
            "--summary",
            summary_root,
            "--analysis_dir",
            analysis_dir,
            "--output_dir",
            figures_dir,
            "--top_k",
            str(args.top_k),
        ]
        paper_command = [
            sys.executable,
            "-m",
            "code.scripts.run_paper_experiments",
            "--summary",
            summary_root,
            "--sample_level",
            sample_level_root,
            "--output_dir",
            paper_dir,
            "--num_bins",
            str(args.num_bins),
            "--bootstrap_rounds",
            str(args.bootstrap_rounds),
            "--seed",
            str(args.seed),
        ]
        records = []
        for label, command in [
            ("batch", batch_command),
            ("analyze", analyze_command),
            ("plot", plot_command),
            ("paper", paper_command),
        ]:
            started_at = datetime.now().isoformat(timespec="seconds")
            try:
                _run_command(command)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                print(f"[full] failed stage={label} error={error_text}")
                records.append(
                    {
                        "status": "failed",
                        "label": label,
                        "started_at": started_at,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": error_text,
                    }
                )
                continue
            records.append(
                {
                    "status": "ok",
                    "label": label,
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": "",
                }
            )
        _write_stage_report("full_workflow", records, args.result_dir)


if __name__ == "__main__":
    main()
