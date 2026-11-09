import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter

from code.core.runtime import configure_runtime_for_stability

configure_runtime_for_stability()

from code.config.dataset_profiles import DATASET_CATEGORY_NO_REAL_MISSING
from code.config.imputer_families import get_imputer_family
from code.config.args import add_batch_experiment_arguments, build_single_run_namespace, resolve_batch_iterations, resolve_batch_seeds
from code.core.compute_backend import enforce_safe_gpu_parallelism, initialize_compute_backend
from code.core.data_io import build_missing_data_path
from code.core.path_config import apply_workspace_paths


def _missing_file_exists(dataset, missing_percent, max_missing_lenth, mechanism, iteration, seed, missing_data_dir):
    path = build_missing_data_path(
        argparse.Namespace(
            dataset=dataset,
            missing_percent=missing_percent,
            max_missing_lenth=max_missing_lenth,
            missing_mechanism=mechanism,
            iter=iteration,
            seed=seed,
            missing_data_dir=missing_data_dir,
        )
    )
    return os.path.exists(path)


def _generate_missing_if_needed(args, dataset, missing_percent, mechanism, iteration):
    if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
        return
    if _missing_file_exists(
        dataset,
        missing_percent,
        args.max_missing_lenth,
        mechanism,
        iteration,
        getattr(args, "seed", None),
        getattr(args, "missing_data_dir", "data/missing_data"),
    ):
        return

    command = [
        sys.executable,
        "-m",
        "code.generate_missing_data",
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
        str(getattr(args, "mar_logistic_beta", 1.0)),
        "--mar_logistic_clip",
        str(getattr(args, "mar_logistic_clip", 3.0)),
        "--workspace_dir",
        args.workspace_dir,
        "--ground_truth_data_dir",
        args.ground_truth_data_dir,
        "--real_missing_data_dir",
        args.real_missing_data_dir,
        "--real_missing_truth_dir",
        args.real_missing_truth_dir,
        "--missing_data_dir",
        args.missing_data_dir,
        "--clean_data_dir",
        args.clean_data_dir,
        "--result_dir",
        args.result_dir,
    ]
    subprocess.run(command, check=True)


def _ensure_missing_exists(args, dataset, missing_percent, mechanism, iteration):
    if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
        return
    if _missing_file_exists(
        dataset,
        missing_percent,
        args.max_missing_lenth,
        mechanism,
        iteration,
        getattr(args, "seed", None),
        getattr(args, "missing_data_dir", "data/missing_data"),
    ):
        return
    path = build_missing_data_path(
        argparse.Namespace(
            dataset=dataset,
            missing_percent=missing_percent,
            max_missing_lenth=args.max_missing_lenth,
            missing_mechanism=mechanism,
            iter=iteration,
            seed=getattr(args, "seed", None),
            missing_data_dir=getattr(args, "missing_data_dir", "data/missing_data"),
        )
    )
    raise FileNotFoundError(
        f"Missing data file not found: {path}. "
        "Run `python -m code.workflow generate ...` first for this dataset/seed/iter/mechanism configuration."
    )


def _build_run_args(batch_args, dataset, missing_percent, mechanism, iteration):
    return build_single_run_namespace(
        dataset=dataset,
        dataset_category=batch_args.dataset_category,
        iteration=iteration,
        seed=batch_args.seed,
        class_method=batch_args.classifiers,
        imputation_method=batch_args.imputers,
        missing_percent=missing_percent,
        max_missing_lenth=batch_args.max_missing_lenth,
        missing_mechanism=mechanism,
        hard_sample_percent=batch_args.hard_sample_percent,
        tuning_cv_folds=batch_args.tuning_cv_folds,
        tuning_n_jobs=batch_args.tuning_n_jobs,
        tuning_scoring=batch_args.tuning_scoring,
        compute_backend=batch_args.compute_backend,
        gpu_device=batch_args.gpu_device,
        selected_compute_backend=getattr(batch_args, "selected_compute_backend", None),
        workspace_dir=batch_args.workspace_dir,
        ground_truth_data_dir=batch_args.ground_truth_data_dir,
        real_missing_data_dir=batch_args.real_missing_data_dir,
        real_missing_truth_dir=batch_args.real_missing_truth_dir,
        real_missing_label_col=batch_args.real_missing_label_col,
        missing_data_dir=batch_args.missing_data_dir,
        clean_data_dir=batch_args.clean_data_dir,
        result_dir=batch_args.result_dir,
    )


def _resolve_parallel_jobs(requested_parallel_jobs):
    requested = int(requested_parallel_jobs)
    if requested > 0:
        return requested

    cpu_total = os.cpu_count() or 1
    if cpu_total <= 2:
        return 1
    if cpu_total <= 6:
        return max(1, cpu_total - 1)
    return max(1, cpu_total - 2)


def _format_elapsed(seconds):
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _execute_full_task(task):
    label = task["label"]
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        runtime_args = argparse.Namespace(
            compute_backend=task["compute_backend"],
            gpu_device=task["gpu_device"],
            tuning_n_jobs=task["tuning_n_jobs"],
            parallel_jobs=1,
        )
        initialize_compute_backend(runtime_args)
        for note in enforce_safe_gpu_parallelism(runtime_args):
            print(f"[compute-backend] {label} {note}")
        from code.pipeline.runner import run_experiment

        single_args = build_single_run_namespace(
            dataset=task["dataset"],
            dataset_category=task["dataset_category"],
            iteration=task["iteration"],
            seed=task["seed"],
            class_method=task["classifiers"],
            imputation_method=task["imputers"],
            missing_percent=task["missing_percent"],
            max_missing_lenth=task["max_missing_lenth"],
            missing_mechanism=task["mechanism"],
            hard_sample_percent=task["hard_sample_percent"],
            tuning_cv_folds=task["tuning_cv_folds"],
            tuning_n_jobs=runtime_args.tuning_n_jobs,
            tuning_scoring=task["tuning_scoring"],
            compute_backend=task["compute_backend"],
            gpu_device=task["gpu_device"],
            selected_compute_backend=runtime_args.selected_compute_backend,
            workspace_dir=task["workspace_dir"],
            ground_truth_data_dir=task["ground_truth_data_dir"],
            real_missing_data_dir=task["real_missing_data_dir"],
            real_missing_truth_dir=task["real_missing_truth_dir"],
            real_missing_label_col=task["real_missing_label_col"],
            missing_data_dir=task["missing_data_dir"],
            clean_data_dir=task["clean_data_dir"],
            result_dir=task["result_dir"],
        )
        if task.get("skip_missing_generation", False):
            _ensure_missing_exists(single_args, task["dataset"], task["missing_percent"], task["mechanism"], task["iteration"])
        else:
            _generate_missing_if_needed(single_args, task["dataset"], task["missing_percent"], task["mechanism"], task["iteration"])
        run_experiment(single_args)
        return {
            "status": "ok",
            "label": label,
            "dataset": task["dataset"],
            "iter": task["iteration"],
            "seed": task["seed"],
            "missing_percent": task["missing_percent"],
            "missing_mechanism": task["mechanism"],
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        return {
            "status": "failed",
            "label": label,
            "dataset": task["dataset"],
            "iter": task["iteration"],
            "seed": task["seed"],
            "missing_percent": task["missing_percent"],
            "missing_mechanism": task["mechanism"],
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": error_text,
        }


LAYERED_IMPUTER_GROUPS = [
    (
        "level_1_baselines",
        "Simple and classical baselines that establish the lower/medium reference band.",
        ["Mean", "Mode", "KNNI", "Mice", "GMM"],
    ),
    (
        "level_2_tree_ensembles",
        "Strong machine-learning baselines used to test whether ensemble imputers already close most of the gap.",
        ["MISSF", "XGBI"],
    ),
    (
        "level_3_deep_models",
        "Deep or representation-learning imputers used to test whether higher-capacity models improve downstream robustness.",
        ["RRSI", "VAEI", "GINN", "NOMI"],
    ),
    (
        "level_4_focus_method",
        "Priority method group highlighted in the paper narrative.",
        ["GCID"],
    ),
]


SERIAL_UNSAFE_IMPUTERS = {
    "VAEI",
    "MIWAE",
    "MLPI",
}


def _requires_serial_batch_execution(imputers):
    requested = {str(name) for name in imputers}
    return bool(requested & SERIAL_UNSAFE_IMPUTERS)


def _build_layered_imputer_plan(imputers):
    requested = [str(name) for name in imputers]
    requested_set = set(requested)
    groups = []
    covered = set()

    for layer_name, description, members in LAYERED_IMPUTER_GROUPS:
        enabled = [name for name in members if name in requested_set]
        if not enabled:
            continue
        covered.update(enabled)
        groups.append(
            {
                "layer": layer_name,
                "description": description,
                "imputers": enabled,
                "families": {name: get_imputer_family(name) for name in enabled},
            }
        )

    remaining = [name for name in requested if name not in covered]
    if remaining:
        groups.append(
            {
                "layer": "custom_or_remaining",
                "description": "Requested imputers that are outside the default layered-paper grouping.",
                "imputers": remaining,
                "families": {name: get_imputer_family(name) for name in remaining},
            }
        )

    return {
        "strategy": "layered_imputer_evaluation",
        "requested_imputers": requested,
        "total_imputers": len(requested),
        "groups": groups,
    }


def _write_batch_manifest(args, iterations, seeds, mechanisms, missing_percents):
    logs_root = os.path.join(args.result_dir, "logs")
    os.makedirs(logs_root, exist_ok=True)
    missing_settings = {
        "dataset_category": args.dataset_category,
        "missing_percents": [float(value) for value in missing_percents],
        "max_missing_lenth": int(args.max_missing_lenth),
        "missing_mechanisms": list(mechanisms),
    }

    manifest = {
        "run_type": "batch_experiment",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": getattr(args, "workspace_dir", os.getcwd()),
        "dataset_category": args.dataset_category,
        "datasets": list(args.datasets),
        "classifiers": list(args.classifiers),
        "imputers": list(args.imputers),
        "seeds": [int(value) for value in seeds],
        "iterations": [int(value) for value in iterations],
        "repeat_count": len(iterations),
        "missing_settings": missing_settings,
        "hard_sample_percent": float(args.hard_sample_percent),
        "tuning": {
            "cv_folds": int(args.tuning_cv_folds),
            "n_jobs": int(args.tuning_n_jobs),
            "scoring": str(args.tuning_scoring),
        },
        "compute": {
            "requested_backend": str(getattr(args, "compute_backend", "cpu")),
            "selected_backend": str(getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu"))),
            "gpu_device": int(getattr(args, "gpu_device", 0)),
        },
        "layered_design": _build_layered_imputer_plan(args.imputers),
    }

    manifest_path = os.path.join(logs_root, "full_runs.log")
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")


def resolve_batch_scope(args):
    iterations = resolve_batch_iterations(args)
    seeds = resolve_batch_seeds(args)
    mechanisms = list(args.mechanisms)
    missing_percents = list(args.missing_percents)
    if args.dataset_category != DATASET_CATEGORY_NO_REAL_MISSING:
        mechanisms = mechanisms[:1]
        missing_percents = missing_percents[:1]
    return iterations, seeds, mechanisms, missing_percents


def build_batch_tasks(args):
    iterations, seeds, mechanisms, missing_percents = resolve_batch_scope(args)
    tasks = []
    for seed in seeds:
        for iteration in iterations:
            for dataset in args.datasets:
                for missing_percent in missing_percents:
                    for mechanism in mechanisms:
                        tasks.append(
                            {
                                "label": f"{dataset}_{missing_percent}_{args.max_missing_lenth}_{mechanism}_{iteration}_seed{seed}",
                                "dataset": dataset,
                                "dataset_category": args.dataset_category,
                                "iteration": iteration,
                                "seed": seed,
                                "missing_percent": missing_percent,
                                "max_missing_lenth": args.max_missing_lenth,
                                "mechanism": mechanism,
                                "classifiers": list(args.classifiers),
                                "imputers": list(args.imputers),
                                "hard_sample_percent": args.hard_sample_percent,
                                "tuning_cv_folds": args.tuning_cv_folds,
                                "tuning_n_jobs": args.tuning_n_jobs,
                                "tuning_scoring": args.tuning_scoring,
                                "compute_backend": args.compute_backend,
                                "gpu_device": args.gpu_device,
                                "workspace_dir": args.workspace_dir,
                                "ground_truth_data_dir": args.ground_truth_data_dir,
                                "real_missing_data_dir": args.real_missing_data_dir,
                                "real_missing_truth_dir": args.real_missing_truth_dir,
                                "real_missing_label_col": args.real_missing_label_col,
                                "missing_data_dir": args.missing_data_dir,
                                "clean_data_dir": args.clean_data_dir,
                                "result_dir": args.result_dir,
                                "skip_missing_generation": bool(getattr(args, "skip_missing_generation", False)),
                            }
                        )
    return tasks, iterations, seeds, mechanisms, missing_percents


def run_task_records(tasks, parallel_jobs, *, log_prefix="full-batch"):
    total_tasks = len(tasks)
    batch_started = perf_counter()
    records = []

    if parallel_jobs == 1:
        for index, task in enumerate(tasks, start=1):
            print(f"[{log_prefix}] start {index}/{total_tasks} label={task['label']}")
            result = _execute_full_task(task)
            if result["status"] == "failed":
                print(
                    f"[{log_prefix}] failed {index}/{total_tasks} label={result['label']} "
                    f"elapsed={_format_elapsed(perf_counter() - batch_started)} error={result['error']}"
                )
            else:
                print(
                    f"[{log_prefix}] done {index}/{total_tasks} label={result['label']} "
                    f"elapsed={_format_elapsed(perf_counter() - batch_started)}"
                )
            records.append(result)
        return records

    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        future_map = {executor.submit(_execute_full_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            completed += 1
            if result["status"] == "failed":
                print(
                    f"[{log_prefix}] failed {completed}/{total_tasks} label={result['label']} "
                    f"elapsed={_format_elapsed(perf_counter() - batch_started)} error={result['error']}"
                )
            else:
                print(
                    f"[{log_prefix}] done {completed}/{total_tasks} label={result['label']} "
                    f"elapsed={_format_elapsed(perf_counter() - batch_started)}"
                )
            records.append(result)
    return records


def write_batch_report(result_dir, records, *, stage="full_batch", filename="full_batch_report.json"):
    logs_root = os.path.join(result_dir, "logs")
    os.makedirs(logs_root, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "total": len(records),
        "ok": sum(1 for record in records if record["status"] == "ok"),
        "failed": sum(1 for record in records if record["status"] == "failed"),
        "records": records,
    }
    report_path = os.path.join(logs_root, filename)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload, report_path


def main():
    parser = argparse.ArgumentParser(description="run the full staged experiment matrix")
    add_batch_experiment_arguments(parser)
    parser.add_argument("--log_prefix", default="full-batch")
    parser.add_argument("--report_stage", default="full_batch")
    parser.add_argument("--report_filename", default="full_batch_report.json")
    parser.add_argument("--skip_missing_generation", action="store_true")
    args = parser.parse_args()
    apply_workspace_paths(args)
    initialize_compute_backend(args)
    for note in enforce_safe_gpu_parallelism(args):
        print(f"[compute-backend] {note}")
    tasks, iterations, seeds, mechanisms, missing_percents = build_batch_tasks(args)
    _write_batch_manifest(args, iterations, seeds, mechanisms, missing_percents)

    requested_parallel_jobs = int(getattr(args, "parallel_jobs", 1))
    parallel_jobs = _resolve_parallel_jobs(requested_parallel_jobs)
    if parallel_jobs > 1 and _requires_serial_batch_execution(args.imputers):
        parallel_jobs = 1
        requested_names = sorted({name for name in args.imputers if str(name) in SERIAL_UNSAFE_IMPUTERS})
        print(
            f"[{args.log_prefix}] forcing parallel_jobs=1 because TensorFlow-based imputers "
            f"{requested_names} are unstable under outer batch parallelism"
        )
    total_tasks = len(tasks)
    print(
        f"[{args.log_prefix}] prepared total_tasks={total_tasks} "
        f"parallel_jobs={parallel_jobs} tuning_n_jobs={args.tuning_n_jobs}"
    )
    if requested_parallel_jobs <= 0:
        cpu_total = os.cpu_count() or 1
        print(
            f"[{args.log_prefix}] auto parallel selection requested={requested_parallel_jobs} "
            f"cpu_total={cpu_total} selected={parallel_jobs}"
        )
    records = run_task_records(tasks, parallel_jobs, log_prefix=args.log_prefix)
    payload, report_path = write_batch_report(
        args.result_dir,
        records,
        stage=args.report_stage,
        filename=args.report_filename,
    )
    print(
        f"[{args.log_prefix}] summary total={payload['total']} ok={payload['ok']} "
        f"failed={payload['failed']} report={report_path}"
    )


if __name__ == "__main__":
    main()
