import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.config.dataset_profiles import (
    DATASET_CATEGORIES,
    DATASET_CATEGORY_NO_REAL_MISSING,
    DATASET_CATEGORY_REAL_MISSING_NO_TRUTH,
    DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH,
)
from code.core.data_io import build_run_stem
from code.config.settings import cfg

REAL_MISSING_TRUTH_BASES = [
    REPO_ROOT / "data" / "Real_missing_truth",
    REPO_ROOT / "data" / "Real_missing_truth01",
]


def _dataset_lookup_key(name):
    return re.sub(r"[^0-9a-z]+", "", str(name).strip().lower())


def _matching_dataset_dirs(dataset, bases):
    key = _dataset_lookup_key(dataset)
    candidates = []
    for base in bases:
        candidates.append(base / str(dataset))
        if not base.is_dir():
            continue
        for root, _dirs, _files in os.walk(base):
            root_path = Path(root)
            if _dataset_lookup_key(root_path.name) == key:
                candidates.append(root_path)

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _candidate_missing_paths(dataset, missing_percent, max_missing_lenth, missing_mechanism, iteration, seed):
    stem = f"{dataset}_{missing_percent}_{max_missing_lenth}_{missing_mechanism}_{iteration}"
    candidates = []
    if seed is not None:
        candidates.append(REPO_ROOT / "data" / "Missing_data" / dataset / f"{stem}_seed{seed}.csv")
        candidates.append(REPO_ROOT / "data" / "Missing_data" / f"{stem}_seed{seed}.csv")
    candidates.append(REPO_ROOT / "data" / "Missing_data" / dataset / f"{stem}.csv")
    candidates.append(REPO_ROOT / "data" / "Missing_data" / f"{stem}.csv")
    candidates.append(REPO_ROOT / "data" / "Missing_data" / f"{dataset}_holo_{missing_percent}_{max_missing_lenth}_{missing_mechanism}_{iteration}.csv")
    return candidates


def _structured_real_missing_truth_path(dataset):
    for candidate in _matching_dataset_dirs(dataset, REAL_MISSING_TRUTH_BASES):
        for filename in ("missing.csv", "observed_model_ready.csv"):
            path = candidate / filename
            if path.exists():
                return path
    return None


def _resolve_constraint_path(dataset):
    constraint_dir = REPO_ROOT / "imputation_algorithms" / "Holoclean" / "Ground_truth_data"
    exact_path = constraint_dir / f"{dataset}_constraints.txt"
    if exact_path.exists():
        return exact_path

    key = _dataset_lookup_key(dataset)
    for path in sorted(constraint_dir.glob("*_constraints.txt")):
        constraint_dataset = path.name[: -len("_constraints.txt")]
        if _dataset_lookup_key(constraint_dataset) == key:
            return path
    return exact_path


def _resolve_input_path(args):
    if args.input_path:
        path = Path(args.input_path)
        return path if path.is_absolute() else REPO_ROOT / path
    if args.dataset_category == DATASET_CATEGORY_REAL_MISSING_NO_TRUTH:
        return REPO_ROOT / "data" / "Real_missing_data" / f"{args.dataset}.csv"
    if args.dataset_category == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
        flat_path = REPO_ROOT / "data" / "Real_missing_truth" / f"{args.dataset}.csv"
        if flat_path.exists():
            return flat_path
        structured_path = _structured_real_missing_truth_path(args.dataset)
        if structured_path is not None:
            return structured_path
        return flat_path
    for candidate in _candidate_missing_paths(
        args.dataset,
        args.missing_percent,
        args.max_missing_lenth,
        args.missing_mechanism,
        args.iter,
        args.seed,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a Holoclean input CSV under data/Missing_data/. "
        "Pass --input_path explicitly or generate the missing-data file first."
    )


def _default_output_dir(args):
    if args.dataset_category == DATASET_CATEGORY_NO_REAL_MISSING:
        category_dir = "no_real_missing"
    elif args.dataset_category == DATASET_CATEGORY_REAL_MISSING_NO_TRUTH:
        category_dir = "real_missing_data"
    elif args.dataset_category == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
        category_dir = "real_missing_truth"
    else:
        category_dir = args.dataset_category
    return REPO_ROOT / "result_data" / category_dir / f"{args.dataset}_Holoclean"


def build_parser():
    parser = argparse.ArgumentParser(description="Run the legacy Holoclean entrypoint against the current repo layout")
    parser.add_argument("--dataset", type=str, default=cfg["dataset"])
    parser.add_argument(
        "--dataset_category",
        type=str,
        choices=DATASET_CATEGORIES,
        default=DATASET_CATEGORY_NO_REAL_MISSING,
    )
    parser.add_argument("--missing_percent", type=float, default=cfg["missing_percent"])
    parser.add_argument("--max_missing_lenth", type=int, default=cfg["max_missing_lenth"])
    parser.add_argument(
        "--missing_mechanism",
        type=str,
        choices=["MCAR", "MAR", "MNAR"],
        default=cfg["missing_mechanism"],
    )
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=cfg.get("base_seed", 42))
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--domain-thresh-1", type=float, default=None)
    parser.add_argument("--domain-thresh-2", type=float, default=None)
    parser.add_argument("--weak-label-thresh", type=float, default=None)
    parser.add_argument("--max-domain", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    input_path = _resolve_input_path(args)
    save_name = build_run_stem(
        args.dataset,
        args.missing_percent,
        args.max_missing_lenth,
        args.missing_mechanism,
        args.iter,
        seed=args.seed,
    )
    constraint_path = _resolve_constraint_path(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{save_name}_Holoclean_clean.csv"
    timing_path = output_dir / f"{save_name}_Holoclean_timing.json"

    program_path = REPO_ROOT / "imputation_algorithms" / "Holoclean" / "Holoclean" / "Holoclean_repair.py"
    subprocess.run(
        [
            sys.executable,
            str(program_path),
            "--dataset-name",
            save_name,
            "--save-name",
            save_name,
            "--dataset",
            args.dataset,
            "--input-path",
            str(input_path),
            "--constraint-path",
            str(constraint_path),
            "--output-path",
            str(output_path),
            "--timing-path",
            str(timing_path),
        ]
        + (["--input-has-header"] if input_path.name in {"missing.csv", "observed_model_ready.csv"} else [])
        + (["--domain-thresh-1", str(args.domain_thresh_1)] if args.domain_thresh_1 is not None else [])
        + (["--domain-thresh-2", str(args.domain_thresh_2)] if args.domain_thresh_2 is not None else [])
        + (["--weak-label-thresh", str(args.weak_label_thresh)] if args.weak_label_thresh is not None else [])
        + (["--max-domain", str(args.max_domain)] if args.max_domain is not None else [])
        + (["--threads", str(args.threads)] if args.threads is not None else [])
        + (["--timeout-ms", str(args.timeout_ms)] if args.timeout_ms is not None else []),
        check=True,
    )


if __name__ == "__main__":
    main()
