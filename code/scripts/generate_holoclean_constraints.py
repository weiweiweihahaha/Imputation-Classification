import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_DIR = REPO_ROOT / "imputation_algorithms" / "Holoclean" / "Denial_Constraint_Discovery"
DISCOVERY_INPUT_DIR = DISCOVERY_DIR / "generated_inputs"
DISCOVERY_OUTPUT_DIR = DISCOVERY_DIR / "generated_outputs"
CONSTRAINT_OUTPUT_DIR = REPO_ROOT / "imputation_algorithms" / "Holoclean" / "Ground_truth_data"
MAVEN_LOCAL_REPO = REPO_ROOT / "tmp" / "m2_repository"
DEFAULT_DATASETS = [
    "glass",
    "hcv",
    "iris",
    "mobile",
    "obesity",
    "rice",
    "seed",
    "wine",
    "heart_disease",
    "horse_colic",
    "hcc_survival",
]
DEFAULT_JAVA_XMX = os.environ.get("FDCD_JAVA_XMX", "8g")
DEFAULT_PRE_ENUM_MAX_PREDICATES = int(os.environ.get("FDCD_PRE_ENUM_MAX_PREDICATES", "48"))
DATASET_PREFERRED_COLUMNS = {
    "obesity": [
        "fcvc",
        "ncp",
        "ch2o",
        "faf",
        "tue",
        "favc",
        "familyhistory",
        "caec",
        "calc",
        "mtrans",
        "age",
        "weight",
        "height",
        "gender",
        "smoke",
        "scc",
    ],
}


def _normalize_name(name):
    return re.sub(r"[^0-9a-z]+", "", str(name).strip().lower())


def _make_unique(names):
    counts = Counter()
    unique = []
    for name in names:
        base = name or "col"
        counts[base] += 1
        if counts[base] == 1:
            unique.append(base)
        else:
            unique.append(f"{base}{counts[base]}")
    return unique


def _read_json_if_exists(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _move_label_column_to_front(frame, label_column):
    if label_column is None:
        return frame
    if isinstance(label_column, int):
        if label_column < 0 or label_column >= frame.shape[1]:
            return frame
        label_name = frame.columns[label_column]
    else:
        label_name = label_column
    if label_name not in frame.columns:
        return frame
    ordered = [label_name] + [column for column in frame.columns if column != label_name]
    return frame.loc[:, ordered]


def _load_structured_real_missing_truth(dataset):
    bases = [
        REPO_ROOT / "data" / "Real_missing_truth",
        REPO_ROOT / "data" / "Real_missing_truth01",
    ]
    candidates = []
    for base in bases:
        candidates.append(base / str(dataset))
        if base.is_dir():
            for root, _dirs, _files in os.walk(base):
                root_path = Path(root)
                if root_path.name == str(dataset):
                    candidates.append(root_path)

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        truth_path = candidate / "truth.csv"
        if not truth_path.exists():
            continue
        metadata = _read_json_if_exists(candidate / "dataset_description.json")
        frame = pd.read_csv(truth_path)
        identifier_columns = [
            column for column in metadata.get("identifier_columns", []) if column in frame.columns
        ]
        if identifier_columns:
            frame = frame.drop(columns=identifier_columns)
        frame = _move_label_column_to_front(frame, metadata.get("label_column"))
        return truth_path, frame, True, "real_missing_truth"

    return None


def _resolve_source(dataset):
    ground_truth_path = REPO_ROOT / "data" / "Ground_truth_data" / f"{dataset}.csv"
    if ground_truth_path.exists():
        return ground_truth_path, None, True, "ground_truth_data"
    structured_real_missing_truth = _load_structured_real_missing_truth(dataset)
    if structured_real_missing_truth is not None:
        return structured_real_missing_truth
    real_missing_path = REPO_ROOT / "data" / "Real_missing_data" / f"{dataset}.csv"
    if real_missing_path.exists():
        return real_missing_path, None, False, "real_missing_data"
    raise FileNotFoundError(f"No source CSV found for dataset={dataset}")


def _load_source_frame(path, has_header):
    if has_header:
        return pd.read_csv(path)
    return pd.read_csv(path, header=None)


def _infer_fdcd_type(series):
    cleaned = series.dropna()
    if cleaned.empty:
        return "str"
    as_text = cleaned.astype(str).str.strip()
    as_text = as_text[as_text.ne("")]
    if as_text.empty:
        return "str"
    numeric = pd.to_numeric(as_text, errors="coerce")
    if numeric.notna().all():
        rounded = numeric.round()
        if ((numeric - rounded).abs() < 1e-9).all():
            return "int"
        return "float"
    return "str"


def _estimate_predicate_cost(fdcd_type):
    return 6 if fdcd_type in {"int", "float"} else 2


def _normalized_non_empty_text(series):
    text = series.dropna().astype(str).str.strip()
    return text[text.ne("")]


def _column_profile(series, fdcd_type):
    non_empty = _normalized_non_empty_text(series)
    non_null_ratio = float(len(non_empty)) / float(len(series)) if len(series) else 0.0
    distinct_non_empty = int(non_empty.nunique(dropna=True))
    unique_ratio = float(distinct_non_empty) / float(len(non_empty)) if len(non_empty) else 0.0
    score = (3.0 * non_null_ratio) + min(unique_ratio, 0.5)
    if fdcd_type in {"int", "float"}:
        score += 0.15
    return {
        "non_null_ratio": round(non_null_ratio, 6),
        "distinct_non_empty": distinct_non_empty,
        "unique_ratio": round(unique_ratio, 6),
        "estimated_predicate_cost": _estimate_predicate_cost(fdcd_type),
        "score": round(score, 6),
    }


def _dataset_priority(dataset, normalized_name):
    ordered = DATASET_PREFERRED_COLUMNS.get(str(dataset), [])
    try:
        return ordered.index(normalized_name)
    except ValueError:
        return len(ordered) + 100


def _adjusted_column_score(dataset, normalized_name, fdcd_type, profile):
    score = float(profile["score"])
    # Extremely high-cardinality numeric columns often dominate the generic
    # ranking but contribute fewer compact DCs than medium-cardinality behavior
    # variables, especially on obesity.
    if fdcd_type in {"int", "float"} and profile["unique_ratio"] > 0.6:
        score -= 0.35
    if str(dataset) == "obesity":
        priority = _dataset_priority(dataset, normalized_name)
        if priority < 5:
            score += 1.5
        elif priority < 10:
            score += 0.8
        elif priority < 13:
            score += 0.3
    return round(score, 6)


def _prune_feature_columns(dataset, feature_frame, fdcd_types, *, max_columns=None, max_predicates=None):
    kept = []
    dropped = []

    for name in feature_frame.columns:
        fdcd_type = fdcd_types[name]
        profile = _column_profile(feature_frame[name], fdcd_type)
        entry = {
            "normalized_name": name,
            "fdcd_type": fdcd_type,
            "dataset_priority": _dataset_priority(dataset, name),
            **profile,
        }
        entry["adjusted_score"] = _adjusted_column_score(dataset, name, fdcd_type, profile)
        if profile["distinct_non_empty"] <= 1:
            entry["drop_reason"] = "constant_or_empty"
            dropped.append(entry)
            continue
        kept.append(entry)

    kept.sort(
        key=lambda item: (
            item["dataset_priority"],
            -item["adjusted_score"],
            item["estimated_predicate_cost"],
            -item["distinct_non_empty"],
            item["normalized_name"],
        )
    )

    selected = []
    selected_names = set()
    estimated_predicates = 0

    for entry in kept:
        over_column_budget = max_columns is not None and len(selected) >= max_columns
        over_predicate_budget = (
            max_predicates is not None
            and selected
            and estimated_predicates + entry["estimated_predicate_cost"] > max_predicates
        )
        if over_column_budget:
            entry = {**entry, "drop_reason": "max_columns_budget"}
            dropped.append(entry)
            continue
        if over_predicate_budget:
            entry = {**entry, "drop_reason": "max_predicates_budget"}
            dropped.append(entry)
            continue
        selected.append(entry)
        selected_names.add(entry["normalized_name"])
        estimated_predicates += entry["estimated_predicate_cost"]

    if not selected and kept:
        fallback = kept[0]
        selected = [fallback]
        selected_names = {fallback["normalized_name"]}
        estimated_predicates = fallback["estimated_predicate_cost"]
        dropped = [
            item
            for item in dropped
            if item["normalized_name"] != fallback["normalized_name"]
        ]

    pruned_frame = feature_frame.loc[:, [entry["normalized_name"] for entry in selected]].copy()
    return pruned_frame, selected, dropped, estimated_predicates


def _prepare_discovery_frame(dataset, pre_enum_max_columns=None, pre_enum_max_predicates=None):
    source_path, source_frame, has_header, source_kind = _resolve_source(dataset)
    frame = source_frame.copy() if source_frame is not None else _load_source_frame(source_path, has_header)
    if frame.shape[1] < 2:
        raise ValueError(f"Dataset {dataset} must have at least one label column and one feature column")

    feature_frame = frame.iloc[:, 1:].copy()
    if has_header:
        original_names = [str(col) for col in frame.columns[1:]]
        normalized_names = _make_unique([_normalize_name(col) for col in original_names])
    else:
        original_names = [f"column_{idx}" for idx in range(1, frame.shape[1])]
        normalized_names = [f"c{idx}" for idx in range(1, frame.shape[1])]

    feature_frame.columns = normalized_names
    fdcd_types = {name: _infer_fdcd_type(feature_frame[name]) for name in feature_frame.columns}
    original_feature_count = int(feature_frame.shape[1])
    original_estimated_predicates = int(sum(_estimate_predicate_cost(fdcd_types[name]) for name in feature_frame.columns))

    feature_frame, selected_columns, dropped_columns, kept_estimated_predicates = _prune_feature_columns(
        dataset,
        feature_frame,
        fdcd_types,
        max_columns=pre_enum_max_columns,
        max_predicates=pre_enum_max_predicates,
    )
    fdcd_types = {name: fdcd_types[name] for name in feature_frame.columns}
    typed_headers = [f"{name} {fdcd_types[name]}" for name in feature_frame.columns]

    metadata = {
        "dataset": dataset,
        "source_path": str(source_path),
        "source_kind": source_kind,
        "has_header": has_header,
        "row_count": int(feature_frame.shape[0]),
        "original_feature_count": original_feature_count,
        "feature_count": int(feature_frame.shape[1]),
        "label_column_assumed": 0,
        "pre_enum_max_columns": pre_enum_max_columns,
        "pre_enum_max_predicates": pre_enum_max_predicates,
        "estimated_predicate_count_before_pruning": original_estimated_predicates,
        "estimated_predicate_count_after_pruning": int(kept_estimated_predicates),
        "feature_name_mapping": [
            {
                "normalized_name": normalized,
                "source_name": original,
            }
            for original, normalized in zip(original_names, normalized_names)
        ],
        "selected_columns": selected_columns,
        "dropped_columns": dropped_columns,
        "fdcd_types": {
            name: fdcd_types[name]
            for name in feature_frame.columns
        },
    }
    return feature_frame, typed_headers, metadata


def _write_discovery_csv(dataset, feature_frame, typed_headers):
    DISCOVERY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DISCOVERY_INPUT_DIR / f"{dataset}.csv"
    feature_frame.to_csv(output_path, index=False, header=typed_headers, na_rep="")
    return output_path


def _convert_fdcd_line_to_holoclean(line):
    text = line.strip()
    if not text:
        return None
    if text.lower().startswith("not(") and text.endswith(")"):
        text = text[4:-1]
    elif text.lower().startswith("not ") and text.endswith(")"):
        text = text[4:-1]

    parts = text.split()
    holo = ["t1", "t2"]
    for part in parts:
        if part.lower() == "and":
            continue
        tokens = re.split(r"\s*(==|<=|>=|<>|<|>)\s*", part)
        if len(tokens) != 3:
            raise ValueError(f"Unsupported FDCD predicate: {part}")
        left, op, right = tokens
        if op == "==":
            holo.append(f"EQ({left},{right})")
        elif op == "<=":
            holo.append(f"LTE({left},{right})")
        elif op == ">=":
            holo.append(f"GTE({left},{right})")
        elif op == "<":
            holo.append(f"LT({left},{right})")
        elif op == ">":
            holo.append(f"GT({left},{right})")
        elif op == "<>":
            holo.append(f"IQ({left},{right})")
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return "&".join(holo)


def _predicate_count(constraint_line):
    return max(0, constraint_line.count("&") - 1)


def _convert_fdcd_output(raw_dc_path, constraint_path, max_predicates=None, max_constraints=None):
    converted = []
    with open(raw_dc_path, "r", encoding="utf-8") as handle:
        for line in handle:
            result = _convert_fdcd_line_to_holoclean(line)
            if result:
                converted.append(result)
    raw_count = len(converted)
    if max_predicates is not None:
        converted = [line for line in converted if _predicate_count(line) <= max_predicates]
    converted = sorted(dict.fromkeys(converted), key=lambda line: (_predicate_count(line), len(line), line))
    if max_constraints is not None:
        converted = converted[:max_constraints]
    constraint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(constraint_path, "w", encoding="utf-8") as handle:
        for line in converted:
            handle.write(line + "\n")
    return converted, raw_count


def _build_discovery_tool(skip_build):
    jar_path = DISCOVERY_DIR / "target" / "discoverDCs.jar"
    if skip_build and jar_path.exists():
        return jar_path
    MAVEN_LOCAL_REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mvn", "-q", f"-Dmaven.repo.local={MAVEN_LOCAL_REPO}", "-DskipTests", "clean", "package"],
        cwd=DISCOVERY_DIR,
        check=True,
    )
    if not jar_path.exists():
        raise FileNotFoundError(f"FDCD build completed but jar is missing: {jar_path}")
    return jar_path


def _run_discovery(jar_path, dataset, discovery_csv, algorithm, max_rows, java_xmx):
    DISCOVERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dc_path = DISCOVERY_OUTPUT_DIR / f"{dataset}_fdcd.txt"
    cmd = ["java"]
    if java_xmx:
        cmd.append(f"-Xmx{java_xmx}")
    cmd.extend(["-jar", str(jar_path), str(discovery_csv), "-o", str(raw_dc_path), "-e", algorithm])
    if max_rows is not None:
        cmd.extend(["-n", str(max_rows)])
    subprocess.run(cmd, cwd=DISCOVERY_DIR, check=True)
    if not raw_dc_path.exists():
        raise FileNotFoundError(f"FDCD run finished but raw output is missing: {raw_dc_path}")
    return raw_dc_path


def generate_for_dataset(
    dataset,
    jar_path,
    algorithm,
    max_rows,
    max_predicates,
    max_constraints,
    java_xmx,
    pre_enum_max_columns=None,
    pre_enum_max_predicates=None,
):
    feature_frame, typed_headers, metadata = _prepare_discovery_frame(
        dataset,
        pre_enum_max_columns=pre_enum_max_columns,
        pre_enum_max_predicates=pre_enum_max_predicates,
    )
    discovery_csv = _write_discovery_csv(dataset, feature_frame, typed_headers)
    raw_dc_path = _run_discovery(jar_path, dataset, discovery_csv, algorithm, max_rows, java_xmx)
    constraint_path = CONSTRAINT_OUTPUT_DIR / f"{dataset}_constraints.txt"
    converted, raw_constraint_count = _convert_fdcd_output(
        raw_dc_path,
        constraint_path,
        max_predicates=max_predicates,
        max_constraints=max_constraints,
    )

    metadata.update(
        {
            "algorithm": algorithm,
            "max_rows": max_rows,
            "max_predicates": max_predicates,
            "max_constraints": max_constraints,
            "java_xmx": java_xmx,
            "pre_enum_max_columns": pre_enum_max_columns,
            "pre_enum_max_predicates": pre_enum_max_predicates,
            "discovery_csv": str(discovery_csv),
            "raw_dc_path": str(raw_dc_path),
            "constraint_path": str(constraint_path),
            "raw_constraint_count": raw_constraint_count,
            "constraint_count": len(converted),
        }
    )
    metadata_path = CONSTRAINT_OUTPUT_DIR / f"{dataset}_constraints_meta.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Holoclean denial constraints with the bundled FDCD tool")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--algorithm", default="INCS")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_predicates", type=int, default=None)
    parser.add_argument("--max_constraints", type=int, default=None)
    parser.add_argument(
        "--pre_enum_max_columns",
        type=int,
        default=None,
        help="Maximum number of feature columns kept before writing the FDCD input CSV.",
    )
    parser.add_argument(
        "--pre_enum_max_predicates",
        type=int,
        default=DEFAULT_PRE_ENUM_MAX_PREDICATES,
        help=(
            "Estimated predicate-space budget applied before enumeration. "
            "Numeric columns cost 6 predicates, categorical columns cost 2. "
            "Set to 0 to disable. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--java_xmx",
        default=DEFAULT_JAVA_XMX,
        help="JVM max heap passed to FDCD as -Xmx. Default: %(default)s. Set to empty string to omit.",
    )
    parser.add_argument("--skip_build", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.java_xmx = args.java_xmx.strip() if isinstance(args.java_xmx, str) else args.java_xmx
    if args.pre_enum_max_columns is not None and args.pre_enum_max_columns <= 0:
        args.pre_enum_max_columns = None
    if args.pre_enum_max_predicates is not None and args.pre_enum_max_predicates <= 0:
        args.pre_enum_max_predicates = None
    jar_path = _build_discovery_tool(skip_build=args.skip_build)
    results = []
    for dataset in args.datasets:
        print(f"[constraints] generating dataset={dataset}", flush=True)
        metadata = generate_for_dataset(
            dataset=dataset,
            jar_path=jar_path,
            algorithm=args.algorithm,
            max_rows=args.max_rows,
            max_predicates=args.max_predicates,
            max_constraints=args.max_constraints,
            java_xmx=args.java_xmx,
            pre_enum_max_columns=args.pre_enum_max_columns,
            pre_enum_max_predicates=args.pre_enum_max_predicates,
        )
        print(
            (
                f"[constraints] done dataset={dataset} count={metadata['constraint_count']} "
                f"columns={metadata['feature_count']}/{metadata['original_feature_count']} "
                f"estimated_predicates={metadata['estimated_predicate_count_after_pruning']}/"
                f"{metadata['estimated_predicate_count_before_pruning']} "
                f"output={metadata['constraint_path']}"
            ),
            flush=True,
        )
        results.append(
            {
                "dataset": dataset,
                "constraint_count": metadata["constraint_count"],
                "constraint_path": metadata["constraint_path"],
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
