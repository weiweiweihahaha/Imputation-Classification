import json
import os
import re

import numpy as np
import pandas as pd

from code.config.dataset_profiles import (
    DATASET_CATEGORY_NO_REAL_MISSING,
    DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH,
)


def get_dataset_category(args):
    return getattr(args, "dataset_category", DATASET_CATEGORY_NO_REAL_MISSING)


def has_ground_truth(args):
    return True


def uses_synthetic_missing(args):
    return get_dataset_category(args) == DATASET_CATEGORY_NO_REAL_MISSING


def uses_real_missing(args):
    return get_dataset_category(args) == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH


def _read_csv_strict(path, **kwargs):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _read_json_if_exists(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _metadata_column_names(metadata):
    columns = []
    label_name = metadata.get("label_column")
    if label_name:
        columns.append(label_name)

    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        feature_columns = [
            *metadata.get("numeric_columns", []),
            *metadata.get("categorical_columns", []),
        ]

    for column in feature_columns:
        if column not in columns:
            columns.append(column)
    return columns


def _metadata_identifier_columns(metadata):
    columns = list(metadata.get("identifier_columns", []))
    for key in ("id_column", "row_id_column", "split_column"):
        value = metadata.get(key)
        if value and value not in columns:
            columns.append(value)
    return columns


def _discrete_columns_from_metadata(metadata):
    discrete = set(metadata.get("discrete_columns", []))
    for name, field in metadata.get("fields", {}).items():
        if not isinstance(field, dict):
            continue
        model_type = str(field.get("model_field_type", ""))
        original_type = str(field.get("original_field_type", ""))
        if model_type.startswith("discrete") or original_type in {"boolean", "categorical_text"}:
            discrete.add(name)
    return discrete


def _apply_metadata_dtypes(frame, metadata, *, include_label=True):
    if frame is None or frame.empty or not metadata:
        return frame

    typed = frame.copy()
    discrete_columns = _discrete_columns_from_metadata(metadata)
    if not include_label:
        discrete_columns.discard(metadata.get("label_column"))

    for column in typed.columns:
        if str(column) not in discrete_columns:
            continue
        typed[column] = pd.Categorical(typed[column])
    return typed


def _metadata_for_dataset(args):
    bundle = _structured_real_missing_bundle(args)
    if not bundle:
        return {}
    return _read_json_if_exists(bundle["description_path"])


def apply_real_missing_metadata_dtypes(args, frame, *, include_label=True):
    metadata = _metadata_for_dataset(args)
    if not metadata:
        return frame

    typed = frame.copy()
    metadata_columns = _metadata_column_names(metadata)
    if len(metadata_columns) == typed.shape[1]:
        typed.columns = metadata_columns
        typed = _apply_metadata_dtypes(typed, metadata, include_label=include_label)
        typed.columns = list(frame.columns)
        return typed

    return _apply_metadata_dtypes(typed, metadata, include_label=include_label)


def get_real_missing_discrete_column_positions(args, frame):
    metadata = _metadata_for_dataset(args)
    if not metadata:
        return []

    metadata_columns = _metadata_column_names(metadata)
    discrete_columns = _discrete_columns_from_metadata(metadata)
    positions = []
    if len(metadata_columns) == frame.shape[1]:
        for idx, name in enumerate(metadata_columns):
            if name == metadata.get("label_column"):
                continue
            if name in discrete_columns:
                positions.append(idx)
        return positions

    for idx, column in enumerate(frame.columns):
        if idx == 0:
            continue
        if str(column) in discrete_columns:
            positions.append(idx)
    return positions


def _move_label_column_to_front(frame, label_col):
    normalized_label_col = int(label_col)
    column_count = frame.shape[1]
    if normalized_label_col < 0:
        normalized_label_col += column_count
    if normalized_label_col < 0 or normalized_label_col >= column_count:
        raise ValueError(
            f"real_missing_label_col={label_col} is out of bounds for a dataset with {column_count} columns."
        )
    if normalized_label_col == 0:
        return frame
    ordered_columns = [normalized_label_col] + [idx for idx in range(column_count) if idx != normalized_label_col]
    return frame.iloc[:, ordered_columns].reset_index(drop=True)


def _drop_columns_by_name(frame, column_names):
    if not column_names:
        return frame
    drop_names = [column for column in column_names if column in frame.columns]
    if not drop_names:
        return frame
    return frame.drop(columns=drop_names)


def _normalize_real_missing_columns(frame, metadata, fallback_label_col, *, restore_dtypes=True):
    normalized = frame.copy()
    normalized = _drop_columns_by_name(normalized, _metadata_identifier_columns(metadata))

    metadata_columns = _metadata_column_names(metadata)
    if len(metadata_columns) > 1 and all(column in normalized.columns for column in metadata_columns):
        normalized = normalized.loc[:, metadata_columns].reset_index(drop=True)
        return _apply_metadata_dtypes(normalized, metadata) if restore_dtypes else normalized

    label_name = metadata.get("label_column")
    if label_name and label_name in normalized.columns:
        ordered_columns = [label_name] + [column for column in normalized.columns if column != label_name]
        normalized = normalized.loc[:, ordered_columns].reset_index(drop=True)
        return _apply_metadata_dtypes(normalized, metadata) if restore_dtypes else normalized

    normalized = _move_label_column_to_front(normalized, fallback_label_col)
    return _apply_metadata_dtypes(normalized, metadata) if restore_dtypes else normalized


def _ground_truth_csv_path(args):
    base = getattr(args, "ground_truth_data_dir", "data/Ground_truth_data")
    return os.path.join(base, f"{args.dataset}.csv")


def _dataset_lookup_key(name):
    return re.sub(r"[^0-9a-z]+", "", str(name).strip().lower())


def _real_missing_truth_bases(args):
    primary = getattr(args, "real_missing_truth_dir", "data/Real_missing_truth")
    bases = [primary]
    parent = os.path.dirname(primary.rstrip(os.sep)) or "."
    fallback = os.path.join(parent, "Real_missing_truth01")
    if fallback not in bases:
        bases.append(fallback)
    return bases


def _structured_real_missing_bundle(args):
    dataset = str(getattr(args, "dataset"))
    dataset_key = _dataset_lookup_key(dataset)
    candidates = []

    for base in _real_missing_truth_bases(args):
        candidates.append(os.path.join(base, dataset))
        if os.path.isdir(base):
            for root, dirs, _files in os.walk(base):
                if _dataset_lookup_key(os.path.basename(root)) == dataset_key:
                    candidates.append(root)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        observed_path = None
        oracle_path = None
        for observed_name in ("missing.csv", "observed_model_ready.csv"):
            candidate_path = os.path.join(candidate, observed_name)
            if os.path.exists(candidate_path):
                observed_path = candidate_path
                break
        for oracle_name in ("truth.csv", "oracle_model_ready.csv"):
            candidate_path = os.path.join(candidate, oracle_name)
            if os.path.exists(candidate_path):
                oracle_path = candidate_path
                break
        if observed_path and oracle_path:
            return {
                "dataset_dir": candidate,
                "observed_path": observed_path,
                "oracle_path": oracle_path,
                "description_path": os.path.join(candidate, "dataset_description.json"),
            }
    return None


def _load_structured_bundle_frame(args, kind, *, restore_dtypes=True):
    bundle = _structured_real_missing_bundle(args)
    if not bundle:
        return None

    frame_key = "observed_path" if kind == "observed" else "oracle_path"
    frame = _read_csv_strict(bundle[frame_key], delimiter=",", header=0)
    metadata = _read_json_if_exists(bundle["description_path"])
    return _normalize_real_missing_columns(
        frame,
        metadata,
        getattr(args, "real_missing_label_col", 0),
        restore_dtypes=restore_dtypes,
    )


def _normalize_seed(seed):
    if seed in (None, "", "None"):
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return str(seed)


def _stringify_path_part(value):
    return str(value).replace(os.sep, "_")


def build_imputation_group_label(imputation_method):
    if imputation_method is None:
        return None
    if isinstance(imputation_method, str):
        names = [imputation_method]
    else:
        names = [str(name) for name in imputation_method if str(name)]
    if not names:
        return None
    label = "_".join(_stringify_path_part(name) for name in names)
    if len(label) <= 120:
        return label
    preview = "_".join(_stringify_path_part(name) for name in names[:3])
    return f"{preview}_plus_{len(names) - 3}_more"


def build_run_stem(dataset, missing_percent, max_missing_lenth, mechanism, iteration, seed=None):
    parts = [
        str(dataset),
        str(missing_percent),
        str(max_missing_lenth),
        str(mechanism),
        str(iteration),
    ]
    normalized_seed = _normalize_seed(seed)
    if normalized_seed is not None:
        parts.append(f"seed{normalized_seed}")
    return "_".join(parts)


def parse_run_stem(run_stem):
    parts = str(run_stem).split("_")
    seed = None
    if parts and parts[-1].startswith("seed"):
        seed_token = parts.pop()[4:]
        seed = _normalize_seed(seed_token)
    if len(parts) < 5:
        raise ValueError(f"Invalid run stem: {run_stem}")

    iteration = int(parts.pop())
    mechanism = parts.pop()
    max_missing_lenth = int(parts.pop())
    missing_percent = float(parts.pop())
    dataset = "_".join(parts)
    return {
        "dataset": dataset,
        "missing_percent": missing_percent,
        "max_missing_lenth": max_missing_lenth,
        "mechanism": mechanism,
        "iteration": iteration,
        "seed": seed,
    }


def build_missing_filename(dataset, missing_percent, max_missing_lenth, mechanism, iteration, seed=None):
    return f"{build_run_stem(dataset, missing_percent, max_missing_lenth, mechanism, iteration, seed=seed)}.csv"


def _missing_root_candidates(args):
    configured_root = getattr(args, "missing_data_dir", "data/Missing_data")
    candidates = [configured_root]

    normalized = configured_root.replace("\\", "/").rstrip("/")
    if normalized.endswith("/Missing_data"):
        candidates.append(f"{configured_root[:-len('Missing_data')]}missing_data")
    elif normalized.endswith("/missing_data"):
        candidates.append(f"{configured_root[:-len('missing_data')]}Missing_data")

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _select_missing_root(args, *, ensure_parent=False):
    candidates = _missing_root_candidates(args)
    if ensure_parent:
        return candidates[0]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def build_missing_data_path(args, mechanism=None, iteration=None, *, ensure_parent=False):
    dataset = getattr(args, "dataset")
    mechanism = mechanism or getattr(args, "missing_mechanism")
    iteration = getattr(args, "iter", 0) if iteration is None else iteration
    seed = getattr(args, "seed", None)
    missing_root = _select_missing_root(args, ensure_parent=ensure_parent)
    dataset_root = os.path.join(missing_root, dataset)
    path = os.path.join(
        dataset_root,
        build_missing_filename(
            dataset,
            getattr(args, "missing_percent"),
            getattr(args, "max_missing_lenth"),
            mechanism,
            iteration,
            seed=seed,
        ),
    )
    if ensure_parent:
        os.makedirs(dataset_root, exist_ok=True)
    return path


def _build_legacy_missing_data_path(args, mechanism=None, iteration=None):
    dataset = getattr(args, "dataset")
    mechanism = mechanism or getattr(args, "missing_mechanism")
    iteration = getattr(args, "iter", 0) if iteration is None else iteration
    filename = build_missing_filename(
        dataset,
        getattr(args, "missing_percent"),
        getattr(args, "max_missing_lenth"),
        mechanism,
        iteration,
        seed=None,
    )
    for missing_root in _missing_root_candidates(args):
        candidate = os.path.join(missing_root, dataset, filename)
        if os.path.exists(candidate):
            return candidate

    fallback_root = _select_missing_root(args, ensure_parent=False)
    return os.path.join(fallback_root, dataset, filename)


def resolve_missing_data_path(args, mechanism=None, iteration=None):
    if uses_real_missing(args):
        observed_override = getattr(args, "real_missing_observed_path", None) or os.getenv(
            "FRAMEWORK_REAL_MISSING_OBSERVED_PATH"
        )
        if observed_override:
            return observed_override

        flat_path = os.path.join(
            getattr(args, "real_missing_data_dir", "data/Real_missing_data"),
            f"{getattr(args, 'dataset')}.csv",
        )
        if os.path.exists(flat_path):
            return flat_path
        bundle = _structured_real_missing_bundle(args)
        if bundle:
            return bundle["observed_path"]
        return flat_path

    preferred_path = build_missing_data_path(args, mechanism=mechanism, iteration=iteration)
    if os.path.exists(preferred_path):
        return preferred_path

    legacy_path = _build_legacy_missing_data_path(args, mechanism=mechanism, iteration=iteration)
    if legacy_path != preferred_path and os.path.exists(legacy_path):
        return legacy_path

    return preferred_path


def build_clean_filename(dataset, missing_percent, max_missing_lenth, mechanism, iteration, imputation, seed=None):
    stem = build_run_stem(
        dataset,
        missing_percent,
        max_missing_lenth,
        mechanism,
        iteration,
        seed=seed,
    )
    return f"{stem}_{imputation}_clean.csv"


def build_clean_timing_filename(dataset, missing_percent, max_missing_lenth, mechanism, iteration, imputation, seed=None):
    stem = build_run_stem(
        dataset,
        missing_percent,
        max_missing_lenth,
        mechanism,
        iteration,
        seed=seed,
    )
    return f"{stem}_{imputation}_timing.json"


def _clean_data_category_dir(args):
    category = get_dataset_category(args)
    if category == DATASET_CATEGORY_NO_REAL_MISSING:
        return "no_real_missing"
    if category == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
        return "real_missing_truth"
    return _stringify_path_part(category)


def build_clean_data_path(args, imputation, mechanism=None, iteration=None, *, ensure_parent=False):
    dataset = getattr(args, "dataset")
    mechanism = mechanism or getattr(args, "missing_mechanism")
    iteration = getattr(args, "iter", 0) if iteration is None else iteration
    seed = getattr(args, "seed", None)
    clean_root = getattr(args, "clean_data_dir", "result_data")
    dataset_root = os.path.join(
        clean_root,
        _clean_data_category_dir(args),
        f"{_stringify_path_part(dataset)}_{_stringify_path_part(imputation)}",
    )
    path = os.path.join(
        dataset_root,
        build_clean_filename(
            dataset,
            getattr(args, "missing_percent"),
            getattr(args, "max_missing_lenth"),
            mechanism,
            iteration,
            imputation,
            seed=seed,
        ),
    )
    if ensure_parent:
        os.makedirs(dataset_root, exist_ok=True)
    return path


def build_clean_timing_path(args, imputation, mechanism=None, iteration=None, *, ensure_parent=False):
    dataset = getattr(args, "dataset")
    mechanism = mechanism or getattr(args, "missing_mechanism")
    iteration = getattr(args, "iter", 0) if iteration is None else iteration
    seed = getattr(args, "seed", None)
    clean_root = getattr(args, "clean_data_dir", "result_data")
    dataset_root = os.path.join(
        clean_root,
        _clean_data_category_dir(args),
        f"{_stringify_path_part(dataset)}_{_stringify_path_part(imputation)}",
    )
    path = os.path.join(
        dataset_root,
        build_clean_timing_filename(
            dataset,
            getattr(args, "missing_percent"),
            getattr(args, "max_missing_lenth"),
            mechanism,
            iteration,
            imputation,
            seed=seed,
        ),
    )
    if ensure_parent:
        os.makedirs(dataset_root, exist_ok=True)
    return path


def load_real_missing_data(args, *, restore_dtypes=True):
    observed_override = getattr(args, "real_missing_observed_path", None) or os.getenv(
        "FRAMEWORK_REAL_MISSING_OBSERVED_PATH"
    )
    if observed_override:
        frame = _read_csv_strict(
            observed_override,
            delimiter=",",
            header=None,
        )
        return frame

    base = getattr(args, "real_missing_data_dir", "data/Real_missing_data")
    flat_path = os.path.join(base, f"{args.dataset}.csv")
    if os.path.exists(flat_path):
        frame = _read_csv_strict(
            flat_path,
            delimiter=",",
            header=None,
        )
        frame = _move_label_column_to_front(frame, getattr(args, "real_missing_label_col", 0))
        return apply_real_missing_metadata_dtypes(args, frame) if restore_dtypes else frame

    structured_frame = _load_structured_bundle_frame(args, "observed")
    if structured_frame is not None:
        return structured_frame if restore_dtypes else _load_structured_bundle_frame(args, "observed", restore_dtypes=False)

    raise FileNotFoundError(
        f"Dataset file not found: {flat_path}. "
        "Also did not find a matching observed/oracle bundle under real_missing_truth_dir."
    )


def load_real_truth_data(args, *, restore_dtypes=True):
    base = getattr(args, "real_missing_truth_dir", "data/Real_missing_truth")
    flat_path = os.path.join(base, f"{args.dataset}.csv")
    if os.path.exists(flat_path):
        frame = _read_csv_strict(
            flat_path,
            delimiter=",",
            header=0,
        )
        return apply_real_missing_metadata_dtypes(args, frame) if restore_dtypes else frame

    structured_frame = _load_structured_bundle_frame(args, "oracle")
    if structured_frame is not None:
        return structured_frame if restore_dtypes else _load_structured_bundle_frame(args, "oracle", restore_dtypes=False)

    raise FileNotFoundError(f"Dataset file not found: {flat_path}")


def load_data(args, mechanism, imputation):
    name = args.dataset
    iteration = getattr(args, "iter", 0)
    if imputation == "Ground":
        ground_data = _read_csv_strict(
            _ground_truth_csv_path(args),
            delimiter=",",
            header=0,
        )
        return ground_data
    if imputation is None:
        missing_data = pd.read_csv(
            resolve_missing_data_path(args, mechanism=mechanism, iteration=iteration),
            delimiter=",",
            header=None,
        )
        if uses_real_missing(args):
            return apply_real_missing_metadata_dtypes(args, missing_data)
        return missing_data

    file_path = build_clean_data_path(args, imputation, mechanism=mechanism, iteration=iteration)
    clean_data = pd.read_csv(file_path, delimiter=",", header=None)
    if uses_real_missing(args):
        return apply_real_missing_metadata_dtypes(args, clean_data)
    return clean_data


def load_non_missing_data(args):
    res = _read_csv_strict(_ground_truth_csv_path(args)).to_numpy()
    return res


def load_labels(args):
    if args.dataset == "hcv":
        data = _read_csv_strict(_ground_truth_csv_path(args), delimiter=",")
        start_index = int(0.8 * len(data))
        class_id_data = data["class_id"].iloc[start_index:]
        return np.array(class_id_data)
    raise ValueError(f"load_labels is only implemented for dataset={args.dataset}")
