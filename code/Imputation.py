import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from code.core.runtime import build_safe_subprocess_env, configure_runtime_for_stability

configure_runtime_for_stability()

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from code.config.settings import cfg
from code.config.dataset_profiles import DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH
from code.core.data_io import (
    build_clean_data_path,
    build_clean_timing_path,
    get_real_missing_discrete_column_positions,
    load_real_missing_data,
    load_real_truth_data,
    resolve_missing_data_path,
    uses_real_missing,
)
from code.core.path_config import build_workspace_subprocess_env
# import matlab.engine


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _maybe_silence_r_console():
    if not _env_flag("FRAMEWORK_SILENCE_R_CONSOLE", default=False):
        return
    try:
        from rpy2.rinterface_lib import callbacks
    except Exception:
        return

    def _discard_console_output(*args, **kwargs):
        return None

    callbacks.consolewrite_print = _discard_console_output
    callbacks.consolewrite_warnerror = _discard_console_output
    callbacks.showmessage = _discard_console_output


def _load_rpy2():
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri

    _maybe_silence_r_console()
    return robjects, rpackages, pandas2ri


_R_FUNCTIONS_LOADED = set()


def _normalize_blank_missing_values(frame):
    if frame is None or frame.empty:
        return frame
    normalized = frame.copy()
    return normalized.replace(r"^\s*$", np.nan, regex=True)


def _coerce_numeric_like_columns(frame):
    if frame is None or frame.empty:
        return frame
    coerced = frame.copy()
    for column in coerced.columns:
        numeric = pd.to_numeric(coerced[column], errors="coerce")
        non_missing = coerced[column].notna()
        if not non_missing.any() or numeric.loc[non_missing].notna().all():
            coerced[column] = numeric
    return coerced


def _normalize_imputer_input_frame(frame):
    return _coerce_numeric_like_columns(_normalize_blank_missing_values(frame))


def _ensure_r_function(robjects, name, code):
    if name in _R_FUNCTIONS_LOADED:
        return
    robjects.r(code)
    _R_FUNCTIONS_LOADED.add(name)


def _normalize_constraint_column_name(name):
    return re.sub(r"[^0-9a-z]+", "", str(name).strip().lower())


def _infer_holoclean_column_names(dataset, raw_input, constraint_path):
    if not os.path.exists(constraint_path):
        return [f"c{idx}" for idx in range(raw_input.shape[1])]

    repo_root = Path(__file__).resolve().parents[1]
    ground_truth_path = repo_root / "data" / "Ground_truth_data" / f"{dataset}.csv"
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



def _prepare_r_dataframe(robjects, pandas2ri, dataframe):
    """Compatibility bridge for old/new rpy2 pandas conversion APIs."""
    try:
        pandas2ri.activate()
        return dataframe
    except DeprecationWarning:
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter

        with localconverter(default_converter + pandas2ri.converter):
            return robjects.conversion.py2rpy(dataframe)


def _run_checked(command, context, extra_env=None):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=build_safe_subprocess_env(extra_env=extra_env),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        output_sections = []
        if stderr:
            stderr_lines = stderr.splitlines()
            stderr_tail = "\n".join(stderr_lines[-40:])
            output_sections.append(f"[stderr tail]\n{stderr_tail}")
        if stdout:
            stdout_lines = stdout.splitlines()
            stdout_tail = "\n".join(stdout_lines[-40:])
            output_sections.append(f"[stdout tail]\n{stdout_tail}")
        details = "\n\n".join(output_sections) if output_sections else "no subprocess output captured"
        raise RuntimeError(
            f"{context} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"{details}"
        )


def _tf_imputer_python():
    return os.environ.get("FRAMEWORK_TF_PYTHON") or sys.executable


def _r_result_to_dataframe(robjects, pandas2ri, result):
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    with localconverter(default_converter + pandas2ri.converter):
        converted = robjects.conversion.rpy2py(result)

    if isinstance(converted, pd.DataFrame):
        return converted.copy()
    return pd.DataFrame(np.asarray(converted))


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


def _finalize_imputed_frame(missing_data, imputed, *, keep_label=True):
    if isinstance(imputed, pd.DataFrame):
        frame = imputed.copy()
    else:
        frame = pd.DataFrame(np.asarray(imputed))

    if frame.shape == missing_data.shape[::-1]:
        frame = frame.T

    if frame.shape != missing_data.shape:
        raise ValueError(
            f"Imputed data shape mismatch: expected {missing_data.shape}, got {frame.shape}"
        )

    frame = frame.reset_index(drop=True)
    frame.columns = list(missing_data.columns)
    result = missing_data.combine_first(frame)

    if keep_label and 0 in result.columns and 0 in missing_data.columns:
        result[0] = missing_data[0]

    for column in result.columns:
        if column == 0 and keep_label:
            continue
        if result[column].isnull().any():
            numeric_series = pd.to_numeric(result[column], errors="coerce")
            if numeric_series.notna().any():
                result[column] = result[column].fillna(numeric_series.mean())
            else:
                modes = result[column].mode(dropna=True)
                if not modes.empty:
                    result[column] = result[column].fillna(modes.iloc[0])

    return result


def _column_fallback_value(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        modes = series.mode(dropna=True)
        if not modes.empty:
            return modes.iloc[0]
        categories = list(series.cat.categories)
        return categories[0] if categories else 0

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().any():
        mean_value = numeric_series.mean()
        return 0.0 if pd.isna(mean_value) else mean_value

    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return 0


def _fill_frame_from_fallbacks(frame, source):
    filled = frame.copy()
    for column in filled.columns:
        if filled[column].isnull().any():
            fallback = _column_fallback_value(source[column] if column in source.columns else filled[column])
            if isinstance(filled[column].dtype, pd.CategoricalDtype) and fallback not in filled[column].cat.categories:
                filled[column] = filled[column].cat.add_categories([fallback])
            filled[column] = filled[column].fillna(fallback)
    return filled


def _mode_fallback_value(series):
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return 0


def _fill_mode_only_columns(frame, columns):
    if not columns:
        return frame
    filled = frame.copy()
    for column in columns:
        if column not in filled.columns or not filled[column].isnull().any():
            continue
        fallback = _mode_fallback_value(filled[column])
        if isinstance(filled[column].dtype, pd.CategoricalDtype) and fallback not in filled[column].cat.categories:
            filled[column] = filled[column].cat.add_categories([fallback])
        filled[column] = filled[column].fillna(fallback)
    return filled


def _finite_numeric_mask(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float))


def _is_constant_for_mice(series, *, rank_tol=1e-8):
    non_null = series.dropna()
    if non_null.nunique(dropna=True) <= 1:
        return True
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        std_value = numeric.std(ddof=0)
        return not np.isfinite(std_value) or std_value <= rank_tol
    return False


def _is_high_cardinality_or_mixed_for_mice(series, *, max_categories=20):
    non_null = series.dropna()
    if non_null.empty:
        return True

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        return False

    return True


def _numeric_collinearity_drop_columns(frame, *, rank_tol=1e-8, corr_threshold=0.999):
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    candidate_columns = []
    candidate_series = []
    standardized_columns = []

    for column in numeric.columns:
        series = numeric[column]
        if series.notna().sum() < 2:
            continue
        mean_value = series.mean()
        filled = series.fillna(mean_value)
        std_value = filled.std(ddof=0)
        if not np.isfinite(std_value) or std_value <= rank_tol:
            continue
        candidate_columns.append(column)
        candidate_series.append(series)
        standardized_columns.append(((filled - mean_value) / std_value).to_numpy(dtype=float))

    kept_vectors = []
    kept_series = []
    dropped = []
    current_rank = 0
    for column, series, vector in zip(candidate_columns, candidate_series, standardized_columns):
        if kept_vectors:
            pairwise_collinear = False
            for existing in kept_series:
                mask = series.notna() & existing.notna()
                if mask.sum() < 2:
                    continue
                left = series.loc[mask].to_numpy(dtype=float)
                right = existing.loc[mask].to_numpy(dtype=float)
                if np.std(left) <= rank_tol or np.std(right) <= rank_tol:
                    continue
                corr = np.corrcoef(left, right)[0, 1]
                if np.isfinite(corr) and abs(corr) >= corr_threshold:
                    pairwise_collinear = True
                    break
            if pairwise_collinear:
                dropped.append(column)
                continue

            kept_matrix = np.column_stack(kept_vectors)
            correlations = np.abs(np.corrcoef(kept_matrix, vector, rowvar=False)[-1, :-1])
            if np.nanmax(correlations) >= corr_threshold:
                dropped.append(column)
                continue
            candidate_matrix = np.column_stack(kept_vectors + [vector])
        else:
            candidate_matrix = vector.reshape(-1, 1)

        next_rank = np.linalg.matrix_rank(candidate_matrix, tol=rank_tol)
        if next_rank <= current_rank:
            dropped.append(column)
            continue
        kept_vectors.append(vector)
        kept_series.append(series)
        current_rank = next_rank

    return dropped


def _prepare_mice_family_input(frame, *, keep_label_in_input=False, max_categories=20):
    original = frame.copy().reset_index(drop=True)
    original.columns = list(frame.columns)
    label_columns = [0] if 0 in original.columns else []
    excluded_columns = [] if keep_label_in_input else label_columns
    force_kept_columns = label_columns if keep_label_in_input else []
    constant_columns = [
        column for column in original.columns
        if column not in force_kept_columns and _is_constant_for_mice(original[column])
    ]
    mixed_columns = [
        column for column in original.columns
        if column not in force_kept_columns
        and column not in constant_columns
        and _is_high_cardinality_or_mixed_for_mice(original[column], max_categories=max_categories)
    ]
    collinearity_candidates = [
        column for column in original.columns
        if column not in excluded_columns
        and column not in label_columns
        and column not in constant_columns
        and column not in mixed_columns
    ]
    reduced = original.loc[:, collinearity_candidates]
    collinear_columns = _numeric_collinearity_drop_columns(reduced)
    kept_columns = [
        column for column in original.columns
        if column not in set(excluded_columns).union(constant_columns, mixed_columns, collinear_columns)
    ]
    removed_columns = [column for column in original.columns if column not in kept_columns]
    metadata = {
        "original_columns": list(original.columns),
        "kept_columns": kept_columns,
        "removed_columns": removed_columns,
        "fallback_values": {
            column: _column_fallback_value(original[column])
            for column in removed_columns
        },
    }
    return original.loc[:, kept_columns].copy(), metadata


def _restore_mice_family_output(original, imputed, metadata, *, keep_label=True):
    original = original.copy().reset_index(drop=True)
    original.columns = metadata["original_columns"]
    kept_columns = metadata["kept_columns"]
    restored = pd.DataFrame(index=original.index, columns=metadata["original_columns"])

    if kept_columns:
        imputed_frame = pd.DataFrame(imputed).copy()
        if imputed_frame.shape == (len(kept_columns), len(original)):
            imputed_frame = imputed_frame.T
        if imputed_frame.shape != (len(original), len(kept_columns)):
            raise ValueError(
                "MICE-family imputed shape mismatch after pruning: "
                f"expected {(len(original), len(kept_columns))}, got {imputed_frame.shape}"
            )
        imputed_frame = imputed_frame.reset_index(drop=True)
        imputed_frame.columns = kept_columns
        for column in kept_columns:
            restored[column] = imputed_frame[column]

    for column in metadata["removed_columns"]:
        restored[column] = original[column].fillna(metadata["fallback_values"][column])

    result = original.combine_first(restored)
    if keep_label and 0 in result.columns:
        result[0] = original[0]
    return _fill_frame_from_fallbacks(result, original)


def _mice_family_can_skip_r(frame):
    return frame.shape[1] == 0 or not frame.isnull().any().any()


def _mice_family_fallback_output(original, pruned, metadata):
    filled = _fill_frame_from_fallbacks(pruned, pruned)
    return _restore_mice_family_output(
        original,
        filled,
        metadata,
        keep_label=True,
    )


def _run_numeric_java_imputer(args, missing_data, *, dataset_name, name, path, program_path, class_value, context, extra_env):
    original_missing_data = missing_data.copy()
    java_input, java_metadata = _prepare_mice_family_input(original_missing_data)
    if _mice_family_can_skip_r(java_input):
        df = _restore_mice_family_output(
            original_missing_data,
            java_input,
            java_metadata,
            keep_label=True,
        )
        df.to_csv(path, sep=",", header=False, index=False)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "java_numeric_input.csv")
        output_path = os.path.join(tmpdir, "java_numeric_output.csv")
        java_input.to_csv(input_path, sep=",", header=False, index=False, na_rep="")
        command = [
            "java",
            "-classpath",
            class_value,
            program_path,
            dataset_name,
            name,
            input_path,
            output_path,
        ]
        _run_checked(command, context, extra_env=extra_env)
        java_output = pd.read_csv(output_path, header=None)

    df = _restore_mice_family_output(
        original_missing_data,
        java_output,
        java_metadata,
        keep_label=True,
    )
    df.to_csv(path, sep=",", header=False, index=False)


def _load_missing_input(args):
    if uses_real_missing(args):
        frame = load_real_missing_data(args, restore_dtypes=False).copy()
        frame.columns = list(range(frame.shape[1]))
        frame = frame.reset_index(drop=True)
        frame = _normalize_imputer_input_frame(frame)

        if getattr(args, "real_missing_observed_path", None) or os.getenv(
            "FRAMEWORK_REAL_MISSING_OBSERVED_PATH"
        ):
            return frame

        if getattr(args, "dataset_category", None) == DATASET_CATEGORY_REAL_MISSING_WITH_TRUTH:
            truth = load_real_truth_data(args, restore_dtypes=False).copy()
            truth.columns = list(range(truth.shape[1]))
            truth = truth.reset_index(drop=True)
            truth = _normalize_imputer_input_frame(truth)
            if truth.shape[0] != frame.shape[0]:
                raise ValueError(
                    f"Observed/truth row mismatch for dataset={args.dataset}: "
                    f"observed_rows={frame.shape[0]} truth_rows={truth.shape[0]}"
                )
            frame[0] = pd.to_numeric(frame[0], errors="coerce")
            truth[0] = pd.to_numeric(truth[0], errors="coerce")
            valid_label_mask = _finite_numeric_mask(truth[0])
            frame_label_valid_mask = _finite_numeric_mask(frame[0])
            label_fill_mask = valid_label_mask & ~frame_label_valid_mask
            if bool(label_fill_mask.any()):
                frame.loc[label_fill_mask, 0] = truth.loc[label_fill_mask, 0]
            frame = frame.loc[valid_label_mask].reset_index(drop=True)

        return frame
    frame = pd.read_csv(
        resolve_missing_data_path(args, mechanism=args.missing_mechanism, iteration=args.iter),
        delimiter=",",
        header=None,
    )
    return _normalize_imputer_input_frame(frame)


def _encode_mixed_features_for_knn(feature_df):
    encoded = pd.DataFrame(index=feature_df.index)
    metadata = {}

    for column in feature_df.columns:
        series = feature_df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        non_null_count = int(series.notna().sum())
        numeric_non_null_count = int(numeric.notna().sum())

        if non_null_count == 0 or numeric_non_null_count == non_null_count:
            encoded[column] = numeric.astype(float)
            metadata[column] = {"kind": "numeric"}
            continue

        categories = pd.Index(pd.unique(series.dropna()))
        code_map = {value: idx for idx, value in enumerate(categories)}
        encoded[column] = series.map(code_map).astype(float)
        metadata[column] = {
            "kind": "categorical",
            "categories": list(categories),
        }

    return encoded, metadata


def _decode_mixed_features_from_knn(encoded_df, metadata):
    decoded = pd.DataFrame(index=encoded_df.index)

    for column in encoded_df.columns:
        column_meta = metadata.get(column, {"kind": "numeric"})
        if column_meta["kind"] == "numeric":
            decoded[column] = pd.to_numeric(encoded_df[column], errors="coerce")
            continue

        categories = list(column_meta.get("categories", []))
        numeric = pd.to_numeric(encoded_df[column], errors="coerce")
        if categories:
            numeric = numeric.round().clip(lower=0, upper=len(categories) - 1)
            decoded[column] = numeric.map(
                lambda value: categories[int(value)] if pd.notna(value) else np.nan
            )
        else:
            decoded[column] = np.nan

    return decoded


@contextmanager
def _temporary_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _impute_with_datawig(missing_data):
    try:
        import datawig
    except ImportError as exc:
        raise RuntimeError(
            "Datawig is not installed. Install it in the current environment with `pip install datawig`."
        ) from exc

    working = missing_data.copy()
    working.columns = [f"c{idx}" for idx in range(working.shape[1])]
    imputed = datawig.SimpleImputer.complete(working)
    return _finalize_imputed_frame(missing_data, imputed, keep_label=True)


def _build_smcfcs_method_list(dataframe, *, label_column=0, default_method="norm"):
    methods = []
    for column in dataframe.columns:
        if column == label_column:
            methods.append("")
            continue
        methods.append(default_method if dataframe[column].isnull().any() else "")
    return methods


def _run_holoclean_repair(session, featurizers):
    status, _ = session.repair_engine.setup_featurized_ds(featurizers)
    status, _ = session.repair_engine.setup_repair_model()
    status, _ = session.repair_engine.fit_repair_model()
    status, _ = session.repair_engine.infer_repairs()
    status, _ = session.ds.get_inferred_values()
    repaired_df, status, _ = session.ds.get_repaired_dataset()
    return repaired_df


def _holoclean_runtime_config(dataset_name):
    config = {
        "domain_thresh_1": 0.0,
        "domain_thresh_2": 0.0,
        "weak_label_thresh": 0.99,
        "max_domain": 10000,
        "threads": 1,
        "timeout": 3 * 60000,
    }
    if str(dataset_name) == "obesity":
        # Obesity's dense denial constraints create very large SQL workloads.
        # Keep domains small and prune low-probability candidates earlier.
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
    return config


def _impute_with_holoclean(args, missing_data, imputation_seed):
    holoclean_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "imputation_algorithms", "Holoclean", "Holoclean")
    )
    constraint_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "imputation_algorithms",
            "Holoclean",
            "Ground_truth_data",
            f"{args.dataset}_constraints.txt",
        )
    )

    with _temporary_sys_path(holoclean_root):
        try:
            import holoclean
            from detect import NullDetector, ViolationDetector
            from repair.featurize import (
                ConstraintFeaturizer,
                FreqFeaturizer,
                InitAttrFeaturizer,
                OccurAttrFeaturizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Holoclean dependencies are not available. It requires the vendored holoclean package and its "
                "database dependencies."
            ) from exc

        with tempfile.TemporaryDirectory(
            prefix="holoclean_",
            dir=getattr(args, "workspace_dir", None) or None,
        ) as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            holoclean_input = missing_data.copy().reset_index(drop=True)
            holoclean_input.columns = _infer_holoclean_column_names(
                args.dataset,
                holoclean_input,
                constraint_path,
            )
            holoclean_input.to_csv(input_path, index=False)

            try:
                runtime_config = _holoclean_runtime_config(args.dataset)
                session = holoclean.HoloClean(
                    db_port=int(os.environ.get("DB_PORT", "5432")),
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
                    verbose=False,
                    timeout=runtime_config["timeout"],
                    feature_norm=False,
                    weight_norm=False,
                    print_fw=False,
                    seed=imputation_seed,
                ).session
                session_name = f"hc_{args.dataset}_{os.getpid()}_{int(time.time() * 1000)}"
                session.load_data(session_name, input_path)

                detectors = [NullDetector()]
                featurizers = [
                    InitAttrFeaturizer(),
                    OccurAttrFeaturizer(),
                    FreqFeaturizer(),
                ]

                if os.path.exists(constraint_path):
                    session.load_dcs(constraint_path)
                    constraints = session.get_dcs()
                    session.ds.set_constraints(constraints)
                    if constraints:
                        detectors.append(ViolationDetector())
                        featurizers.append(ConstraintFeaturizer())
                else:
                    session.ds.set_constraints([])

                session.detect_errors(detectors)
                session.setup_domain()
                repaired_df = _run_holoclean_repair(session, featurizers)
            except Exception as exc:
                raise RuntimeError(
                    "Holoclean failed. Make sure the PostgreSQL backend is available and any optional "
                    f"constraint file is valid. Original error: {exc}"
                ) from exc

    repaired_values = repaired_df.iloc[:, 1:].copy()
    repaired_values.columns = list(missing_data.columns)
    return _finalize_imputed_frame(missing_data, repaired_values, keep_label=True)


def impute(args, imputation, dataset_name, name):
    missing_data = _load_missing_input(args)
    mode_only_columns = (
        get_real_missing_discrete_column_positions(args, missing_data)
        if uses_real_missing(args)
        else []
    )
    missing_data = _fill_mode_only_columns(missing_data, mode_only_columns)
    path = build_clean_data_path(args, imputation, ensure_parent=True)
    timing_path = build_clean_timing_path(args, imputation, ensure_parent=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start_time = time.perf_counter()
    imputation_seed = int(getattr(args, "seed", cfg["base_seed"]))
    if not missing_data.isnull().any().any():
        missing_data.to_csv(path, sep=",", header=False, index=False)
        _write_imputation_timing_sidecar(
            timing_path,
            args,
            dataset_name,
            imputation,
            time.perf_counter() - start_time,
        )
        return
    subprocess_env = build_workspace_subprocess_env(args)
    observed_override = getattr(args, "real_missing_observed_path", None)
    if observed_override:
        subprocess_env["FRAMEWORK_REAL_MISSING_OBSERVED_PATH"] = str(observed_override)
    if imputation == 'Mean':
        filled_df = missing_data.copy()
        for column in filled_df.columns:
            if column == 0:
                continue
            numeric_series = pd.to_numeric(filled_df[column], errors="coerce")
            if numeric_series.notna().any():
                filled_df[column] = filled_df[column].where(
                    filled_df[column].notna(),
                    numeric_series.mean(),
                )
            else:
                modes = filled_df[column].mode(dropna=True)
                if not modes.empty:
                    filled_df[column] = filled_df[column].fillna(modes.iloc[0])
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'Mice':
        robjects, rpackages, pandas2ri = _load_rpy2()
        original_missing_data = missing_data.copy()
        missing_data_r_input, mice_metadata = _prepare_mice_family_input(original_missing_data)
        if _mice_family_can_skip_r(missing_data_r_input):
            df = _restore_mice_family_output(
                original_missing_data,
                missing_data_r_input,
                mice_metadata,
                keep_label=True,
            )
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        temp_columns = [f"V{i}" for i in range(missing_data_r_input.shape[1])]
        missing_data_r_input.columns = temp_columns
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data_r_input)
        robjects.globalenv['dataframe'] = missing_data_r
        MICE = rpackages.importr('mice')
        func = MICE.mice
        M = 5
        try:
            imputed = func(
                missing_data_r,
                m=M,
                seed=imputation_seed,
                remove_collinear=True,
                printFlag=False,
            )
            imputed = robjects.r['complete'](imputed, M)
            df = _r_result_to_dataframe(robjects, pandas2ri, imputed)
            df = _restore_mice_family_output(
                original_missing_data,
                df,
                mice_metadata,
                keep_label=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Mice failed after constant/collinear pruning ({exc}); "
                "falling back to per-column mean/mode fills."
            )
            df = _mice_family_fallback_output(
                original_missing_data,
                missing_data_r_input,
                mice_metadata,
            )
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'Mode':
        def fill_missing_with_mode(df):
            filled_df = df.copy()
            for column in filled_df.columns:
                if filled_df[column].isnull().any():
                    mode_value = filled_df[column].mode()[0]
                    filled_df[column] = filled_df[column].fillna(mode_value)
            return filled_df
        missing_data = _load_missing_input(args)
        df = pd.DataFrame(missing_data)
        filled_df = fill_missing_with_mode(df)
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'KNNI':
        label_df = missing_data.iloc[:, [0]].copy()
        feature_df = missing_data.iloc[:, 1:].copy()
        encoded_features, knn_meta = _encode_mixed_features_for_knn(feature_df)
        X = encoded_features.to_numpy(dtype=float, copy=False)
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        imputer.fit(X)
        X_imputed = imputer.transform(X)
        imputed_features = pd.DataFrame(X_imputed, columns=feature_df.columns, index=feature_df.index)
        decoded_features = _decode_mixed_features_from_knn(imputed_features, knn_meta)
        filled_df = pd.concat([label_df, decoded_features], axis=1)
        filled_df.columns = missing_data.columns
        filled_df = _finalize_imputed_frame(missing_data, filled_df, keep_label=True)
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'CMI':
        from imputation_algorithms.Imputation_Algorithm.CMI import CMI

        merged_data_sorted = CMI(missing_data)
        df = pd.DataFrame(merged_data_sorted, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False, float_format='%.8f')
    if imputation == 'Datawig':
        df = _impute_with_datawig(missing_data)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'CI':
        robjects, rpackages, pandas2ri = _load_rpy2()
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data)
        robjects.globalenv['dataframe'] = missing_data_r
        ClustImpute = rpackages.importr('ClustImpute')
        clust_impute_func = ClustImpute.ClustImpute
        nr_cluster = 3
        imputed = clust_impute_func(missing_data_r, nr_cluster)
        list_df = list(imputed)
        df = pd.DataFrame(list_df)
        num_columns = len(missing_data.columns)
        filled_df = pd.DataFrame()
        for idx, cell in enumerate(df.iloc[0]):
            if idx + 1 > num_columns:
                break
            for index, value in enumerate(cell):
                filled_df.loc[index,idx] = value
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'XGBI':
        from imputation_algorithms.Imputation_Algorithm.XGBI import XGBImputer

        imputer = XGBImputer(categorical_index=[0])
        imputed = imputer.transform(missing_data)
        df = pd.DataFrame(imputed)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'GMM':
        from imputation_algorithms.Imputation_Algorithm.GMM import GMM

        merged_data_sorted = GMM(missing_data)
        df = pd.DataFrame(merged_data_sorted, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == "MIDA":
        from imputation_algorithms.Imputation_Algorithm.MIDA import MIDA

        if missing_data.shape[1] <= 1:
            df = missing_data.copy()
        else:
            feature_columns = list(missing_data.columns[1:])
            imputed_features = MIDA(missing_data.loc[:, feature_columns])
            feature_frame = pd.DataFrame(
                imputed_features,
                index=missing_data.index,
                columns=feature_columns,
            )
            df = missing_data.copy()
            df.loc[:, feature_columns] = feature_frame
            df = _finalize_imputed_frame(missing_data, df, keep_label=True)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == "SOM":
        from imputation_algorithms.Imputation_Algorithm.SOM import SOM

        imputed = SOM(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MISSF':
        from imputation_algorithms.Imputation_Algorithm.MISSF import MissForest

        df = pd.DataFrame(missing_data)
        mf = MissForest()
        imputed_df = mf.fit_transform(df)
        imputed_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MIWAE':
        _run_checked(
            [_tf_imputer_python(), "-m", "imputation_algorithms.Imputation_Algorithm.MIWAE", dataset_name, name],
            "MIWAE subprocess",
            extra_env=subprocess_env,
        )
    if imputation == 'RRSI':
        from imputation_algorithms.Imputation_Algorithm.RRSI import RRSI

        imputed = RRSI(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'LOESS':
        from imputation_algorithms.Imputation_Algorithm.LOESS import LOESS

        imputed = LOESS(missing_data)
        imputed.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'HDI':
        from imputation_algorithms.Imputation_Algorithm.HDI import HDI

        imputed = HDI(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MLPI':
        _run_checked(
            [sys.executable, "-m", "imputation_algorithms.Imputation_Algorithm.MLPI", dataset_name, name],
            "MLPI subprocess",
            extra_env=subprocess_env,
        )
    if imputation == 'VAEI':
        _run_checked(
            [_tf_imputer_python(), "-m", "imputation_algorithms.Imputation_Algorithm.VAEI", dataset_name, name],
            "VAEI subprocess",
            extra_env=subprocess_env,
        )
    if imputation == 'CDI':
        robjects, rpackages, pandas2ri = _load_rpy2()
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data)
        robjects.globalenv['dataframe'] = missing_data_r
        ColdDeck = rpackages.importr('nftbart')
        func = ColdDeck.CDimpute
        imputed = func(missing_data_r, missing_data_r)
        df = pd.DataFrame(imputed)
        num_columns = len(missing_data.columns)
        filled_df = pd.DataFrame()
        for idx, cell in enumerate(df.iloc[0]):
            if idx + 1 > num_columns:
                break
            for index, value in enumerate(cell):
                filled_df.loc[index,idx] = value
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'IIM':
        program_path = "Imputation_Algorithm.IIM.IIMTest"
        class_value = os.pathsep.join(
            [
                "imputation_algorithms",
                os.path.join("imputation_algorithms", "Imputation_Algorithm", "IIM", "Jama.jar"),
            ]
        )
        _run_numeric_java_imputer(
            args,
            missing_data,
            dataset_name=dataset_name,
            name=name,
            path=path,
            program_path=program_path,
            class_value=class_value,
            context="IIM java process",
            extra_env=subprocess_env,
        )
    if imputation == 'DLM':
        program_path = "Imputation_Algorithm.DLM.DLMTest"
        class_value = os.pathsep.join(
            [
                "imputation_algorithms",
                os.path.join("imputation_algorithms", "Imputation_Algorithm", "DLM", "Jama.jar"),
            ]
        )
        _run_numeric_java_imputer(
            args,
            missing_data,
            dataset_name=dataset_name,
            name=name,
            path=path,
            program_path=program_path,
            class_value=class_value,
            context="DLM java process",
            extra_env=subprocess_env,
        )
    if imputation == 'GINN':
        os.environ.setdefault("DGLBACKEND", "pytorch")
        os.environ.setdefault("DGL_DOWNLOAD_DIR", os.path.join(os.getcwd(), ".dgl_cache"))
        os.makedirs(os.environ["DGL_DOWNLOAD_DIR"], exist_ok=True)
        try:
            from imputation_algorithms.Imputation_Algorithm.GINN import GINN_impute

            imputed = GINN_impute(missing_data)
            df = pd.DataFrame(imputed, index=missing_data.index)
        except Exception as exc:
            # Keep experiment flow alive when DGL/Graphbolt runtime is incompatible.
            print(f"[WARN] GINN unavailable ({exc}); falling back to KNNI.")
            X = missing_data.values
            imputer = KNNImputer(n_neighbors=4, weights="uniform")
            df = pd.DataFrame(imputer.fit_transform(X), columns=missing_data.columns)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'NOMI':
        from imputation_algorithms.Imputation_Algorithm.NOMI import NOMI_impute

        imputed = NOMI_impute(missing_data, random_state=imputation_seed)
        df = _finalize_imputed_frame(missing_data, imputed, keep_label=True)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'KNNE':
        from imputation_algorithms.Imputation_Algorithm.KNNE import KNNE

        imputed = KNNE(missing_data, random_state=imputation_seed)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'GCID':
        try:
            import matlab.engine
        except ImportError as exc:
            raise RuntimeError("GCID requires matlabengine package and a local MATLAB installation.") from exc
        input_path = resolve_missing_data_path(args, mechanism=args.missing_mechanism, iteration=args.iter)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        eng = matlab.engine.start_matlab()
        try:
            python_dir = os.getcwd()
            relative_path = 'imputation_algorithms/Imputation_Algorithm/GCID'
            absolute_path = os.path.join(python_dir, relative_path)
            eng.cd(absolute_path)
            eng.demo(dataset_name, name, input_path, path, nargout=0)
        finally:
            eng.exit()
        gcid_output = pd.read_csv(path, header=None)
        df = _finalize_imputed_frame(missing_data, gcid_output, keep_label=True)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'IFC':
        from imputation_algorithms.Imputation_Algorithm.IFC import IFC

        imputed = IFC(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MVNI':
        robjects, rpackages, pandas2ri = _load_rpy2()
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data)
        robjects.globalenv['dataframe'] = missing_data_r
        NormImpute = rpackages.importr('mlmi')
        impute_func = NormImpute.normImp
        imputed = impute_func(missing_data_r, M = 10, pd = False, rseed = imputation_seed)
        list_df = list(imputed)
        df = pd.DataFrame(list_df)
        num_columns = len(missing_data.columns)
        filled_df = pd.DataFrame()
        for idx, cell in enumerate(df.iloc[0]):
            if idx + 1 > num_columns:
                break
            for index, value in enumerate(cell):
                filled_df.loc[index,idx] = value
        filled_df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'SMC-FCS':
        robjects, rpackages, pandas2ri = _load_rpy2()
        original_missing_data = missing_data.copy()
        smcfcs_input, smcfcs_metadata = _prepare_mice_family_input(
            original_missing_data,
            keep_label_in_input=True,
        )
        if 0 not in smcfcs_input.columns:
            df = _mice_family_fallback_output(
                original_missing_data,
                smcfcs_input,
                smcfcs_metadata,
            )
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        if smcfcs_input.shape[1] <= 1:
            df = _restore_mice_family_output(
                original_missing_data,
                smcfcs_input,
                smcfcs_metadata,
                keep_label=True,
            )
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        smcfcs_kept_columns = list(smcfcs_input.columns)
        smcfcs_input.columns = [f'Col{i}' for i in range(len(smcfcs_input.columns))]
        column_names = smcfcs_input.columns.astype(str)
        formula = (column_names[0]) + '~' + '+'.join(column_names[1:])
        method_list = _build_smcfcs_method_list(smcfcs_input, label_column="Col0")
        if all(method == "" for method in method_list):
            df = original_missing_data.copy()
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, smcfcs_input)
        robjects.globalenv['dataframe'] = missing_data_r
        FCSImpute = rpackages.importr('smcfcs')
        impute_func = FCSImpute.smcfcs
        options_func = robjects.r["options"]
        robjects.r(f"set.seed({imputation_seed})")
        itr = 5
        old_warn = options_func("warn")[0]
        try:
            options_func(warn=-1)
            imputed = impute_func(
                missing_data_r,
                smtype="lm",
                smformula=str(formula),
                method=method_list,
                m=itr,
                numit=10,
                rjlimit=50000,
                noisy=False,
            )
        finally:
            options_func(warn=old_warn)
        imputed = imputed.rx('impDatasets')
        list_df = list(imputed)
        df = pd.DataFrame(list_df)
        df = df.iloc[0][itr-1]
        df = pd.DataFrame(df).T
        df.columns = smcfcs_kept_columns
        df = _restore_mice_family_output(
            original_missing_data,
            df,
            smcfcs_metadata,
            keep_label=True,
        )
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MIGEC':
        from imputation_algorithms.Imputation_Algorithm.MIGEC import MIGEC

        imputed = MIGEC(missing_data, random_seed=imputation_seed)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'Sice':
        robjects, rpackages, pandas2ri = _load_rpy2()
        original_missing_data = missing_data.copy()
        missing_data_r_input, mice_metadata = _prepare_mice_family_input(original_missing_data)
        if _mice_family_can_skip_r(missing_data_r_input):
            df = _restore_mice_family_output(
                original_missing_data,
                missing_data_r_input,
                mice_metadata,
                keep_label=True,
            )
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        filled = pd.DataFrame(
            0.0,
            index=missing_data_r_input.index,
            columns=missing_data_r_input.columns,
        )
        temp_columns = [f"V{i}" for i in range(missing_data_r_input.shape[1])]
        missing_data_r_input.columns = temp_columns
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data_r_input)
        robjects.globalenv['dataframe'] = missing_data_r
        MICE = rpackages.importr('mice')
        func = MICE.mice
        as_data_frame = robjects.r["as.data.frame"]
        M = 5
        try:
            mice_run = func(
                missing_data_r,
                m=M,
                seed=imputation_seed,
                remove_collinear=True,
                printFlag=False,
            )
            for i in range(1, M+1):
                imputed = robjects.r['complete'](mice_run, i)
                df = _r_result_to_dataframe(robjects, pandas2ri, as_data_frame(imputed))
                df = df.reset_index(drop=True)
                if df.shape != missing_data_r_input.shape:
                    raise ValueError(
                        f"Sice returned unexpected shape: expected {missing_data_r_input.shape}, got {df.shape}"
                    )
                df.columns = list(filled.columns)
                df = df.apply(pd.to_numeric, errors="coerce")
                filled += df
            filled = filled / M
            filled = _restore_mice_family_output(
                original_missing_data,
                filled,
                mice_metadata,
                keep_label=True,
            )
        except Exception as exc:
            print(
                f"[WARN] Sice failed after constant/collinear pruning ({exc}); "
                "falling back to per-column mean/mode fills."
            )
            filled = _mice_family_fallback_output(
                original_missing_data,
                missing_data_r_input,
                mice_metadata,
            )
        filled.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'TWDA':
        robjects, rpackages, pandas2ri = _load_rpy2()
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data)
        robjects.globalenv['dataframe'] = missing_data_r
        Miceadds = rpackages.importr('miceadds', on_conflict="warn")
        impute_func = Miceadds.tw_mcmc_imputation
        imputed = impute_func(missing_data_r, iter = 100)
        df = pd.DataFrame(imputed)
        df = df.T
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'MIPCA':
        robjects, rpackages, pandas2ri = _load_rpy2()
        original_missing_data = missing_data.copy()
        mipca_input, mipca_metadata = _prepare_mice_family_input(original_missing_data)
        if _mice_family_can_skip_r(mipca_input):
            df = _restore_mice_family_output(
                original_missing_data,
                mipca_input,
                mipca_metadata,
                keep_label=True,
            )
            df.to_csv(path, sep=",", header=False, index=False)
            end_time = time.perf_counter()
            _write_imputation_timing_sidecar(
                timing_path,
                args,
                dataset_name,
                imputation,
                end_time - start_time,
            )
            return
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, mipca_input)
        robjects.globalenv['dataframe'] = missing_data_r
        MIPCA = rpackages.importr('missMDA')
        impute_func = MIPCA.MIPCA
        imputed = impute_func(missing_data_r)
        imputed = imputed.rx('res.imputePCA')
        df = _restore_mice_family_output(
            original_missing_data,
            _r_result_to_dataframe(robjects, pandas2ri, imputed[0]),
            mipca_metadata,
            keep_label=True,
        )
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'NMF':
        robjects, rpackages, pandas2ri = _load_rpy2()
        r_code = """
        run_nmf_imputation <- function(data_matrix, z) {
        data_matrix[is.na(data_matrix)] <- 0
        data_matrix <- data.matrix(data_matrix)
        nmf <- RcppML::nmf(data_matrix, z)
        data_nmf_imputed <- as.matrix(nmf$w %*% Matrix::Diagonal(x = nmf$d) %*% nmf$h)
        return(data_nmf_imputed)
        }
        """
        missing_data_r = _prepare_r_dataframe(robjects, pandas2ri, missing_data)
        robjects.globalenv['dataframe'] = missing_data_r
        _ensure_r_function(robjects, "run_nmf_imputation", r_code)
        z = 3
        imputed = robjects.r['run_nmf_imputation'](missing_data_r, z)
        df = _finalize_imputed_frame(missing_data, _r_result_to_dataframe(robjects, pandas2ri, imputed))
        result = missing_data.combine_first(df)
        result.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'POP':
        from imputation_algorithms.Imputation_Algorithm.POP import POP

        imputed = POP(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'RIBG':
        from imputation_algorithms.Imputation_Algorithm.RIBG import RIBG

        imputed = RIBG(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == 'CVBKNN':
        from imputation_algorithms.Imputation_Algorithm.CVBKNN import CVBKNN

        imputed = CVBKNN(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="BPCA":
        from imputation_algorithms.Imputation_Algorithm.BPCA import BPCA

        imputed = BPCA(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="LSA":
        from imputation_algorithms.Imputation_Algorithm.LSA import LSA

        imputed = _finalize_imputed_frame(missing_data, LSA(missing_data))
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="PMM":
        from imputation_algorithms.Imputation_Algorithm.PMM import PMM

        imputed = PMM(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="CCF":
        from imputation_algorithms.Imputation_Algorithm.CCF import CCF

        imputed = CCF(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation == "Holoclean":
        df = _impute_with_holoclean(args, missing_data, imputation_seed)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="ESD":
        from imputation_algorithms.Imputation_Algorithm.ESD import ESDImputer

        imputer = ESDImputer(epsilon=1e-3, max_iter=100)
        imputed = imputer.fit_transform(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="MIST":
        from imputation_algorithms.Imputation_Algorithm.MIST import MIST

        robjects, _, _ = _load_rpy2()
        robjects.r(f"set.seed({imputation_seed})")
        imputed = MIST(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="CLRMC":
        from imputation_algorithms.Imputation_Algorithm.CLRMC import CLRMC

        imputed = CLRMC(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="ECLRMC":
        from imputation_algorithms.Imputation_Algorithm.ECLRMC import ECLRMC

        # 假设第一列不进行填补
        first_col = missing_data.iloc[:, 0]  # 提取第一列
        data_to_impute = missing_data.iloc[:, 1:]  # 其余列进行填补
        # 调用 R 进行填补
        imputed_data = ECLRMC(data_to_impute)
        # 如果 imputed_data 仍是 NumPy 数组，转换成 DataFrame
        if isinstance(imputed_data, np.ndarray):
            imputed_data = pd.DataFrame(imputed_data)

        # 确保 first_col 也是 DataFrame
        first_col = pd.DataFrame(first_col)
        # 合并回第一列
        df = pd.concat([first_col, imputed_data], axis=1)
        # 保存数据
        df.to_csv(path, sep=",", header=False, index=False)
    if imputation=="NIIA":
        from imputation_algorithms.Imputation_Algorithm.NIIA import NIIAImputer

        imputer = NIIAImputer(epsilon=1e-3, max_iter=100)
        imputed = imputer.fit_transform(missing_data)
        df = pd.DataFrame(imputed, index=missing_data.index)
        df.to_csv(path, sep=",", header=False, index=False)

    imputation_runtime_seconds = time.perf_counter() - start_time
    _write_imputation_timing_sidecar(
        timing_path,
        args,
        dataset_name,
        imputation,
        imputation_runtime_seconds,
    )
