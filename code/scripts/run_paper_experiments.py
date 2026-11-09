import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from code.scripts.experiment3_fe_library import fit_linearmodels_absorbingls
from code.scripts.paper_experiment_5 import run_experiment_5
from code.config.settings import get_default_config
from code.scripts.plotting.paper_plots import (
    PAPER_EXPERIMENT_GRID_WIDTH_PT,
    PAPER_EXPERIMENT1_WIDTH_PT,
    PAPER_PT_PER_INCH,
    save_mechanism_dataset_bar_grid,
    save_mechanism_dataset_boxplot_grid,
    save_nrmse_delta_scatter,
)
from code.scripts.plot_experiment4_triptych_preview import generate_cross_dataset_triptych_artifacts, generate_triptych_artifacts

PAPER_DEFAULT_CONFIG = get_default_config()
PAPER_IMPUTER_ORDER = [str(name) for name in PAPER_DEFAULT_CONFIG["imputation_method"]]
PAPER_CLASSIFIER_ORDER = [str(name) for name in PAPER_DEFAULT_CONFIG["class_method"]]
PAPER_MECHANISM_ORDER = [str(name) for name in PAPER_DEFAULT_CONFIG["missing_mechanisms"]]

STRUCTURE_METRICS = ["m1_w1_mean", "m1_corr_frob", "m1_fisher_delta"]
PRIMARY_STRUCTURE = "m1_corr_frob"
PRIMARY_HARD = "m2_hard_inconsistency_ref_input"
PRIMARY_TARGET = "harm_f1"
EXPERIMENT4_TRIPTYCH_M1_METRICS = ["m1_corr_frob", "m1_fisher_delta"]
EXPERIMENT4_TRIPTYCH_BASE_M2_METRICS = [PRIMARY_HARD]
EXPERIMENT4_SAMPLE_LEVEL_FLIP_PRIMARY = "is_flip_ref_input"
EXPERIMENT4_TRIPTYCH_ZH_M1_METRICS = ["m1_corr_frob", "m1_fisher_delta"]
EXPERIMENT4_TRIPTYCH_ZH_M2_METRICS = [PRIMARY_HARD]
EXPERIMENT4_FOCUS_DATASET = "wine"
EXPERIMENT4_FOCUS_CLASSIFIER = "LogisticRegression"
EXPERIMENT4_FOCUS_MISSING_PERCENT = 0.3
EXPERIMENT2_SEEDS = [42, 52, 62]
EXPERIMENT1_METRIC_LABEL_MAP = {
    "m1_corr_frob": r"$\Delta{Corr}$",
    "m1_fisher_delta": r"$\Delta\mathrm{FR}$",
    "m1_w1_mean": r"$\overline{W_1}$",
    "m1_w1_iqr_scaled_mean": r"$\overline{W}_{1,\mathrm{IQR}}$",
    "m1_w1_relative_iqr_mean": r"$\overline{W_1}$",
}
EXPERIMENT1_ZH_METRIC_LABEL_MAP = {
    "m1_corr_frob": "相\n关\n矩\n阵\n差",
    "m1_fisher_delta": "平\n均\n费\n舍\n尔\n判\n别\n率",
    "m1_w1_mean": "平\n均\n瓦\n瑟\n斯\n坦\n距\n离",
    "m1_w1_iqr_scaled_mean": "IQR\n缩\n放\nW1",
    "m1_w1_relative_iqr_mean": "平\n均\nW1",
}


def _is_zh_label_mode(label_mode):
    return str(label_mode).strip().lower() == "zh"


def _structure_metrics_for_w1(w1_column):
    return [str(w1_column), "m1_corr_frob", "m1_fisher_delta"]


def _ordered_present_names(names, preferred_order):
    seen = {str(name) for name in names}
    ordered = [str(name) for name in preferred_order if str(name) in seen]
    ordered.extend(sorted(name for name in seen if name not in set(ordered)))
    return ordered


def _require_columns(frame, required_cols, frame_name):
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _ensure_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _remove_stale_macos_files(path):
    root = Path(path)
    if not root.exists():
        return
    for item in root.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def _load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _common_parent(path_or_paths):
    if isinstance(path_or_paths, list):
        parents = [str(Path(item).parent) for item in path_or_paths]
        return Path(os.path.commonpath(parents))
    return Path(path_or_paths).parent


def _resolve_result_paths(path, stem):
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        return [str(candidate)]

    if candidate.exists() and candidate.is_dir():
        direct_path = candidate / f"{stem}.csv"
        if direct_path.exists():
            return [str(direct_path)]
        matches = sorted(item for item in candidate.rglob(f"{stem}.csv") if item.is_file())
        if matches:
            return [str(item) for item in matches]
        return None

    return None


def _resolve_latest_result_path(path, stem):
    paths = _resolve_result_paths(path, stem)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    direct_matches = [item for item in paths if Path(item).parent == Path(path)]
    if direct_matches:
        return direct_matches[0]
    return paths


def _resolve_result_root(resolved_path):
    return str(_common_parent(resolved_path))


def _resolve_default_input_root(path, stem):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    default_map = {
        "run_summary": "result/run_summary.csv",
        "sample_level": "result/sample_level.csv",
    }
    if str(path) != default_map.get(stem):
        return candidate

    result_root = Path("result")
    if (result_root / f"{stem}.csv").exists():
        return result_root
    combined_dir = result_root / "combined_csv_cleaned"
    if combined_dir.exists():
        return combined_dir
    return candidate


def _discover_category_result_roots(summary_input):
    candidate = Path(summary_input)
    if not candidate.exists() or not candidate.is_dir():
        return []
    category_roots = []
    for child in sorted(item for item in candidate.iterdir() if item.is_dir()):
        if (child / "run_summary.csv").exists():
            category_roots.append(child)
    return category_roots


def _load_summary(path):
    if isinstance(path, list):
        frames = [_load_csv(item) for item in path]
        frames = [frame for frame in frames if not frame.empty]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        resolved = _resolve_latest_result_path(path, "run_summary") if not os.path.exists(path) else path
        frame = _load_csv(resolved)
    numeric_cols = [
        "missing_percent",
        "max_missing_lenth",
        "f1_missing",
        "f1_imputed",
        "delta_f1",
        "runtime_seconds",
        "rmse_masked",
        "nrmse",
        "masked_cell_count",
        "m1_w1_mean",
        "m1_w1_iqr_scaled_mean",
        "m1_w1_relative_iqr_mean",
        "m1_corr_frob",
        "m1_fisher_delta",
        "m2_inconsistency_ref_input",
        "m2_inconsistency_retrained",
        "m2_hard_inconsistency_ref_input",
        "m2_hard_inconsistency_retrained",
        "m2_margin_shift",
        "m2_hard_margin_shift",
        "m2_flip_pos",
        "m2_flip_neg",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if {"f1_ground_ref", "f1_imputed"}.issubset(frame.columns):
        frame["f1_ground_ref"] = pd.to_numeric(frame["f1_ground_ref"], errors="coerce")
        frame["f1_imputed"] = pd.to_numeric(frame["f1_imputed"], errors="coerce")
        frame["harm_f1"] = frame["f1_ground_ref"] - frame["f1_imputed"]
    return frame


def _load_sample_level(path):
    if isinstance(path, list):
        frames = [_load_csv(item) for item in path]
        frames = [frame for frame in frames if not frame.empty]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        resolved = _resolve_latest_result_path(path, "sample_level") if not os.path.exists(path) else path
        frame = _load_csv(resolved)
    numeric_cols = [
        "missing_percent",
        "sample_id",
        "margin_ref_ground",
        "margin_ref_imputed",
        "margin_shift",
        "is_hard",
        "is_flip_ref_input",
        "is_flip_retrained",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return _normalize_sample_level_margin_direction(frame)


def _normalize_real_missing_truth_placeholder_mechanism(summary):
    if summary is None or summary.empty:
        return summary
    if "dataset_category" not in summary.columns or "missing_mechanism" not in summary.columns:
        return summary
    mask = summary["dataset_category"].astype(str).eq("real_missing_with_truth")
    if not mask.any():
        return summary
    working = summary.copy()
    working.loc[mask, "missing_mechanism"] = "Real missing"
    return working


def _is_real_missing_with_truth_summary(summary):
    if summary is None or summary.empty or "dataset_category" not in summary.columns:
        return False
    categories = set(summary["dataset_category"].dropna().astype(str).unique())
    return bool(categories) and categories.issubset({"real_missing_with_truth"})


def _normalize_sample_level_margin_direction(frame):
    if frame is None or frame.empty:
        return frame
    if {"margin_ref_ground", "margin_ref_imputed"}.issubset(frame.columns):
        working = frame.copy()
        working["margin_ref_ground"] = pd.to_numeric(working["margin_ref_ground"], errors="coerce")
        working["margin_ref_imputed"] = pd.to_numeric(working["margin_ref_imputed"], errors="coerce")
        working["margin_shift"] = working["margin_ref_imputed"] - working["margin_ref_ground"]
        return working
    return frame


def _refresh_summary_margin_metrics_from_sample_level(summary, sample_level):
    if summary is None or summary.empty or sample_level is None or sample_level.empty:
        return summary
    if not {"margin_ref_ground", "margin_ref_imputed"}.issubset(sample_level.columns):
        return summary

    working = _normalize_sample_level_margin_direction(sample_level)
    group_cols = [
        col
        for col in [
            "source_dir",
            "dataset",
            "iter",
            "seed",
            "missing_percent",
            "missing_mechanism",
            "imputer",
            "classifier",
        ]
        if col in working.columns and col in summary.columns
    ]
    if not group_cols:
        return summary

    rows = []
    for group_key, group_frame in working.groupby(group_cols, as_index=False, dropna=False, sort=False):
        row = {}
        if isinstance(group_key, tuple):
            for idx, col in enumerate(group_cols):
                row[col] = group_key[idx]
        else:
            row[group_cols[0]] = group_key

        margin_series = pd.to_numeric(group_frame["margin_shift"], errors="coerce").dropna()
        hard_mask = (
            pd.to_numeric(group_frame["is_hard"], errors="coerce").fillna(0).astype(int) == 1
            if "is_hard" in group_frame.columns
            else pd.Series(False, index=group_frame.index)
        )
        hard_margin_series = pd.to_numeric(group_frame.loc[hard_mask, "margin_shift"], errors="coerce").dropna()

        row["m2_margin_shift"] = float(margin_series.mean()) if not margin_series.empty else np.nan
        row["m2_hard_margin_shift"] = float(hard_margin_series.mean()) if not hard_margin_series.empty else np.nan
        rows.append(row)

    if not rows:
        return summary

    margin_frame = pd.DataFrame(rows)
    merged = summary.drop(columns=["m2_margin_shift", "m2_hard_margin_shift"], errors="ignore").merge(
        margin_frame,
        on=group_cols,
        how="left",
    )
    return merged


def _run_all_paper_experiments(
    summary_path,
    sample_level_path,
    output_dir,
    num_bins,
    bootstrap_rounds,
    seed,
    require_sample_level,
    selected_experiments=None,
    experiment3_missing_percent=None,
    label_mode="en",
    main_figures_only=False,
    w1_column="m1_w1_mean",
    experiment3_layout="legacy",
    experiment4_correlations_only=False,
    hsr_paired_path=None,
):
    selected = set(int(item) for item in selected_experiments) if selected_experiments else {1, 2, 3, 4}
    needs_summary = bool(selected.intersection({1, 2, 3, 4}))
    if needs_summary and summary_path is None:
        raise FileNotFoundError("No run_summary file found.")
    if sample_level_path is None and require_sample_level:
        raise FileNotFoundError("No sample_level file found.")
    if 5 in selected and hsr_paired_path is None:
        raise FileNotFoundError("Experiment 5 requires an HSR paired result file.")
    os.makedirs(output_dir, exist_ok=True)
    _remove_stale_macos_files(output_dir)
    summary = _load_summary(summary_path) if needs_summary else pd.DataFrame()
    summary = _normalize_real_missing_truth_placeholder_mechanism(summary)
    if needs_summary and w1_column not in summary.columns:
        raise ValueError(f"run_summary missing requested W1 column: {w1_column}")
    structure_metrics = _structure_metrics_for_w1(w1_column)
    config = {
        "summary": (
            [str(item) for item in summary_path]
            if isinstance(summary_path, list)
            else (str(summary_path) if summary_path is not None else None)
        ),
        "sample_level": (
            [str(item) for item in sample_level_path]
            if isinstance(sample_level_path, list)
            else (str(sample_level_path) if sample_level_path is not None else None)
        ),
        "w1_column": str(w1_column),
        "input_rows": int(len(summary)),
        "w1_nonnull_rows": int(
            pd.to_numeric(summary[w1_column], errors="coerce").notna().sum()
        ) if w1_column in summary.columns else 0,
        "structure_metrics": structure_metrics,
        "selected_experiments": sorted(selected),
        "experiment3_missing_percent": experiment3_missing_percent,
        "experiment3_layout": str(experiment3_layout),
        "experiment4_correlations_only": bool(experiment4_correlations_only),
        "hsr_paired": str(hsr_paired_path) if hsr_paired_path is not None else None,
    }
    Path(output_dir, "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if 1 in selected:
        real_mask = (
            summary["dataset_category"].astype(str).eq("real_missing_with_truth")
            if "dataset_category" in summary.columns
            else pd.Series(False, index=summary.index)
        )
        mixed_categories = bool(real_mask.any() and (~real_mask).any())
        synthetic_summary = summary.loc[~real_mask].copy() if mixed_categories else summary
        real_summary = summary.loc[real_mask].copy() if mixed_categories else pd.DataFrame()
        run_experiment_1(
            synthetic_summary,
            output_dir,
            label_mode=label_mode,
            main_figures_only=main_figures_only,
            structure_metrics=structure_metrics,
            experiment_subdir="experiment_1",
            include_rate_figures=True,
        )
        if mixed_categories:
            run_experiment_1(
                real_summary,
                output_dir,
                label_mode=label_mode,
                main_figures_only=main_figures_only,
                structure_metrics=structure_metrics,
                experiment_subdir=os.path.join(
                    "experiment_1", "real_missing_truth"
                ),
                include_rate_figures=False,
            )
            os.makedirs(os.path.join(output_dir, "experiment_1"), exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "scope": "no_real_missing",
                        "output_dir": "experiment_1",
                        "input_rows": int((~real_mask).sum()),
                    },
                    {
                        "scope": "real_missing_with_truth",
                        "output_dir": "experiment_1/real_missing_truth",
                        "input_rows": int(real_mask.sum()),
                    },
                ]
            ).to_csv(
                os.path.join(output_dir, "experiment_1", "scope_manifest.csv"),
                index=False,
            )

        dotplot_root = os.path.join(output_dir, "experiment_1_dotplot")
        _ensure_dir(dotplot_root)
        try:
            from code.scripts.exp1_dotplot_figures import (
                generate_exp1_dotplot_combined_figures,
                generate_exp1_dotplot_figures,
            )

            generate_exp1_dotplot_figures(
                synthetic_summary,
                dotplot_root,
                per_rate=not main_figures_only,
                metrics={
                    metric: EXPERIMENT1_METRIC_LABEL_MAP.get(metric, metric)
                    for metric in structure_metrics
                },
            )
            if mixed_categories:
                generate_exp1_dotplot_figures(
                    real_summary,
                    os.path.join(dotplot_root, "real_missing_truth"),
                    per_rate=False,
                    metrics={
                        metric: EXPERIMENT1_METRIC_LABEL_MAP.get(metric, metric)
                        for metric in structure_metrics
                    },
                )
                combined_synthetic_summary = synthetic_summary
                if "missing_percent" in synthetic_summary.columns:
                    missing_percent = pd.to_numeric(
                        synthetic_summary["missing_percent"], errors="coerce"
                    )
                    combined_synthetic_summary = synthetic_summary.loc[
                        np.isclose(missing_percent, 0.3, equal_nan=False)
                    ].copy()
                    if combined_synthetic_summary.empty:
                        raise ValueError(
                            "Experiment 1 combined dot plot requires synthetic "
                            "results at the 30% missing rate"
                        )
                generate_exp1_dotplot_combined_figures(
                    combined_synthetic_summary,
                    real_summary,
                    dotplot_root,
                    metrics={
                        metric: EXPERIMENT1_METRIC_LABEL_MAP.get(metric, metric)
                        for metric in structure_metrics
                    },
                )
        except Exception as exc:  # never break the classic pipeline outputs
            print(f"[warn] experiment_1 dot-plot figures skipped: {exc}")
    if 2 in selected:
        real_mask = (
            summary["dataset_category"].astype(str).eq("real_missing_with_truth")
            if "dataset_category" in summary.columns
            else pd.Series(False, index=summary.index)
        )
        if real_mask.any() and (~real_mask).any():
            synthetic_exp2_points = run_experiment_2(
                summary.loc[~real_mask].copy(),
                output_dir,
                label_mode=label_mode,
                main_figures_only=main_figures_only,
                experiment_subdir="experiment_2",
            )
            real_exp2_points = run_experiment_2(
                summary.loc[real_mask].copy(),
                output_dir,
                label_mode=label_mode,
                main_figures_only=True,
                experiment_subdir=os.path.join(
                    "experiment_2", "real_missing_truth"
                ),
            )
            if (
                isinstance(synthetic_exp2_points, pd.DataFrame)
                and not synthetic_exp2_points.empty
                and isinstance(real_exp2_points, pd.DataFrame)
                and not real_exp2_points.empty
            ):
                save_nrmse_delta_scatter(
                    synthetic_exp2_points,
                    os.path.join(
                        output_dir,
                        "experiment_2",
                        "nrmse_vs_harm_f1_with_real_world.png",
                    ),
                    y_col="harm_f1",
                    y_label=(
                        "F1损失"
                        if _is_zh_label_mode(label_mode)
                        else "Harm F1"
                    ),
                    label_mode=label_mode,
                    imputer_order=PAPER_IMPUTER_ORDER,
                    classifier_order=PAPER_CLASSIFIER_ORDER,
                    mechanism_title_style="abbr",
                    show_axis_labels_every_panel=True,
                    real_world_subset=real_exp2_points,
                )
            os.makedirs(os.path.join(output_dir, "experiment_2"), exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "scope": "no_real_missing",
                        "output_dir": "experiment_2",
                        "input_rows": int((~real_mask).sum()),
                    },
                    {
                        "scope": "real_missing_with_truth",
                        "output_dir": "experiment_2/real_missing_truth",
                        "input_rows": int(real_mask.sum()),
                    },
                ]
            ).to_csv(
                os.path.join(output_dir, "experiment_2", "scope_manifest.csv"),
                index=False,
            )
        else:
            run_experiment_2(
                summary,
                output_dir,
                label_mode=label_mode,
                main_figures_only=main_figures_only,
            )
    if 3 in selected:
        run_experiment_3(
            summary,
            output_dir,
            restrict_missing_percent=experiment3_missing_percent,
            label_mode=label_mode,
            main_figures_only=main_figures_only,
            structure_metrics=structure_metrics,
            analysis_layout=experiment3_layout,
        )
    if 4 in selected:
        experiment4_m1_metrics = (
            structure_metrics
            if experiment4_correlations_only
            else (
                EXPERIMENT4_TRIPTYCH_ZH_M1_METRICS
                if _is_zh_label_mode(label_mode)
                else None
            )
        )
        experiment4_m2_metrics = EXPERIMENT4_TRIPTYCH_ZH_M2_METRICS if _is_zh_label_mode(label_mode) else None
        selected_m2_metrics = list(experiment4_m2_metrics or EXPERIMENT4_TRIPTYCH_BASE_M2_METRICS)
        needs_sample_level = any("margin" in str(metric) for metric in selected_m2_metrics)
        sample_level = _load_sample_level(sample_level_path) if needs_sample_level and sample_level_path is not None else pd.DataFrame()
        run_experiment_4(
            summary,
            sample_level,
            output_dir,
            bx=num_bins,
            by=num_bins,
            bootstrap_rounds=bootstrap_rounds,
            seed=seed,
            label_mode=label_mode,
            experiment4_m1_metrics=experiment4_m1_metrics,
            experiment4_m2_metrics=experiment4_m2_metrics,
            main_figures_only=main_figures_only,
            correlations_only=experiment4_correlations_only,
        )
    if 5 in selected:
        run_experiment_5(hsr_paired_path, output_dir)

def _sanitize_name(value):
    text = str(value)
    safe_chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    return "".join(safe_chars).strip("_") or "unknown"


def _generate_experiment4_triptych_combo_figures(
    summary,
    exp_dir,
    label_mode="en",
    m1_metrics=None,
    m2_metrics=None,
    max_figures=None,
):
    combo_root = os.path.join(exp_dir, "triptych_full_conditions")
    os.makedirs(combo_root, exist_ok=True)
    filtered = summary.copy()
    status = filtered["status"].fillna("").astype(str).str.lower() if "status" in filtered.columns else pd.Series(["ok"] * len(filtered))
    filtered = filtered[status.isin(["ok", "success"])].copy()
    if max_figures is not None:
        focus_filtered = filtered[
            (filtered["dataset"].astype(str) == EXPERIMENT4_FOCUS_DATASET)
            & (filtered["classifier"].astype(str) == EXPERIMENT4_FOCUS_CLASSIFIER)
            & np.isclose(pd.to_numeric(filtered["missing_percent"], errors="coerce"), EXPERIMENT4_FOCUS_MISSING_PERCENT)
        ].copy()
        if not focus_filtered.empty:
            filtered = focus_filtered
    combos = (
        filtered[["dataset", "classifier", "missing_percent"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["dataset", "classifier", "missing_percent"])
    )
    manifest_rows = []
    selected_m1_metrics = list(m1_metrics or EXPERIMENT4_TRIPTYCH_M1_METRICS)
    selected_m2_metrics = list(m2_metrics or EXPERIMENT4_TRIPTYCH_BASE_M2_METRICS)
    figure_count = 0
    for row in combos.itertuples(index=False):
        dataset = str(row.dataset)
        classifier = str(row.classifier)
        missing_percent = float(row.missing_percent)
        mp_tag = str(missing_percent).replace(".", "_")
        target_dir = Path(combo_root) / dataset / classifier / f"mp_{mp_tag}"
        for m1_metric in selected_m1_metrics:
            for m2_metric in selected_m2_metrics:
                if max_figures is not None and figure_count >= max_figures:
                    pd.DataFrame(manifest_rows).to_csv(os.path.join(combo_root, "figure_manifest.csv"), index=False)
                    return
                points_path, plot_path = generate_triptych_artifacts(
                    summary_df=summary,
                    output_dir=target_dir,
                    dataset=dataset,
                    classifier=classifier,
                    missing_percent=missing_percent,
                    m1_column=m1_metric,
                    m2_column=m2_metric,
                    label_mode=label_mode,
                )
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "missing_percent": missing_percent,
                        "m1_metric": m1_metric,
                        "m2_metric": m2_metric,
                        "points_csv": str(points_path),
                        "plot_png": str(plot_path),
                    }
                )
                figure_count += 1
    pd.DataFrame(manifest_rows).to_csv(os.path.join(combo_root, "figure_manifest.csv"), index=False)


def _generate_experiment4_cross_dataset_figures(
    summary,
    exp_dir,
    label_mode="en",
    m1_metrics=None,
    m2_metrics=None,
):
    output_root = Path(exp_dir) / "cross_dataset_triptych"
    os.makedirs(output_root, exist_ok=True)
    filtered = summary.copy()
    status = filtered["status"].fillna("").astype(str).str.lower() if "status" in filtered.columns else pd.Series(["ok"] * len(filtered))
    filtered = filtered[status.isin(["ok", "success"])].copy()
    combos = (
        filtered[["classifier", "missing_percent"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["classifier", "missing_percent"])
    )
    manifest_rows = []
    selected_m1_metrics = list(m1_metrics or EXPERIMENT4_TRIPTYCH_M1_METRICS)
    selected_m2_metrics = list(m2_metrics or EXPERIMENT4_TRIPTYCH_BASE_M2_METRICS)
    for row in combos.itertuples(index=False):
        classifier = str(row.classifier)
        missing_percent = float(row.missing_percent)
        mp_tag = str(missing_percent).replace(".", "_")
        target_dir = output_root / classifier / f"mp_{mp_tag}"
        for m1_metric in selected_m1_metrics:
            for m2_metric in selected_m2_metrics:
                points_path, plot_path = generate_cross_dataset_triptych_artifacts(
                    summary_df=summary,
                    output_dir=target_dir,
                    classifier=classifier,
                    missing_percent=missing_percent,
                    m1_column=m1_metric,
                    m2_column=m2_metric,
                    label_mode=label_mode,
                )
                manifest_rows.append(
                    {
                        "classifier": classifier,
                        "missing_percent": missing_percent,
                        "m1_metric": m1_metric,
                        "m2_metric": m2_metric,
                        "points_csv": str(points_path),
                        "plot_png": str(plot_path),
                    }
                )
    pd.DataFrame(manifest_rows).to_csv(output_root / "figure_manifest.csv", index=False)


def _format_missing_percent_tag(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "missing_percent_unknown"
    return f"missing_percent_{float(numeric):g}".replace(".", "_")


def _as_float_or_nan(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def _filter_frame_by_restriction(frame, col, value):
    if col not in frame.columns:
        return frame.iloc[0:0].copy()

    target_numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(target_numeric):
        series_numeric = pd.to_numeric(frame[col], errors="coerce")
        mask = series_numeric.notna() & np.isclose(
            series_numeric.to_numpy(dtype=float),
            float(target_numeric),
            rtol=1e-9,
            atol=1e-12,
        )
        return frame.loc[mask].copy()

    return frame.loc[frame[col].astype(str) == str(value)].copy()


def _sorted_missing_percent_values(frame):
    if "missing_percent" not in frame.columns:
        return []
    values = pd.to_numeric(frame["missing_percent"], errors="coerce").dropna().unique()
    return sorted(float(value) for value in values)


def _iqr_bounds(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    return float(values.quantile(0.25)), float(values.quantile(0.75))


def _summarize_experiment4_hard_samples(sample_level):
    group_cols = [
        col
        for col in ["dataset", "classifier", "missing_mechanism", "missing_percent", "imputer"]
        if col in sample_level.columns
    ]
    if not group_cols:
        return pd.DataFrame()

    working = sample_level.copy()
    for col in ["sample_id", "is_hard", EXPERIMENT4_SAMPLE_LEVEL_FLIP_PRIMARY, "margin_shift"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    rows = []
    for group_key, group_frame in working.groupby(group_cols, as_index=False, dropna=False, sort=True):
        row = {}
        if isinstance(group_key, tuple):
            for idx, col in enumerate(group_cols):
                row[col] = group_key[idx]
        else:
            row[group_cols[0]] = group_key

        hard_indicator = group_frame["is_hard"] if "is_hard" in group_frame.columns else pd.Series(index=group_frame.index, dtype=float)
        flip_indicator = (
            group_frame[EXPERIMENT4_SAMPLE_LEVEL_FLIP_PRIMARY]
            if EXPERIMENT4_SAMPLE_LEVEL_FLIP_PRIMARY in group_frame.columns
            else pd.Series(index=group_frame.index, dtype=float)
        )
        margin_values = group_frame["margin_shift"] if "margin_shift" in group_frame.columns else pd.Series(index=group_frame.index, dtype=float)

        hard_mask = pd.to_numeric(hard_indicator, errors="coerce").fillna(0).astype(int) == 1
        hard_frame = group_frame.loc[hard_mask].copy()
        margin_series = pd.to_numeric(margin_values.loc[hard_frame.index], errors="coerce").dropna()
        flip_series = pd.to_numeric(flip_indicator.loc[hard_frame.index], errors="coerce").dropna()

        row["samples"] = int(len(group_frame))
        row["hard_samples"] = int(hard_mask.sum())
        row["hard_rate"] = float(hard_mask.mean()) if len(group_frame) else np.nan
        row[PRIMARY_HARD] = float(flip_series.mean()) if not flip_series.empty else np.nan
        row["hard_median_margin_shift_ref_input"] = float(margin_series.median()) if not margin_series.empty else np.nan
        row["hard_mean_margin_shift_ref_input"] = float(margin_series.mean()) if not margin_series.empty else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _fit_ols_with_multi_fe(frame, y_col, x_cols, fe_cols, model_name, requested_x_cols=None, metadata=None, cluster_cols=None):
    """Fit OLS with multiple absorbed fixed effects via linearmodels."""
    requested_x_cols = list(requested_x_cols or x_cols)
    metadata = dict(metadata or {})
    cluster_cols = list(cluster_cols or [])

    def _nan_rows(terms, n, dof, fit_status):
        rows = []
        for term in terms:
            row = {
                "model_name": model_name, "term": term,
                "coefficient": np.nan, "std_error": np.nan,
                "t_value": np.nan, "p_value": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "significant_05": np.nan, "n": n, "dof": dof,
                "r2": np.nan, "x_cols": ",".join(x_cols),
                "fe_cols": ",".join(fe_cols), "fit_status": fit_status,
            }
            row.update(metadata)
            rows.append(row)
        return pd.DataFrame(rows)

    fit_result = fit_linearmodels_absorbingls(
        frame=frame,
        y_col=y_col,
        x_cols=x_cols,
        fe_cols=fe_cols,
        cluster_cols=cluster_cols,
    )
    fit_status = str(fit_result.get("fit_status", ""))
    if fit_status != "ok":
        return _nan_rows(
            ["intercept"] + requested_x_cols,
            int(fit_result.get("n", 0) or 0),
            fit_result.get("dof", np.nan),
            fit_status or "fit_failed",
        )

    n = int(fit_result["n"])
    r2 = float(fit_result["r2"])
    dof = int(fit_result["dof"])
    params = fit_result["params"]
    bse = fit_result["bse"]
    tvalues = fit_result["tvalues"]
    pvalues = fit_result["pvalues"]
    conf_int = fit_result["conf_int"]
    name_to_idx = fit_result["name_to_idx"]
    se_type = str(fit_result["se_type"])

    # ── 组装返回行（intercept + x_cols） ──
    param_names = ["intercept"] + list(x_cols)
    # statsmodels 的 add_constant 默认列名为 "const"
    sm_name_map = {"intercept": "const"}

    rows = []
    for name in param_names:
        sm_name = sm_name_map.get(name, name)
        if sm_name in name_to_idx:
            idx = name_to_idx[sm_name]
            coef = float(params[idx])
            se = float(bse[idx])
            t_val = float(tvalues[idx])
            p_val = float(pvalues[idx])
            ci_lo = float(conf_int[idx, 0])
            ci_hi = float(conf_int[idx, 1])
            sig = int(p_val < 0.05) if np.isfinite(p_val) else np.nan
        else:
            coef = se = t_val = p_val = ci_lo = ci_hi = np.nan
            sig = np.nan
        row = {
            "model_name": model_name, "term": name,
            "coefficient": coef, "std_error": se,
            "t_value": t_val, "p_value": p_val,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "significant_05": sig, "n": int(n), "dof": dof,
            "r2": r2, "x_cols": ",".join(x_cols),
            "fe_cols": ",".join(fe_cols), "se_type": se_type,
            "fit_status": "ok",
        }
        row.update(metadata)
        rows.append(row)

    # ── 请求但未参与拟合的变量 ──
    for term in requested_x_cols:
        if term in set(x_cols):
            continue
        row = {
            "model_name": model_name, "term": term,
            "coefficient": np.nan, "std_error": np.nan,
            "t_value": np.nan, "p_value": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
            "significant_05": np.nan, "n": int(n), "dof": dof,
            "r2": r2, "x_cols": ",".join(x_cols),
            "fe_cols": ",".join(fe_cols), "fit_status": "dropped_before_fit",
        }
        row.update(metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def _term_row(long_frame, term):
    """Pluck a single OLS coefficient row (or a NaN shell) out of a long-format fit result."""
    if long_frame is None or long_frame.empty:
        return {}
    subset = long_frame.loc[long_frame["term"].astype(str) == str(term)]
    if subset.empty:
        return {}
    row = subset.iloc[0].to_dict()
    return row


def _run_nrmse_structure_regression(
    summary,
    structure_col,
    fe_cols,
    cluster_cols,
    model_name,
    restrict=None,
    target_col=PRIMARY_TARGET,
):
    """Fit two OLS models sharing the same FE set and return a one-row summary.

    baseline:  target = α·NRMSE + FE + ε
    full:      target = α·NRMSE + β·S + FE + ε

    The summary row contains α (NRMSE), β (S), their CIs / p-values,
    R² of both models, ΔR² = R²_full - R²_baseline, and cluster-robust SE info.
    """
    working = summary.copy()
    if restrict:
        for col, val in restrict.items():
            working = _filter_frame_by_restriction(working, col, val)
            if working.empty:
                break
    base_needed = [target_col, "nrmse"]
    if structure_col not in working.columns or any(col not in working.columns for col in base_needed):
        return pd.DataFrame([
            {
                "model_name": model_name,
                "restrict": ";".join(f"{k}={v}" for k, v in (restrict or {}).items()) or "all",
                "structure_metric": structure_col,
                "target_metric": target_col,
                "n": 0,
                "fit_status": "missing_columns",
            }
        ])
    available_fe_cols = [col for col in fe_cols if col in working.columns]
    available_cluster_cols = [col for col in cluster_cols if col in working.columns]
    working = working.dropna(subset=base_needed + [structure_col] + available_fe_cols)
    available_fe_cols = [col for col in available_fe_cols if working[col].nunique(dropna=True) > 1]
    available_cluster_cols = [
        col
        for col in available_cluster_cols
        if col in working.columns and working[col].nunique(dropna=True) > 1
    ]
    if working.empty or len(working) < 8:
        return pd.DataFrame([
            {
                "model_name": model_name,
                "restrict": ";".join(f"{k}={v}" for k, v in (restrict or {}).items()) or "all",
                "structure_metric": structure_col,
                "target_metric": target_col,
                "n": int(len(working)),
                "fe_cols": ",".join(available_fe_cols),
                "cluster_cols": ",".join(available_cluster_cols),
                "fit_status": "insufficient_rows",
            }
        ])

    baseline_fit = _fit_ols_with_multi_fe(
        frame=working,
        y_col=target_col,
        x_cols=["nrmse"],
        fe_cols=available_fe_cols,
        model_name=f"{model_name}_baseline",
        requested_x_cols=["nrmse"],
        cluster_cols=available_cluster_cols,
    )
    full_fit = _fit_ols_with_multi_fe(
        frame=working,
        y_col=target_col,
        x_cols=["nrmse", structure_col],
        fe_cols=available_fe_cols,
        model_name=f"{model_name}_full",
        requested_x_cols=["nrmse", structure_col],
        cluster_cols=available_cluster_cols,
    )
    baseline_row = baseline_fit.iloc[0].to_dict() if not baseline_fit.empty else {}
    full_row = full_fit.iloc[0].to_dict() if not full_fit.empty else {}
    baseline_r2 = _as_float_or_nan(baseline_row.get("r2"))
    full_r2 = _as_float_or_nan(full_row.get("r2"))
    delta_r2 = float(full_r2 - baseline_r2) if np.isfinite(full_r2) and np.isfinite(baseline_r2) else np.nan

    alpha_row = _term_row(full_fit, "nrmse")
    beta_row = _term_row(full_fit, structure_col)

    summary_row = {
        "model_name": model_name,
        "restrict": ";".join(f"{k}={v}" for k, v in (restrict or {}).items()) or "all",
        "structure_metric": structure_col,
        "target_metric": target_col,
        "n": int(_as_float_or_nan(full_row.get("n")) or len(working)),
        "dof_full": _as_float_or_nan(full_row.get("dof")),
        "fe_cols": ",".join(available_fe_cols),
        "cluster_cols": ",".join(available_cluster_cols),
        "se_type": str(full_row.get("se_type", "")),
        "alpha_nrmse": _as_float_or_nan(alpha_row.get("coefficient")),
        "alpha_nrmse_se": _as_float_or_nan(alpha_row.get("std_error")),
        "alpha_nrmse_ci_low": _as_float_or_nan(alpha_row.get("ci_low")),
        "alpha_nrmse_ci_high": _as_float_or_nan(alpha_row.get("ci_high")),
        "alpha_nrmse_p": _as_float_or_nan(alpha_row.get("p_value")),
        "beta_structure": _as_float_or_nan(beta_row.get("coefficient")),
        "beta_structure_se": _as_float_or_nan(beta_row.get("std_error")),
        "beta_structure_ci_low": _as_float_or_nan(beta_row.get("ci_low")),
        "beta_structure_ci_high": _as_float_or_nan(beta_row.get("ci_high")),
        "beta_structure_p": _as_float_or_nan(beta_row.get("p_value")),
        "beta_structure_significant_05": int(_as_float_or_nan(beta_row.get("p_value")) < 0.05) if np.isfinite(_as_float_or_nan(beta_row.get("p_value"))) else np.nan,
        "r2_baseline": baseline_r2,
        "r2_full": full_r2,
        "delta_r2": delta_r2,
        "fit_status": str(full_row.get("fit_status", "")),
    }
    return pd.DataFrame([summary_row])


def run_experiment_1(
    summary,
    output_dir,
    label_mode="en",
    main_figures_only=False,
    structure_metrics=None,
    experiment_subdir="experiment_1",
    include_rate_figures=True,
):
    _require_columns(summary, ["dataset", "imputer"], "run_summary")
    exp_dir = os.path.join(output_dir, experiment_subdir)
    _ensure_dir(exp_dir)

    structure_keys = [
        col
        for col in [
            "dataset",
            "missing_mechanism",
            "missing_percent",
            "max_missing_lenth",
            "imputer",
            "iter",
            "seed",
        ]
        if col in summary.columns
    ]
    requested_structure_metrics = list(structure_metrics or STRUCTURE_METRICS)
    available_structure_metrics = [
        metric for metric in requested_structure_metrics if metric in summary.columns
    ]
    if structure_keys and available_structure_metrics:
        exp1_frame = (
            summary[structure_keys + available_structure_metrics]
            .groupby(structure_keys, dropna=False, as_index=False)
            .first()
        )
    else:
        exp1_frame = summary.copy()

    def _format_missing_percent_tag(value):
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return _sanitize_name(value)
        return f"mp_{numeric:g}"

    summary_rows = []
    is_real_missing_truth = _is_real_missing_with_truth_summary(summary)
    metric_label_map = EXPERIMENT1_ZH_METRIC_LABEL_MAP if _is_zh_label_mode(label_mode) else EXPERIMENT1_METRIC_LABEL_MAP
    metrics_to_plot = (
        [PRIMARY_STRUCTURE] if main_figures_only else requested_structure_metrics
    )
    for metric in metrics_to_plot:
        if metric not in exp1_frame.columns:
            continue
        if "missing_mechanism" in exp1_frame.columns:
            grouped_frames = [(None, exp1_frame)]
            if (
                "missing_percent" in exp1_frame.columns
                and not main_figures_only
                and include_rate_figures
            ):
                grouped_frames.extend(
                    (missing_percent, frame.copy())
                    for missing_percent, frame in exp1_frame.groupby("missing_percent", dropna=False)
                )

            for missing_percent, plot_frame in grouped_frames:
                suffix = ""
                if missing_percent is not None:
                    suffix = f"_{_format_missing_percent_tag(missing_percent)}"
                show_axis_labels_every_panel = True
                if is_real_missing_truth:
                    save_mechanism_dataset_bar_grid(
                        plot_frame,
                        metric,
                        os.path.join(exp_dir, f"{metric}_bar_mechanism_dataset_grid{suffix}.png"),
                        label_mode=label_mode,
                        metric_label_map=metric_label_map,
                        force_zero_ymin=True,
                        imputer_order=PAPER_IMPUTER_ORDER,
                    )
                else:
                    save_mechanism_dataset_boxplot_grid(
                        plot_frame,
                        metric,
                        os.path.join(exp_dir, f"{metric}_boxplot_mechanism_dataset_grid{suffix}.png"),
                        sort_by="median",
                        show_counts=False,
                        show_x_tick_labels=False,
                        x_tick_rotation=90,
                        x_tick_fontsize=8,
                        label_mode=label_mode,
                        metric_label_map=metric_label_map,
                        force_zero_ymin=True,
                        imputer_order=PAPER_IMPUTER_ORDER,
                        show_axis_labels_every_panel=show_axis_labels_every_panel,
                    )

    for dataset, dataset_frame in exp1_frame.groupby("dataset"):
        for metric in requested_structure_metrics:
            if metric not in dataset_frame.columns:
                continue
            for imputer in _ordered_present_names(dataset_frame["imputer"].astype(str).unique(), PAPER_IMPUTER_ORDER):
                values = dataset_frame.loc[dataset_frame["imputer"] == imputer, metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                if "missing_percent" in dataset_frame.columns:
                    imputer_frame = dataset_frame.loc[dataset_frame["imputer"] == imputer]
                    if "missing_mechanism" in imputer_frame.columns:
                        for mechanism in _ordered_present_names(
                            imputer_frame["missing_mechanism"].astype(str).unique(),
                            PAPER_MECHANISM_ORDER,
                        ):
                            mechanism_frame = imputer_frame.loc[imputer_frame["missing_mechanism"].astype(str) == mechanism]
                            for missing_percent, percent_frame in mechanism_frame.groupby("missing_percent", dropna=False):
                                percent_values = percent_frame[metric].dropna().to_numpy(dtype=float)
                                if len(percent_values) == 0:
                                    continue
                                summary_rows.append(
                                    {
                                        "dataset": dataset,
                                        "imputer": imputer,
                                        "metric": metric,
                                        "median": float(np.median(percent_values)),
                                        "mean": float(np.mean(percent_values)),
                                        "std": float(np.std(percent_values)),
                                        "runs": int(len(percent_values)),
                                        "missing_mechanism": mechanism,
                                        "missing_percent": missing_percent,
                                    }
                                )
                    else:
                        for missing_percent, percent_frame in imputer_frame.groupby("missing_percent", dropna=False):
                            percent_values = percent_frame[metric].dropna().to_numpy(dtype=float)
                            if len(percent_values) == 0:
                                continue
                            summary_rows.append(
                                {
                                    "dataset": dataset,
                                    "imputer": imputer,
                                    "metric": metric,
                                    "median": float(np.median(percent_values)),
                                    "mean": float(np.mean(percent_values)),
                                    "std": float(np.std(percent_values)),
                                    "runs": int(len(percent_values)),
                                    "missing_percent": missing_percent,
                                }
                            )
                else:
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "imputer": imputer,
                            "metric": metric,
                            "median": float(np.median(values)),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "runs": int(len(values)),
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(os.path.join(exp_dir, "structure_damage_summary.csv"), index=False)


def run_experiment_2(
    summary,
    output_dir,
    label_mode="en",
    main_figures_only=False,
    experiment_subdir="experiment_2",
):
    _require_columns(summary, ["nrmse", "harm_f1", "dataset", "classifier", "imputer", "missing_mechanism"], "run_summary")
    exp_dir = os.path.join(output_dir, experiment_subdir)
    _ensure_dir(exp_dir)
    is_real_missing_truth = _is_real_missing_with_truth_summary(summary)

    target_col = "harm_f1"
    filename = "nrmse_vs_harm_f1"
    subset_cols = ["nrmse", target_col, "dataset", "classifier", "imputer", "missing_mechanism"]
    if "missing_percent" in summary.columns:
        subset_cols.append("missing_percent")
    if "seed" in summary.columns:
        subset_cols.append("seed")
    if "iter" in summary.columns:
        subset_cols.append("iter")
    if "source_file" in summary.columns:
        subset_cols.append("source_file")
    if PRIMARY_STRUCTURE in summary.columns:
        subset_cols.append(PRIMARY_STRUCTURE)
    subset = summary[subset_cols].dropna(subset=["nrmse", target_col, "dataset", "classifier", "imputer", "missing_mechanism"])

    if "seed" in subset.columns:
        seed_numeric = pd.to_numeric(subset["seed"], errors="coerce")
        subset = subset.loc[seed_numeric.isin(EXPERIMENT2_SEEDS)].copy()
        subset["seed"] = seed_numeric.loc[subset.index].astype("Int64")

    dedupe_cols = ["dataset", "missing_mechanism", "classifier", "imputer"]
    if "missing_percent" in subset.columns:
        dedupe_cols.append("missing_percent")
    if "seed" in subset.columns:
        dedupe_cols.append("seed")
    if "iter" in subset.columns:
        subset["_experiment2_iter_sort"] = pd.to_numeric(subset["iter"], errors="coerce")
    else:
        subset["_experiment2_iter_sort"] = np.nan
    if "source_file" not in subset.columns:
        subset["source_file"] = ""
    subset = (
        subset.sort_values(
            by=[*dedupe_cols, "_experiment2_iter_sort", "source_file"],
            na_position="last",
        )
        .drop_duplicates(subset=dedupe_cols, keep="first")
        .drop(columns=["_experiment2_iter_sort"])
        .copy()
    )

    group_cols = ["dataset", "missing_mechanism", "classifier", "imputer"]
    if "missing_percent" in subset.columns:
        group_cols.append("missing_percent")
    aggregations = {
        "nrmse": ("nrmse", "mean"),
        target_col: (target_col, "mean"),
        "runs": ("imputer", "size"),
    }
    if PRIMARY_STRUCTURE in subset.columns:
        aggregations[PRIMARY_STRUCTURE] = (PRIMARY_STRUCTURE, "mean")
    plot_subset = subset.groupby(group_cols, as_index=False).agg(**aggregations)
    overall_dataset_count = int(plot_subset["dataset"].astype(str).nunique()) if not plot_subset.empty else 0
    overall_mechanism_count = int(plot_subset["missing_mechanism"].astype(str).nunique()) if not plot_subset.empty else 0
    overall_dataset_cols = max(1, min(4, overall_dataset_count))
    overall_dataset_chunks = max(1, (overall_dataset_count + 3) // 4)
    overall_rows = max(overall_mechanism_count, 1) * overall_dataset_chunks
    if is_real_missing_truth:
        panel_size = 3.05
        legend_rows_estimate = 5.0
        overall_figsize = (
            max(overall_dataset_cols * panel_size + 1.25, 8.0),
            max(overall_rows * panel_size + 1.25 + 0.25 * legend_rows_estimate, 5.1),
        )
    else:
        paper_width = PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH
        overall_figsize = (
            max(paper_width / 4.0 * overall_dataset_cols, paper_width),
            max(2.05 * overall_rows + 4.45, 16.75),
        )

    subset.to_csv(os.path.join(exp_dir, f"{filename}_points_raw.csv"), index=False)
    plot_subset.to_csv(os.path.join(exp_dir, f"{filename}_points.csv"), index=False)
    by_missing_percent_dir = os.path.join(exp_dir, "by_missing_percent")

    grouped_plot_frames = [(None, plot_subset)]
    if "missing_percent" in plot_subset.columns and not main_figures_only:
        os.makedirs(by_missing_percent_dir, exist_ok=True)
        grouped_plot_frames.extend(
            (missing_percent, frame.copy())
            for missing_percent, frame in plot_subset.groupby("missing_percent", dropna=False)
        )

    def _build_summary_rows(group_frame, missing_percent_value=None):
        summary_rows = []
        summary_groups = [("overall", group_frame)]
        summary_groups.extend((f"mechanism::{mechanism}", frame) for mechanism, frame in group_frame.groupby("missing_mechanism"))
        summary_groups.extend((f"dataset::{dataset}", frame) for dataset, frame in group_frame.groupby("dataset"))
        summary_groups.extend(
            (
                f"dataset_mechanism::{dataset}::{mechanism}",
                frame,
            )
            for (dataset, mechanism), frame in group_frame.groupby(["dataset", "missing_mechanism"])
        )
        for group_name, nested_frame in summary_groups:
            dataset_value = "overall"
            mechanism_value = "overall"
            if group_name.startswith("mechanism::"):
                mechanism_value = group_name.split("::", 1)[1]
            elif group_name.startswith("dataset::"):
                dataset_value = group_name.split("::", 1)[1]
            elif group_name.startswith("dataset_mechanism::"):
                _, dataset_value, mechanism_value = group_name.split("::", 2)
            row = {
                "x": "nrmse",
                "y": target_col,
                "dataset": str(dataset_value),
                "missing_mechanism": str(mechanism_value),
                "pearson_corr": (
                    float(nested_frame["nrmse"].corr(nested_frame[target_col]))
                    if len(nested_frame) >= 2
                    else np.nan
                ),
                "n": int(len(nested_frame)),
            }
            if missing_percent_value is not None:
                row["missing_percent"] = _as_float_or_nan(missing_percent_value)
            summary_rows.append(row)
        return summary_rows

    summary_rows = []
    for missing_percent_value, frame in grouped_plot_frames:
        target_dir = exp_dir
        if missing_percent_value is not None:
            target_dir = os.path.join(by_missing_percent_dir, _format_missing_percent_tag(missing_percent_value))
            os.makedirs(target_dir, exist_ok=True)
        frame.to_csv(os.path.join(target_dir, f"{filename}_points.csv"), index=False)
        summary_frame = pd.DataFrame(_build_summary_rows(frame, missing_percent_value))
        summary_frame.to_csv(os.path.join(target_dir, f"{filename}_summary.csv"), index=False)
        save_nrmse_delta_scatter(
            frame,
            os.path.join(target_dir, f"{filename}.png"),
            fixed_figsize=overall_figsize,
            y_col=target_col,
            y_label="F1损失" if _is_zh_label_mode(label_mode) else "Harm F1",
            label_mode=label_mode,
            imputer_order=PAPER_IMPUTER_ORDER,
            classifier_order=PAPER_CLASSIFIER_ORDER,
            mechanism_title_style="abbr",
            show_axis_labels_every_panel=True,
            compact_canvas=is_real_missing_truth,
            square_panels=is_real_missing_truth,
        )
        summary_rows.extend(_build_summary_rows(frame, missing_percent_value))
    pd.DataFrame(summary_rows).to_csv(os.path.join(exp_dir, f"{filename}_summary.csv"), index=False)
    return plot_subset


def _build_experiment_3_model_configs(base_fe_cols, structure_metrics, missing_percent_scope=None, mechanism_values=None):
    pooled_restrict = {}
    if missing_percent_scope is not None:
        pooled_restrict["missing_percent"] = float(missing_percent_scope)

    mechanism_values = list(mechanism_values or [])
    model_configs = []
    for metric in structure_metrics:
        metric_role = "main" if metric == PRIMARY_STRUCTURE else "robustness_structure_metric"
        metric_suffix = "corr_frob" if metric == PRIMARY_STRUCTURE else metric
        pooled_prefix = "main" if metric == PRIMARY_STRUCTURE else "robust"

        model_configs.append({
            "model_name": f"{pooled_prefix}_all_mechanisms_{metric_suffix}",
            "structure_col": metric,
            "restrict": pooled_restrict or None,
            "fe_cols": base_fe_cols,
            "role": metric_role,
            "mechanism_scope": "All",
        })

        for mech in mechanism_values:
            restrict = dict(pooled_restrict)
            restrict["missing_mechanism"] = mech
            model_configs.append({
                "model_name": f"{pooled_prefix}_{mech}_{metric_suffix}",
                "structure_col": metric,
                "restrict": restrict,
                # Drop missing_mechanism from FE because it is constant inside each
                # mechanism-restricted slice and would only collinearize the design.
                "fe_cols": [col for col in base_fe_cols if col != "missing_mechanism"],
                "role": metric_role,
                "mechanism_scope": mech,
            })
    return model_configs


def _build_experiment_3_appendix_configs(base_fe_cols, structure_metrics, missing_percent_scope=None):
    pooled_restrict = {}
    if missing_percent_scope is not None:
        pooled_restrict["missing_percent"] = float(missing_percent_scope)

    appendix_configs = [
        {
            "model_name": "appendix_pooled_dataset_fe_corr_frob",
            "structure_col": PRIMARY_STRUCTURE,
            "restrict": pooled_restrict or None,
            "fe_cols": base_fe_cols,
            "role": "appendix_pooled_dataset_fe",
            "mechanism_scope": "All",
        }
    ]
    for alt in structure_metrics:
        if alt == PRIMARY_STRUCTURE:
            continue
        appendix_configs.append({
            "model_name": f"appendix_pooled_dataset_fe_{alt}",
            "structure_col": alt,
            "restrict": pooled_restrict or None,
            "fe_cols": base_fe_cols,
            "role": "appendix_pooled_dataset_fe",
            "mechanism_scope": "All",
        })
    return appendix_configs


def _run_experiment_3_scope(
    summary,
    exp_dir,
    missing_percent_scope=None,
    label_mode="en",
    structure_metrics=None,
):
    appendix_dir = os.path.join(exp_dir, "appendix")
    _ensure_dir(exp_dir)
    os.makedirs(appendix_dir, exist_ok=True)

    # Keep classifier fixed effects for downstream-model differences. Do not
    # absorb imputer effects here: structure metrics are primarily imputer-level
    # signals, so imputer FE can fully absorb the coefficient of interest.
    dataset_fe_cols = [
        col
        for col in ["classifier", "missing_mechanism", "missing_percent"]
        if col in summary.columns
    ]
    dataset_cluster_cols = [col for col in ["classifier"] if col in summary.columns]
    pooled_fe_cols = [
        col
        for col in ["dataset", "classifier", "missing_mechanism", "missing_percent"]
        if col in summary.columns
    ]
    pooled_cluster_cols = [col for col in ["dataset", "classifier"] if col in summary.columns]

    # The main claim in Experiment 3: after controlling for NRMSE via a regression
    # covariate, the structure metric S still carries significant explanatory power
    # for harm_f1. We fit a paired (baseline, full) model for each row and summarize
    # β_S along with ΔR² so the table stands on its own without any bucketing.
    requested_metrics = list(structure_metrics or STRUCTURE_METRICS)
    structure_metrics = [PRIMARY_STRUCTURE]
    for metric in requested_metrics:
        if metric in summary.columns and metric not in structure_metrics:
            structure_metrics.append(metric)
    mechanism_values = _ordered_present_names(
        summary["missing_mechanism"].dropna().astype(str).unique() if "missing_mechanism" in summary.columns else [],
        PAPER_MECHANISM_ORDER,
    )

    dataset_values = _ordered_present_names(
        summary["dataset"].dropna().astype(str).unique() if "dataset" in summary.columns else [],
        [],
    )
    table_rows = []
    for dataset in dataset_values:
        dataset_summary = _filter_frame_by_restriction(summary, "dataset", dataset)
        if dataset_summary.empty:
            continue
        dataset_prefix = _sanitize_name(dataset)
        for cfg in _build_experiment_3_model_configs(
            base_fe_cols=dataset_fe_cols,
            structure_metrics=structure_metrics,
            missing_percent_scope=missing_percent_scope,
            mechanism_values=mechanism_values,
        ):
            row_frame = _run_nrmse_structure_regression(
                dataset_summary,
                structure_col=cfg["structure_col"],
                fe_cols=cfg["fe_cols"],
                cluster_cols=dataset_cluster_cols,
                model_name=f"{dataset_prefix}_{cfg['model_name']}",
                restrict=cfg["restrict"],
            )
            if not row_frame.empty:
                row_frame = row_frame.copy()
                row_frame["role"] = cfg["role"]
                row_frame["dataset_scope"] = dataset
                row_frame["mechanism_scope"] = cfg.get("mechanism_scope", "All")
                row_frame["missing_percent_scope"] = (
                    float(missing_percent_scope) if missing_percent_scope is not None else np.nan
                )
                table_rows.append(row_frame)

    table3a = pd.concat(table_rows, ignore_index=True) if table_rows else pd.DataFrame()
    table3a.to_csv(os.path.join(exp_dir, "table3a_nrmse_structure_regression.csv"), index=False)

    appendix_rows = []
    for cfg in _build_experiment_3_appendix_configs(
        base_fe_cols=pooled_fe_cols,
        structure_metrics=structure_metrics,
        missing_percent_scope=missing_percent_scope,
    ):
        row_frame = _run_nrmse_structure_regression(
            summary,
            structure_col=cfg["structure_col"],
            fe_cols=cfg["fe_cols"],
            cluster_cols=pooled_cluster_cols,
            model_name=cfg["model_name"],
            restrict=cfg["restrict"],
        )
        if not row_frame.empty:
            row_frame = row_frame.copy()
            row_frame["role"] = cfg["role"]
            row_frame["dataset_scope"] = "Pooled"
            row_frame["mechanism_scope"] = cfg.get("mechanism_scope", "All")
            row_frame["missing_percent_scope"] = (
                float(missing_percent_scope) if missing_percent_scope is not None else np.nan
            )
            appendix_rows.append(row_frame)
    if appendix_rows:
        appendix_table = pd.concat(appendix_rows, ignore_index=True)
        appendix_table.to_csv(
            os.path.join(appendix_dir, "table3a_pooled_dataset_fe_regression.csv"),
            index=False,
        )


def _prepare_experiment_3_tiered_scope(summary, synthetic_missing_percent):
    working = summary.copy()
    if "status" in working.columns:
        status = working["status"].fillna("").astype(str).str.lower()
        working = working.loc[status.isin(["ok", "success"])].copy()

    if "dataset_category" not in working.columns:
        return _filter_frame_by_restriction(
            working, "missing_percent", synthetic_missing_percent
        )

    real_mask = working["dataset_category"].astype(str).eq(
        "real_missing_with_truth"
    )
    synthetic = working.loc[~real_mask].copy()
    real = working.loc[real_mask].copy()
    synthetic = _filter_frame_by_restriction(
        synthetic, "missing_percent", synthetic_missing_percent
    )
    return pd.concat([synthetic, real], ignore_index=True, sort=False)


def _experiment_3_within_standardised_effect(
    frame,
    structure_col,
    beta,
    fe_cols,
):
    required = [PRIMARY_TARGET, "nrmse", structure_col, *fe_cols]
    working = frame[required].dropna().copy()
    if len(working) < 3 or not np.isfinite(beta):
        return np.nan, int(len(working))

    values = working[[structure_col, PRIMARY_TARGET]].astype(float).to_numpy()
    active_fe = [col for col in fe_cols if working[col].nunique(dropna=True) > 1]
    if active_fe:
        import pyhdfe

        residuals = pyhdfe.create(
            working[active_fe].astype(str).to_numpy(),
            drop_singletons=False,
        ).residualize(values)
    else:
        residuals = values - values.mean(axis=0, keepdims=True)

    structure_sd = float(np.std(residuals[:, 0], ddof=1))
    target_sd = float(np.std(residuals[:, 1], ddof=1))
    if (
        not np.isfinite(structure_sd)
        or not np.isfinite(target_sd)
        or target_sd == 0.0
    ):
        return np.nan, int(len(working))
    return float(beta) * structure_sd / target_sd, int(len(working))


def _run_experiment_3_tiered(
    summary,
    output_dir,
    synthetic_missing_percent,
    structure_metrics,
):
    exp_dir = os.path.join(output_dir, "experiment_3")
    _ensure_dir(exp_dir)
    synthetic_missing_percent = (
        0.3
        if synthetic_missing_percent is None
        else float(synthetic_missing_percent)
    )
    scoped = _prepare_experiment_3_tiered_scope(
        summary, synthetic_missing_percent
    )
    if scoped.empty:
        raise ValueError("Experiment 3 tiered scope contains no rows")

    requested_metrics = [
        metric
        for metric in list(structure_metrics or STRUCTURE_METRICS)
        if metric in scoped.columns
    ]
    if not requested_metrics:
        raise ValueError("Experiment 3 tiered scope has no structure metrics")

    pooled_dir = os.path.join(exp_dir, "pooled")
    mechanism_dir = os.path.join(exp_dir, "by_mechanism")
    dataset_dir = os.path.join(exp_dir, "by_dataset")
    for path in [pooled_dir, mechanism_dir, dataset_dir]:
        os.makedirs(path, exist_ok=True)

    def _fit_row(frame, metric, fe_cols, cluster_cols, model_name, metadata):
        result = _run_nrmse_structure_regression(
            frame,
            structure_col=metric,
            fe_cols=fe_cols,
            cluster_cols=cluster_cols,
            model_name=model_name,
        )
        if result.empty:
            return result
        result = result.copy()
        beta = _as_float_or_nan(result.iloc[0].get("beta_structure"))
        effect, effect_n = _experiment_3_within_standardised_effect(
            frame,
            structure_col=metric,
            beta=beta,
            fe_cols=[col for col in fe_cols if col in frame.columns],
        )
        regression_n_value = _as_float_or_nan(result.iloc[0].get("n"))
        regression_n = (
            int(round(regression_n_value))
            if np.isfinite(regression_n_value)
            else -1
        )
        if (
            str(result.iloc[0].get("fit_status", "")) == "ok"
            and regression_n != effect_n
        ):
            raise ValueError(
                f"Experiment 3 effect-size sample mismatch for {model_name}: "
                f"regression n={regression_n}, effect n={effect_n}"
            )
        result["beta_within_std_effect"] = effect
        result["effect_n"] = effect_n
        result["synthetic_missing_percent"] = synthetic_missing_percent
        result["real_missing_scope"] = "native_pattern"
        for key, value in metadata.items():
            result[key] = value
        return result

    pooled_rows = []
    pooled_fe = [
        col
        for col in ["dataset", "classifier", "missing_mechanism"]
        if col in scoped.columns
    ]
    pooled_clusters = [
        col for col in ["dataset", "classifier"] if col in scoped.columns
    ]
    for metric in requested_metrics:
        pooled_rows.append(
            _fit_row(
                scoped,
                metric,
                pooled_fe,
                pooled_clusters,
                f"pooled_all_datasets_{_sanitize_name(metric)}",
                {
                    "analysis_level": "pooled",
                    "dataset_scope": "Pooled",
                },
            )
        )
    pooled_table = pd.concat(pooled_rows, ignore_index=True)
    pooled_path = os.path.join(pooled_dir, "table3a_pooled_all_datasets.csv")
    pooled_table.to_csv(pooled_path, index=False)

    if "dataset_category" in scoped.columns:
        synthetic = scoped.loc[
            ~scoped["dataset_category"].astype(str).eq(
                "real_missing_with_truth"
            )
        ].copy()
    else:
        synthetic = scoped.copy()
    mechanism_rows = []
    mechanism_values = _ordered_present_names(
        synthetic["missing_mechanism"].dropna().astype(str).unique(),
        PAPER_MECHANISM_ORDER,
    )
    for mechanism in mechanism_values:
        mechanism_frame = synthetic.loc[
            synthetic["missing_mechanism"].astype(str).eq(mechanism)
        ].copy()
        fe_cols = [
            col for col in ["dataset", "classifier"] if col in mechanism_frame
        ]
        cluster_cols = [
            col for col in ["dataset", "classifier"] if col in mechanism_frame
        ]
        for metric in requested_metrics:
            mechanism_rows.append(
                _fit_row(
                    mechanism_frame,
                    metric,
                    fe_cols,
                    cluster_cols,
                    f"pooled_{_sanitize_name(mechanism)}_{_sanitize_name(metric)}",
                    {
                        "analysis_level": "mechanism",
                        "dataset_scope": "Pooled",
                        "missing_mechanism": mechanism,
                    },
                )
            )
    mechanism_table = pd.concat(mechanism_rows, ignore_index=True)
    mechanism_path = os.path.join(
        mechanism_dir, "table3b_pooled_by_mechanism.csv"
    )
    mechanism_table.to_csv(mechanism_path, index=False)

    dataset_rows = []
    for dataset in _ordered_present_names(scoped["dataset"].unique(), []):
        dataset_frame = scoped.loc[
            scoped["dataset"].astype(str).eq(dataset)
        ].copy()
        category = (
            str(dataset_frame["dataset_category"].iloc[0])
            if "dataset_category" in dataset_frame.columns
            else "unknown"
        )
        is_real = category == "real_missing_with_truth"
        fe_cols = ["classifier"]
        if not is_real and "missing_mechanism" in dataset_frame.columns:
            fe_cols.append("missing_mechanism")
        fe_cols = [col for col in fe_cols if col in dataset_frame.columns]
        cluster_cols = [
            col for col in ["classifier"] if col in dataset_frame.columns
        ]
        for metric in requested_metrics:
            dataset_rows.append(
                _fit_row(
                    dataset_frame,
                    metric,
                    fe_cols,
                    cluster_cols,
                    f"{_sanitize_name(dataset)}_{_sanitize_name(metric)}",
                    {
                        "analysis_level": "dataset",
                        "dataset": dataset,
                        "dataset_category": category,
                    },
                )
            )
    dataset_table = pd.concat(dataset_rows, ignore_index=True)
    dataset_path = os.path.join(dataset_dir, "table3c_by_dataset.csv")
    dataset_table.to_csv(dataset_path, index=False)

    combined = pd.concat(
        [pooled_table, mechanism_table, dataset_table],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(os.path.join(exp_dir, "table3_all_levels.csv"), index=False)
    count_cols = [
        col
        for col in ["dataset_category", "dataset", "missing_mechanism"]
        if col in scoped.columns
    ]
    scoped.groupby(count_cols, dropna=False).size().rename("rows").reset_index().to_csv(
        os.path.join(exp_dir, "analysis_scope_counts.csv"),
        index=False,
    )


def run_experiment_3(
    summary,
    output_dir,
    restrict_missing_percent=None,
    label_mode="en",
    main_figures_only=False,
    structure_metrics=None,
    analysis_layout="legacy",
):
    _require_columns(
        summary,
        ["imputer", "dataset", "classifier", "missing_mechanism", "missing_percent", "nrmse", PRIMARY_STRUCTURE, PRIMARY_TARGET],
        "run_summary",
    )
    exp_dir = os.path.join(output_dir, "experiment_3")

    if analysis_layout == "tiered":
        _run_experiment_3_tiered(
            summary,
            output_dir,
            synthetic_missing_percent=restrict_missing_percent,
            structure_metrics=structure_metrics,
        )
        return
    if analysis_layout != "legacy":
        raise ValueError(f"Unknown experiment 3 analysis layout: {analysis_layout}")

    if restrict_missing_percent is not None:
        scoped_summary = _filter_frame_by_restriction(summary, "missing_percent", restrict_missing_percent)
        if scoped_summary.empty:
            available = ", ".join(str(value) for value in _sorted_missing_percent_values(summary))
            raise ValueError(
                "No rows found for experiment 3 missing_percent="
                f"{restrict_missing_percent}. Available missing_percent values: {available or 'none'}"
            )
        _run_experiment_3_scope(
            summary=scoped_summary,
            exp_dir=exp_dir,
            missing_percent_scope=float(restrict_missing_percent),
            label_mode=label_mode,
            structure_metrics=structure_metrics,
        )
        return

    _run_experiment_3_scope(
        summary=summary,
        exp_dir=exp_dir,
        missing_percent_scope=None,
        label_mode=label_mode,
        structure_metrics=structure_metrics,
    )

    if main_figures_only:
        return

    available_missing_percents = _sorted_missing_percent_values(summary)
    if not available_missing_percents:
        return

    by_missing_percent_dir = os.path.join(exp_dir, "by_missing_percent")
    os.makedirs(by_missing_percent_dir, exist_ok=True)
    manifest_rows = []
    for missing_percent in available_missing_percents:
        scope_summary = _filter_frame_by_restriction(summary, "missing_percent", missing_percent)
        if scope_summary.empty:
            continue
        scope_dir = os.path.join(by_missing_percent_dir, _format_missing_percent_tag(missing_percent))
        _run_experiment_3_scope(
            summary=scope_summary,
            exp_dir=scope_dir,
            missing_percent_scope=float(missing_percent),
            label_mode=label_mode,
            structure_metrics=structure_metrics,
        )
        manifest_rows.append(
            {
                "missing_percent": float(missing_percent),
                "output_dir": scope_dir,
                "table_path": os.path.join(scope_dir, "table3a_nrmse_structure_regression.csv"),
                "appendix_table_path": os.path.join(
                    scope_dir,
                    "appendix",
                    "table3a_pooled_dataset_fe_regression.csv",
                ),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(by_missing_percent_dir, "manifest.csv"),
        index=False,
    )


def _experiment_4_pearson_pair(x, y):
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    n_points = int(len(pair))
    if (
        n_points < 2
        or pair["x"].nunique(dropna=True) < 2
        or pair["y"].nunique(dropna=True) < 2
    ):
        return np.nan, n_points
    return float(pair["x"].corr(pair["y"])), n_points


def _format_experiment_4_aggregate_latex(summary, scope_metadata):
    rows = {row["path_key"]: row for row in summary.to_dict("records")}
    panel_count = scope_metadata["panel_count"]
    synthetic_panel_count = scope_metadata["synthetic_panel_count"]
    real_panel_count = scope_metadata["real_panel_count"]
    min_imputers = scope_metadata["min_imputers"]
    median_imputers = scope_metadata["median_imputers"]
    max_imputers = scope_metadata["max_imputers"]
    latex_label = scope_metadata.get("latex_label", "tab:exp4-aggregate")

    def stats(path_key):
        row = rows[path_key]
        return (
            f"{row['median_r']:.3f} & "
            f"[{row['q1_r']:.2f},{row['q3_r']:.2f}] & "
            f"{row['pct_gt_03']:.1f} & {row['pct_gt_05']:.1f}"
        )

    def probability(path_key):
        return f"{rows[path_key]['probability_pct']:.1f}"

    if synthetic_panel_count and real_panel_count:
        scope_text = (
            rf"for synthetic datasets at missing rate $0.3$ and real-missing-with-truth "
            rf"datasets at their native missingness, aggregated over ${panel_count}$ panels "
            rf"(${synthetic_panel_count}$ synthetic and ${real_panel_count}$ real; dataset "
            r"$\times$ classifier $\times$ mechanism); within each panel $r$ is computed "
            rf"across the available imputers (${min_imputers}$--${max_imputers}$, median "
            rf"${median_imputers:g}$). "
        )
    elif synthetic_panel_count:
        scope_text = (
            rf"at missing rate $0.3$, aggregated over ${panel_count}$ panels (dataset "
            r"$\times$ classifier $\times$ mechanism); within each panel $r$ is computed "
            rf"across the available imputers (${min_imputers}$--${max_imputers}$, median "
            rf"${median_imputers:g}$). "
        )
    else:
        scope_text = (
            rf"for real-missing-with-truth datasets at their native missingness, aggregated "
            rf"over ${panel_count}$ panels (dataset $\times$ classifier); within each panel "
            rf"$r$ is computed across the available imputers (${min_imputers}$--"
            rf"${max_imputers}$, median ${median_imputers:g}$). "
        )

    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\caption{Pearson $r$ for the $M_1\!\to\!M_2\!\to\!Y$ chain "
                + scope_text
                + r"$M_2 = \mathrm{HSIR}$ (hard-sample inconsistency rate), "
                r"$Y = \mathrm{Harm\:F1}$. Two choices of $M_1$ are shown: "
                r"$\Delta\mathrm{Corr}$ (Panel A) and $\Delta\mathrm{FR}$ (Panel B).}"
            ),
            rf"\label{{{latex_label}}}",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Path & Med.\ $r$ & $[Q_1,Q_3]$ & $\%\!>\!0.3$ & $\%\!>\!0.5$ \\",
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\textit{Shared decision-layer link}}\\",
            r"$\mathrm{HSIR}\!\to\!\mathrm{Harm\:F1}$\ ($M_2\!\to\!Y$) & "
            + stats("shared_m2_to_y") + r" \\",
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\textit{Panel A.}\ $M_1 = \Delta\mathrm{Corr}$}\\",
            r"$M_1\!\to\!\mathrm{HSIR}$\ ($M_1\!\to\!M_2$) & "
            + stats("corr_m1_to_m2") + r" \\",
            r"$M_1\!\to\!\mathrm{Harm\:F1}$\ ($M_1\!\to\!Y$) & "
            + stats("corr_m1_to_y") + r" \\",
            r"\textbf{$\Pr(\,r_{M_2\to Y} > r_{M_1\to Y}\,)$} & "
            + rf"\multicolumn{{4}}{{c}}{{\textbf{{${probability('corr_m2_beats_m1')}\%$}}}} \\",
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\textit{Panel B.}\ $M_1 = \Delta\mathrm{FR}$}\\",
            r"$M_1\!\to\!\mathrm{HSIR}$\ ($M_1\!\to\!M_2$) & "
            + stats("fr_m1_to_m2") + r" \\",
            r"$M_1\!\to\!\mathrm{Harm\:F1}$\ ($M_1\!\to\!Y$) & "
            + stats("fr_m1_to_y") + r" \\",
            r"\textbf{$\Pr(\,r_{M_2\to Y} > r_{M_1\to Y}\,)$} & "
            + rf"\multicolumn{{4}}{{c}}{{\textbf{{${probability('fr_m2_beats_m1')}\%$}}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _generate_experiment_4_aggregate_table(
    long_table,
    exp_dir,
    file_stem="table4",
    latex_label="tab:exp4-aggregate",
):
    """Summarize synthetic 0.3 and real native-missingness panels."""
    has_categories = "dataset_category" in long_table.columns
    if has_categories:
        synthetic_mask = long_table["dataset_category"].eq("no_real_missing") & np.isclose(
            long_table["missing_percent"], 0.3
        )
        real_mask = long_table["dataset_category"].eq("real_missing_with_truth")
        analysis_mask = synthetic_mask | real_mask
        panel_columns = ["dataset_category", "dataset", "classifier", "missing_mechanism"]
    else:
        synthetic_mask = np.isclose(long_table["missing_percent"], 0.3)
        real_mask = pd.Series(False, index=long_table.index)
        analysis_mask = synthetic_mask
        panel_columns = ["dataset", "classifier", "missing_mechanism"]

    scope = long_table.loc[
        analysis_mask
        & long_table["m2_metric"].eq(PRIMARY_HARD)
        & long_table["m1_metric"].isin(["m1_corr_frob", "m1_fisher_delta"])
    ].copy()
    if scope.empty:
        return None

    wide = scope.pivot(
        index=[*panel_columns, "m1_metric"],
        columns="relation",
        values="pearson_r",
    ).reset_index()
    expected_relations = {"m1_to_m2", "m2_to_harm_f1", "m1_to_harm_f1"}
    if not expected_relations.issubset(wide.columns):
        raise ValueError("Experiment 4 aggregate table is missing a chain correlation")
    panel_count = int(wide[panel_columns].drop_duplicates().shape[0])
    panel_points = scope.loc[
        scope["m1_metric"].eq("m1_corr_frob") & scope["relation"].eq("m1_to_m2"),
        [*panel_columns, "missing_percent", "n_points"],
    ].drop_duplicates(panel_columns)
    if len(panel_points) != panel_count:
        raise ValueError("Experiment 4 aggregate table has inconsistent panel point counts")

    if has_categories:
        synthetic_panel_count = int(
            panel_points["dataset_category"].eq("no_real_missing").sum()
        )
        real_panel_count = int(
            panel_points["dataset_category"].eq("real_missing_with_truth").sum()
        )
    else:
        synthetic_panel_count = panel_count
        real_panel_count = 0
    scope_metadata = {
        "panel_count": panel_count,
        "synthetic_panel_count": synthetic_panel_count,
        "real_panel_count": real_panel_count,
        "min_imputers": int(panel_points["n_points"].min()),
        "median_imputers": float(panel_points["n_points"].median()),
        "max_imputers": int(panel_points["n_points"].max()),
        "latex_label": str(latex_label),
    }

    def correlation_row(path_key, metric, relation):
        values = wide.loc[wide["m1_metric"].eq(metric), relation].dropna()
        return {
            "path_key": path_key,
            "metric": metric,
            "relation": relation,
            "panel_count": int(len(values)),
            "median_r": float(values.median()),
            "q1_r": float(values.quantile(0.25)),
            "q3_r": float(values.quantile(0.75)),
            "pct_gt_03": float((values > 0.3).mean() * 100),
            "pct_gt_05": float((values > 0.5).mean() * 100),
            "probability_pct": np.nan,
        }

    summary_rows = [
        correlation_row("shared_m2_to_y", "m1_corr_frob", "m2_to_harm_f1"),
        correlation_row("corr_m1_to_m2", "m1_corr_frob", "m1_to_m2"),
        correlation_row("corr_m1_to_y", "m1_corr_frob", "m1_to_harm_f1"),
        correlation_row("fr_m1_to_m2", "m1_fisher_delta", "m1_to_m2"),
        correlation_row("fr_m1_to_y", "m1_fisher_delta", "m1_to_harm_f1"),
    ]
    for path_key, metric in [("corr_m2_beats_m1", "m1_corr_frob"), ("fr_m2_beats_m1", "m1_fisher_delta")]:
        metric_rows = wide.loc[wide["m1_metric"].eq(metric)]
        summary_rows.append(
            {
                "path_key": path_key,
                "metric": metric,
                "relation": "m2_to_harm_f1_gt_m1_to_harm_f1",
                "panel_count": int(len(metric_rows)),
                "median_r": np.nan,
                "q1_r": np.nan,
                "q3_r": np.nan,
                "pct_gt_03": np.nan,
                "pct_gt_05": np.nan,
                "probability_pct": float(
                    (metric_rows["m2_to_harm_f1"] > metric_rows["m1_to_harm_f1"]).mean() * 100
                ),
            }
        )
    aggregate = pd.DataFrame(summary_rows)
    aggregate.to_csv(os.path.join(exp_dir, f"{file_stem}_aggregate.csv"), index=False)
    wide.to_csv(os.path.join(exp_dir, f"{file_stem}_panel_correlations.csv"), index=False)
    scope_group_columns = ["dataset"]
    if has_categories:
        scope_group_columns.insert(0, "dataset_category")
    scope_counts = (
        panel_points.groupby(scope_group_columns, as_index=False)
        .agg(
            panel_count=("classifier", "size"),
            classifier_count=("classifier", "nunique"),
            mechanism_count=("missing_mechanism", "nunique"),
            missing_percent_min=("missing_percent", "min"),
            missing_percent_max=("missing_percent", "max"),
            min_imputers=("n_points", "min"),
            median_imputers=("n_points", "median"),
            max_imputers=("n_points", "max"),
        )
    )
    scope_counts.to_csv(os.path.join(exp_dir, f"{file_stem}_analysis_scope.csv"), index=False)
    return scope_metadata


def _generate_experiment_4_correlation_tables(
    summary,
    exp_dir,
    m1_metrics,
    m2_metrics,
):
    filtered = summary.copy()
    if "status" in filtered.columns:
        status = filtered["status"].fillna("").astype(str).str.lower()
        filtered = filtered.loc[status.isin(["ok", "success"])].copy()

    selected_m1 = [metric for metric in m1_metrics if metric in filtered.columns]
    selected_m2 = [metric for metric in m2_metrics if metric in filtered.columns]
    if not selected_m1 or not selected_m2:
        raise ValueError("Experiment 4 correlation table has no usable M1/M2 metrics")

    combo_cols = ["dataset", "classifier", "missing_percent"]
    if "dataset_category" in filtered.columns:
        combo_cols.insert(0, "dataset_category")
    combos = (
        filtered[combo_cols]
        .dropna()
        .drop_duplicates()
        .sort_values(combo_cols)
    )
    relations = [
        ("m1_to_m2", "m1", "m2"),
        ("m2_to_harm_f1", "m2", PRIMARY_TARGET),
        ("m1_to_harm_f1", "m1", PRIMARY_TARGET),
    ]
    rows = []
    for combo in combos.itertuples(index=False):
        combo_values = dict(zip(combo_cols, combo, strict=True))
        combo_frame = filtered.copy()
        for col, value in combo_values.items():
            combo_frame = _filter_frame_by_restriction(combo_frame, col, value)
        for m1_metric in selected_m1:
            for m2_metric in selected_m2:
                required = [
                    "missing_mechanism",
                    "imputer",
                    m1_metric,
                    m2_metric,
                    PRIMARY_TARGET,
                ]
                working = combo_frame[required].copy()
                for col in [m1_metric, m2_metric, PRIMARY_TARGET]:
                    working[col] = pd.to_numeric(working[col], errors="coerce")
                working = working.dropna(subset=required)
                points = (
                    working.groupby(
                        ["missing_mechanism", "imputer"],
                        as_index=False,
                        dropna=False,
                    )
                    .agg(
                        **{
                            m1_metric: (m1_metric, "mean"),
                            m2_metric: (m2_metric, "mean"),
                            PRIMARY_TARGET: (PRIMARY_TARGET, "mean"),
                            "run_rows": ("imputer", "size"),
                        }
                    )
                )
                for mechanism, panel in points.groupby(
                    "missing_mechanism", sort=True, dropna=False
                ):
                    value_map = {
                        "m1": panel[m1_metric],
                        "m2": panel[m2_metric],
                        PRIMARY_TARGET: panel[PRIMARY_TARGET],
                    }
                    for relation, x_key, y_key in relations:
                        pearson_r, n_points = _experiment_4_pearson_pair(
                            value_map[x_key], value_map[y_key]
                        )
                        row = {
                            **combo_values,
                            "missing_mechanism": str(mechanism),
                            "m1_metric": m1_metric,
                            "m2_metric": m2_metric,
                            "relation": relation,
                            "x_metric": (
                                m1_metric
                                if x_key == "m1"
                                else m2_metric
                            ),
                            "y_metric": (
                                m2_metric
                                if y_key == "m2"
                                else PRIMARY_TARGET
                            ),
                            "pearson_r": pearson_r,
                            "n_points": n_points,
                            "run_rows": int(panel["run_rows"].sum()),
                        }
                        rows.append(row)

    long_table = pd.DataFrame(rows).sort_values(
        [
            *combo_cols,
            "missing_mechanism",
            "m1_metric",
            "m2_metric",
            "relation",
        ]
    )
    long_path = os.path.join(exp_dir, "pearson_correlations_by_condition.csv")
    long_table.to_csv(long_path, index=False)

    index_cols = [
        *combo_cols,
        "missing_mechanism",
        "m1_metric",
        "m2_metric",
    ]
    corr_wide = long_table.pivot(
        index=index_cols,
        columns="relation",
        values="pearson_r",
    ).rename(columns=lambda value: f"pearson_r_{value}")
    n_wide = long_table.pivot(
        index=index_cols,
        columns="relation",
        values="n_points",
    ).rename(columns=lambda value: f"n_points_{value}")
    wide_table = corr_wide.join(n_wide).reset_index()
    wide_table.to_csv(
        os.path.join(exp_dir, "pearson_correlations_by_condition_wide.csv"),
        index=False,
    )
    aggregate_metadata = _generate_experiment_4_aggregate_table(long_table, exp_dir)
    stratified_aggregate_metadata = {}
    if "dataset_category" in long_table.columns:
        aggregate_scopes = [
            (
                "no_real_missing",
                "table4_no_real_missing",
                "tab:exp4-no-real-missing",
            ),
            (
                "real_missing_with_truth",
                "table4_real_missing_truth",
                "tab:exp4-real-missing-truth",
            ),
        ]
        for category, file_stem, latex_label in aggregate_scopes:
            category_table = long_table.loc[
                long_table["dataset_category"].eq(category)
            ].copy()
            category_metadata = _generate_experiment_4_aggregate_table(
                category_table,
                exp_dir,
                file_stem=file_stem,
                latex_label=latex_label,
            )
            if category_metadata is not None:
                stratified_aggregate_metadata[category] = category_metadata
    manifest = {
        "mode": "correlations_only",
        "long_row_count": int(len(long_table)),
        "wide_row_count": int(len(wide_table)),
        "nonnull_pearson_count": int(long_table["pearson_r"].notna().sum()),
        "null_pearson_count": int(long_table["pearson_r"].isna().sum()),
        "m1_metrics": selected_m1,
        "m2_metrics": selected_m2,
        "triptych_png_count": 0,
    }
    if aggregate_metadata is not None:
        manifest["aggregate_table"] = aggregate_metadata
    if stratified_aggregate_metadata:
        manifest["stratified_aggregate_tables"] = stratified_aggregate_metadata
    Path(exp_dir, "correlation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_experiment_4(
    summary,
    sample_level,
    output_dir,
    bx=4,
    by=4,
    bootstrap_rounds=500,
    seed=42,
    label_mode="en",
    experiment4_m1_metrics=None,
    experiment4_m2_metrics=None,
    main_figures_only=False,
    correlations_only=False,
):
    _require_columns(
        summary,
        ["dataset", "classifier", "missing_mechanism", "missing_percent", "imputer", PRIMARY_STRUCTURE, PRIMARY_HARD, PRIMARY_TARGET],
        "run_summary",
    )
    exp_dir = os.path.join(output_dir, "experiment_4")
    _ensure_dir(exp_dir)
    is_real_missing_truth = _is_real_missing_with_truth_summary(summary)
    hard_sample_summary_path = os.path.join(exp_dir, "hard_sample_summary.csv")
    selected_m2_metrics = list(experiment4_m2_metrics or EXPERIMENT4_TRIPTYCH_BASE_M2_METRICS)
    needs_sample_level_margin = any("margin" in str(metric) for metric in selected_m2_metrics)
    sample_level = _normalize_sample_level_margin_direction(sample_level)
    if needs_sample_level_margin:
        summary = _refresh_summary_margin_metrics_from_sample_level(summary, sample_level)

    if correlations_only:
        _generate_experiment_4_correlation_tables(
            summary,
            exp_dir,
            m1_metrics=list(
                experiment4_m1_metrics or EXPERIMENT4_TRIPTYCH_M1_METRICS
            ),
            m2_metrics=selected_m2_metrics,
        )
        return

    if (
        needs_sample_level_margin
        and (not main_figures_only)
        and (not sample_level.empty)
        and (not os.path.exists(hard_sample_summary_path))
    ):
        _require_columns(
            sample_level,
            ["is_hard", EXPERIMENT4_SAMPLE_LEVEL_FLIP_PRIMARY, "margin_shift"],
            "sample_level",
        )
        sample_level_summary = _summarize_experiment4_hard_samples(sample_level)
        sample_level_summary.to_csv(hard_sample_summary_path, index=False)

    _generate_experiment4_triptych_combo_figures(
        summary,
        exp_dir,
        label_mode=label_mode,
        m1_metrics=experiment4_m1_metrics,
        m2_metrics=selected_m2_metrics,
        max_figures=1 if main_figures_only else None,
    )
    if is_real_missing_truth and not main_figures_only:
        _generate_experiment4_cross_dataset_figures(
            summary,
            exp_dir,
            label_mode=label_mode,
            m1_metrics=experiment4_m1_metrics,
            m2_metrics=selected_m2_metrics,
        )


def main():
    parser = argparse.ArgumentParser(
        description="offline paper analysis only: reads existing result CSVs and produces tables/figures"
    )
    parser.add_argument("--summary", default="result", help="run_summary path, batch dir, or result dir")
    parser.add_argument(
        "--sample_level",
        default="result",
        help="sample_level path, batch dir, or result dir (used only by experiment 4; ignored if missing unless --require_sample_level is set)",
    )
    parser.add_argument(
        "--hsr-paired",
        "--hsr_paired",
        default="result/hsai_experiment/merged_hsr/hsr_experiment_paired.csv",
        help="canonical paired base-vs-HSR CSV used only by experiment 5",
    )
    parser.add_argument("--require_sample_level", action="store_true", help="fail if sample_level is missing")
    parser.add_argument("--output_dir", default="result/paper_experiments", help="directory for experiment outputs")
    parser.add_argument("--num_bins", type=int, default=4, help="reserved for legacy supplemental analyses")
    parser.add_argument("--bootstrap_rounds", type=int, default=500, help="reserved for legacy supplemental analyses")
    parser.add_argument("--seed", type=int, default=42, help="random seed for bootstrap")
    parser.add_argument(
        "--experiment3_missing_percent",
        type=float,
        default=None,
        help="optional filter for experiment 3 only; when set, fit fixed-effect regressions on one missing_percent value",
    )
    parser.add_argument(
        "--experiment3-layout",
        choices=["legacy", "tiered"],
        default="legacy",
        help=(
            "legacy outputs, or tiered pooled/by-mechanism/by-dataset fixed-effect "
            "tables for mixed synthetic and real-truth data"
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="optional subset of paper experiments to run, e.g. --experiments 5 or --experiments 3 4",
    )
    parser.add_argument(
        "--label-mode",
        choices=["en", "zh"],
        default="en",
        help="text label mode for generated figures",
    )
    parser.add_argument(
        "--main_figures_only",
        action="store_true",
        help="generate only the representative main figure/table set for each selected experiment",
    )
    parser.add_argument(
        "--experiment4-correlations-only",
        action="store_true",
        help=(
            "skip Experiment 4 triptych PNGs and write per-condition Pearson "
            "correlation tables instead"
        ),
    )
    parser.add_argument(
        "--w1-column",
        default="m1_w1_mean",
        help="run_summary column to use as the W1 structure metric",
    )
    args = parser.parse_args()

    summary_input = _resolve_default_input_root(args.summary, "run_summary")
    sample_level_input = _resolve_default_input_root(args.sample_level, "sample_level")
    requested_experiments = set(args.experiments or [])
    experiment5_requested = 5 in requested_experiments
    if (
        args.output_dir == "result/paper_experiments"
        and Path(summary_input).exists()
        and Path(summary_input).is_dir()
        and not experiment5_requested
    ):
        category_roots = _discover_category_result_roots(summary_input)
        if category_roots:
            for category_root in category_roots:
                summary_path = _resolve_latest_result_path(str(category_root), "run_summary")
                sample_level_path = _resolve_latest_result_path(str(category_root), "sample_level")
                output_dir = str(category_root / "paper_experiments")
                _run_all_paper_experiments(
                    summary_path,
                    sample_level_path,
                    output_dir,
                    args.num_bins,
                    args.bootstrap_rounds,
                    args.seed,
                    args.require_sample_level,
                    args.experiments,
                    args.experiment3_missing_percent,
                    args.label_mode,
                    args.main_figures_only,
                    args.w1_column,
                    args.experiment3_layout,
                    args.experiment4_correlations_only,
                    args.hsr_paired,
                )
            return

    needs_summary = not args.experiments or bool(requested_experiments.intersection({1, 2, 3, 4}))
    summary_path = (
        _resolve_latest_result_path(summary_input, "run_summary")
        if needs_summary
        else None
    )
    sample_level_path = (
        _resolve_latest_result_path(sample_level_input, "sample_level")
        if needs_summary
        else None
    )
    if needs_summary and summary_path is None:
        raise FileNotFoundError(f"No run_summary file found for {args.summary}")
    if experiment5_requested and not Path(args.hsr_paired).is_file():
        raise FileNotFoundError(f"No HSR paired result file found for {args.hsr_paired}")
    only_experiment5 = requested_experiments == {5}
    if args.output_dir == "result/paper_experiments":
        if only_experiment5:
            output_dir = str(Path(args.hsr_paired).parent / "paper_experiments")
        else:
            output_dir = str(Path(_resolve_result_root(summary_path)) / "paper_experiments")
    else:
        output_dir = args.output_dir
    _run_all_paper_experiments(
        summary_path,
        sample_level_path,
        output_dir,
        args.num_bins,
        args.bootstrap_rounds,
        args.seed,
        args.require_sample_level,
        args.experiments,
        args.experiment3_missing_percent,
        args.label_mode,
        args.main_figures_only,
        args.w1_column,
        args.experiment3_layout,
        args.experiment4_correlations_only,
        args.hsr_paired,
    )


if __name__ == "__main__":
    main()
