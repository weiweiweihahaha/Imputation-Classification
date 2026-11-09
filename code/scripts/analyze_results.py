import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


M1_COLUMNS = [
    "m1_corr_frob",
    "m1_fisher_delta",
]

M2_COLUMNS = [
    "m2_inconsistency_ref_input",
    "m2_hard_inconsistency_ref_input",
    "m2_hard_margin_shift",
]

TARGET_COLUMNS = ["delta_f1"]


def _ensure_output_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _resolve_stem_paths(path, stem):
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        return [candidate], candidate.parent

    if candidate.exists() and candidate.is_dir():
        direct_path = candidate / f"{stem}.csv"
        if direct_path.exists():
            return [direct_path], candidate
        matches = sorted(item for item in candidate.rglob(f"{stem}.csv") if item.is_file())
        if matches:
            return matches, candidate
        raise FileNotFoundError(f"No result file found for '{stem}.csv' under {candidate}")

    if candidate.suffix == ".csv":
        raise FileNotFoundError(f"No such file: {candidate}")
    raise FileNotFoundError(f"No such directory: {candidate}")


def _resolve_default_input(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if str(path) != "result/run_summary.csv":
        return candidate

    result_root = Path("result")
    if (result_root / "run_summary.csv").exists():
        return result_root
    combined_dir = result_root / "combined_csv_cleaned"
    if combined_dir.exists():
        return combined_dir
    return candidate


def _load_stem_frame(path, stem):
    paths, summary_root = _resolve_stem_paths(path, stem)
    frames = [pd.read_csv(item) for item in paths]
    return pd.concat(frames, ignore_index=True), summary_root


def _resolve_output_dir(requested_output_dir, summary_root):
    if requested_output_dir == "result/analysis":
        return str(Path(summary_root) / "analysis")
    return requested_output_dir


def _pairwise_corr(frame, left_cols, right_cols):
    rows = []
    for left in left_cols:
        if left not in frame.columns:
            continue
        for right in right_cols:
            if right not in frame.columns:
                continue
            subset = frame[[left, right]].dropna()
            if len(subset) < 2:
                corr_value = np.nan
            else:
                corr_value = float(subset[left].corr(subset[right]))
            rows.append({"x": left, "y": right, "pearson_corr": corr_value, "n": int(len(subset))})
    return pd.DataFrame(rows)


def _fit_linear(frame, feature_cols, target_col):
    usable_cols = [col for col in feature_cols if col in frame.columns]
    subset = frame[usable_cols + [target_col]].dropna()
    if len(subset) <= len(usable_cols) or not usable_cols:
        return None

    x = subset[usable_cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(subset)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ beta
    residual = y - y_hat
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    rows = [{"target": target_col, "feature": "intercept", "coefficient": float(beta[0]), "r2": float(r2) if not np.isnan(r2) else np.nan, "n": int(len(subset))}]
    for idx, feature in enumerate(usable_cols, start=1):
        rows.append({"target": target_col, "feature": feature, "coefficient": float(beta[idx]), "r2": float(r2) if not np.isnan(r2) else np.nan, "n": int(len(subset))})
    return pd.DataFrame(rows)


def _group_regressions(frame, group_col):
    rows = []
    for group_value, group_frame in frame.groupby(group_col):
        m1_to_m2 = _fit_linear(group_frame, M1_COLUMNS, "m2_inconsistency_ref_input")
        m2_to_delta = _fit_linear(group_frame, M2_COLUMNS, "delta_f1")
        if m1_to_m2 is not None:
            m1_to_m2.insert(0, group_col, group_value)
            m1_to_m2.insert(1, "model_stage", "M1_to_M2")
            rows.append(m1_to_m2)
        if m2_to_delta is not None:
            m2_to_delta.insert(0, group_col, group_value)
            m2_to_delta.insert(1, "model_stage", "M2_to_DeltaF1")
            rows.append(m2_to_delta)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_method_ranking(frame):
    return (
        frame.groupby([col for col in ["dataset", "classifier", "imputer_family", "imputer"] if col in frame.columns], as_index=False)
        .agg(
            mean_delta_f1=("delta_f1", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
            mean_hard_disagree=("m2_hard_inconsistency_ref_input", "mean"),
            runs=("imputer", "count"),
        )
        .sort_values(["dataset", "classifier", "mean_delta_f1"], ascending=[True, True, False])
    )


def _write_markdown_report(output_dir, summary, corr_m1_m2, corr_m2_target, nrmse_corr, overall_reg, by_classifier, ranking):
    best_methods = ranking.head(10)
    strongest_m1_m2 = corr_m1_m2.reindex(corr_m1_m2["pearson_corr"].abs().sort_values(ascending=False).index).head(5)
    strongest_m2_delta = corr_m2_target[corr_m2_target["y"] == "delta_f1"].reindex(
        corr_m2_target[corr_m2_target["y"] == "delta_f1"]["pearson_corr"].abs().sort_values(ascending=False).index
    ).head(5)

    lines = [
        "# Experiment Analysis Report",
        "",
        f"- Total runs: {len(summary)}",
        f"- Datasets: {', '.join(sorted(summary['dataset'].astype(str).unique()))}",
        f"- Classifiers: {', '.join(sorted(summary['classifier'].astype(str).unique()))}",
        f"- Imputers: {', '.join(sorted(summary['imputer'].astype(str).unique()))}",
        "",
        "## NRMSE baseline check",
    ]
    if not nrmse_corr.empty:
        for _, row in nrmse_corr.iterrows():
            if pd.notna(row["pearson_corr"]):
                lines.append(f"- nrmse -> {row['y']}: corr={row['pearson_corr']:.4f} (n={int(row['n'])})")
            else:
                lines.append(f"- nrmse -> {row['y']}: corr=nan (n={int(row['n'])})")
    else:
        lines.append("- No valid NRMSE correlations available.")

    lines.extend([
        "",
        "## Strongest M1 -> M2 correlations",
    ])
    for _, row in strongest_m1_m2.iterrows():
        if pd.notna(row["pearson_corr"]):
            lines.append(f"- {row['x']} -> {row['y']}: corr={row['pearson_corr']:.4f} (n={int(row['n'])})")
        else:
            lines.append(f"- {row['x']} -> {row['y']}: corr=nan (n={int(row['n'])})")

    lines.extend(["", "## Strongest M2 -> delta_f1 correlations"])
    for _, row in strongest_m2_delta.iterrows():
        if pd.notna(row["pearson_corr"]):
            lines.append(f"- {row['x']} -> delta_f1: corr={row['pearson_corr']:.4f} (n={int(row['n'])})")
        else:
            lines.append(f"- {row['x']} -> delta_f1: corr=nan (n={int(row['n'])})")

    lines.extend(["", "## Top configurations by delta F1"])
    for _, row in best_methods.iterrows():
        family_text = f", family={row['imputer_family']}" if "imputer_family" in row else ""
        lines.append(
            f"- dataset={row['dataset']}, classifier={row['classifier']}, imputer={row['imputer']}{family_text}, "
            f"mean_delta_f1={row['mean_delta_f1']:.4f}"
        )

    lines.extend(["", "## Regression outputs"])
    if not overall_reg.empty:
        for _, row in overall_reg.iterrows():
            if row["feature"] == "intercept":
                continue
            lines.append(f"- {row['model_stage']}: {row['feature']} coef={row['coefficient']:.4f}, r2={row['r2']:.4f}, n={int(row['n'])}")

    if not by_classifier.empty:
        lines.extend(["", "## By-classifier regression snapshots"])
        for _, row in by_classifier[by_classifier["feature"] != "intercept"].head(10).iterrows():
            lines.append(
                f"- classifier={row['classifier']}, stage={row['model_stage']}, {row['feature']} coef={row['coefficient']:.4f}, r2={row['r2']:.4f}"
            )

def main():
    parser = argparse.ArgumentParser(description="analyze staged experiment outputs")
    parser.add_argument("--input", default="result", help="run_summary path, batch dir, or result dir")
    parser.add_argument("--output_dir", default="result/analysis", help="directory for analysis outputs")
    args = parser.parse_args()

    summary, summary_root = _load_stem_frame(_resolve_default_input(args.input), "run_summary")
    output_dir = _resolve_output_dir(args.output_dir, summary_root)
    _ensure_output_dir(output_dir)
    numeric_cols = [
        col
        for col in M1_COLUMNS
        + M2_COLUMNS
        + TARGET_COLUMNS
        + [
            "runtime_seconds",
            "imputation_runtime_seconds",
            "train_runtime_seconds",
            "inference_runtime_seconds",
            "total_runtime_seconds",
            "nrmse",
            "rmse_masked",
            "masked_cell_count",
        ]
        if col in summary.columns
    ]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    corr_m1_m2 = _pairwise_corr(summary, M1_COLUMNS, M2_COLUMNS)
    corr_m2_target = _pairwise_corr(summary, M2_COLUMNS, TARGET_COLUMNS)
    corr_m1_target = _pairwise_corr(summary, M1_COLUMNS, TARGET_COLUMNS)
    corr_nrmse_target = _pairwise_corr(summary, ["nrmse"], TARGET_COLUMNS)

    overall_parts = []
    model_m1_m2 = _fit_linear(summary, M1_COLUMNS, "m2_inconsistency_ref_input")
    if model_m1_m2 is not None:
        model_m1_m2.insert(0, "model_stage", "M1_to_M2")
        overall_parts.append(model_m1_m2)
    model_m2_delta = _fit_linear(summary, M2_COLUMNS, "delta_f1")
    if model_m2_delta is not None:
        model_m2_delta.insert(0, "model_stage", "M2_to_DeltaF1")
        overall_parts.append(model_m2_delta)
    overall_reg = pd.concat(overall_parts, ignore_index=True) if overall_parts else pd.DataFrame()

    by_classifier = _group_regressions(summary, "classifier")
    ranking = _build_method_ranking(summary)

    corr_m1_m2.to_csv(os.path.join(output_dir, "corr_m1_to_m2.csv"), index=False)
    corr_m2_target.to_csv(os.path.join(output_dir, "corr_m2_to_targets.csv"), index=False)
    corr_m1_target.to_csv(os.path.join(output_dir, "corr_m1_to_targets.csv"), index=False)
    corr_nrmse_target.to_csv(os.path.join(output_dir, "corr_nrmse_to_targets.csv"), index=False)
    overall_reg.to_csv(os.path.join(output_dir, "regression_overall.csv"), index=False)
    by_classifier.to_csv(os.path.join(output_dir, "regression_by_classifier.csv"), index=False)
    ranking.to_csv(os.path.join(output_dir, "method_ranking.csv"), index=False)

if __name__ == "__main__":
    main()
