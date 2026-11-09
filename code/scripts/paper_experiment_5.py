"""Paper Experiment 5: HSR activation and paired outcome statistics.

The unit of analysis is one paired base-vs-HSR experimental cell.  Rates use
strict inequalities: ties are not counted as either an improvement or a loss.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT5_KEY_COLUMNS = [
    "base",
    "dataset",
    "missing_percent",
    "mechanism",
    "iteration",
    "seed",
    "classifier",
]
EXPERIMENT5_METRIC_COLUMNS = ["selected_lam", "delta_hsi", "delta_f1"]
EXPERIMENT5_REQUIRED_COLUMNS = EXPERIMENT5_KEY_COLUMNS + EXPERIMENT5_METRIC_COLUMNS
EXPERIMENT5_DATASET_SAMPLE_COUNTS = {
    "iris": 150,
    "wine": 178,
    "seed": 199,
    "glass": 214,
    "air_quality": 500,
    "hcv": 615,
    "fifa": 850,
    "college": 1092,
    "cvss": 1617,
    "mobile": 2000,
    "obesity": 2111,
    "rice": 3810,
}
EXPERIMENT5_DATASET_LABELS = {
    "iris": "Iris",
    "wine": "Wine",
    "seed": "Seed",
    "glass": "Glass",
    "air_quality": "AirQuality",
    "hcv": "HCV",
    "fifa": "FIFA",
    "college": "College",
    "cvss": "CVSS",
    "mobile": "Mobile",
    "obesity": "Obesity",
    "rice": "Rice",
    "All datasets": "All datasets",
}
EXPERIMENT5_LATEX_DATASETS_PER_GROUP = 7


def _require_experiment5_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in EXPERIMENT5_REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"HSR paired results missing required columns: {missing}")


def load_hsr_paired(path: str | Path) -> pd.DataFrame:
    """Load and validate the canonical HSR paired-result table."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HSR paired result file not found: {source}")

    frame = pd.read_csv(source, low_memory=False)
    _require_experiment5_columns(frame)
    frame = frame.copy()
    for column in EXPERIMENT5_METRIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    missing_metrics = {
        column: int(frame[column].isna().sum())
        for column in EXPERIMENT5_METRIC_COLUMNS
        if frame[column].isna().any()
    }
    if missing_metrics:
        raise ValueError(f"HSR paired results contain missing/non-numeric metrics: {missing_metrics}")

    duplicate_count = int(frame.duplicated(EXPERIMENT5_KEY_COLUMNS, keep=False).sum())
    if duplicate_count:
        raise ValueError(
            "HSR paired results contain duplicate experiment keys: "
            f"{duplicate_count} rows across {EXPERIMENT5_KEY_COLUMNS}"
        )
    return frame


def _summarize_scope(frame: pd.DataFrame) -> dict[str, int | float]:
    total = int(len(frame))
    lambda_nonzero = frame["selected_lam"].ne(0)
    hsi_decrease = frame["delta_hsi"].lt(0)
    downstream_increase = frame["delta_f1"].gt(0)
    joint_improvement = hsi_decrease & downstream_increase

    lambda_count = int(lambda_nonzero.sum())
    hsi_count = int(hsi_decrease.sum())
    joint_count = int(joint_improvement.sum())
    return {
        "total_paired_cells": total,
        "lambda_nonzero_count": lambda_count,
        "lambda_nonzero_rate": float(lambda_count / total) if total else np.nan,
        "hsi_decrease_count": hsi_count,
        "hsi_decrease_rate": float(hsi_count / total) if total else np.nan,
        "hsi_decrease_f1_increase_count": joint_count,
        "hsi_decrease_f1_increase_given_hsi_decrease_rate": (
            float(joint_count / hsi_count) if hsi_count else np.nan
        ),
    }


def summarize_experiment_5(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-dataset and pooled Experiment 5 summaries."""
    _require_experiment5_columns(paired)
    rows = []
    present = {str(dataset) for dataset in paired["dataset"].drop_duplicates()}
    dataset_order = [
        dataset
        for dataset in EXPERIMENT5_DATASET_SAMPLE_COUNTS
        if dataset in present
    ]
    dataset_order.extend(sorted(present.difference(dataset_order)))
    dataset_names = paired["dataset"].astype(str)
    for dataset in dataset_order:
        group = paired.loc[dataset_names.eq(dataset)]
        rows.append({"dataset": str(dataset), **_summarize_scope(group)})
    by_dataset = pd.DataFrame(rows)
    overall = pd.DataFrame([{"dataset": "All datasets", **_summarize_scope(paired)}])
    return by_dataset, overall


def _format_count_rate(count: int, denominator: int, rate: float) -> str:
    if denominator == 0 or not np.isfinite(rate):
        return f"{count}/{denominator} (n/a)"
    return f"{count}/{denominator} ({100.0 * rate:.2f}\\%)"


def _format_rate(rate: float) -> str:
    return "n/a" if not np.isfinite(rate) else f"{100.0 * rate:.2f}\\%"


def format_experiment_5_latex(by_dataset: pd.DataFrame, overall: pd.DataFrame) -> str:
    """Build the paper-ready LaTeX table for Experiment 5."""
    del overall  # The compact paper table reports datasets only; pooled results stay in CSV.
    column_spec = "@{}l" + ("c" * EXPERIMENT5_LATEX_DATASETS_PER_GROUP) + "@{}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HSR activation and paired hard-sample outcomes by dataset, ordered by sample size.}",
        r"\label{tab:experiment5_hsr_by_dataset}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
    ]
    records = list(by_dataset.itertuples(index=False))
    for start in range(0, len(records), EXPERIMENT5_LATEX_DATASETS_PER_GROUP):
        group = records[start : start + EXPERIMENT5_LATEX_DATASETS_PER_GROUP]
        while len(group) < EXPERIMENT5_LATEX_DATASETS_PER_GROUP:
            group.append(None)
        dataset_labels = [
            EXPERIMENT5_DATASET_LABELS.get(str(row.dataset), str(row.dataset))
            .replace("_", r"\_")
            if row is not None
            else ""
            for row in group
        ]
        lines.append("Dataset & " + " & ".join(dataset_labels) + r" \\")
        lines.append(r"\midrule")

        lambda_values = []
        joint_values = []
        for row in group:
            if row is None:
                lambda_values.append("")
                joint_values.append("")
                continue
            lambda_values.append(_format_rate(float(row.lambda_nonzero_rate)))
            joint_values.append(
                _format_rate(
                    float(row.hsi_decrease_f1_increase_given_hsi_decrease_rate)
                )
            )
        lines.append(
            r"$\lambda \neq 0$ & " + " & ".join(lambda_values) + r" \\"
        )
        lines.append(
            r"HSI$\downarrow$ \& F1$\uparrow$ $\mid$ HSI$\downarrow$ & "
            + " & ".join(joint_values)
            + r" \\"
        )
        if start + EXPERIMENT5_LATEX_DATASETS_PER_GROUP < len(records):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize Notes: one observation is one paired base-vs-HSR experimental "
            r"cell. HSI denotes hard-sample inconsistency. Strict inequalities are used: "
            r"$\Delta\mathrm{HSI}<0$ and $\Delta\mathrm{F1}>0$; ties are excluded. "
            r"The conditional row uses cells with $\Delta\mathrm{HSI}<0$ as its denominator; "
            r"counts are available in the accompanying CSV table.",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment_5(hsr_paired_path: str | Path, output_dir: str | Path) -> dict:
    """Generate all paper Experiment 5 tables and reproducibility metadata."""
    paired_path = Path(hsr_paired_path)
    paired = load_hsr_paired(paired_path)
    by_dataset, overall = summarize_experiment_5(paired)

    exp_dir = Path(output_dir) / "experiment_5"
    if exp_dir.is_dir():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    by_dataset_path = exp_dir / "table5_hsr_by_dataset.csv"
    overall_path = exp_dir / "table5_hsr_overall.csv"
    manifest_path = exp_dir / "experiment5_manifest.json"
    by_dataset.to_csv(by_dataset_path, index=False)
    overall.to_csv(overall_path, index=False)

    manifest = {
        "experiment": 5,
        "title": "HSR activation and paired hard-sample outcomes",
        "input": str(paired_path.resolve()),
        "input_rows": int(len(paired)),
        "dataset_count": int(paired["dataset"].nunique(dropna=False)),
        "base_imputer_count": int(paired["base"].nunique(dropna=False)),
        "classifier_count": int(paired["classifier"].nunique(dropna=False)),
        "experimental_unit": "one paired base-vs-HSR experimental cell",
        "unique_key": EXPERIMENT5_KEY_COLUMNS,
        "rules": {
            "lambda_nonzero": "selected_lam != 0",
            "hsi_decrease": "delta_hsi < 0",
            "downstream_improvement": "delta_f1 > 0",
            "joint_denominator": "rows satisfying delta_hsi < 0",
            "ties": "excluded from improvements",
        },
        "generated_files": [
            by_dataset_path.name,
            overall_path.name,
        ],
        "reported_table_metrics": [
            "lambda_nonzero_rate",
            "hsi_decrease_f1_increase_given_hsi_decrease_rate",
        ],
        "dataset_order": list(EXPERIMENT5_DATASET_SAMPLE_COUNTS),
        "dataset_sample_counts": EXPERIMENT5_DATASET_SAMPLE_COUNTS,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest
