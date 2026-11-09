import argparse
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_M1 = "m1_corr_frob"
PRIMARY_M2 = "m2_hard_inconsistency_ref_input"
PRIMARY_TARGET = "delta_f1"


def _ensure_output_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

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


def _resolve_default_summary_input(path):
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if str(path) != "result/run_summary.csv":
        return path

    result_root = Path("result")
    if (result_root / "run_summary.csv").exists():
        return str(result_root)
    combined_dir = result_root / "combined_csv_cleaned"
    if combined_dir.exists():
        return str(combined_dir)
    return path


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


def _load_latest_prefixed_csv(directory, stem):
    base = Path(directory)
    if not base.exists():
        return pd.DataFrame()
    direct_path = base / f"{stem}.csv"
    if direct_path.exists():
        return pd.read_csv(direct_path)
    candidates = sorted(base.glob(f"*/{stem}.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[-1])


def _load_summary(path):
    if isinstance(path, list):
        frames = [_load_csv(item) for item in path]
        frames = [frame for frame in frames if not frame.empty]
        summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        summary = _load_csv(path)
    numeric_cols = [
        "missing_percent",
        "runtime_seconds",
        "delta_f1",
        "nrmse",
        "m1_corr_frob",
        "m1_fisher_delta",
        "m2_inconsistency_ref_input",
        "m2_hard_inconsistency_ref_input",
        "m2_inconsistency_retrained",
        "m2_hard_inconsistency_retrained",
        "m2_margin_shift",
        "m2_hard_margin_shift",
    ]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary


def _scatter_with_fit(ax, frame, x_col, y_col, color_col=None, title=None):
    usable_cols = [x_col, y_col]
    if color_col is not None and color_col in frame.columns:
        usable_cols.append(color_col)
    subset = frame[usable_cols].dropna()
    if subset.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or f"{x_col} vs {y_col}")
        return

    if color_col is not None and color_col in subset.columns:
        scatter = ax.scatter(
            subset[x_col],
            subset[y_col],
            c=subset[color_col],
            cmap="coolwarm",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label(color_col)
    else:
        ax.scatter(
            subset[x_col],
            subset[y_col],
            color="#1f4e79",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )

    if len(subset) >= 2:
        x = subset[x_col].to_numpy(dtype=float)
        y = subset[y_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#b22222", linewidth=1.6)
        corr = float(subset[x_col].corr(subset[y_col]))
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {corr:.4f}\nn = {len(subset)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{x_col} vs {y_col}")
    ax.grid(alpha=0.25, linewidth=0.4)


def _save_primary_scatter(summary, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _scatter_with_fit(
        axes[0],
        summary,
        PRIMARY_M1,
        PRIMARY_M2,
        color_col=PRIMARY_TARGET,
        title="Structural Perturbation to Hard-Disagree",
    )
    _scatter_with_fit(
        axes[1],
        summary,
        PRIMARY_M2,
        PRIMARY_TARGET,
        color_col=None,
        title="Hard-Disagree to Delta F1",
    )
    fig.savefig(os.path.join(output_dir, "causal_chain_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_nrmse_scatter(summary, output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    _scatter_with_fit(
        ax,
        summary,
        "nrmse",
        "delta_f1",
        color_col="m1_corr_frob" if "m1_corr_frob" in summary.columns else None,
        title="NRMSE to Delta F1",
    )
    fig.savefig(os.path.join(output_dir, "nrmse_vs_delta_f1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_top_methods(summary, output_dir, top_k):
    if summary.empty:
        frame = pd.DataFrame()
    else:
        frame = (
            summary.groupby(["dataset", "classifier", "imputer"], as_index=False)
            .agg(
                mean_delta_f1=("delta_f1", "mean"),
                std_delta_f1=("delta_f1", "std"),
                runs=("imputer", "count"),
            )
            .sort_values("mean_delta_f1", ascending=False)
            .head(top_k)
        )

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if frame.empty:
        ax.text(0.5, 0.5, "No ranking data", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [
            f"{row.dataset} | {row.classifier} | {row.imputer}"
            for row in frame.itertuples(index=False)
        ]
        y_pos = np.arange(len(frame))
        error = frame["std_delta_f1"].fillna(0.0).to_numpy(dtype=float)
        ax.barh(
            y_pos,
            frame["mean_delta_f1"],
            xerr=error,
            color="#2e8b57",
            alpha=0.85,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    ax.set_xlabel("Mean delta F1")
    ax.set_title(f"Top {top_k} Configurations by Delta F1")
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    fig.savefig(os.path.join(output_dir, "top_methods_by_delta_f1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_heatmap(corr_df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if corr_df.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(os.path.join(output_dir, "m1_m2_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = corr_df.pivot(index="x", columns="y", values="pearson_corr")
    pivot = pivot.sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("M1 to M2 Correlation Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "nan" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)

    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation")
    fig.savefig(os.path.join(output_dir, "m1_m2_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_missing_percent_profile(summary, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    required = {"missing_percent", "delta_f1", "classifier"}
    if summary.empty or not required.issubset(summary.columns):
        ax.text(0.5, 0.5, "No missing-percent profile data", ha="center", va="center", transform=ax.transAxes)
    else:
        grouped = (
            summary.groupby(["classifier", "missing_percent"], as_index=False)
            .agg(mean_delta_f1=("delta_f1", "mean"))
            .sort_values(["classifier", "missing_percent"])
        )
        for classifier, group in grouped.groupby("classifier"):
            ax.plot(
                group["missing_percent"],
                group["mean_delta_f1"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=str(classifier),
            )
        ax.legend(frameon=False)
    ax.set_xlabel("Missing percent")
    ax.set_ylabel("Mean delta_f1")
    ax.set_title("Delta F1 by Missing Rate")
    ax.grid(alpha=0.25, linewidth=0.4)
    fig.savefig(os.path.join(output_dir, "delta_f1_by_missing_rate.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_output_dir(requested_output_dir, summary_path):
    if requested_output_dir == "result/figures":
        return str(_common_parent(summary_path) / "figures")
    return requested_output_dir


def _resolve_analysis_dir(requested_analysis_dir, summary_path):
    if requested_analysis_dir == "result/analysis":
        return str(_common_parent(summary_path) / "analysis")
    return requested_analysis_dir


def run(summary_path, analysis_dir, output_dir, top_k):
    summary_path = _resolve_default_summary_input(summary_path)
    resolved_summary_path = _resolve_latest_result_path(summary_path, "run_summary")
    if resolved_summary_path is None:
        raise FileNotFoundError(f"No run_summary file found for {summary_path}")
    output_dir = _resolve_output_dir(output_dir, resolved_summary_path)
    analysis_dir = _resolve_analysis_dir(analysis_dir, resolved_summary_path)
    _ensure_output_dir(output_dir)
    summary = _load_summary(resolved_summary_path)
    corr_df = _load_latest_prefixed_csv(analysis_dir, "corr_m1_to_m2")

    _save_primary_scatter(summary, output_dir)
    _save_nrmse_scatter(summary, output_dir)
    _save_top_methods(summary, output_dir, top_k)
    _save_correlation_heatmap(corr_df, output_dir)
    _save_missing_percent_profile(summary, output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="generate publication figures from results")
    parser.add_argument("--summary", default="result")
    parser.add_argument("--analysis_dir", default="result/analysis")
    parser.add_argument("--output_dir", default="result/figures")
    parser.add_argument("--top_k", type=int, default=10)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(
        summary_path=args.summary,
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
