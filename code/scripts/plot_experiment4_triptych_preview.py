from pathlib import Path
from math import ceil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from code.scripts.plotting.paper_plots import (
    DEFAULT_IMPUTER_ORDER,
    PAPER_EN_FONT_FAMILY,
    _add_top_legend,
    _build_color_map,
    _configure_plot_locale,
)


MECHANISM_ORDER = ["MCAR", "MAR", "MNAR"]
TRIPTYCH_COLUMNS = [
    ("m1_m2", "M1 -> M2"),
    ("m2_y", "M2 -> Y"),
    ("m1_y", "M1 -> Y"),
]


def _fit_png_to_size(path, target_size):
    if not target_size:
        return
    image = Image.open(path).convert("RGBA")
    if image.size == target_size:
        return
    target_width, target_height = target_size
    fitted = Image.new("RGBA", target_size, "white")
    crop_left = max((image.width - target_width) // 2, 0)
    crop_top = max((image.height - target_height) // 2, 0)
    crop_right = crop_left + min(image.width, target_width)
    crop_bottom = crop_top + min(image.height, target_height)
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    paste_left = max((target_width - cropped.width) // 2, 0)
    paste_top = max((target_height - cropped.height) // 2, 0)
    fitted.paste(cropped, (paste_left, paste_top))
    fitted.convert("RGB").save(path)


def _label_for_m1(metric, label_mode="en"):
    labels = {
        "m1_w1_mean": "Mean W1",
        "m1_corr_frob": r"$\Delta Corr$",
        "m1_fisher_delta": r"$\Delta FR$",
    }
    return labels.get(str(metric), str(metric))


def _label_for_m2(metric, label_mode="en"):
    labels = {
        "m2_inconsistency_ref_input": "SIR",
        "m2_inconsistency_retrained": "SIR retrained",
        "m2_hard_inconsistency_ref_input": "HSIR",
        "m2_hard_inconsistency_retrained": "HSIR retrained",
        "m2_margin_shift": "Margin shift",
        "m2_hard_margin_shift": "Hard margin shift",
        "m2_flip_pos": "Positive flip rate",
        "m2_flip_neg": "Negative flip rate",
    }
    return labels.get(str(metric), str(metric))


def _safe_name(value):
    text = str(value)
    safe = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    return "".join(safe).strip("_") or "unknown"


def _ordered_imputers(names):
    present = {str(name) for name in names}
    ordered = [str(name) for name in DEFAULT_IMPUTER_ORDER if str(name) in present]
    ordered.extend(sorted(name for name in present if name not in set(ordered)))
    return ordered


def _add_imputer_legend(fig, color_map, imputer_order, fig_width=None, y_anchor=0.995):
    ordered = [str(name) for name in imputer_order if str(name) in color_map]
    if not ordered:
        return 0

    ncols = min(9, len(ordered))
    legend_rows = max(1, ceil(len(ordered) / ncols))
    _add_top_legend(
        fig,
        color_map,
        item_order=ordered,
        ncols=ncols,
        y_anchor=y_anchor,
        x_anchor=0.030,
        width=0.935,
        fontsize=10.2,
        columnspacing=0.78,
        handlelength=1.55,
        handleheight=1.10,
        handletextpad=0.42,
        labelspacing=0.10,
        mode="expand",
    )
    return legend_rows


def _filter_summary(summary_df, dataset, classifier, missing_percent):
    frame = summary_df.copy()
    required = [
        "dataset",
        "classifier",
        "missing_percent",
        "missing_mechanism",
        "imputer",
        "harm_f1",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    frame = frame[
        (frame["dataset"].astype(str) == str(dataset))
        & (frame["classifier"].astype(str) == str(classifier))
        & np.isclose(pd.to_numeric(frame["missing_percent"], errors="coerce"), float(missing_percent))
    ].copy()
    return frame


def _filter_summary_for_classifier(summary_df, classifier, missing_percent):
    frame = summary_df.copy()
    required = [
        "dataset",
        "classifier",
        "missing_percent",
        "missing_mechanism",
        "imputer",
        "harm_f1",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    return frame[
        (frame["classifier"].astype(str) == str(classifier))
        & np.isclose(pd.to_numeric(frame["missing_percent"], errors="coerce"), float(missing_percent))
    ].copy()


def _aggregate_points(frame, m1_column, m2_column):
    required = ["missing_mechanism", "imputer", m1_column, m2_column, "harm_f1"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    working = frame[required].copy()
    for col in [m1_column, m2_column, "harm_f1"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=[m1_column, m2_column, "harm_f1", "missing_mechanism", "imputer"])
    if working.empty:
        return pd.DataFrame(columns=["missing_mechanism", "imputer", m1_column, m2_column, "harm_f1", "runs"])

    return (
        working.groupby(["missing_mechanism", "imputer"], as_index=False, dropna=False)
        .agg(
            **{
                m1_column: (m1_column, "mean"),
                m2_column: (m2_column, "mean"),
                "harm_f1": ("harm_f1", "mean"),
                "runs": ("imputer", "size"),
            }
        )
        .sort_values(["missing_mechanism", "imputer"])
    )


def _aggregate_cross_dataset_points(frame, m1_column, m2_column):
    required = ["dataset", "imputer", m1_column, m2_column, "harm_f1"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    working = frame[required].copy()
    for col in [m1_column, m2_column, "harm_f1"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=["dataset", "imputer", m1_column, m2_column, "harm_f1"])
    if working.empty:
        return pd.DataFrame(columns=["dataset", "imputer", m1_column, m2_column, "harm_f1", "runs"])

    return (
        working.groupby(["dataset", "imputer"], as_index=False, dropna=False)
        .agg(
            **{
                m1_column: (m1_column, "mean"),
                m2_column: (m2_column, "mean"),
                "harm_f1": ("harm_f1", "mean"),
                "runs": ("imputer", "size"),
            }
        )
        .sort_values(["dataset", "imputer"])
    )


def _available_mechanisms(points):
    present = {str(item) for item in points["missing_mechanism"].dropna().unique()} if not points.empty else set()
    ordered = [item for item in MECHANISM_ORDER if item in present]
    ordered.extend(sorted(item for item in present if item not in set(ordered)))
    return ordered or MECHANISM_ORDER


def _axis_pair(points, relation_key, m1_column, m2_column):
    if relation_key == "m1_m2":
        return points[m1_column], points[m2_column], _label_for_m1(m1_column), _label_for_m2(m2_column)
    if relation_key == "m2_y":
        return points[m2_column], points["harm_f1"], _label_for_m2(m2_column), "Y: Harm F1"
    return points[m1_column], points["harm_f1"], _label_for_m1(m1_column), "Y: Harm F1"


def _pearson_corr_text(x, y):
    numeric = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    if len(numeric) < 2 or numeric["x"].nunique() < 2 or numeric["y"].nunique() < 2:
        return "Pearson r = n/a"
    return f"Pearson r = {numeric['x'].corr(numeric['y']):.3f}"


def _panel_title(prefix, relation_title, x, y):
    return f"{prefix} {relation_title}\n{_pearson_corr_text(x, y)}"


def _style_triptych_axis(ax):
    ax.grid(True, color="#f0f0f0", linewidth=0.45, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#9a9a9a")
        spine.set_linewidth(0.70)
    ax.tick_params(axis="both", labelsize=8.6, width=0.70, length=2.2, pad=1.2)


def _draw_panel(ax, panel_points, relation_key, m1_column, m2_column, color_map):
    x, y, x_label, y_label = _axis_pair(panel_points, relation_key, m1_column, m2_column)
    if panel_points.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = [color_map.get(str(name), "#4c78a8") for name in panel_points["imputer"]]
        ax.scatter(x, y, s=34, c=colors, edgecolors="none", linewidths=0.0, alpha=0.95)
        if len(panel_points) >= 2 and x.nunique(dropna=True) > 1:
            coeff = np.polyfit(x.astype(float), y.astype(float), deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#555555", linewidth=1.05, alpha=0.90)
    _style_triptych_axis(ax)
    ax.set_xlabel(x_label, fontsize=9.0, fontweight="bold", labelpad=2)
    ax.set_ylabel(y_label, fontsize=9.0, fontweight="bold", labelpad=2)
    return x, y


def _save_triptych(points, plot_path, dataset, classifier, missing_percent, m1_column, m2_column, label_mode):
    _configure_plot_locale(label_mode)
    plt.rcParams["font.family"] = PAPER_EN_FONT_FAMILY
    target_size = Image.open(plot_path).size if Path(plot_path).exists() else None
    mechanisms = _available_mechanisms(points)
    imputer_order = _ordered_imputers(points["imputer"].astype(str).unique() if not points.empty else [])
    fig_width = 10.9
    legend_cols = min(9, max(len(imputer_order), 1))
    legend_rows = max(1, ceil(max(len(imputer_order), 1) / legend_cols))
    fig_height = max(8.70, 1.86 * len(mechanisms) + 0.52 * legend_rows)
    fig, axes = plt.subplots(len(mechanisms), len(TRIPTYCH_COLUMNS), figsize=(fig_width, fig_height), squeeze=False)
    color_map = _build_color_map(imputer_order, DEFAULT_IMPUTER_ORDER)

    for row_idx, mechanism in enumerate(mechanisms):
        panel_points = points[points["missing_mechanism"].astype(str) == str(mechanism)]
        for col_idx, (relation_key, relation_title) in enumerate(TRIPTYCH_COLUMNS):
            ax = axes[row_idx][col_idx]
            x, y = _draw_panel(ax, panel_points, relation_key, m1_column, m2_column, color_map)
            panel_prefix = f"({_subplot_letter(row_idx * len(TRIPTYCH_COLUMNS) + col_idx)}) {mechanism} |"
            ax.set_title(
                _panel_title(panel_prefix, relation_title, x, y),
                fontsize=9.6,
                fontweight="bold",
                pad=2,
            )

    _add_imputer_legend(fig, color_map, imputer_order, y_anchor=0.995)
    legend_height_fraction = (0.34 * legend_rows + 0.15) / fig_height
    top_rect = min(0.875, max(0.68, 0.975 - legend_height_fraction))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.065, top=top_rect, hspace=0.38, wspace=0.22)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    _fit_png_to_size(plot_path, target_size)


def _subplot_letter(index):
    label = ""
    value = int(index)
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def generate_triptych_artifacts(
    summary_df,
    output_dir,
    dataset,
    classifier,
    missing_percent,
    m1_column,
    m2_column,
    label_mode="en",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _filter_summary(summary_df, dataset, classifier, missing_percent)
    points = _aggregate_points(frame, m1_column, m2_column)

    dataset_tag = _safe_name(dataset)
    classifier_tag = _safe_name(classifier)
    mp_tag = _safe_name(f"{float(missing_percent):g}")
    m1_tag = _safe_name(m1_column)
    m2_tag = _safe_name(m2_column)
    stem = f"{dataset_tag}_{classifier_tag}_mp{mp_tag}"

    points_path = output_dir / f"{stem}_points_{m1_tag}_{m2_tag}.csv"
    plot_path = output_dir / f"{stem}_all_mechanisms_triptych_{m1_tag}_{m2_tag}.png"
    points.to_csv(points_path, index=False)
    _save_triptych(points, plot_path, dataset, classifier, missing_percent, m1_column, m2_column, label_mode)
    return points_path, plot_path


def _draw_cross_dataset_panel(ax, panel_points, relation_key, m1_column, m2_column, color_map):
    x, y, x_label, y_label = _axis_pair(panel_points, relation_key, m1_column, m2_column)
    if panel_points.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = [color_map.get(str(name), "#4c78a8") for name in panel_points["imputer"]]
        ax.scatter(x, y, s=34, c=colors, edgecolors="none", linewidths=0.0, alpha=0.95)
        if len(panel_points) >= 2 and x.nunique(dropna=True) > 1:
            coeff = np.polyfit(x.astype(float), y.astype(float), deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#555555", linewidth=1.05, alpha=0.90)
    _style_triptych_axis(ax)
    ax.set_xlabel(x_label, fontsize=9.0, fontweight="bold", labelpad=2)
    ax.set_ylabel(y_label, fontsize=9.0, fontweight="bold", labelpad=2)
    return x, y


def _save_cross_dataset_triptych(points, plot_path, classifier, missing_percent, m1_column, m2_column, label_mode):
    _configure_plot_locale(label_mode)
    plt.rcParams["font.family"] = PAPER_EN_FONT_FAMILY
    target_size = Image.open(plot_path).size if Path(plot_path).exists() else None
    datasets = sorted(points["dataset"].dropna().astype(str).unique()) if not points.empty else []
    rows = max(len(datasets), 1)
    imputer_order = _ordered_imputers(points["imputer"].astype(str).unique() if not points.empty else [])
    fig_width = 10.9
    legend_cols = min(9, max(len(imputer_order), 1))
    legend_rows = max(1, ceil(max(len(imputer_order), 1) / legend_cols))
    fig_height = max(8.70, 1.86 * rows + 0.52 * legend_rows)
    fig, axes = plt.subplots(rows, len(TRIPTYCH_COLUMNS), figsize=(fig_width, fig_height), squeeze=False)
    color_map = _build_color_map(imputer_order, DEFAULT_IMPUTER_ORDER)

    if not datasets:
        datasets = ["No data"]
    for row_idx, dataset in enumerate(datasets):
        panel_points = points[points["dataset"].astype(str) == str(dataset)] if not points.empty else points
        for col_idx, (relation_key, relation_title) in enumerate(TRIPTYCH_COLUMNS):
            ax = axes[row_idx][col_idx]
            x, y = _draw_cross_dataset_panel(ax, panel_points, relation_key, m1_column, m2_column, color_map)
            panel_prefix = f"({_subplot_letter(row_idx * len(TRIPTYCH_COLUMNS) + col_idx)}) {str(dataset).replace('_', ' ')} |"
            ax.set_title(
                _panel_title(panel_prefix, relation_title, x, y),
                fontsize=9.6,
                fontweight="bold",
                pad=2,
            )

    _add_imputer_legend(fig, color_map, imputer_order, y_anchor=0.995)
    legend_height_fraction = (0.34 * legend_rows + 0.15) / fig_height
    top_rect = min(0.875, max(0.68, 0.975 - legend_height_fraction))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.065, top=top_rect, hspace=0.38, wspace=0.22)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    _fit_png_to_size(plot_path, target_size)


def generate_cross_dataset_triptych_artifacts(
    summary_df,
    output_dir,
    classifier,
    missing_percent,
    m1_column,
    m2_column,
    label_mode="en",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _filter_summary_for_classifier(summary_df, classifier, missing_percent)
    points = _aggregate_cross_dataset_points(frame, m1_column, m2_column)

    classifier_tag = _safe_name(classifier)
    mp_tag = _safe_name(f"{float(missing_percent):g}")
    m1_tag = _safe_name(m1_column)
    m2_tag = _safe_name(m2_column)
    stem = f"all_datasets_{classifier_tag}_mp{mp_tag}"

    points_path = output_dir / f"{stem}_points_{m1_tag}_{m2_tag}.csv"
    plot_path = output_dir / f"{stem}_triptych_{m1_tag}_{m2_tag}.png"
    points.to_csv(points_path, index=False)
    _save_cross_dataset_triptych(points, plot_path, classifier, missing_percent, m1_column, m2_column, label_mode)
    return points_path, plot_path
