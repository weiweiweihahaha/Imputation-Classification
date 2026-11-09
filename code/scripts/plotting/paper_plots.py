import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from code.config.settings import get_default_config


DEFAULT_IMPUTER_ORDER = [str(name) for name in get_default_config()["imputation_method"]]
DEFAULT_CLASSIFIER_ORDER = [str(name) for name in get_default_config()["class_method"]]
PLOT_EXCLUDED_IMPUTERS = set()

# Central paper-figure style knobs. Experiments 1 and 2 share these values so
# their PNG outputs stay visually consistent after batch regeneration.
PAPER_AXIS_LABEL_FONTSIZE = 20.2
PAPER_TICK_FONTSIZE = 17.2
PAPER_PANEL_TITLE_FONTSIZE = 21.0
PAPER_ROW_LABEL_FONTSIZE = 25.8
PAPER_LEGEND_FONTSIZE = 15.6
PAPER_LEGEND_TITLE_FONTSIZE = 18.0
PAPER_Y_TICK_FONTSIZE_COMPACT = 16.2
PAPER_SCATTER_SIZE = 58
PAPER_BOX_WIDTH = 0.94
PAPER_EN_FONT_FAMILY = ["Times New Roman", "Times", "DejaVu Serif"]
PAPER_ZH_FONT_FAMILY = ["Songti SC"]
PAPER_ZH_MIN_FONTSIZE = 16.0
# Shared full-width canvas for Experiment 1 and no-real-missing Experiment 2.
PAPER_EXPERIMENT_GRID_WIDTH_PT = 1120.0
PAPER_EXPERIMENT1_WIDTH_PT = 1400.0
PAPER_PT_PER_INCH = 72.27


def _ordered_names(names, default_order, preferred_order=None, include_all_preferred=False):
    seen = {str(name) for name in names}
    base_order = [str(name) for name in (preferred_order if preferred_order is not None else default_order)]
    ordered = []
    seen_ordered = set()
    for name in base_order:
        if include_all_preferred or name in seen:
            if name not in seen_ordered:
                ordered.append(name)
                seen_ordered.add(name)
    ordered.extend(sorted(name for name in seen if name not in seen_ordered))
    return ordered


def _is_zh(label_mode):
    return str(label_mode).strip().lower() == "zh"


def _configure_plot_locale(label_mode="en"):
    # Use the requested manuscript font family before any artists are created.
    if _is_zh(label_mode):
        plt.rcParams["font.family"] = PAPER_ZH_FONT_FAMILY
        plt.rcParams["font.serif"] = PAPER_ZH_FONT_FAMILY
        plt.rcParams["font.sans-serif"] = PAPER_ZH_FONT_FAMILY
        plt.rcParams["font.size"] = PAPER_ZH_MIN_FONTSIZE
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
    else:
        plt.rcParams["font.family"] = PAPER_EN_FONT_FAMILY
        plt.rcParams["font.serif"] = PAPER_EN_FONT_FAMILY
        plt.rcParams["font.sans-serif"] = PAPER_EN_FONT_FAMILY
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def _apply_paper_text_style(fig, label_mode="en"):
    # Re-apply font settings to all text artists; some Matplotlib helpers create
    # text after rcParams are set and otherwise keep fallback defaults.
    family = PAPER_ZH_FONT_FAMILY if _is_zh(label_mode) else PAPER_EN_FONT_FAMILY
    for text in fig.findobj(match=Text):
        text.set_fontfamily(family)
        text.set_color("black")
        text.set_fontweight("bold")
        if _is_zh(label_mode):
            text.set_fontsize(max(float(text.get_fontsize()), PAPER_ZH_MIN_FONTSIZE))


def _apply_bold_black_text(fig):
    _apply_paper_text_style(fig, label_mode="en")


def _save_figure(fig, output_path, dpi=300, bbox_inches="tight"):
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


def _no_data_text(label_mode="en"):
    return "无有效数据" if _is_zh(label_mode) else "No valid data"


def _mechanism_row_title_map(label_mode="en"):
    return {
        "MCAR": "MCAR",
        "MAR": "MAR",
        "MNAR": "MNAR",
    }


def _ensure_axes_grid(axes, rows, cols):
    return np.asarray(axes, dtype=object).reshape(rows, cols)


def _build_color_map(names, preferred_order=None, include_all_preferred=False):
    ordered = _ordered_names(
        names,
        default_order=DEFAULT_IMPUTER_ORDER,
        preferred_order=preferred_order,
        include_all_preferred=include_all_preferred,
    )
    if not ordered:
        return {}

    # Use a multi-hue continuous palette so many imputers can still occupy
    # distinct color regions instead of collapsing into one hue family.
    cmap = plt.get_cmap("turbo")
    if len(ordered) == 1:
        positions = [0.55]
    else:
        positions = np.linspace(0.05, 0.95, len(ordered))
    return {name: to_hex(cmap(position)) for name, position in zip(ordered, positions)}


def _style_reference_panel(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color="#e9e9e9", alpha=0.72, linewidth=0.42)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#626262")
        spine.set_linewidth(0.78)
    ax.tick_params(axis="both", color="#4f4f4f", width=0.75, length=2.6, pad=1.5)


def _build_marker_map(names, preferred_order=None, include_all_preferred=False):
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p", "d"]
    ordered = _ordered_names(
        names,
        default_order=DEFAULT_CLASSIFIER_ORDER,
        preferred_order=preferred_order,
        include_all_preferred=include_all_preferred,
    )
    return {name: markers[idx % len(markers)] for idx, name in enumerate(ordered)}


def _style_y_tick_labels_outside_vertical(ax, pad=1, rotation=90, ha="center", va="center"):
    # Keep y tick labels outside the panel while allowing panel-specific
    # rotation control for dense grids.
    ax.tick_params(axis="y", direction="out", pad=pad)
    for label in ax.get_yticklabels():
        label.set_rotation(rotation)
        label.set_ha(ha)
        label.set_va(va)
        label.set_fontsize(PAPER_Y_TICK_FONTSIZE_COMPACT)


def _style_boxplot(
    ax,
    labels,
    values,
    colors=None,
    positions=None,
    total_slots=None,
    show_x_tick_labels=True,
    x_tick_rotation=28,
    x_tick_fontsize=9,
    empty_text="No valid data",
):
    if not values:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
        if total_slots:
            ax.set_xlim(0.5, total_slots + 0.5)
        return

    box = ax.boxplot(
        values,
        labels=labels,
        positions=positions,
        patch_artist=True,
        widths=PAPER_BOX_WIDTH,
        showfliers=False,
        medianprops={"color": "#2f2f2f", "linewidth": 2.35},
        boxprops={"facecolor": "#f4f4f4", "edgecolor": "none", "linewidth": 0.0},
        whiskerprops={"color": "#3d3d3d", "linewidth": 1.05},
        capprops={"color": "#3d3d3d", "linewidth": 1.05},
    )
    for idx, patch in enumerate(box["boxes"]):
        facecolor = "#f7f7f7"
        if colors and idx < len(colors):
            facecolor = colors[idx]
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("none")
        patch.set_linewidth(0.0)
        patch.set_alpha(0.92)
    ax.tick_params(axis="y", labelsize=PAPER_TICK_FONTSIZE - 2.4, width=0.75)
    _style_y_tick_labels_outside_vertical(ax, pad=1)
    if show_x_tick_labels:
        ax.tick_params(axis="x", pad=2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(x_tick_rotation)
            tick.set_fontsize(x_tick_fontsize)
            tick.set_ha("center" if abs(x_tick_rotation - 90) <= 5 else "right")
            tick.set_va("top")
    else:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.set_xticklabels([])
    if total_slots:
        ax.set_xlim(0.5, total_slots + 0.5)


def _ordered_imputers(names, preferred_order=None, include_all_preferred=False):
    return _ordered_names(
        names,
        default_order=DEFAULT_IMPUTER_ORDER,
        preferred_order=preferred_order,
        include_all_preferred=include_all_preferred,
    )


def _ordered_classifiers(names, preferred_order=None, include_all_preferred=False):
    return _ordered_names(
        names,
        default_order=DEFAULT_CLASSIFIER_ORDER,
        preferred_order=preferred_order,
        include_all_preferred=include_all_preferred,
    )


def _exclude_plot_imputers(frame, imputer_col="imputer"):
    if imputer_col not in frame.columns:
        return frame
    return frame[~frame[imputer_col].astype(str).isin(PLOT_EXCLUDED_IMPUTERS)].copy()


def _prepare_panel_boxplot_data(
    panel_frame,
    metric,
    imputer_col,
    sort_by="fixed",
    show_counts=True,
    color_map=None,
    global_imputer_order=None,
):
    grouped = {}
    for imputer, group in panel_frame.groupby(imputer_col):
        values = group[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        grouped[str(imputer)] = {
            "imputer": str(imputer),
            "values": values,
            "median": float(np.median(values)),
            "n": int(len(values)),
        }

    if sort_by == "median":
        ordered_names = [
            item["imputer"]
            for item in sorted(grouped.values(), key=lambda item: (item["median"], item["imputer"]))
        ]
    elif sort_by == "name":
        ordered_names = sorted(grouped)
    else:
        base_order = global_imputer_order or _ordered_imputers(grouped.keys())
        ordered_names = [name for name in base_order if name in grouped]

    labels = []
    values = []
    colors = []
    positions = []
    slot_order = global_imputer_order or ordered_names
    slot_index = {name: idx + 1 for idx, name in enumerate(slot_order)}
    for name in ordered_names:
        item = grouped[name]
        label = item["imputer"]
        if show_counts:
            label = f"{label}\n(n={item['n']})"
        labels.append(label)
        values.append(item["values"])
        colors.append(color_map.get(item["imputer"], "#f7f7f7") if color_map else "#f7f7f7")
        positions.append(slot_index[item["imputer"]])
    return labels, values, colors, positions


def _normalize_mechanism_order(mechanisms):
    preferred = ["MCAR", "MAR", "MNAR"]
    present = [item for item in preferred if item in mechanisms]
    remaining = sorted(item for item in mechanisms if item not in preferred)
    return present + remaining


def _compute_panel_ylim(values):
    if not values:
        return None
    merged = np.concatenate(values).astype(float)
    if merged.size == 0:
        return None
    if merged.size == 1:
        padding = max(abs(float(merged[0])) * 0.10, 0.03)
        return float(merged[0]) - padding, float(merged[0]) + padding
    raw_min = float(np.min(merged))
    raw_max = float(np.max(merged))
    span = raw_max - raw_min
    if span <= 1e-12:
        padding = max(abs(raw_max) * 0.08, 0.03)
        return raw_min - padding, raw_max + padding
    q1, q3 = np.percentile(merged, [25, 75])
    iqr = q3 - q1
    upper_padding = max(span * 0.08, iqr * 0.28, abs(raw_max) * 0.02, 0.02)
    lower_padding = max(span * 0.05, iqr * 0.20, abs(raw_min) * 0.02, 0.015)
    return raw_min - lower_padding, raw_max + upper_padding


def _compute_numeric_limits(values, lower_padding_ratio=0.06, upper_padding_ratio=0.08, min_padding=0.02):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return None
    if numeric.size == 1:
        padding = max(abs(float(numeric[0])) * upper_padding_ratio, min_padding)
        return float(numeric[0]) - padding, float(numeric[0]) + padding
    raw_min = float(np.min(numeric))
    raw_max = float(np.max(numeric))
    span = raw_max - raw_min
    if span <= 1e-12:
        padding = max(abs(raw_max) * upper_padding_ratio, min_padding)
        return raw_min - padding, raw_max + padding
    lower_padding = max(span * lower_padding_ratio, min_padding)
    upper_padding = max(span * upper_padding_ratio, min_padding)
    return raw_min - lower_padding, raw_max + upper_padding


def _compute_dataset_scatter_limits(frame, y_col):
    if frame.empty or "dataset" not in frame.columns:
        return {}

    limits = {}
    for dataset, dataset_frame in frame.groupby("dataset", dropna=False):
        dataset_name = str(dataset)
        x_limits = _compute_numeric_limits(dataset_frame["nrmse"], lower_padding_ratio=0.06, upper_padding_ratio=0.08)
        y_limits = _compute_numeric_limits(dataset_frame[y_col], lower_padding_ratio=0.08, upper_padding_ratio=0.10)
        limits[dataset_name] = {
            "x": x_limits,
            "y": y_limits,
        }
    return limits


def _add_top_legend(
    fig,
    color_map,
    item_order=None,
    ncols=6,
    y_anchor=1.01,
    x_anchor=0.035,
    width=0.93,
    fontsize=7.5,
    columnspacing=0.8,
    handlelength=0.8,
    handleheight=0.7,
    handletextpad=0.25,
    labelspacing=0.25,
    mode=None,
):
    # Experiment 1 has many imputers; a single compact top legend preserves
    # panel space while keeping color order stable.
    ordered_names = item_order or list(color_map.keys())
    handles = [
        Patch(facecolor=color, edgecolor="#333333", linewidth=0.7, alpha=0.86, label=name)
        for name in ordered_names
        if name in color_map
        for color in [color_map[name]]
    ]
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(x_anchor, y_anchor, width, 0.001) if mode == "expand" else (0.5, y_anchor),
        ncol=min(ncols, max(1, len(handles))),
        mode=mode,
        frameon=False,
        columnspacing=columnspacing,
        handlelength=handlelength,
        handleheight=handleheight,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
        fontsize=fontsize,
    )


def _wrap_dataset_grid(dataset_names, max_datasets_per_row=4):
    dataset_names = list(dataset_names)
    if not dataset_names:
        return []
    per_row = max(1, min(max_datasets_per_row, len(dataset_names)))
    return [dataset_names[idx : idx + per_row] for idx in range(0, len(dataset_names), per_row)]


def _suggest_legend_ncols(item_count, fig_width, approx_item_width=1.2):
    if item_count <= 0:
        return 1
    max_cols_from_width = max(1, int(fig_width / approx_item_width))
    return min(item_count, max_cols_from_width)


def _legend_grid(item_count, fig_width, target_rows=4, approx_item_width=1.45, max_cols=None):
    if item_count <= 0:
        return 1, 1
    usable_width = max(fig_width * 0.90, 1.0)
    width_limited_cols = max(1, int(usable_width / approx_item_width))
    desired_cols = max(1, ceil(item_count / max(target_rows, 1)))
    col_limit = width_limited_cols if max_cols is None else min(width_limited_cols, max_cols)
    cols = max(1, min(item_count, desired_cols, col_limit))
    rows = max(1, ceil(item_count / cols))
    return cols, rows


def _top_margin_for_legend(fig_height, legend_rows, row_height=0.35, base_height=0.56, min_top=0.48, max_top=0.86):
    reserved_height = base_height + row_height * max(legend_rows, 1)
    return min(max_top, max(min_top, 1.0 - reserved_height / max(fig_height, 1.0)))


def _compute_mechanism_chunk_row(mechanism_idx, dataset_chunk_idx, total_dataset_chunks):
    return mechanism_idx * total_dataset_chunks + dataset_chunk_idx


def _format_dataset_title(dataset_name):
    dataset_name = str(dataset_name)
    display_names = {
        "college_scorecard": "college",
        "nycrestaurants": "restaurant",
    }
    if dataset_name in display_names:
        return display_names[dataset_name]
    return dataset_name.replace("_", " ")


def _format_vertical_metric_label(metric_label):
    if "W_1" in str(metric_label) or "W1" in str(metric_label):
        return r"$\overline{W_1}$"
    return str(metric_label)


def _format_stacked_mechanism_label(mechanism):
    return "\n".join(str(mechanism))


def _subplot_label_from_index(index):
    # Excel-style lowercase labels: a, b, ..., z, aa, ab, ...
    label = ""
    value = int(index)
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def save_mechanism_dataset_boxplot_grid(
    summary_frame,
    metric,
    output_path,
    dataset_col="dataset",
    mechanism_col="missing_mechanism",
    imputer_col="imputer",
    sort_by="median",
    show_counts=False,
    show_x_tick_labels=False,
    x_tick_rotation=90,
    x_tick_fontsize=8,
    label_mode="en",
    metric_label_map=None,
    force_zero_ymin=False,
    imputer_order=None,
    show_axis_labels_every_panel=False,
    square_panels=False,
    compact_canvas=False,
):
    _configure_plot_locale(label_mode=label_mode)
    required_cols = [dataset_col, mechanism_col, imputer_col, metric]
    missing_cols = [col for col in required_cols if col not in summary_frame.columns]
    if missing_cols:
        raise ValueError(f"summary_frame missing required columns: {missing_cols}")

    working = summary_frame[required_cols].dropna().copy()
    working = _exclude_plot_imputers(working, imputer_col=imputer_col)
    dataset_names = sorted(working[dataset_col].astype(str).unique())
    mechanism_names = _normalize_mechanism_order(sorted(working[mechanism_col].astype(str).unique()))
    present_imputer_order = _ordered_imputers(
        working[imputer_col].astype(str).unique(),
        preferred_order=imputer_order,
    )
    global_imputer_order = _ordered_imputers(
        working[imputer_col].astype(str).unique(),
        preferred_order=imputer_order,
        include_all_preferred=imputer_order is not None,
    )
    color_map = _build_color_map(global_imputer_order)

    if not dataset_names or not mechanism_names:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.text(0.5, 0.5, _no_data_text(label_mode), ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _apply_paper_text_style(fig, label_mode=label_mode)
        _save_figure(fig, output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    dataset_chunks = _wrap_dataset_grid(dataset_names, max_datasets_per_row=4)
    outer_rows = len(dataset_chunks)
    outer_cols = max(len(chunk) for chunk in dataset_chunks)
    rows_per_group = len(mechanism_names)
    if compact_canvas:
        panel_size = 3.05
        fig_width = max(outer_cols * panel_size + 1.25, 8.0)
        row_height = panel_size
        base_height = 1.25
        min_height = 5.1
    else:
        fig_width = PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH
        row_height = 1.60 if show_axis_labels_every_panel else 1.44
        base_height = 1.94 if show_axis_labels_every_panel else 1.80
        min_height = 11.4 if show_axis_labels_every_panel else 10.8
    fig_height = max(row_height * outer_rows * rows_per_group + base_height, min_height)
    legend_item_count = len(present_imputer_order)
    legend_cols = min(12, max(1, legend_item_count))
    legend_rows = max(1, ceil(legend_item_count / legend_cols))
    # Leave just enough room for the top legend; dataset titles sit below it.
    if compact_canvas:
        top_margin = _top_margin_for_legend(
            fig_height,
            legend_rows,
            row_height=0.34,
            base_height=0.54,
            min_top=0.52,
            max_top=0.84,
        )
    else:
        top_margin = _top_margin_for_legend(
            fig_height,
            legend_rows,
            row_height=0.18,
            base_height=0.50,
            min_top=0.86,
            max_top=0.87,
        )
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    outer_grid = fig.add_gridspec(outer_rows, outer_cols)
    axes = np.empty((outer_rows, outer_cols, rows_per_group), dtype=object)
    group_hspace = 0.14
    group_wspace = 0.40
    inner_hspace = 0.20 if show_axis_labels_every_panel else 0.14
    bottom_margin = 0.055 if show_x_tick_labels else 0.035
    if show_axis_labels_every_panel:
        bottom_margin = max(bottom_margin, 0.11)
    # The metric label stays on the canvas edge. Mechanism labels are attached
    # beside every panel so each subplot remains self-identifying.
    left_margin = 0.062 if show_axis_labels_every_panel else 0.080
    panel_mechanism_label_x = -0.255 if show_axis_labels_every_panel else -0.112
    metric_label_x = 0.018
    fig.subplots_adjust(
        top=top_margin,
        bottom=bottom_margin,
        left=left_margin,
        right=0.965,
        hspace=group_hspace,
        wspace=group_wspace,
    )
    for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
        for col_idx in range(outer_cols):
            if col_idx >= len(dataset_chunk):
                for mechanism_idx in range(rows_per_group):
                    axes[chunk_idx, col_idx, mechanism_idx] = fig.add_subplot(outer_grid[chunk_idx, col_idx])
                    axes[chunk_idx, col_idx, mechanism_idx].set_axis_off()
                continue
            inner_grid = outer_grid[chunk_idx, col_idx].subgridspec(rows_per_group, 1, hspace=inner_hspace)
            for mechanism_idx in range(rows_per_group):
                axes[chunk_idx, col_idx, mechanism_idx] = fig.add_subplot(inner_grid[mechanism_idx, 0])

    mechanism_title_map = _mechanism_row_title_map(label_mode=label_mode)
    metric_label = _format_vertical_metric_label((metric_label_map or {}).get(metric, metric))
    metric_label_is_stacked = "\n" in str(metric_label)
    metric_label_rotation = 0 if metric_label_is_stacked else 90
    metric_label_fontsize = PAPER_ZH_MIN_FONTSIZE if _is_zh(label_mode) else PAPER_AXIS_LABEL_FONTSIZE
    canvas_metric_label_fontsize = PAPER_AXIS_LABEL_FONTSIZE + (1.0 if _is_zh(label_mode) else 2.8)
    mechanism_label_fontsize = PAPER_ROW_LABEL_FONTSIZE - (7.4 if _is_zh(label_mode) else 7.0)
    panel_xlabel = "填补方法" if _is_zh(label_mode) else "Imputers"
    dataset_ylim_map = {}
    # Share y-limits within each dataset column so MCAR/MAR/MNAR panels can be
    # compared without visual scale changes.
    for dataset in dataset_names:
        dataset_values = []
        for mechanism in mechanism_names:
            panel_frame = working[
                (working[dataset_col].astype(str) == dataset)
                & (working[mechanism_col].astype(str) == mechanism)
            ]
            _, values, _, _ = _prepare_panel_boxplot_data(
                panel_frame,
                metric,
                imputer_col,
                sort_by=sort_by,
                show_counts=show_counts,
                color_map=color_map,
                global_imputer_order=global_imputer_order,
            )
            dataset_values.extend(values)
        limits = _compute_panel_ylim(dataset_values)
        if limits is not None and force_zero_ymin:
            limits = (0.0, limits[1])
        dataset_ylim_map[str(dataset)] = limits

    for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
        for col_idx in range(outer_cols):
            for mechanism_idx, mechanism in enumerate(mechanism_names):
                ax = axes[chunk_idx, col_idx, mechanism_idx]
                if col_idx >= len(dataset_chunk):
                    ax.set_axis_off()
                    continue
                dataset = dataset_chunk[col_idx]
                panel_frame = working[
                    (working[dataset_col].astype(str) == dataset)
                    & (working[mechanism_col].astype(str) == mechanism)
                ]

                labels, values, colors, positions = _prepare_panel_boxplot_data(
                    panel_frame,
                    metric,
                    imputer_col,
                    sort_by=sort_by,
                    show_counts=show_counts,
                    color_map=color_map,
                    global_imputer_order=global_imputer_order,
                )
                _style_boxplot(
                    ax,
                    labels,
                    values,
                    colors=colors,
                    positions=positions,
                    total_slots=len(global_imputer_order),
                    show_x_tick_labels=show_x_tick_labels,
                    x_tick_rotation=x_tick_rotation,
                    x_tick_fontsize=x_tick_fontsize,
                    empty_text=_no_data_text(label_mode),
                )
                if square_panels:
                    ax.set_box_aspect(1)
                _style_reference_panel(ax, grid_axis="y")
                _style_y_tick_labels_outside_vertical(ax, pad=1, rotation=0, ha="right", va="center")
                ax.text(
                    panel_mechanism_label_x,
                    0.5,
                    _format_stacked_mechanism_label(mechanism_title_map.get(mechanism, str(mechanism))),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=mechanism_label_fontsize,
                    fontweight="bold",
                    linespacing=0.82,
                    clip_on=False,
                )
                limits = dataset_ylim_map.get(str(dataset))
                if limits is not None:
                    ax.set_ylim(*limits)
                if mechanism_idx == 0:
                    ax.set_title(
                        _format_dataset_title(dataset),
                        fontsize=PAPER_PANEL_TITLE_FONTSIZE,
                        fontweight="bold",
                        pad=4,
                    )

                if show_axis_labels_every_panel:
                    ax.set_xlabel(
                        panel_xlabel,
                        fontsize=PAPER_AXIS_LABEL_FONTSIZE - 3.2,
                        fontweight="bold",
                        labelpad=2,
                    )
                    ax.set_ylabel(
                        metric_label,
                        fontsize=metric_label_fontsize,
                        fontweight="bold",
                        rotation=metric_label_rotation,
                        labelpad=8 if metric_label_is_stacked else 4,
                    )
                    if metric_label_is_stacked:
                        ax.yaxis.set_label_coords(-0.105, 0.5)
                    else:
                        ax.yaxis.set_label_coords(-0.100, 0.5)
                else:
                    ax.set_ylabel("")

                if not show_axis_labels_every_panel and (chunk_idx != outer_rows - 1 or mechanism_idx != rows_per_group - 1 or not show_x_tick_labels):
                    ax.tick_params(axis="x", labelbottom=False)
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(
                        panel_xlabel,
                        fontsize=PAPER_AXIS_LABEL_FONTSIZE - 3.2,
                        fontweight="bold",
                        labelpad=2,
                    )

    subplot_label_idx = 0
    for col_idx in range(outer_cols):
        for chunk_idx in range(outer_rows):
            for mechanism_idx in range(rows_per_group):
                ax = axes[chunk_idx, col_idx, mechanism_idx]
                if not ax.axison:
                    continue
                ax.text(
                    0.018,
                    0.965,
                    f"({_subplot_label_from_index(subplot_label_idx)})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=PAPER_PANEL_TITLE_FONTSIZE - 4.4,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
                    zorder=10,
                )
                subplot_label_idx += 1

    for chunk_idx in range(len(dataset_chunks)):
        chunk_top_bbox = axes[chunk_idx, 0, 0].get_position()
        chunk_bottom_bbox = axes[chunk_idx, 0, rows_per_group - 1].get_position()
        if not show_axis_labels_every_panel:
            fig.text(
                metric_label_x,
                0.5 * (chunk_top_bbox.y1 + chunk_bottom_bbox.y0),
                metric_label,
                ha="center",
                va="center",
                rotation=metric_label_rotation,
                fontsize=canvas_metric_label_fontsize,
                fontweight="bold",
            )

    _add_top_legend(
        fig,
        color_map,
        item_order=present_imputer_order,
        ncols=legend_cols,
        y_anchor=0.995,
        x_anchor=0.006,
        width=0.959,
        fontsize=PAPER_LEGEND_FONTSIZE - 1.0,
        columnspacing=0.92,
        handlelength=1.80,
        handleheight=1.08,
        handletextpad=0.46,
        labelspacing=0.20,
        mode="expand",
    )
    _apply_paper_text_style(fig, label_mode=label_mode)
    _save_figure(fig, output_path, dpi=300, bbox_inches=None)
    plt.close(fig)


def save_nrmse_delta_scatter(
    subset,
    output_path,
    fixed_figsize=None,
    y_col="delta_f1",
    y_label=r"$\Delta$F1",
    label_mode="en",
    axis_source_subset=None,
    imputer_order=None,
    classifier_order=None,
    mechanism_title_style="default",
    show_axis_labels_every_panel=True,
    compact_canvas=False,
    square_panels=False,
    real_world_subset=None,
):
    _configure_plot_locale(label_mode=label_mode)
    required_cols = ["nrmse", y_col, "dataset", "missing_mechanism", "imputer", "classifier"]
    missing_cols = [col for col in required_cols if col not in subset.columns]
    if missing_cols:
        raise ValueError(f"subset missing required columns: {missing_cols}")

    working = subset[required_cols].dropna().copy()
    working = _exclude_plot_imputers(working, imputer_col="imputer")
    real_world_working = pd.DataFrame(columns=required_cols)
    if real_world_subset is not None:
        real_world_missing_cols = [col for col in required_cols if col not in real_world_subset.columns]
        if real_world_missing_cols:
            raise ValueError(
                "real_world_subset missing required columns: "
                f"{real_world_missing_cols}"
            )
        real_world_working = real_world_subset[required_cols].dropna().copy()
        real_world_working = _exclude_plot_imputers(
            real_world_working,
            imputer_col="imputer",
        )
    legend_working = pd.concat(
        [working, real_world_working],
        ignore_index=True,
    )
    dataset_names = sorted(working["dataset"].astype(str).unique())
    real_world_dataset_names = sorted(
        real_world_working["dataset"].astype(str).unique()
    )
    mechanism_order = _normalize_mechanism_order(sorted(working["missing_mechanism"].astype(str).unique()))
    present_imputer_order = _ordered_imputers(
        legend_working["imputer"].astype(str).unique(),
        preferred_order=imputer_order,
    )
    full_imputer_order = _ordered_imputers(
        legend_working["imputer"].astype(str).unique(),
        preferred_order=imputer_order,
        include_all_preferred=imputer_order is not None,
    )
    present_classifier_order = _ordered_classifiers(
        legend_working["classifier"].astype(str).unique(),
        preferred_order=classifier_order,
    )
    full_classifier_order = _ordered_classifiers(
        legend_working["classifier"].astype(str).unique(),
        preferred_order=classifier_order,
        include_all_preferred=classifier_order is not None,
    )
    color_map = _build_color_map(full_imputer_order)
    marker_map = _build_marker_map(full_classifier_order)
    dataset_chunks = _wrap_dataset_grid(dataset_names, max_datasets_per_row=4)
    synthetic_cols = max((len(chunk) for chunk in dataset_chunks), default=1)
    has_real_world_column = bool(real_world_dataset_names)
    cols = synthetic_cols + int(has_real_world_column)
    rows_per_chunk = max(len(mechanism_order), 1)
    dataset_chunk_count = max(len(dataset_chunks), 1)
    rows = rows_per_chunk * dataset_chunk_count
    if len(real_world_dataset_names) > rows:
        raise ValueError(
            "real_world_subset has more datasets than available grid rows: "
            f"{len(real_world_dataset_names)} > {rows}"
        )
    chunk_gap_ratio = 0.0 if show_axis_labels_every_panel else 0.18
    chunk_gap_count = (
        max(dataset_chunk_count - 1, 0)
        if chunk_gap_ratio > 0
        else 0
    )
    if fixed_figsize is None:
        if compact_canvas:
            panel_size = 3.05 if show_axis_labels_every_panel else 2.75
            base_height = 2.40 if show_axis_labels_every_panel else 2.10
            figure_width = max(panel_size * cols + 1.25, 8.0)
            figure_height = max(
                panel_size * (rows + chunk_gap_ratio * chunk_gap_count) + base_height,
                5.1 if show_axis_labels_every_panel else 4.8,
            )
        else:
            row_height = 2.05 if show_axis_labels_every_panel else 1.85
            base_height = 4.45 if show_axis_labels_every_panel else 4.0
            col_width = PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH / 4.0
            figure_width = max(col_width * cols, PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH)
            figure_height = max(
                row_height * (rows + chunk_gap_ratio * chunk_gap_count) + base_height,
                17.2 if show_axis_labels_every_panel else 15.8,
            )
            if has_real_world_column and show_axis_labels_every_panel:
                figure_height = max(figure_height, 18.9)
    else:
        figure_width, figure_height = fixed_figsize
        legend_item_count_for_size = len(present_imputer_order)
        if compact_canvas and legend_item_count_for_size > 24:
            figure_width = max(figure_width, PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH)
            size_imputer_cols, size_imputer_rows = _legend_grid(
                legend_item_count_for_size,
                figure_width,
                target_rows=4,
            )
            size_classifier_count = len(present_classifier_order)
            size_classifier_cols = min(size_classifier_count, 9) if size_classifier_count else 1
            size_classifier_rows = max(1, ceil(size_classifier_count / size_classifier_cols)) if size_classifier_count else 1
            size_legend_rows = size_imputer_rows + size_classifier_rows
            figure_height = max(
                figure_height,
                3.05 * (rows + chunk_gap_ratio * chunk_gap_count)
                + 0.92
                + 0.42 * size_legend_rows,
            )
        if show_axis_labels_every_panel and not compact_canvas:
            figure_width = max(figure_width, PAPER_EXPERIMENT_GRID_WIDTH_PT / PAPER_PT_PER_INCH)
            figure_height = max(
                figure_height,
                max(2.05 * (rows + chunk_gap_ratio * chunk_gap_count) + 4.45, 16.75),
            )
        elif (not show_axis_labels_every_panel) and chunk_gap_count:
            figure_height += 0.36 * chunk_gap_count
    if not dataset_names or not mechanism_order:
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
        ax.text(0.5, 0.5, _no_data_text(label_mode), ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _apply_paper_text_style(fig, label_mode=label_mode)
        _save_figure(fig, output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    imputer_item_count = len(present_imputer_order)
    classifier_item_count = len(present_classifier_order)
    exp2_tick_fontsize = PAPER_TICK_FONTSIZE - 3.6
    exp2_axis_label_fontsize = PAPER_AXIS_LABEL_FONTSIZE
    exp2_panel_title_fontsize = PAPER_PANEL_TITLE_FONTSIZE
    exp2_row_label_fontsize = PAPER_ROW_LABEL_FONTSIZE
    exp2_x_label_fontsize = PAPER_AXIS_LABEL_FONTSIZE - 3.2
    exp2_y_label_fontsize = PAPER_AXIS_LABEL_FONTSIZE + 2.8
    exp2_imputer_legend_fontsize = PAPER_LEGEND_FONTSIZE - 1.0
    exp2_imputer_legend_title_fontsize = PAPER_LEGEND_TITLE_FONTSIZE - 0.4
    exp2_classifier_legend_fontsize = PAPER_LEGEND_FONTSIZE - 1.0
    exp2_classifier_legend_title_fontsize = PAPER_LEGEND_TITLE_FONTSIZE - 0.4
    # Experiment 2 keeps classifier shapes, but places both legends in the
    # same compact top band used by Experiment 1.
    imputer_cols = min(12, max(1, imputer_item_count))
    imputer_rows = max(1, ceil(imputer_item_count / imputer_cols))
    classifier_cols = min(classifier_item_count, 9) if classifier_item_count else 1
    classifier_rows = max(1, ceil(classifier_item_count / classifier_cols)) if classifier_item_count else 1
    legend_rows = imputer_rows + classifier_rows
    top_margin = _top_margin_for_legend(
        figure_height,
        legend_rows,
        row_height=0.34,
        base_height=0.84,
        min_top=0.48 if compact_canvas else 0.62,
        max_top=0.84 if compact_canvas else 0.90,
    )
    if show_axis_labels_every_panel and not compact_canvas:
        top_margin = 0.897 if has_real_world_column else 0.885
    bottom_margin = 0.085 if show_axis_labels_every_panel else 0.060
    imputer_legend_x = 0.006
    imputer_legend_y = 0.995
    imputer_legend_width = 0.959
    classifier_legend_x = imputer_legend_x
    classifier_legend_y = imputer_legend_y - (0.28 * imputer_rows / max(figure_height, 1.0))
    classifier_legend_width = imputer_legend_width
    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=False)
    grid_rows = rows + chunk_gap_count
    height_ratios = []
    for chunk_idx in range(dataset_chunk_count):
        height_ratios.extend([1.0] * rows_per_chunk)
        if chunk_gap_count and chunk_idx < dataset_chunk_count - 1:
            height_ratios.append(chunk_gap_ratio)
    grid = fig.add_gridspec(grid_rows, cols, height_ratios=height_ratios)
    axes = np.empty((rows, cols), dtype=object)
    for row_idx in range(rows):
        chunk_idx = row_idx // rows_per_chunk
        mechanism_idx = row_idx % rows_per_chunk
        grid_row_idx = (
            chunk_idx * (rows_per_chunk + int(bool(chunk_gap_count)))
            + mechanism_idx
        )
        for col_idx in range(cols):
            axes[row_idx, col_idx] = fig.add_subplot(grid[grid_row_idx, col_idx])

    left_margin = 0.062 if show_axis_labels_every_panel else 0.122
    y_label_x = 0.014
    panel_hspace = (
        0.39
        if has_real_world_column and show_axis_labels_every_panel
        else (0.28 if show_axis_labels_every_panel else 0.14)
    )
    panel_wspace = 0.40 if show_axis_labels_every_panel else 0.12
    fig.subplots_adjust(
        top=top_margin,
        bottom=bottom_margin,
        left=left_margin,
        right=0.965,
        hspace=panel_hspace,
        wspace=panel_wspace,
    )
    if mechanism_title_style == "abbr":
        mechanism_title_map = {"MCAR": "MCAR", "MAR": "MAR", "MNAR": "MNAR"}
    else:
        mechanism_title_map = _mechanism_row_title_map(label_mode=label_mode)
    stacked_mechanism_title_map = {
        mechanism: _format_stacked_mechanism_label(mechanism_title_map.get(mechanism, str(mechanism)))
        for mechanism in mechanism_order
    }
    axis_frame = axis_source_subset if axis_source_subset is not None else legend_working
    axis_frame = axis_frame[required_cols].dropna().copy()
    axis_frame = _exclude_plot_imputers(axis_frame, imputer_col="imputer")
    dataset_axis_limits = _compute_dataset_scatter_limits(axis_frame, y_col=y_col)

    def _draw_panel_points(ax, panel_frame):
        if panel_frame.empty:
            ax.text(
                0.5,
                0.5,
                _no_data_text(label_mode),
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return
        classifier_groups = {
            str(classifier): classifier_frame
            for classifier, classifier_frame in panel_frame.groupby(
                "classifier",
                sort=False,
            )
        }
        for classifier in present_classifier_order:
            classifier_frame = classifier_groups.get(str(classifier))
            if classifier_frame is None or classifier_frame.empty:
                continue
            edge_colors = [
                color_map[str(item)]
                for item in classifier_frame["imputer"].astype(str)
            ]
            x_values = pd.to_numeric(
                classifier_frame["nrmse"],
                errors="coerce",
            ).to_numpy(dtype=float)
            y_values = pd.to_numeric(
                classifier_frame[y_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            ax.scatter(
                x_values,
                y_values,
                facecolors="none",
                edgecolors=edge_colors,
                marker=marker_map[str(classifier)],
                linewidths=0.82,
                alpha=0.92,
                s=PAPER_SCATTER_SIZE,
                zorder=3,
            )

    for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
        for mechanism_idx, mechanism in enumerate(mechanism_order):
            row_idx = chunk_idx * rows_per_chunk + mechanism_idx
            for col_idx in range(synthetic_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(dataset_chunk):
                    ax.set_axis_off()
                    continue
                dataset = dataset_chunk[col_idx]
                dataset_limits = dataset_axis_limits.get(str(dataset), {})
                panel_frame = working[
                    (working["missing_mechanism"].astype(str) == mechanism)
                    & (working["dataset"].astype(str) == dataset)
                ]
                _draw_panel_points(ax, panel_frame)
                if square_panels:
                    ax.set_box_aspect(1)
                if dataset_limits.get("x") is not None:
                    ax.set_xlim(*dataset_limits["x"])
                else:
                    ax.margins(x=0.06)
                if dataset_limits.get("y") is not None:
                    ax.set_ylim(*dataset_limits["y"])
                else:
                    ax.margins(y=0.10)
                _style_reference_panel(ax, grid_axis="both")
                show_x_tick_labels = show_axis_labels_every_panel or (
                    chunk_idx == len(dataset_chunks) - 1
                    and mechanism_idx == rows_per_chunk - 1
                )
                ax.tick_params(axis="x", labelbottom=show_x_tick_labels)
                ax.tick_params(axis="y", labelleft=True)
                _style_y_tick_labels_outside_vertical(ax, pad=1, rotation=0, ha="right", va="center")
                ax.text(
                    -0.292,
                    0.5,
                    stacked_mechanism_title_map.get(mechanism, _format_stacked_mechanism_label(mechanism)),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=exp2_row_label_fontsize - 7.0,
                    fontweight="bold",
                    linespacing=0.82,
                    clip_on=False,
                )
                if has_real_world_column or mechanism_idx == 0:
                    ax.set_title(
                        _format_dataset_title(dataset),
                        fontsize=exp2_panel_title_fontsize,
                        fontweight="bold",
                        pad=2,
                    )
                if show_axis_labels_every_panel:
                    ax.set_ylabel(y_label, fontsize=exp2_axis_label_fontsize, fontweight="bold", labelpad=4)
                    ax.yaxis.set_label_coords(-0.128, 0.5)
                    ax.yaxis.label.set_visible(True)
                else:
                    ax.set_ylabel("")
                    ax.yaxis.label.set_visible(False)
                if show_x_tick_labels:
                    ax.set_xlabel("NRMSE", fontsize=exp2_x_label_fontsize, fontweight="bold", labelpad=1)
                else:
                    ax.set_xlabel("")
                ax.tick_params(axis="both", labelsize=exp2_tick_fontsize, width=0.75)

    if has_real_world_column:
        real_world_col_idx = cols - 1
        for row_idx in range(rows):
            ax = axes[row_idx, real_world_col_idx]
            if row_idx >= len(real_world_dataset_names):
                ax.set_axis_off()
                continue
            dataset = real_world_dataset_names[row_idx]
            dataset_limits = dataset_axis_limits.get(str(dataset), {})
            panel_frame = real_world_working[
                real_world_working["dataset"].astype(str) == dataset
            ]
            _draw_panel_points(ax, panel_frame)
            if square_panels:
                ax.set_box_aspect(1)
            if dataset_limits.get("x") is not None:
                ax.set_xlim(*dataset_limits["x"])
            else:
                ax.margins(x=0.06)
            if dataset_limits.get("y") is not None:
                ax.set_ylim(*dataset_limits["y"])
            else:
                ax.margins(y=0.10)
            _style_reference_panel(ax, grid_axis="both")
            _style_y_tick_labels_outside_vertical(
                ax,
                pad=1,
                rotation=0,
                ha="right",
                va="center",
            )
            ax.set_title(
                _format_dataset_title(dataset),
                fontsize=exp2_panel_title_fontsize,
                fontweight="bold",
                pad=2,
            )
            if show_axis_labels_every_panel:
                ax.set_ylabel(
                    y_label,
                    fontsize=exp2_axis_label_fontsize,
                    fontweight="bold",
                    labelpad=4,
                )
                ax.yaxis.set_label_coords(-0.128, 0.5)
                ax.yaxis.label.set_visible(True)
            else:
                ax.set_ylabel("")
                ax.yaxis.label.set_visible(False)
            ax.set_xlabel(
                "NRMSE",
                fontsize=exp2_x_label_fontsize,
                fontweight="bold",
                labelpad=1,
            )
            ax.tick_params(
                axis="both",
                labelsize=exp2_tick_fontsize,
                width=0.75,
            )

    subplot_label_idx = 0
    for col_idx in range(cols):
        for row_idx in range(rows):
            ax = axes[row_idx, col_idx]
            if not ax.axison:
                continue
            ax.text(
                0.018,
                0.965,
                f"({_subplot_label_from_index(subplot_label_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PAPER_PANEL_TITLE_FONTSIZE - 4.4,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
                zorder=10,
            )
            subplot_label_idx += 1

    for chunk_idx in range(len(dataset_chunks)):
        chunk_start = chunk_idx * rows_per_chunk
        chunk_end = chunk_start + rows_per_chunk - 1
        chunk_top_bbox = axes[chunk_start, 0].get_position()
        chunk_bottom_bbox = axes[chunk_end, 0].get_position()
        if not show_axis_labels_every_panel:
            fig.text(
                y_label_x,
                0.5 * (chunk_top_bbox.y1 + chunk_bottom_bbox.y0),
                y_label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=exp2_y_label_fontsize,
                fontweight="bold",
            )

    imputer_handles = [
        Patch(facecolor=color, edgecolor="#333333", linewidth=0.7, alpha=0.86, label=name)
        for name in present_imputer_order
        if name in color_map
        for color in [color_map[name]]
    ]
    classifier_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="white",
            markerfacecolor="none",
            markeredgecolor="#666666",
            markeredgewidth=1.25,
            markersize=17.0,
            linewidth=0,
            label=name,
        )
        for name in present_classifier_order
        if name in marker_map
        for marker in [marker_map[name]]
    ]
    if imputer_handles:
        _add_top_legend(
            fig,
            color_map,
            item_order=present_imputer_order,
            ncols=imputer_cols,
            y_anchor=imputer_legend_y,
            x_anchor=imputer_legend_x,
            width=imputer_legend_width,
            fontsize=exp2_imputer_legend_fontsize,
            columnspacing=0.92,
            handlelength=1.80,
            handleheight=1.08,
            handletextpad=0.46,
            labelspacing=0.20,
            mode="expand",
        )
    if classifier_handles:
        fig.legend(
            handles=classifier_handles,
            loc="upper center",
            bbox_to_anchor=(classifier_legend_x, classifier_legend_y, classifier_legend_width, 0.001),
            ncol=classifier_cols,
            mode="expand",
            frameon=False,
            columnspacing=0.82,
            handletextpad=0.44,
            labelspacing=0.18,
            fontsize=exp2_classifier_legend_fontsize,
        )
    _apply_paper_text_style(fig, label_mode=label_mode)
    _save_figure(fig, output_path, dpi=300, bbox_inches=None)
    plt.close(fig)


def save_mechanism_dataset_bar_grid(
    summary_frame,
    metric,
    output_path,
    dataset_col="dataset",
    mechanism_col="missing_mechanism",
    imputer_col="imputer",
    label_mode="en",
    metric_label_map=None,
    force_zero_ymin=False,
    imputer_order=None,
):
    _configure_plot_locale(label_mode=label_mode)
    required_cols = [dataset_col, mechanism_col, imputer_col, metric]
    missing_cols = [col for col in required_cols if col not in summary_frame.columns]
    if missing_cols:
        raise ValueError(f"summary_frame missing required columns: {missing_cols}")

    working = summary_frame[required_cols].dropna().copy()
    working = _exclude_plot_imputers(working, imputer_col=imputer_col)
    dataset_names = sorted(working[dataset_col].astype(str).unique())
    mechanism_names = _normalize_mechanism_order(sorted(working[mechanism_col].astype(str).unique()))
    present_imputer_order = _ordered_imputers(
        working[imputer_col].astype(str).unique(),
        preferred_order=imputer_order,
    )
    global_imputer_order = _ordered_imputers(
        working[imputer_col].astype(str).unique(),
        preferred_order=imputer_order,
        include_all_preferred=imputer_order is not None,
    )
    color_map = _build_color_map(global_imputer_order)

    dataset_chunks = _wrap_dataset_grid(dataset_names, max_datasets_per_row=4)
    cols = max((len(chunk) for chunk in dataset_chunks), default=1)
    rows_per_chunk = max(len(mechanism_names), 1)
    rows = max(len(dataset_chunks), 1) * rows_per_chunk
    legend_item_count = len(present_imputer_order)
    paper_width = PAPER_EXPERIMENT1_WIDTH_PT / PAPER_PT_PER_INCH
    panel_size = 3.05
    fig_width = max(cols * panel_size + 1.25, 8.0)
    if legend_item_count > 24:
        fig_width = max(fig_width, paper_width)
    legend_cols, legend_rows = _legend_grid(legend_item_count, fig_width, target_rows=5)
    fig_height = max(rows * panel_size + 1.55 + 0.36 * legend_rows, 4.8)
    top_margin = _top_margin_for_legend(
        fig_height,
        legend_rows,
        row_height=0.36,
        base_height=0.62,
        min_top=0.50,
        max_top=0.84,
    )

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(
        top=top_margin,
        bottom=0.105,
        left=0.095,
        right=0.965,
        hspace=0.32,
        wspace=0.34,
    )

    metric_label = _format_vertical_metric_label((metric_label_map or {}).get(metric, metric))
    mechanism_title_map = _mechanism_row_title_map(label_mode=label_mode)
    x_positions = np.arange(len(global_imputer_order))
    slot_index = {name: idx for idx, name in enumerate(global_imputer_order)}

    dataset_ylim_map = {}
    for dataset in dataset_names:
        values = pd.to_numeric(
            working.loc[working[dataset_col].astype(str) == dataset, metric],
            errors="coerce",
        ).dropna()
        limits = _compute_numeric_limits(values, lower_padding_ratio=0.04, upper_padding_ratio=0.12)
        if limits is not None and force_zero_ymin:
            limits = (0.0, limits[1])
        dataset_ylim_map[str(dataset)] = limits

    for chunk_idx, dataset_chunk in enumerate(dataset_chunks):
        for mechanism_idx, mechanism in enumerate(mechanism_names):
            row_idx = chunk_idx * rows_per_chunk + mechanism_idx
            for col_idx in range(cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(dataset_chunk):
                    ax.set_axis_off()
                    continue
                dataset = dataset_chunk[col_idx]
                panel_frame = working[
                    (working[dataset_col].astype(str) == dataset)
                    & (working[mechanism_col].astype(str) == mechanism)
                ].copy()
                grouped = (
                    panel_frame.groupby(imputer_col, dropna=False)[metric]
                    .mean()
                    .dropna()
                    .rename("value")
                )
                if grouped.empty:
                    ax.text(0.5, 0.5, _no_data_text(label_mode), ha="center", va="center", transform=ax.transAxes)
                else:
                    bar_names = [name for name in global_imputer_order if name in grouped.index.astype(str)]
                    bar_x = [slot_index[name] for name in bar_names]
                    bar_y = [float(grouped.loc[name]) for name in bar_names]
                    bar_colors = [color_map.get(name, "#4c78a8") for name in bar_names]
                    ax.bar(bar_x, bar_y, color=bar_colors, edgecolor="#333333", linewidth=0.35, width=0.78)
                ax.set_box_aspect(1)
                ax.set_xlim(-0.6, max(len(global_imputer_order) - 0.4, 0.6))
                limits = dataset_ylim_map.get(str(dataset))
                if limits is not None:
                    ax.set_ylim(*limits)
                ax.set_xticks(x_positions)
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
                ax.tick_params(axis="y", labelsize=PAPER_TICK_FONTSIZE - 3.0, width=0.8)
                _style_y_tick_labels_outside_vertical(ax, pad=1, rotation=0, ha="right", va="center")
                ax.grid(axis="y", alpha=0.25, linewidth=0.4)
                ax.set_axisbelow(True)
                ax.set_ylabel(metric_label, fontsize=PAPER_AXIS_LABEL_FONTSIZE - 2.4, fontweight="bold", labelpad=4)
                ax.yaxis.set_label_coords(-0.128, 0.5)
                ax.text(
                    -0.245,
                    0.5,
                    _format_stacked_mechanism_label(mechanism_title_map.get(mechanism, str(mechanism))),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=PAPER_ROW_LABEL_FONTSIZE - 12.0,
                    fontweight="bold",
                    linespacing=0.82,
                    clip_on=False,
                )
                if mechanism_idx == 0:
                    ax.set_title(
                        _format_dataset_title(dataset),
                        fontsize=PAPER_PANEL_TITLE_FONTSIZE - 4.0,
                        fontweight="bold",
                        pad=4,
                    )

    subplot_label_idx = 0
    for col_idx in range(cols):
        for row_idx in range(rows):
            ax = axes[row_idx, col_idx]
            if not ax.axison:
                continue
            ax.text(
                0.022,
                0.960,
                f"({_subplot_label_from_index(subplot_label_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PAPER_PANEL_TITLE_FONTSIZE - 5.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.22},
                zorder=10,
            )
            subplot_label_idx += 1

    _add_top_legend(
        fig,
        color_map,
        item_order=present_imputer_order,
        ncols=legend_cols,
        y_anchor=0.992,
        fontsize=PAPER_LEGEND_FONTSIZE - 1.2,
        columnspacing=1.10,
        handlelength=1.55,
        handleheight=1.05,
        handletextpad=0.44,
        labelspacing=0.38,
        mode="expand",
    )
    _apply_paper_text_style(fig, label_mode=label_mode)
    _save_figure(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _structure_metric_plot_label(metric, label_mode="en"):
    if _is_zh(label_mode):
        return {
            "m1_corr_frob": r"相关矩阵差 ($\Delta{Corr}$)",
            "m1_fisher_delta": r"Fisher 判别比变化 ($\Delta\mathrm{FR}$)",
            "m1_w1_mean": r"边际分布差 ($\overline{W_1}$)",
            "m1_w1_iqr_scaled_mean": r"IQR 缩放边际分布差 ($\overline{W}_{1,\mathrm{IQR}}$)",
            "m1_w1_relative_iqr_mean": r"边际分布差 ($\overline{W_1}$)",
        }.get(str(metric), str(metric))
    return {
        "m1_corr_frob": r"$\Delta{Corr}$",
        "m1_fisher_delta": r"$\Delta\mathrm{FR}$",
        "m1_w1_mean": r"$\overline{W_1}$",
        "m1_w1_iqr_scaled_mean": r"$\overline{W}_{1,\mathrm{IQR}}$",
        "m1_w1_relative_iqr_mean": r"$\overline{W_1}$",
    }.get(str(metric), str(metric))


def _mechanism_plot_label(mechanism, label_mode="en"):
    text = str(mechanism)
    if _is_zh(label_mode) and text == "All":
        return "总体"
    return text


def save_structure_regression_forest_plot(table_frame, output_path, label_mode="en"):
    _configure_plot_locale(label_mode=label_mode)
    required_cols = [
        "structure_metric",
        "mechanism_scope",
        "beta_structure",
        "beta_structure_ci_low",
        "beta_structure_ci_high",
    ]
    missing_cols = [col for col in required_cols if col not in table_frame.columns]
    if missing_cols:
        raise ValueError(f"table_frame missing required columns: {missing_cols}")

    working = table_frame.copy()
    if "fit_status" in working.columns:
        working = working[working["fit_status"].astype(str).str.lower().eq("ok")].copy()
    if "role" in working.columns:
        working = working[
            working["role"].astype(str).isin(["main", "robustness_structure_metric"])
        ].copy()

    numeric_cols = ["beta_structure", "beta_structure_ci_low", "beta_structure_ci_high", "beta_structure_p"]
    for col in numeric_cols:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=["beta_structure", "beta_structure_ci_low", "beta_structure_ci_high"])

    metric_order = [
        "m1_corr_frob",
        "m1_fisher_delta",
        "m1_w1_mean",
        "m1_w1_iqr_scaled_mean",
        "m1_w1_relative_iqr_mean",
    ]
    mechanism_order = ["All", "MCAR", "MAR", "MNAR"]
    working["metric_order"] = working["structure_metric"].astype(str).map(
        {name: idx for idx, name in enumerate(metric_order)}
    ).fillna(len(metric_order))
    working["mechanism_order"] = working["mechanism_scope"].astype(str).map(
        {name: idx for idx, name in enumerate(mechanism_order)}
    ).fillna(len(mechanism_order))
    working = working.sort_values(["metric_order", "mechanism_order", "structure_metric", "mechanism_scope"])

    if working.empty:
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        ax.text(0.5, 0.5, _no_data_text(label_mode), ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _save_figure(fig, output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    y_positions = np.arange(len(working))[::-1]
    fig_height = max(5.2, 0.48 * len(working) + 1.6)
    fig, ax = plt.subplots(figsize=(9.6, fig_height), constrained_layout=False)
    significant = (
        pd.to_numeric(working.get("beta_structure_p", pd.Series(np.nan, index=working.index)), errors="coerce") < 0.05
    )
    colors = np.where(significant.to_numpy(dtype=bool), "#2f6f9f", "#9a9a9a")

    x = working["beta_structure"].to_numpy(dtype=float)
    xerr = np.vstack([
        x - working["beta_structure_ci_low"].to_numpy(dtype=float),
        working["beta_structure_ci_high"].to_numpy(dtype=float) - x,
    ])
    for y, x_value, error, color in zip(y_positions, x, xerr.T, colors):
        ax.errorbar(
            [x_value],
            [y],
            xerr=np.asarray(error, dtype=float).reshape(2, 1),
            fmt="o",
            markersize=7.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.8,
            capthick=1.5,
            zorder=3,
        )

    labels = [
        f"{_structure_metric_plot_label(row.structure_metric, label_mode)} | "
        f"{_mechanism_plot_label(row.mechanism_scope, label_mode)}"
        for row in working.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=PAPER_TICK_FONTSIZE)
    ax.axvline(0.0, color="#555555", linewidth=1.1, linestyle="--", alpha=0.85, zorder=1)
    ax.grid(axis="x", alpha=0.25, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=PAPER_TICK_FONTSIZE, width=0.8)
    ax.set_xlabel(
        "结构指标回归系数 β（已控制 NRMSE 与固定效应）"
        if _is_zh(label_mode)
        else r"Structure coefficient $\beta$ (controlling NRMSE and fixed effects)",
        fontsize=PAPER_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.set_title(
        "实验三：结构扰动的独立解释力"
        if _is_zh(label_mode)
        else "Experiment 3: Partial Slopes of Structural Distortion",
        fontsize=PAPER_PANEL_TITLE_FONTSIZE + 0.8,
        fontweight="bold",
        pad=12,
    )
    handles = [
        Line2D([0], [0], marker="o", color="#2f6f9f", linestyle="", markersize=8.2, label="p < 0.05"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#9a9a9a",
            linestyle="",
            markersize=8.2,
            label="p ≥ 0.05" if _is_zh(label_mode) else "p >= 0.05",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.88, bottom=0.16)
    _save_figure(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
