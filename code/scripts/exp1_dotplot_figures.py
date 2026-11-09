#!/usr/bin/env python3
"""Experiment 1 supplementary figures: horizontal sorted dot plots.

Additive module for the paper-experiments pipeline. It does NOT replace the
existing experiment_1 boxplot/bar figures; `run_paper_experiments.py` calls
`generate_exp1_dotplot_figures` after `run_experiment_1` and writes into a
separate `experiment_1_dotplot/` folder.

Design: methods on the y axis sorted by pooled median structural damage
(best on top), one dot per dataset, red tick = cross-dataset mean, black
tick = cross-dataset median, log-scaled x axis. Synthetic-missing summaries
get one panel per mechanism (MCAR/MAR/MNAR); real-missing summaries get a
single "Real-world missing" panel. A 4-panel MCAR/MAR/MNAR + Real-world
combined figure is available through the CLI (`--combine-real`).

Standalone usage (same behaviour as scripts/plot_exp1_sorted_dotplot.py):
    python3 code/scripts/exp1_dotplot_figures.py \
        --run-summary result/no_real_missing/run_summary.csv \
        --out-dir fig/exp/1 [--per-rate] [--combine-real <real_run_summary>]
"""

import argparse
import math
import os

import matplotlib

try:  # pipeline may already have selected a backend
    matplotlib.use("Agg")
except Exception:  # pragma: no cover
    pass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

METRICS = {
    "m1_corr_frob": r"$\Delta Corr$",
    "m1_fisher_delta": r"$\Delta \mathrm{FR}$",
    "m1_w1_mean": r"$\overline{W_1}$",
    "m1_w1_iqr_scaled_mean": r"$\overline{W}_{1,\mathrm{IQR}}$",
    "m1_w1_relative_iqr_mean": r"$\overline{W_1}$",
}
MECHANISMS = ["MCAR", "MAR", "MNAR"]
RUN_KEYS = ["dataset", "missing_mechanism", "missing_percent", "imputer", "iter", "seed"]
DEFAULT_DROP_IMPUTERS = ()

DATASET_COLORS = {
    "glass": "#1f77b4",
    "hcv": "#ff7f0e",
    "iris": "#2ca02c",
    "mobile": "#d62728",
    "obesity": "#9467bd",
    "rice": "#8c564b",
    "seed": "#e377c2",
    "wine": "#7f7f7f",
    "airquality": "#17becf",
    "college_scorecard": "#bcbd22",
    "cvss": "#003c8f",
    "fifa": "#b8860b",
    "food": "#a51e6a",
    "nycrestaurants": "#4b0082",
}
_FALLBACK_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
DATASET_LABELS = {
    "college_scorecard": "college",
    "nycrestaurants": "restaurant",
}
MEAN_COLOR = "#d62728"
MEDIAN_COLOR = "black"

# Categories follow Table "imputation_methods" in the paper (46 methods, 8 groups).
CATEGORY_MAP = {
    # Statistical
    "Mean": "Statistical", "Mode": "Statistical", "BPCA": "Statistical",
    "LSA": "Statistical", "POP": "Statistical", "RIBG": "Statistical",
    "NIIA": "Statistical", "ESD": "Statistical", "CLRMC": "Statistical",
    "ECLRMC": "Statistical", "PMM": "Statistical", "CCF": "Statistical",
    "NMF": "Statistical", "CDI": "Statistical",
    # Multiple imputation
    "Mice": "Multiple", "MVNI": "Multiple", "SMC-FCS": "Multiple",
    "Sice": "Multiple", "TWDA": "Multiple", "MIPCA": "Multiple", "MIGEC": "Multiple",
    # Tree-based
    "MISSF": "Tree", "MIST": "Tree", "XGBI": "Tree",
    # Cluster-based
    "CMI": "Cluster", "CI": "Cluster", "IFC": "Cluster",
    "SOM": "Cluster", "GCID": "Cluster", "GMM": "Cluster",
    # Neighbor-based
    "KNNI": "Neighbor", "KNNE": "Neighbor", "CVBKNN": "Neighbor",
    "HDI": "Neighbor", "RRSI": "Neighbor",
    # Regression-based
    "LOESS": "Regression", "IIM": "Regression", "DLM": "Regression", "MLPI": "Regression",
    # Constraint-based
    "Holoclean": "Constraint", "HoloClean": "Constraint",
    # Deep learning
    "MIDA": "DL", "MIWAE": "DL", "NOMI": "DL", "VAEI": "DL",
    "GINN": "DL", "Datawig": "DL",
}
FAMILY_COLORS = {
    "Statistical": "#1a4f8b",
    "Multiple": "#b35900",
    "Tree": "#1e7a1e",
    "Cluster": "#6a3d9a",
    "Neighbor": "#a31515",
    "Regression": "#0e7c7b",
    "Constraint": "#8a6d00",
    "DL": "#c2185b",
}

# scoped style: applied via rc_context so the pipeline's global rcParams stay untouched
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
}


def dataset_color_map(datasets):
    datasets = sorted(str(d) for d in datasets)
    if all(d in DATASET_COLORS for d in datasets):
        return {d: DATASET_COLORS[d] for d in datasets}
    return {d: _FALLBACK_CYCLE[i % len(_FALLBACK_CYCLE)] for i, d in enumerate(datasets)}


def prepare_runs(df, drop_imputers=DEFAULT_DROP_IMPUTERS, metrics=None):
    """Deduplicate per-run rows (one row per classifier in run_summary),
    drop excluded imputers and attach paper-table categories."""
    metric_map = dict(metrics or METRICS)
    keep = [c for c in RUN_KEYS + list(metric_map) + ["imputer_family"] if c in df.columns]
    df = df[keep].copy()
    df = df.drop_duplicates(subset=[k for k in RUN_KEYS if k in df.columns])
    if drop_imputers:
        df = df[~df["imputer"].astype(str).isin(set(drop_imputers))]
    mapped = df["imputer"].astype(str).map(CATEGORY_MAP)
    if "imputer_family" in df.columns:
        mapped = mapped.fillna(df["imputer_family"])
    df["imputer_family"] = mapped.fillna("Other")
    for metric in metric_map:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def load_runs(path, drop_imputers=DEFAULT_DROP_IMPUTERS, metrics=None):
    metric_map = dict(metrics or METRICS)
    cols = RUN_KEYS + list(metric_map) + ["imputer_family"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    return prepare_runs(df, drop_imputers, metric_map)


def grouped_order(df, metric):
    """Bottom-to-top row order: categories sorted by their median damage,
    methods sorted within each category. Returns (order, separator_positions)."""
    med = df.groupby("imputer")[metric].median()
    fam = df.drop_duplicates("imputer").set_index("imputer")["imputer_family"]
    cat_med = med.groupby(fam).median()
    frame = pd.DataFrame({"med": med, "cat": fam.reindex(med.index)})
    frame["cat_med"] = frame["cat"].map(cat_med)
    frame = frame.sort_values(["cat_med", "cat", "med"], ascending=[False, False, False])
    order = frame.index.tolist()
    seps = [i for i in range(1, len(order))
            if frame["cat"].iloc[i] != frame["cat"].iloc[i - 1]]
    return order, seps


def _draw_panel(ax, sub, metric, ypos, n, colors, title, label, floor=None,
                separators=()):
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0, lw=0)
    if floor is not None and floor > 0:
        ax.axvline(floor, ls="--", lw=1.3, color="0.4", zorder=2)
    for ds, color in colors.items():
        pts = sub[sub["dataset"] == ds]
        ax.scatter(
            pts[metric], pts["imputer"].map(ypos),
            s=48, color=color, alpha=0.85, linewidths=0, zorder=3,
        )
    mea = sub.groupby("imputer")[metric].mean()
    ax.scatter(
        mea.to_numpy(), [ypos[i] for i in mea.index],
        marker="|", s=250, color=MEAN_COLOR, linewidths=2.6, zorder=4,
    )
    med = sub.groupby("imputer")[metric].median()
    ax.scatter(
        med.to_numpy(), [ypos[i] for i in med.index],
        marker="|", s=250, color=MEDIAN_COLOR, linewidths=2.6, zorder=5,
    )
    ax.set_xscale("log")
    if title:
        ax.set_title(title, fontsize=17, fontweight="bold", pad=5)
    ax.set_xlabel(label, fontsize=16, labelpad=1)
    ax.grid(axis="x", which="major", ls=":", lw=0.5, color="0.75", zorder=1)
    ax.tick_params(axis="x", labelsize=11, pad=2)
    ax.set_ylim(-0.6, n - 0.4)
    if floor is not None and floor > 0:
        # expose the baseline value as an extra (grey) tick on the x axis
        xmin, xmax = ax.get_xlim()
        decades = [10.0 ** k for k in range(-8, 9) if xmin <= 10.0 ** k <= xmax]
        floor_label = f"{floor:.2g}"
        ax.set_xticks(decades + [floor])
        ax.set_xticklabels(
            [
                # hide decade labels that would collide with the floor label
                "" if abs(math.log10(d) - math.log10(floor)) < 0.35
                else f"$10^{{{int(round(math.log10(d)))}}}$"
                for d in decades
            ]
            + [floor_label]
        )
        for t in ax.get_xticklabels():
            if t.get_text() == floor_label:
                t.set_color("0.35")


def _median_mean_handles():
    return [
        Line2D([], [], marker="|", ls="", color=MEDIAN_COLOR, markersize=16,
               markeredgewidth=3, label="median"),
        Line2D([], [], marker="|", ls="", color=MEAN_COLOR, markersize=16,
               markeredgewidth=3, label="mean"),
        Line2D([], [], ls="--", lw=1.6, color="0.4", label="min"),
    ]


def _stack_figure_legends(fig, legends, *, x=0.02, width=0.96):
    """Stack figure legends and return a safe top bound for the plot axes.

    Legend heights depend on label lengths and column wrapping, so fixed
    figure-relative y positions are not reliable for the narrower real-data
    figure. Measure each rendered legend instead and reserve physical spacing
    for the panel title below the legend block.
    """
    figure_height = fig.get_figheight()
    y = 1.0 - 0.06 / figure_height
    row_gap = 0.05 / figure_height
    title_clearance = 0.34 / figure_height

    for legend in legends:
        legend.set_bbox_to_anchor((x, y, width, 0), transform=fig.transFigure)
        fig.canvas.draw()
        bbox = legend.get_window_extent(fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.transFigure.inverted())
        y = bbox.y0 - row_gap

    return y - title_clearance


def plot_metric(df, metric, label, out_base, suffix="", panel_title=None):
    with plt.rc_context(STYLE):
        return _plot_metric_inner(df, metric, label, out_base, suffix, panel_title)


def _plot_metric_inner(df, metric, label, out_base, suffix, panel_title):
    order, seps = grouped_order(df, metric)
    ypos = {imp: i for i, imp in enumerate(order)}
    n = len(order)

    families = df.drop_duplicates("imputer").set_index("imputer")["imputer_family"]
    colors = dataset_color_map(df["dataset"].unique())

    mechs = [m for m in MECHANISMS if m in set(df["missing_mechanism"].astype(str))]
    single = len(mechs) <= 1
    if single:
        mechs = mechs or [None]

    cell = (
        df.groupby(["missing_mechanism", "dataset", "imputer"])[metric]
        .median()
        .reset_index()
    )

    floor = float(cell[metric].min())
    print(f"[exp1-dotplot] {os.path.basename(out_base)}{suffix}: min observed = {floor:.4g}")

    fig, axes = plt.subplots(
        1, len(mechs),
        figsize=(6.8 if single else 13.5, 0.21 * n + (3.2 if single else 2.6)),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    for i, (ax, mech) in enumerate(zip(axes, mechs)):
        sub = cell if mech is None else cell[cell["missing_mechanism"] == mech]
        title = panel_title if single and panel_title else mech
        if not single and title:
            title = f"({chr(97 + i)}) {title}"
        _draw_panel(ax, sub, metric, ypos, n, colors, title, label, floor=floor,
                    separators=seps)
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(order, fontsize=13)
    for tick, imp in zip(axes[0].get_yticklabels(), order):
        tick.set_color(FAMILY_COLORS.get(str(families.get(imp)), "black"))
    axes[0].tick_params(axis="y", length=0)

    dataset_handles = [
        Line2D([], [], marker="o", ls="", color=c, markersize=16,
               label=DATASET_LABELS.get(ds, ds))
        for ds, c in colors.items()
    ]
    leg1 = fig.legend(
        handles=dataset_handles, loc="upper left",
        ncol=len(dataset_handles) if not single else 3,
        frameon=False, fontsize=21 if not single else 16, mode="expand",
        bbox_to_anchor=(0.02, 0.99, 0.96, 0),
        handletextpad=0.1, handlelength=1.2,
    )
    family_handles = [
        Line2D([], [], marker="s", ls="", color=c, markersize=13, label=f)
        for f, c in FAMILY_COLORS.items()
    ] + _median_mean_handles()
    leg2 = fig.legend(
        handles=family_handles, loc="upper left",
        ncol=len(family_handles) if not single else 4,
        frameon=False, fontsize=17 if not single else 14, mode="expand",
        bbox_to_anchor=(0.02, 0.90, 0.96, 0),
        handletextpad=0.1, handlelength=1.2, title=None,
    )

    axes_top = _stack_figure_legends(fig, (leg1, leg2))
    # Let tight_layout determine label-aware left/bottom margins, then reserve
    # the measured legend block without asking it to solve an undersized rect.
    fig.tight_layout()
    fig.subplots_adjust(top=axes_top, wspace=0.05)
    written = []
    for ext in ("png", "pdf"):
        path = f"{out_base}{suffix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def plot_metric_combined(df_syn, df_real, metric, label, out_base,
                         real_title="Real-world", suffix=""):
    with plt.rc_context(STYLE):
        return _plot_metric_combined_inner(
            df_syn, df_real, metric, label, out_base, real_title, suffix
        )


def _plot_metric_combined_inner(df_syn, df_real, metric, label, out_base,
                                real_title, suffix):
    pooled = pd.concat(
        [df_syn[["imputer", "imputer_family", metric]],
         df_real[["imputer", "imputer_family", metric]]],
        ignore_index=True,
    )
    order, seps = grouped_order(pooled, metric)
    ypos = {imp: i for i, imp in enumerate(order)}
    n = len(order)
    families = pooled.drop_duplicates("imputer").set_index("imputer")["imputer_family"]

    colors_syn = dataset_color_map(df_syn["dataset"].unique())
    colors_real = dataset_color_map(df_real["dataset"].unique())

    cell_syn = (
        df_syn.groupby(["missing_mechanism", "dataset", "imputer"])[metric]
        .median().reset_index()
    )
    cell_real = df_real.groupby(["dataset", "imputer"])[metric].median().reset_index()
    cell_real["missing_mechanism"] = "__real__"
    floor = float(min(cell_syn[metric].min(), cell_real[metric].min()))
    print(f"[exp1-dotplot] {os.path.basename(out_base)}{suffix}: min observed = {floor:.4g}")

    fig = plt.figure(figsize=(16.5, 0.21 * n + 2.8))
    gs = fig.add_gridspec(
        1, 4, wspace=0.05,
        left=0.072, right=0.997, top=0.962, bottom=0.034,
    )
    ax0 = fig.add_subplot(gs[0])
    axes = [ax0] + [fig.add_subplot(gs[i], sharey=ax0) for i in (1, 2, 3)]

    for i, (ax, mech) in enumerate(zip(axes[:3], MECHANISMS)):
        _draw_panel(ax, cell_syn[cell_syn["missing_mechanism"] == mech],
                    metric, ypos, n, colors_syn, f"({chr(97 + i)}) {mech}", label,
                    floor=floor, separators=seps)
    _draw_panel(axes[3], cell_real, metric, ypos, n, colors_real,
                f"(d) {real_title}", label, floor=floor, separators=seps)
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)

    ax0.set_yticks(range(n))
    ax0.set_yticklabels(order, fontsize=13)
    for tick, imp in zip(ax0.get_yticklabels(), order):
        tick.set_color(FAMILY_COLORS.get(str(families.get(imp)), "black"))
    ax0.tick_params(axis="y", length=0)

    all_colors = {**colors_syn, **colors_real}
    row1 = [
        Line2D([], [], marker="o", ls="", color=c, markersize=16,
               label=DATASET_LABELS.get(d, d))
        for d, c in all_colors.items()
    ]
    row2 = [
        Line2D([], [], marker="s", ls="", color=c, markersize=13, label=f)
        for f, c in FAMILY_COLORS.items()
    ] + _median_mean_handles()
    for row, y in ((row1, 1.006), (row2, 0.976)):
        leg = fig.legend(
            handles=row, loc="lower left", ncol=len(row), frameon=False,
            fontsize=21, mode="expand", bbox_to_anchor=(0.005, y, 0.99, 0.028),
            handletextpad=0.1, handlelength=1.2,
        )
        fig.add_artist(leg)

    written = []
    for ext in ("png", "pdf"):
        path = f"{out_base}{suffix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def generate_exp1_dotplot_figures(summary, out_dir, per_rate=True,
                                  drop_imputers=DEFAULT_DROP_IMPUTERS,
                                  real_panel_title="Real-world missing",
                                  metrics=None):
    """Pipeline entry point. `summary` is the already-loaded run_summary frame.

    Writes sorted dot plots for every structure metric into `out_dir`
    (a folder separate from the classic experiment_1 outputs). Returns the
    list of written files.
    """
    os.makedirs(out_dir, exist_ok=True)
    metric_map = dict(metrics or METRICS)
    df = prepare_runs(summary, drop_imputers, metric_map)
    if df.empty or "imputer" not in df.columns:
        return []

    mechs = set(df["missing_mechanism"].astype(str)) & set(MECHANISMS)
    single = len(mechs) <= 1
    name_tag = "real_missing" if single else "mechanism_grid"
    panel_title = real_panel_title if single else None

    written = []
    for metric, label in metric_map.items():
        if metric not in df.columns:
            continue
        base = os.path.join(out_dir, f"{metric}_dotplot_{name_tag}")
        written += plot_metric(df, metric, label, base, panel_title=panel_title)
        if per_rate and not single and "missing_percent" in df.columns:
            for rate, frame in df.groupby("missing_percent", dropna=False):
                written += plot_metric(
                    frame, metric, label, base,
                    suffix=f"_mp_{rate:g}", panel_title=panel_title,
                )
    return written


def generate_exp1_dotplot_combined_figures(
    synthetic_summary,
    real_summary,
    out_dir,
    drop_imputers=DEFAULT_DROP_IMPUTERS,
    real_panel_title="Real-world missing",
    metrics=None,
):
    """Write four-panel synthetic-mechanism plus real-missing dot plots."""
    os.makedirs(out_dir, exist_ok=True)
    metric_map = dict(metrics or METRICS)
    synthetic = prepare_runs(synthetic_summary, drop_imputers, metric_map)
    real = prepare_runs(real_summary, drop_imputers, metric_map)
    if synthetic.empty or real.empty:
        return []

    written = []
    for metric, label in metric_map.items():
        if metric not in synthetic.columns or metric not in real.columns:
            continue
        base = os.path.join(out_dir, f"{metric}_dotplot_mech_plus_real")
        written += plot_metric_combined(
            synthetic,
            real,
            metric,
            label,
            base,
            real_title=real_panel_title,
        )
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-summary", default="result/no_real_missing/run_summary.csv")
    ap.add_argument("--out-dir", default="fig/exp/1")
    ap.add_argument("--per-rate", action="store_true")
    ap.add_argument("--panel-title", default=None)
    ap.add_argument("--name-tag", default="mechanism_grid")
    ap.add_argument("--combine-real", default=None,
                    help="path to real-missing run_summary; emits 4-panel MCAR/MAR/MNAR+Real-world figures")
    ap.add_argument("--drop-imputers", nargs="*", default=list(DEFAULT_DROP_IMPUTERS))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_runs(args.run_summary, args.drop_imputers)

    if args.combine_real:
        df_real = load_runs(args.combine_real, args.drop_imputers)
        for metric, label in METRICS.items():
            if metric not in df.columns or metric not in df_real.columns:
                continue
            base = os.path.join(args.out_dir, f"{metric}_dotplot_mech_plus_real")
            plot_metric_combined(df, df_real, metric, label, base)
            if args.per_rate:
                for rate, frame in df.groupby("missing_percent", dropna=False):
                    plot_metric_combined(frame, df_real, metric, label, base,
                                         suffix=f"_mp_{rate:g}")
            print("done combined", metric)
        return

    for metric, label in METRICS.items():
        if metric not in df.columns:
            continue
        base = os.path.join(args.out_dir, f"{metric}_dotplot_{args.name_tag}")
        plot_metric(df, metric, label, base, panel_title=args.panel_title)
        if args.per_rate:
            for rate, frame in df.groupby("missing_percent", dropna=False):
                plot_metric(frame, metric, label, base, suffix=f"_mp_{rate:g}",
                            panel_title=args.panel_title)
        print("done", metric)


if __name__ == "__main__":
    main()
