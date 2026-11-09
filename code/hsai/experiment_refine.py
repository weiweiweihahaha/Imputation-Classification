"""Experiment: base imputer vs base + HS-Refine (HSR), paired on real data.

Non-invasive. Reads the existing ground-truth and missing-data CSVs (read-only),
uses NRMSE on masked cells, thesis hard-sample inconsistency, train-only
imputation, and clean-test evaluation, then writes to
``result/hsai_experiment/refine/``.

HSR is method-agnostic: every base imputer is invoked through the framework's
``code.Imputation.impute`` entrypoint, then HSR refines the produced matrix.

Usage (from the repo root):
    python -m code.hsai.experiment_refine --datasets obesity mobile wine \
        --missing 0.2 0.3 --mechanisms MCAR MAR MNAR --iters 1 --seeds 42 \
        --bases XGBI --classifiers LogisticRegression RandomForest
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from code.hsai.hs_refine import hs_refine_guarded
from code.Imputation import impute as framework_impute
from code.core.classifiers import fit_model, predict_with_scores
from code.core.data_io import (
    build_clean_data_path,
    build_missing_data_path,
    build_run_stem,
    load_real_missing_data,
    load_real_truth_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "result" / "hsai_experiment" / "refine"
TMP_IO_ROOT = Path(
    os.environ.get("HSR_TMP_IO_ROOT", REPO_ROOT / "tmp" / "hsr_refine_imputer_io")
)


def load_ground_truth(dataset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(REPO_ROOT / "data" / "Ground_truth_data" / f"{dataset}.csv")
    return (
        frame.iloc[:, 1:].to_numpy(dtype=float),
        frame.iloc[:, 0].to_numpy(),
        list(frame.columns[1:]),
    )


def nrmse(X_imputed: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    rmse = float(np.sqrt(np.mean((X_imputed[mask] - X_true[mask]) ** 2)))
    col_std = X_true.std(axis=0, ddof=0)
    return rmse / (float(np.mean(col_std[np.where(mask)[1]])) + 1e-12)


def fit_eval_classifier(X_train, y_train, X_test, y_test, *, classifier, random_state):
    classifier_name = {"LogReg": "LogisticRegression"}.get(classifier, classifier)
    Xtr = pd.DataFrame(np.asarray(X_train))
    Xte = pd.DataFrame(np.asarray(X_test), columns=Xtr.columns)
    model = fit_model(
        classifier_name,
        Xtr,
        y_train,
        tuning_cv_folds=int(os.environ.get("HSR_TUNING_CV_FOLDS", "5")),
        tuning_scoring=os.environ.get("HSR_TUNING_SCORING", "f1_macro"),
        tuning_n_jobs=int(os.environ.get("HSR_TUNING_N_JOBS", "-1")),
        compute_backend=os.environ.get("HSR_COMPUTE_BACKEND", "cpu"),
        gpu_device=int(os.environ.get("HSR_GPU_DEVICE", "0")),
    )
    outputs = predict_with_scores(model, Xte)
    y_pred = np.asarray(outputs["y_pred"])
    return {
        "y_pred": y_pred,
        "margin": np.asarray(outputs["margin"], dtype=float),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def hard_sample_inconsistency(ref_pred_ground, ref_pred_imp, margins_ground, *, quantile=0.2):
    if len(margins_ground) == 0:
        return 0.0
    hard = margins_ground <= float(np.quantile(margins_ground, quantile))
    return float(np.mean(ref_pred_ground[hard] != ref_pred_imp[hard])) if hard.any() else 0.0


REQUESTED_FRAMEWORK_BASES = [
    "Mean", "Mice", "Mode", "KNNI", "CMI", "CI", "XGBI", "GMM", "MIDA",
    "SOM", "Holoclean", "MISSF", "MIWAE", "Datawig", "LOESS", "HDI",
    "MLPI", "VAEI", "CDI", "IIM", "DLM", "GINN", "NOMI", "KNNE",
    "GCID", "IFC", "MVNI", "SMC-FCS", "MIGEC", "Sice", "TWDA",
    "MIPCA", "NMF", "POP", "RIBG", "CVBKNN", "BPCA", "LSA", "PMM",
    "CCF", "ESD", "MIST", "CLRMC", "ECLRMC", "NIIA",
]
BASE_CHOICES = list(REQUESTED_FRAMEWORK_BASES)


def load_missing(
    dataset: str,
    missing_percent: float,
    mechanism: str,
    iteration: int,
    seed: int,
    max_len: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Load synthetic missing data using the same seed-aware filename as the framework."""
    for root_name in ("missing_data", "Missing_data"):
        path = (
            REPO_ROOT / "data" / root_name / dataset /
            f"{dataset}_{missing_percent}_{max_len}_{mechanism}_{iteration}_seed{seed}.csv"
        )
        if path.exists():
            frame = pd.read_csv(path, header=None)
            y = frame.iloc[:, 0].to_numpy()
            X = frame.iloc[:, 1:].to_numpy(dtype=float)
            return X, y
    raise FileNotFoundError(
        f"missing data not found for {dataset} mp={missing_percent} "
        f"mechanism={mechanism} iter={iteration} seed={seed}"
    )


def load_real_missing_truth(dataset: str, real_missing_data_dir: str,
                            real_missing_truth_dir: str,
                            label_col: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one native bundle with a shared numeric representation for both tables."""
    args = SimpleNamespace(
        dataset=dataset,
        real_missing_data_dir=real_missing_data_dir,
        real_missing_truth_dir=real_missing_truth_dir,
        real_missing_label_col=label_col,
    )
    observed = load_real_missing_data(args, restore_dtypes=False).reset_index(drop=True)
    truth = load_real_truth_data(args, restore_dtypes=False).reset_index(drop=True)
    if observed.shape != truth.shape:
        raise ValueError(f"observed/truth shape mismatch: {observed.shape} vs {truth.shape}")

    labels = truth.iloc[:, 0]
    valid = labels.notna() & labels.astype("string").str.strip().ne("")
    observed, truth = observed.loc[valid].reset_index(drop=True), truth.loc[valid].reset_index(drop=True)
    labels = truth.iloc[:, 0]
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    if numeric_labels.notna().all():
        y = numeric_labels.to_numpy()
    else:
        categories = sorted(labels.astype(str).unique())
        y = labels.astype(str).map({value: index for index, value in enumerate(categories)}).to_numpy()
    observed_features = observed.iloc[:, 1:].copy()
    truth_features = truth.iloc[:, 1:].copy()
    encoded_observed, encoded_truth = {}, {}
    for column in truth_features.columns:
        observed_col, truth_col = observed_features[column], truth_features[column]
        observed_numeric = pd.to_numeric(observed_col, errors="coerce")
        truth_numeric = pd.to_numeric(truth_col, errors="coerce")
        observed_is_numeric = observed_col.notna().sum() == observed_numeric.notna().sum()
        truth_is_numeric = truth_col.notna().sum() == truth_numeric.notna().sum()
        if observed_is_numeric and truth_is_numeric:
            encoded_observed[column], encoded_truth[column] = observed_numeric, truth_numeric
            continue
        categories = pd.Index(pd.concat([truth_col, observed_col], ignore_index=True).dropna().astype(str).unique())
        mapping = {value: index for index, value in enumerate(categories)}
        encoded_observed[column] = observed_col.astype("string").map(mapping)
        encoded_truth[column] = truth_col.astype("string").map(mapping)

    X_missing = pd.DataFrame(encoded_observed).to_numpy(dtype=float)
    X_true_raw = pd.DataFrame(encoded_truth).to_numpy(dtype=float)
    # Oracle bundles occasionally retain gaps. They cannot enter sklearn, but
    # they also must not be treated as evaluable truth for NRMSE.
    X_true_model = X_true_raw.copy()
    for column in range(X_true_model.shape[1]):
        values = X_true_model[:, column]
        finite = np.isfinite(values)
        fill_value = float(np.mean(values[finite])) if finite.any() else 0.0
        values[~finite] = fill_value
    return X_true_raw, X_true_model, X_missing, y


def _framework_full_table(X_missing: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Rebuild the framework's ``[label, features]`` frame (column 0 = y)."""
    full = np.concatenate([np.asarray(y).reshape(-1, 1),
                           np.asarray(X_missing, dtype=float)], axis=1)
    return pd.DataFrame(full)


def _fill_remaining_features(values, X_missing: np.ndarray) -> np.ndarray:
    """Return a finite feature matrix, preserving observed cells when possible."""
    arr = np.asarray(pd.DataFrame(values), dtype=float).copy()
    expected = np.asarray(X_missing, dtype=float).shape
    if arr.shape == expected[::-1]:
        arr = arr.T
    if arr.shape != expected:
        raise ValueError(f"imputed feature shape mismatch: expected {expected}, got {arr.shape}")

    observed = ~np.isnan(X_missing)
    arr[observed] = X_missing[observed]
    if np.isnan(arr).any() or not np.isfinite(arr).all():
        source = pd.DataFrame(X_missing)
        fallback = source.apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy(dtype=float)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        bad = np.isnan(arr) | ~np.isfinite(arr)
        arr[bad] = np.take(fallback, np.where(bad)[1])
    return arr


def _framework_args(
    *,
    dataset: str,
    missing_percent: float,
    mechanism: str,
    iteration: int,
    seed: int,
    max_missing_lenth: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_dir=str(REPO_ROOT),
        dataset=dataset,
        dataset_category="no_real_missing",
        iter=int(iteration),
        seed=int(seed),
        missing_percent=float(missing_percent),
        max_missing_lenth=int(max_missing_lenth),
        missing_mechanism=str(mechanism),
        mar_logistic_beta=1.0,
        mar_logistic_clip=3.0,
        ground_truth_data_dir=str(REPO_ROOT / "data" / "Ground_truth_data"),
        real_missing_data_dir=str(REPO_ROOT / "data" / "Real_missing_data"),
        real_missing_truth_dir=str(REPO_ROOT / "data" / "Real_missing_truth"),
        real_missing_label_col=0,
        missing_data_dir=str(TMP_IO_ROOT / "missing_data"),
        clean_data_dir=str(TMP_IO_ROOT / "result_data"),
        result_dir=str(TMP_IO_ROOT / "result"),
    )


def _read_framework_features(path: str | Path, X_missing: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path, header=None)
    arr = np.asarray(frame, dtype=float)
    n, d = np.asarray(X_missing).shape
    if arr.shape == (n, d + 1):
        arr = arr[:, 1:]
    elif arr.shape == (n, d):
        pass
    elif arr.shape == (d + 1, n):
        arr = arr.T[:, 1:]
    elif arr.shape == (d, n):
        arr = arr.T
    else:
        raise ValueError(
            f"framework imputer output shape mismatch: expected {(n, d + 1)} "
            f"or {(n, d)}, got {arr.shape}"
        )
    return _fill_remaining_features(arr, X_missing)


def _impute_base_via_framework(
    name: str,
    X_missing: np.ndarray,
    y: np.ndarray,
    *,
    dataset: str,
    missing_percent: float,
    mechanism: str,
    iteration: int,
    seed: int,
    max_missing_lenth: int,
) -> np.ndarray:
    args = _framework_args(
        dataset=dataset,
        missing_percent=missing_percent,
        mechanism=mechanism,
        iteration=iteration,
        seed=seed,
        max_missing_lenth=max_missing_lenth,
    )
    missing_path = Path(build_missing_data_path(args, ensure_parent=True))
    clean_path = Path(build_clean_data_path(args, name, ensure_parent=True))
    if clean_path.exists():
        clean_path.unlink()
    full = _framework_full_table(X_missing, y)
    full.to_csv(missing_path, header=False, index=False, na_rep="")
    run_stem = build_run_stem(
        dataset,
        missing_percent,
        max_missing_lenth,
        mechanism,
        iteration,
        seed=seed,
    )
    framework_impute(args, name, run_stem, dataset)
    if not clean_path.exists():
        raise FileNotFoundError(f"framework imputer did not write expected output: {clean_path}")
    return _read_framework_features(clean_path, X_missing)


def impute_base(name: str, X_missing: np.ndarray, y: np.ndarray,
                *, random_state: int, dataset: str | None = None,
                missing_percent: float | None = None, mechanism: str | None = None,
                iteration: int | None = None, max_missing_lenth: int = 4) -> np.ndarray:
    """Run a base imputer through the framework's canonical imputation entrypoint."""
    if name not in BASE_CHOICES:
        raise ValueError(f"unknown base imputer: {name}")
    if dataset is None or missing_percent is None or mechanism is None or iteration is None:
        raise ValueError(f"framework-backed imputer {name} requires dataset/run context")
    return _impute_base_via_framework(
        name, X_missing, y,
        dataset=dataset,
        missing_percent=missing_percent,
        mechanism=mechanism,
        iteration=iteration,
        seed=random_state,
        max_missing_lenth=max_missing_lenth,
    )


def run_unit(
    dataset, missing_percent, mechanism, iteration, seed,
    *, base, lam_grid, mu, rounds, steps, lr, clip_radius, hard_quantile,
    guard_min_delta, guard_val_fraction, classifiers, max_missing_lenth,
    dataset_category, real_missing_data_dir, real_missing_truth_dir, real_missing_label_col,
):
    if dataset_category == "real_missing_with_truth":
        X_true_eval, X_true, X_missing, y_true = load_real_missing_truth(
            dataset, real_missing_data_dir, real_missing_truth_dir, real_missing_label_col)
    else:
        X_true, y_true, _ = load_ground_truth(dataset)
        X_true_eval = X_true
        X_missing, _ = load_missing(dataset, missing_percent, mechanism, iteration, seed,
                                    max_len=max_missing_lenth)
    if X_true.shape != X_missing.shape:
        raise ValueError(f"shape mismatch {dataset}: {X_true.shape} vs {X_missing.shape}")

    mask = np.isnan(X_missing) & np.isfinite(X_true_eval)
    indices = np.arange(X_true.shape[0])
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=40, stratify=y_true,
    )

    y_train = y_true[train_idx]
    t0 = time.perf_counter()
    X_base = impute_base(
        base, X_missing[train_idx], y_train,
        random_state=seed,
        dataset=dataset,
        missing_percent=missing_percent,
        mechanism=mechanism,
        iteration=iteration,
        max_missing_lenth=max_missing_lenth,
    )
    base_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = hs_refine_guarded(
        X_missing[train_idx], X_base, y_train,
        lam_grid=lam_grid, mu=mu, rounds=rounds, steps=steps, lr=lr,
        clip_radius=clip_radius, hard_quantile=hard_quantile,
        guard_min_delta=guard_min_delta, guard_val_fraction=guard_val_fraction,
        random_state=seed,
    )
    refine_seconds = time.perf_counter() - t0
    X_ref = ref["X_refined"]

    mask_train = mask[train_idx]
    Xtr_true = X_true[train_idx]
    nrmse_base = nrmse(X_base, X_true_eval[train_idx], mask_train)
    nrmse_ref = nrmse(X_ref, X_true_eval[train_idx], mask_train)

    # structure restoration on the train matrix (thesis M1-style diagnostic)
    with np.errstate(invalid="ignore"):
        corr_true = np.corrcoef(Xtr_true, rowvar=False)
        cf_base = np.linalg.norm(
            np.nan_to_num(np.corrcoef(X_base, rowvar=False) - corr_true))
        cf_ref = np.linalg.norm(
            np.nan_to_num(np.corrcoef(X_ref, rowvar=False) - corr_true))

    X_test_true, y_test = X_true[test_idx], y_true[test_idx]
    rows = []
    for classifier in classifiers:
        ref_ground = fit_eval_classifier(
            Xtr_true, y_train, X_test_true, y_test,
            classifier=classifier, random_state=seed,
        )
        for setting, X_train_imp, nr in (
            (base, X_base, nrmse_base),
            (f"{base}+HSR", X_ref, nrmse_ref),
        ):
            out = fit_eval_classifier(
                X_train_imp, y_train, X_test_true, y_test,
                classifier=classifier, random_state=seed,
            )
            hsi = hard_sample_inconsistency(
                ref_ground["y_pred"], out["y_pred"], ref_ground["margin"],
                quantile=0.2,
            )
            rows.append({
                "dataset": dataset, "missing_percent": missing_percent,
                "mechanism": mechanism, "iteration": iteration, "seed": seed,
                "classifier": classifier, "base": base, "setting": setting,
                "nrmse": float(nr),
                "corr_frob_vs_truth": float(cf_base if setting == base else cf_ref),
                "f1_macro": out["f1_macro"], "accuracy": out["accuracy"],
                "f1_drop_vs_ref": float(ref_ground["f1_macro"] - out["f1_macro"]),
                "hard_sample_inconsistency": float(hsi),
                "f1_reference": ref_ground["f1_macro"],
                "base_seconds": float(base_seconds),
                "refine_seconds": float(refine_seconds),
                "selected_lam": float(ref.get("selected_lam", 0.0)),
                "rounds_used": int(ref.get("rounds_used", 0)),
                "moved_fraction": float(ref.get("moved_fraction", 0.0)),
                "mean_abs_dz": float(ref.get("mean_abs_dz", 0.0)),
                "guard_scores": json.dumps(
                    {str(k): round(float(v), 6)
                     for k, v in ref.get("guard_scores", {}).items()},
                    sort_keys=True),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="base vs base+HS-Refine experiment")
    ap.add_argument("--datasets", nargs="+",
                    default=["obesity", "mobile", "wine"])
    ap.add_argument("--dataset_category", choices=["no_real_missing", "real_missing_with_truth"],
                    default="no_real_missing")
    ap.add_argument("--real_missing_data_dir", default=str(REPO_ROOT / "data" / "Real_missing_data"))
    ap.add_argument("--real_missing_truth_dir", default=str(REPO_ROOT / "data" / "Real_missing_truth"))
    ap.add_argument("--real_missing_label_col", type=int, default=0)
    ap.add_argument("--missing", nargs="+", type=float, default=[0.2, 0.3])
    ap.add_argument("--mechanisms", nargs="+", default=["MCAR", "MAR", "MNAR"])
    ap.add_argument("--iters", nargs="+", type=int, default=[1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--bases", nargs="+", default=["XGBI"],
                    choices=BASE_CHOICES)
    ap.add_argument("--classifiers", nargs="+", default=["LogisticRegression", "RandomForest"])
    ap.add_argument("--lam_grid", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--mu", type=float, default=0.0,
                    help="Compatibility argument; HSR no longer uses L_struct.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--clip_radius", type=float, default=2.0)
    ap.add_argument("--hard_quantile", type=float, default=0.2)
    ap.add_argument("--max_missing_lenth", type=int, default=4)
    ap.add_argument("--guard_min_delta", type=float, default=0.0)
    ap.add_argument("--guard_val_fraction", type=float, default=0.2)
    ap.add_argument("--output_csv", default=str(OUTPUT_ROOT / "hsr_results.csv"))
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = (len(args.datasets) * len(args.missing) * len(args.mechanisms)
             * len(args.iters) * len(args.seeds) * len(args.bases))
    print(f"running {total} cells ...", flush=True)

    all_rows, counter = [], 0
    for base in args.bases:
        for dataset in args.datasets:
            for mp in args.missing:
                for mech in args.mechanisms:
                    for it in args.iters:
                        for seed in args.seeds:
                            counter += 1
                            try:
                                rows = run_unit(
                                    dataset, mp, mech, it, seed,
                                    base=base, lam_grid=args.lam_grid,
                                    mu=args.mu, rounds=args.rounds,
                                    steps=args.steps, lr=args.lr,
                                    clip_radius=args.clip_radius,
                                    hard_quantile=args.hard_quantile,
                                    guard_min_delta=args.guard_min_delta,
                                    guard_val_fraction=args.guard_val_fraction,
                                    max_missing_lenth=args.max_missing_lenth,
                                    classifiers=args.classifiers,
                                    dataset_category=args.dataset_category,
                                    real_missing_data_dir=args.real_missing_data_dir,
                                    real_missing_truth_dir=args.real_missing_truth_dir,
                                    real_missing_label_col=args.real_missing_label_col,
                                )
                                all_rows.extend(rows)
                                lam = rows[0]["selected_lam"] if rows else float("nan")
                                print(f"[{counter:>3}/{total}] {base:5s} {dataset:8s} "
                                      f"mp={mp} mech={mech} it={it} seed={seed} "
                                      f"lam*={lam:g} -> {len(rows)} rows", flush=True)
                            except FileNotFoundError as exc:
                                print(f"[{counter:>3}/{total}] SKIP {base} {dataset} "
                                      f"{mp} {mech} {it} {seed}: {exc}", flush=True)
                            except Exception as exc:
                                print(f"[{counter:>3}/{total}] FAIL {base} {dataset} "
                                      f"{mp} {mech} {it} {seed}: {exc}", flush=True)

    if not all_rows:
        print("no rows produced -- exiting", flush=True)
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {len(out_df)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
