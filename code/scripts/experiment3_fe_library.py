import numpy as np
import pandas as pd
import warnings


def fit_statsmodels_lsdv(frame, y_col, x_cols, fe_cols, cluster_cols=None):
    """Fit OLS with dummy-encoded fixed effects via statsmodels."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for experiment 3 fixed-effect regressions."
        ) from exc

    x_cols = list(x_cols or [])
    fe_cols = list(fe_cols or [])
    cluster_cols = list(cluster_cols or [])

    cols = list(dict.fromkeys([y_col] + x_cols + fe_cols + cluster_cols))
    subset = frame[cols].dropna().copy()
    result = {
        "fit_status": "ok",
        "subset": subset,
        "x_cols": x_cols,
        "fe_cols": fe_cols,
        "cluster_cols": cluster_cols,
    }
    if subset.empty:
        result.update({
            "fit_status": "no_complete_cases",
            "n": 0,
            "p": np.nan,
            "dof": np.nan,
            "r2": np.nan,
            "se_type": "classical",
        })
        return result

    x_continuous = subset[x_cols].astype(float) if x_cols else pd.DataFrame(index=subset.index)
    if fe_cols:
        fe_dummies = pd.get_dummies(subset[fe_cols].astype(str), drop_first=True, dtype=float)
    else:
        fe_dummies = pd.DataFrame(index=subset.index)
    X = pd.concat([x_continuous, fe_dummies], axis=1)
    X = sm.add_constant(X)
    y = subset[y_col].astype(float)

    n, p = X.shape
    result.update({
        "design_columns": list(X.columns),
        "n": int(n),
        "p": int(p),
        "dof": int(n - p),
    })
    if n <= p:
        result.update({
            "fit_status": "insufficient_degrees_of_freedom",
            "r2": np.nan,
            "se_type": "classical",
        })
        return result

    model = sm.OLS(y, X).fit()
    se_type = "classical"
    if cluster_cols:
        cluster_series = subset[cluster_cols].astype(str).agg("||".join, axis=1)
        model = model.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_series.values,
        )
        se_type = f"cluster_robust({'+'.join(cluster_cols)})"

    conf_int = model.conf_int(alpha=0.05)
    if isinstance(conf_int, pd.DataFrame):
        conf_int = conf_int.to_numpy()

    result.update({
        "model": model,
        "r2": float(model.rsquared),
        "params": np.asarray(model.params, dtype=float),
        "bse": np.asarray(model.bse, dtype=float),
        "tvalues": np.asarray(model.tvalues, dtype=float),
        "pvalues": np.asarray(model.pvalues, dtype=float),
        "conf_int": np.asarray(conf_int, dtype=float),
        "name_to_idx": {name: idx for idx, name in enumerate(X.columns)},
        "se_type": se_type,
    })
    return result


def fit_linearmodels_absorbingls(frame, y_col, x_cols, fe_cols, cluster_cols=None):
    """Fit OLS with absorbed fixed effects via linearmodels.AbsorbingLS."""
    try:
        from linearmodels.iv import AbsorbingLS
    except ImportError as exc:
        raise ImportError(
            "linearmodels is required for absorbed fixed-effect regressions."
        ) from exc

    x_cols = list(x_cols or [])
    fe_cols = list(fe_cols or [])
    cluster_cols = list(cluster_cols or [])

    cols = list(dict.fromkeys([y_col] + x_cols + fe_cols + cluster_cols))
    subset = frame[cols].dropna().copy().reset_index(drop=True)
    result = {
        "fit_status": "ok",
        "subset": subset,
        "x_cols": x_cols,
        "fe_cols": fe_cols,
        "cluster_cols": cluster_cols,
    }
    if subset.empty:
        result.update({
            "fit_status": "no_complete_cases",
            "n": 0,
            "p": np.nan,
            "dof": np.nan,
            "r2": np.nan,
            "se_type": "classical",
        })
        return result

    y = subset[[y_col]].astype(float)
    x = subset[x_cols].astype(float) if x_cols else pd.DataFrame(index=subset.index)
    absorb = subset[fe_cols].astype("category") if fe_cols else None

    result.update({
        "n": int(len(subset)),
        "p": int(len(x_cols)),
    })
    if x.empty:
        result.update({
            "fit_status": "no_exog_columns",
            "dof": np.nan,
            "r2": np.nan,
            "se_type": "classical",
        })
        return result

    clusters = None
    se_type = "classical"
    if cluster_cols:
        cluster_series = subset[cluster_cols].astype(str).agg("||".join, axis=1)
        clusters = pd.DataFrame({"cl": pd.Categorical(cluster_series)})
        se_type = f"cluster_robust({'+'.join(cluster_cols)})"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AbsorbingLS(
                dependent=y,
                exog=x,
                absorb=absorb,
            )
            if clusters is not None:
                fit = model.fit(cov_type="clustered", clusters=clusters)
            else:
                fit = model.fit()
    except Exception as exc:
        result.update({
            "fit_status": f"fit_error:{exc.__class__.__name__}",
            "dof": np.nan,
            "r2": np.nan,
            "se_type": se_type,
            "fit_error": str(exc),
        })
        return result

    conf_int = fit.conf_int()
    if isinstance(conf_int, pd.DataFrame):
        conf_int = conf_int.to_numpy()

    result.update({
        "model": fit,
        "dof": int(fit.df_resid),
        "r2": float(fit.rsquared),
        "params": np.asarray(fit.params, dtype=float),
        "bse": np.asarray(fit.std_errors, dtype=float),
        "tvalues": np.asarray(fit.tstats, dtype=float),
        "pvalues": np.asarray(fit.pvalues, dtype=float),
        "conf_int": np.asarray(conf_int, dtype=float),
        "name_to_idx": {name: idx for idx, name in enumerate(fit.params.index)},
        "se_type": se_type,
    })
    return result
