"""HS-Refine (HSR): hard-sample-loss-guided differentiable refinement (Route 2).

A **method-agnostic, plug-in refinement layer**. HSR takes the output of ANY
base imputer, treats ONLY the missing cells as free variables ``z`` and
minimises an *explicit* composite objective by projected Adam:

    J(z) = L_anchor(z) + lam * L_HS(z)

    L_anchor  = mean_{(i,j) in M} (z_ij - z_ij^base)^2          (trust region)
    L_HS      = sum_i g_i * CE(proxy(x_i(z)), y_i) / sum_i g_i  (hard-sample
                surrogate; g_i = h_i * c_i with the SAME soft hard-membership
                h and inconsistency gate c conventions)

Why each term exists (thesis narrative closure):

* the raw hard-sample inconsistency *rate* is piecewise constant in z (zero
  gradient a.e.), so ``L_HS`` replaces it with a differentiable surrogate:
  soft-hard-membership-weighted cross entropy of a *frozen* logistic proxy;
* minimising ``L_HS`` alone is degenerate (it would inject the label into the
  features and fabricate separability), so ``L_anchor`` pins the solution to
  the base imputation and a hard projection ``|z - z_base| <= clip_radius``
  bounds every cell's movement in standardised units;

Optimisation structure (EM-flavoured outer loop, descent inner loop):

    round r:  fit proxy on current X  ->  freeze proxy, h, c
              inner projected Adam on missing cells minimising J
              do-no-harm check (internal stratified validation F1);
              revert + stop if the round did not help

The proxy is REFIT between rounds but FROZEN within the inner loop (the
target-network trick): letting the proxy co-adapt with z during optimisation
is exactly the degenerate collusion we must avoid.

``lam=0`` returns the base imputation byte-identically, so every comparison is
a clean single-factor ablation. The active gradient is analytic and has no
autograd dependency. ``mu`` is accepted by public APIs for backward CLI
compatibility, but the structure term has been removed.

Only training rows are refined.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_EPS = 1e-12


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _guard_f1(X_candidate, y, train_idx, val_idx, *, guard_X_true, random_state):
    if val_idx.size == 0 or train_idx.size == 0:
        return float("-inf")
    X_val = X_candidate[val_idx] if guard_X_true is None else np.asarray(guard_X_true, dtype=float)[val_idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_candidate[train_idx])
    X_eval = scaler.transform(X_val)
    clf = LogisticRegression(solver="lbfgs", max_iter=400, C=1.0, random_state=random_state)
    clf.fit(X_train, y[train_idx])
    pred = clf.predict(X_eval)
    return _macro_f1(y[val_idx], pred)


def hs_consistency_weights(margins, y_pred, y_true, *, hard_quantile=0.2,
                           beta=3.0, soft=True, min_weight=1e-3):
    margins = np.asarray(margins, dtype=float).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    if margins.size == 0:
        return {"weights": margins, "hard": margins, "inconsistency": margins}
    tau = float(np.quantile(margins, hard_quantile))
    if soft:
        scale = max(float(np.std(margins)) * 0.5, 1e-6)
        hard = 1.0 / (1.0 + np.exp((margins - tau) / scale))
    else:
        hard = (margins <= tau).astype(float)
    inconsistency = (y_pred != y_true).astype(float)
    weights = np.clip(1.0 + beta * hard * inconsistency, min_weight, None)
    weights *= margins.size / weights.sum()
    return {"weights": weights, "hard": hard, "inconsistency": inconsistency, "tau": tau}


def _guard_split_prefer_complete(
    y: np.ndarray,
    row_complete: np.ndarray,
    *,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified guard split whose validation rows PREFER fully observed rows.

    Complete rows carry the true data geometry (no imputed cell at all), so
    validating on them measures transfer to the truth rather than to the base
    imputer's geometry. Incomplete rows fill up the validation quota only
    when a class has too few complete rows. Falls back to the plain
    stratified split when (almost) no complete rows exist.
    """
    y = np.asarray(y).ravel()
    row_complete = np.asarray(row_complete, dtype=bool).ravel()
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if idx.size < 2:
            train_idx.extend(idx.tolist())
            continue
        n_val = int(round(idx.size * val_fraction))
        n_val = min(max(1, n_val), idx.size - 1)
        comp = idx[row_complete[idx]]
        incomp = idx[~row_complete[idx]]
        rng.shuffle(comp)
        rng.shuffle(incomp)
        take_c = min(n_val, comp.size)
        val = list(comp[:take_c]) + list(incomp[: n_val - take_c])
        val_set = set(val)
        train = [i for i in idx if i not in val_set]
        val_idx.extend(val)
        train_idx.extend(train)
    return np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)


# ---------------------------------------------------------------------------
# Proxy: multinomial logistic regression with exposed weights (for gradients)
# ---------------------------------------------------------------------------


class _Proxy:
    """Frozen linear-softmax proxy: p = softmax((x - mu)/sd @ Wm + b).

    Exposes everything the analytic gradient needs: ``mu, sd, Wm (d,K), b``.
    """

    def __init__(self, mu, sd, Wm, b, classes):
        self.mu = np.asarray(mu, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.Wm = np.asarray(Wm, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.classes = np.asarray(classes)

    def proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        Z = Xs @ self.Wm + self.b
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def margins_pred(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = self.proba(X)
        if P.shape[1] == 1:
            return np.ones(len(P)), self.classes[np.zeros(len(P), dtype=int)]
        sp = np.sort(P, axis=1)
        margins = sp[:, -1] - sp[:, -2]
        return margins, self.classes[np.argmax(P, axis=1)]


def fit_proxy(X: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> _Proxy:
    """Fit the standardized multinomial logistic-regression proxy."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    classes = np.unique(y)
    clf = LogisticRegression(solver="lbfgs", max_iter=300, C=1.0,
                             random_state=random_state).fit(Xs, y)
    coef = np.asarray(clf.coef_, dtype=float)
    intercept = np.asarray(clf.intercept_, dtype=float)
    if len(clf.classes_) == 2 and coef.shape[0] == 1:
        # Binary sklearn stores one row: score for classes_[1] versus class 0.
        Wm = np.column_stack([np.zeros(coef.shape[1]), coef[0]])
        b = np.array([0.0, intercept[0]])
    else:
        Wm = coef.T
        b = intercept
    return _Proxy(mu, sd, Wm, b, np.asarray(clf.classes_))


# ---------------------------------------------------------------------------
# Structure target: pairwise-complete observed correlation
# ---------------------------------------------------------------------------


def pairwise_observed_corr(
    X_missing: np.ndarray,
    X_fallback: np.ndarray,
    *,
    min_pairs: int = 10,
) -> np.ndarray:
    """Honest data-driven correlation target.

    Entry (a, b) is the Pearson correlation over rows where BOTH columns are
    observed in ``X_missing``; falls back to ``corr(X_fallback)`` when too few
    complete pairs exist or a column is (locally) constant.
    """
    X = np.asarray(X_missing, dtype=float)
    d = X.shape[1]
    obs = ~np.isnan(X)
    with np.errstate(invalid="ignore"):
        Rf = np.corrcoef(np.asarray(X_fallback, dtype=float), rowvar=False)
    Rf = np.atleast_2d(Rf)
    Rf = np.where(np.isfinite(Rf), Rf, 0.0)
    R = np.eye(d)
    for a in range(d):
        for b in range(a + 1, d):
            rows = obs[:, a] & obs[:, b]
            r = np.nan
            if rows.sum() >= min_pairs:
                xa = X[rows, a] - X[rows, a].mean()
                xb = X[rows, b] - X[rows, b].mean()
                den = np.sqrt((xa * xa).sum() * (xb * xb).sum())
                if den > 1e-12:
                    r = float((xa * xb).sum() / den)
            if not np.isfinite(r):
                r = float(Rf[a, b]) if a < Rf.shape[0] and b < Rf.shape[1] else 0.0
            r = float(np.clip(r, -1.0, 1.0))
            R[a, b] = R[b, a] = r
    return R


# ---------------------------------------------------------------------------
# Loss terms with analytic gradients (all in z-space; see design doc)
# ---------------------------------------------------------------------------


def struct_loss_grad(
    Z: np.ndarray,
    R_target: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, np.ndarray]:
    """|| corr(Z_valid) - R*_valid ||_F^2 / d_v^2 and its gradient w.r.t. Z.

    Correlation is invariant to the per-column affine map between raw and
    z-space, so both loss and target live directly in z-space.

    Gradient derivation (verified by finite differences): with centred
    ``Xt = Z_v - mean``, ``m = n - 1``, ``S = Xt'Xt/m``, ``R = S / (s s')``,
    ``P = (2/d_v^2) (R - R*)`` (diagonal zeroed),

        dL/dS = P / (s s')            off-diagonal
        dL/dS_aa = - q_a / S_aa,      q_a = sum_b P_ab R_ab
        dL/dZ_v = (2/m) Xt (dL/dS)
    """
    Z = np.asarray(Z, dtype=float)
    n, d = Z.shape
    grad = np.zeros_like(Z)
    vidx = np.where(valid)[0]
    d_v = vidx.size
    if n < 3 or d_v < 2:
        return 0.0, grad
    Zv = Z[:, vidx]
    Xt = Zv - Zv.mean(axis=0)
    m = n - 1
    S = (Xt.T @ Xt) / m
    diag = np.clip(np.diag(S), 1e-10, None)
    s = np.sqrt(diag)
    R = S / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    D = R - R_target[np.ix_(vidx, vidx)]
    np.fill_diagonal(D, 0.0)
    loss = float((D * D).sum() / (d_v * d_v))
    P = (2.0 / (d_v * d_v)) * D
    G = P / np.outer(s, s)
    q = (P * R).sum(axis=1)
    np.fill_diagonal(G, -q / diag)
    grad_v = (2.0 / m) * (Xt @ G)
    grad[:, vidx] = grad_v
    return loss, grad


def hs_loss_grad(
    Z: np.ndarray,
    proxy: _Proxy,
    y_idx: np.ndarray,
    g: np.ndarray,
    mu0: np.ndarray,
    sd0: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Weighted cross entropy of the FROZEN proxy and its gradient w.r.t. Z.

    loss = sum_i g_i * CE_i / sum_i g_i;  dCE_i/dx = Wm (p_i - e_{y_i}) / sd_p
    chained through x = mu0 + sd0 * z.
    """
    Sg = float(g.sum())
    if Sg < 1e-9:
        return 0.0, np.zeros_like(Z)
    X_raw = mu0 + sd0 * Z
    P = proxy.proba(X_raw)
    n = len(y_idx)
    p_true = np.clip(P[np.arange(n), y_idx], _EPS, 1.0)
    ce = -np.log(p_true)
    loss = float((g * ce).sum() / Sg)
    Y = np.zeros_like(P)
    Y[np.arange(n), y_idx] = 1.0
    grad_raw = (P - Y) @ proxy.Wm.T / proxy.sd  # (n, d) w.r.t. raw x
    grad = grad_raw * (g / Sg)[:, None] * sd0
    return loss, grad


# ---------------------------------------------------------------------------
# Inner solver: projected Adam on the missing cells
# ---------------------------------------------------------------------------


def _inner_adam(
    z_base_flat: np.ndarray,   # anchor values of the missing cells (z units)
    z_init_flat: np.ndarray,   # current values of the missing cells
    Z_fixed: np.ndarray,       # full matrix; observed cells are constants
    mask: np.ndarray,
    proxy: _Proxy,
    y_idx: np.ndarray,
    g: np.ndarray,
    mu0: np.ndarray,
    sd0: np.ndarray,
    *,
    lam: float,
    steps: int,
    lr: float,
    clip_radius: float,
    tol: float,
    patience: int,
) -> tuple[np.ndarray, dict]:
    n_miss = z_base_flat.size
    zv = z_init_flat.copy()
    lo = z_base_flat - clip_radius
    hi = z_base_flat + clip_radius
    m1 = np.zeros(n_miss)
    m2 = np.zeros(n_miss)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss, best_z, bad = np.inf, zv.copy(), 0
    trace = []

    def _objective(zflat: np.ndarray) -> tuple[float, np.ndarray]:
        Z = Z_fixed.copy()
        Z[mask] = zflat
        diff = zflat - z_base_flat
        l_anchor = float((diff * diff).mean())
        grad = 2.0 * diff / n_miss
        total = l_anchor
        if lam > 0:
            l_hs, g_hs = hs_loss_grad(Z, proxy, y_idx, g, mu0, sd0)
            total += lam * l_hs
            grad = grad + lam * g_hs[mask]
        return total, grad

    for t in range(steps):
        loss, grad = _objective(zv)
        trace.append(loss)
        if loss < best_loss - tol:
            best_loss, best_z, bad = loss, zv.copy(), 0
        else:
            bad += 1
            if bad > patience:
                break
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        mhat = m1 / (1 - b1 ** (t + 1))
        vhat = m2 / (1 - b2 ** (t + 1))
        zv = zv - lr * mhat / (np.sqrt(vhat) + eps)
        zv = np.clip(zv, lo, hi)

    return best_z, {"loss_trace": trace, "best_loss": float(best_loss),
                    "steps_run": len(trace)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hs_refine(
    X_missing: np.ndarray,
    X_base: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = 1.0,
    mu: float = 0.0,
    hard_quantile: float = 0.2,
    soft: bool = True,
    rounds: int = 2,
    steps: int = 300,
    lr: float = 0.02,
    clip_radius: float = 2.0,
    tol: float = 1e-7,
    patience: int = 40,
    min_pairs: int = 10,
    round_guard: bool = True,
    guard_val_fraction: float = 0.2,
    random_state: int = 0,
) -> dict:
    """Refine one base imputation of the training-row feature matrix.

    Returns a dict with ``X_refined`` plus diagnostics. ``lam=0`` returns
    ``X_base`` byte-identically. ``mu`` is ignored and kept for compatibility.
    """
    X_missing = np.asarray(X_missing, dtype=float)
    X_base = np.asarray(X_base, dtype=float)
    y = np.asarray(y).ravel()
    mask = np.isnan(X_missing)
    info: dict = {"rounds_used": 0, "n_gated": [], "inner": [],
                  "round_guard_f1": []}

    if not mask.any() or lam <= 0:
        info["X_refined"] = X_base.copy()
        return info

    n, d = X_base.shape
    classes = np.unique(y)
    cls_idx = {c: k for k, c in enumerate(classes)}
    y_idx = np.array([cls_idx[v] for v in y])

    # z-space: standardise columns by base-imputation statistics.
    mu0 = X_base.mean(axis=0)
    sd0 = X_base.std(axis=0)
    sd0 = np.where(sd0 > 1e-8, sd0, 1.0)
    Z_base = (X_base - mu0) / sd0

    z_anchor = Z_base[mask].copy()
    z_cur = z_anchor.copy()

    # Guard validation rows (i) come from the UNREFINED base matrix and
    # (ii) prefer fully observed rows. Validating on candidate rows is
    # self-confirming (a degenerate label-injection solution also "fixes"
    # its own validation rows); complete rows additionally carry the TRUE
    # geometry, so fabricated separability cannot transfer to them.
    guard_train = guard_val = None
    f1_cur = None
    if round_guard:
        guard_train, guard_val = _guard_split_prefer_complete(
            y, ~mask.any(axis=1),
            val_fraction=guard_val_fraction, random_state=random_state + 7919,
        )
        f1_cur = _guard_f1(X_base, y, guard_train, guard_val,
                           guard_X_true=X_base, random_state=random_state)

    for r in range(max(1, rounds)):
        Z = Z_base.copy()
        Z[mask] = z_cur
        X_cur = mu0 + sd0 * Z
        proxy = fit_proxy(X_cur, y, random_state=random_state + r)
        margins, y_pred = proxy.margins_pred(X_cur)
        hs = hs_consistency_weights(margins, y_pred, y,
                                    hard_quantile=hard_quantile,
                                    beta=1.0, soft=soft)
        g = hs["hard"] * hs["inconsistency"]
        n_gated = int(((hs["hard"] > 0.5) & (hs["inconsistency"] > 0.5)).sum())
        info["n_gated"].append(n_gated)
        if lam > 0 and g.sum() < 1e-9:
            break  # nothing hard & inconsistent left

        z_new, inner = _inner_adam(
            z_anchor, z_cur, Z_base, mask, proxy, y_idx, g, mu0, sd0,
            lam=lam, steps=steps, lr=lr, clip_radius=clip_radius,
            tol=tol, patience=patience,
        )
        info["inner"].append(inner)

        if round_guard:
            Zc = Z_base.copy()
            Zc[mask] = z_new
            X_cand = mu0 + sd0 * Zc
            f1_new = _guard_f1(X_cand, y, guard_train, guard_val,
                               guard_X_true=X_base, random_state=random_state)
            info["round_guard_f1"].append((float(f1_cur), float(f1_new)))
            if f1_new < f1_cur:  # do no harm: revert this round and stop
                break
            f1_cur = f1_new
        z_cur = z_new
        info["rounds_used"] = r + 1

    Z = Z_base.copy()
    Z[mask] = z_cur
    X_refined = mu0 + sd0 * Z
    # observed cells are untouched by construction; enforce exactly anyway.
    X_refined[~mask] = X_base[~mask]

    dz = np.abs(z_cur - z_anchor)
    info.update({
        "X_refined": X_refined,
        "moved_fraction": float((dz > 1e-6).mean()),
        "mean_abs_dz": float(dz.mean()),
        "max_abs_dz": float(dz.max() if dz.size else 0.0),
    })
    return info


def hs_refine_guarded(
    X_missing: np.ndarray,
    X_base: np.ndarray,
    y: np.ndarray,
    *,
    lam_grid: list[float] | tuple[float, ...] = (0.3, 1.0, 3.0),
    mu: float = 0.0,
    guard_min_delta: float = 0.0,
    guard_val_fraction: float = 0.2,
    random_state: int = 0,
    **refine_kwargs,
) -> dict:
    """Grid the HS weight ``lam`` under a do-no-harm internal validation guard.

    ``mu`` is ignored and kept for backward CLI compatibility. Mirrors
    Every candidate (and the base itself) is scored
    by validation macro-F1 of a light classifier trained on the candidate
    matrix; if no candidate beats the base by ``guard_min_delta`` the base is
    returned (``selected_lam = 0``). Validation rows always come from the
    UNREFINED base matrix so a degenerate label-injection solution cannot
    game its own acceptance test (see comment in :func:`hs_refine`).
    """
    X_missing = np.asarray(X_missing, dtype=float)
    X_base = np.asarray(X_base, dtype=float)
    y = np.asarray(y).ravel()
    guard_train, guard_val = _guard_split_prefer_complete(
        y, ~np.isnan(X_missing).any(axis=1),
        val_fraction=guard_val_fraction, random_state=random_state + 7919,
    )
    base_f1 = _guard_f1(X_base, y, guard_train, guard_val,
                        guard_X_true=X_base, random_state=random_state)
    guard_scores: dict = {0.0: float(base_f1)}
    best_lam, best_f1, best_out = 0.0, base_f1, None
    for lam in lam_grid:
        out = hs_refine(
            X_missing, X_base, y, lam=float(lam), mu=mu,
            guard_val_fraction=guard_val_fraction,
            random_state=random_state, **refine_kwargs,
        )
        f1 = _guard_f1(out["X_refined"], y, guard_train, guard_val,
                       guard_X_true=X_base, random_state=random_state)
        guard_scores[float(lam)] = float(f1)
        if f1 > best_f1:
            best_lam, best_f1, best_out = float(lam), f1, out

    if best_out is None or best_f1 < base_f1 + guard_min_delta:
        return {"X_refined": X_base.copy(), "selected_lam": 0.0,
                "guard_scores": guard_scores, "rounds_used": 0,
                "n_gated": [], "moved_fraction": 0.0,
                "mean_abs_dz": 0.0, "max_abs_dz": 0.0}
    best_out["selected_lam"] = best_lam
    best_out["guard_scores"] = guard_scores
    return best_out
