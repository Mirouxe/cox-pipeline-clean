from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import KFold
from scipy import stats


def km_step(time: np.ndarray, event_observed: np.ndarray):
    kmf = KaplanMeierFitter().fit(time, event_observed)
    idx = kmf.survival_function_.index.values.astype(float)
    val = kmf.survival_function_.iloc[:, 0].values.astype(float)
    return idx, val


def step_eval(idx: np.ndarray, val: np.ndarray, query):
    query = np.atleast_1d(np.asarray(query, dtype=float))
    pos = np.searchsorted(idx, query, side="right") - 1
    out = np.ones_like(query, dtype=float)
    valid = pos >= 0
    out[valid] = val[np.clip(pos[valid], 0, len(val) - 1)]
    return out


def brier_score_ipcw(T, E, surv_matrix, times):
    cidx, cval = km_step(T, 1 - E)
    G_Ti = np.clip(step_eval(cidx, cval, T), 1e-8, None)
    bs = np.empty(len(times))
    for k, t in enumerate(times):
        S_t = surv_matrix[:, k]
        G_t = max(float(step_eval(cidx, cval, t)[0]), 1e-8)
        cat1 = (T <= t) & (E == 1)
        cat2 = T > t
        contrib = np.zeros(len(T))
        contrib[cat1] = (S_t[cat1] ** 2) / G_Ti[cat1]
        contrib[cat2] = ((1 - S_t[cat2]) ** 2) / G_t
        bs[k] = contrib.mean()
    return bs


def integrated_brier_score(times, bs):
    trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
    return float(trapz(bs, times) / (times[-1] - times[0]))


def cumulative_dynamic_auc_ipcw(T, E, risk, times):
    cidx, cval = km_step(T, 1 - E)
    weights_all = 1.0 / np.clip(step_eval(cidx, cval, T), 1e-8, None)
    aucs = np.full(len(times), np.nan)

    for k, t in enumerate(times):
        cases = (T <= t) & (E == 1)
        controls = T > t
        if not cases.any() or not controls.any():
            continue
        rc = risk[cases][:, None]
        ro = risk[controls][None, :]
        concordant = (rc > ro).astype(float) + 0.5 * (rc == ro)
        w = weights_all[cases][:, None]
        denom = float(w.sum() * controls.sum())
        if denom > 0:
            aucs[k] = float((w * concordant).sum() / denom)

    eidx, evals = km_step(T, E)
    s_t = step_eval(eidx, evals, times)
    s_prev = np.concatenate(([s_t[0]], s_t[:-1]))
    weights = np.clip(s_prev - s_t, 0, None)
    valid = ~np.isnan(aucs)
    mean_auc = float(np.sum(aucs[valid] * weights[valid]) / weights[valid].sum()) if valid.any() and weights[valid].sum() > 0 else float(np.nan)
    return aucs, mean_auc


def validate_cox(X: pd.DataFrame, T: np.ndarray, E: np.ndarray, penalizer: float, n_folds: int, n_boot: int, random_state: int):
    df = X.copy()
    df["T"] = T
    df["E"] = E

    def fit_model(data):
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(data, duration_col="T", event_col="E")
        return model

    def cindex(model, data):
        return float(model.score(data, scoring_method="concordance_index"))

    full_model = fit_model(df)
    apparent = cindex(full_model, df)

    cv_scores = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in kf.split(df):
        model = fit_model(df.iloc[tr])
        cv_scores.append(cindex(model, df.iloc[te]))
    cv_scores = np.asarray(cv_scores, dtype=float)

    rng = np.random.default_rng(random_state)
    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        boot = df.iloc[idx]
        try:
            model = fit_model(boot)
            optimisms.append(cindex(model, boot) - cindex(model, df))
        except Exception:
            continue
    optimisms = np.asarray(optimisms, dtype=float)

    return {
        "c_index_apparent": apparent,
        "c_index_cv_mean": float(cv_scores.mean()),
        "c_index_cv_std": float(cv_scores.std()),
        "c_index_cv_per_fold": [float(x) for x in cv_scores],
        "optimism": float(optimisms.mean()) if len(optimisms) else 0.0,
        "c_index_corrected": float(apparent - optimisms.mean()) if len(optimisms) else float(apparent),
        "bootstrap_n_valid": int(len(optimisms)),
    }


def evaluate_brier_and_calibration(model, X, T, E, horizons):
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    t_grid = np.linspace(np.percentile(T, 5), min(np.percentile(T, 95), T.max() * 0.999), 25)

    surv_df = model.predict_survival_function(X, times=t_grid)
    surv_cox = surv_df.values.T

    km = KaplanMeierFitter().fit(T, E)
    km_surv_at = km.survival_function_at_times(t_grid).values
    surv_km = np.tile(km_surv_at, (len(T), 1))

    bs_cox = brier_score_ipcw(T, E, surv_cox, t_grid)
    bs_km = brier_score_ipcw(T, E, surv_km, t_grid)

    calibration = {}
    valid_horizons = [float(t) for t in horizons if T.min() < float(t) < T.max()]
    for t in valid_horizons:
        surv_t = model.predict_survival_function(X, times=[t]).values.ravel()
        pred_risk = 1.0 - surv_t
        try:
            bins = pd.qcut(pd.Series(pred_risk).rank(method="first"), 5, labels=False)
        except ValueError:
            bins = pd.cut(pred_risk, 5, labels=False)
        obs, prd = [], []
        for b in range(5):
            mask = bins == b
            if mask.sum() < 5:
                continue
            kmb = KaplanMeierFitter().fit(T[mask], E[mask])
            obs.append(1.0 - float(kmb.survival_function_at_times(t).values[0]))
            prd.append(float(pred_risk[mask].mean()))
        calibration[str(t)] = {"predicted": prd, "observed": obs}

    ibs_cox = integrated_brier_score(t_grid, bs_cox)
    ibs_km = integrated_brier_score(t_grid, bs_km)
    return {
        "times": [float(t) for t in t_grid],
        "brier_cox": [float(x) for x in bs_cox],
        "brier_km": [float(x) for x in bs_km],
        "ibs_cox": ibs_cox,
        "ibs_km": ibs_km,
        "skill_score": float(1 - ibs_cox / ibs_km) if ibs_km > 0 else float("nan"),
        "calibration": calibration,
    }


def evaluate_pit_band(model, X, T, E, band=(0.8, 1.0)):
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    sf = model.predict_survival_function(X)
    sf_times = sf.index.values
    sf_vals = sf.values

    event_mask = E == 1
    t_events = T[event_mask]
    idx_cols = np.where(event_mask)[0]
    pos = np.searchsorted(sf_times, t_events, side="right") - 1
    pos = np.clip(pos, 0, len(sf_times) - 1)
    u = 1.0 - sf_vals[pos, idx_cols]

    counts, _ = np.histogram(u, bins=np.linspace(0, 1, 11))
    expected = len(u) / 10
    chi2 = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else float("nan")
    pvalue = float(1 - stats.chi2.cdf(chi2, df=9))

    a, b = band
    return {
        "u_values": u.tolist(),
        "hist_counts": counts.tolist(),
        "fraction_in_band": float(np.mean((u >= a) & (u <= b))),
        "expected_fraction": float(b - a),
        "dcalibration_chi2": chi2,
        "dcalibration_pvalue": pvalue,
        "band": [float(a), float(b)],
    }


def evaluate_time_auc(model, X, T, E, horizons, n_times=20):
    T = np.asarray(T, dtype=float)
    risk = model.predict_partial_hazard(X).values
    auc_times = np.linspace(np.percentile(T, 10), min(np.percentile(T, 90), T.max() * 0.999), n_times)
    auc_values, auc_mean = cumulative_dynamic_auc_ipcw(T, E, risk, auc_times)
    horizon_auc, _ = cumulative_dynamic_auc_ipcw(T, E, risk, np.asarray(horizons, dtype=float))
    return {
        "auc_times": [float(t) for t in auc_times],
        "auc_values": [float(v) for v in auc_values],
        "auc_mean": float(auc_mean),
        "auc_at_horizons": {str(t): float(v) for t, v in zip(horizons, horizon_auc)},
    }


def evaluate_threshold_success(model, X, T, E, threshold=0.6):
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    event_mask = E == 1

    X_event = X.loc[event_mask]
    T_event = T[event_mask]
    surv = model.predict_survival_function(X_event)
    times = surv.index.values.astype(float)
    proba = 1.0 - surv.values

    crossing_times = []
    success_flags = []
    for j in range(proba.shape[1]):
        curve = proba[:, j]
        above = np.where(curve >= threshold)[0]
        if len(above) == 0:
            t_cross = np.nan
            success = False
        else:
            t_cross = float(times[above[0]])
            success = bool(T_event[j] <= t_cross)
        crossing_times.append(t_cross)
        success_flags.append(success)

    success_rate = float(np.mean(success_flags)) if success_flags else float("nan")
    details = [
        {
            "event_index": int(idx),
            "real_event_time": float(t_real),
            "threshold_crossing_time": None if np.isnan(t_cross) else float(t_cross),
            "success": bool(success),
        }
        for idx, t_real, t_cross, success in zip(np.where(event_mask)[0], T_event, crossing_times, success_flags)
    ]
    return {
        "threshold": float(threshold),
        "n_events": int(event_mask.sum()),
        "success_rate": success_rate,
        "n_success": int(np.sum(success_flags)),
        "details": details,
    }


def optimize_probability_threshold(model, X, T, E, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.linspace(0.05, 0.95, 19)

    evaluations = []
    for threshold in threshold_grid:
        result = evaluate_threshold_success(model, X, T, E, threshold=float(threshold))
        evaluations.append({
            "threshold": float(threshold),
            "success_rate": float(result["success_rate"]),
            "n_success": int(result["n_success"]),
            "n_events": int(result["n_events"]),
        })

    max_success_rate = max(row["success_rate"] for row in evaluations)
    best_rows = [row for row in evaluations if row["success_rate"] == max_success_rate]
    best_row = max(best_rows, key=lambda row: row["threshold"])

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "success_rate": float(best_row["success_rate"]),
        "n_success": int(best_row["n_success"]),
        "n_events": int(best_row["n_events"]),
        "grid_results": evaluations,
    }
