from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_model(X: pd.DataFrame, T: np.ndarray, E: np.ndarray, penalizer: float = 0.01):
    df = X.copy()
    df["T"] = T
    df["E"] = E
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(df, duration_col="T", event_col="E")
    return model


def export_cox_formula(model: CoxPHFitter, horizons: list[float], output_path: str | Path):
    baseline_survival = model.baseline_survival_
    baseline_hazard = model.baseline_cumulative_hazard_

    def sample_step(frame, t):
        idx = np.searchsorted(frame.index.values, t)
        idx = min(idx, len(frame) - 1)
        return float(frame.iloc[idx, 0])

    payload = {
        "formula": "P(T<=t|x) = 1 - S0(t)^exp(sum(beta_j * (x_j - mu_j)))",
        "variables": list(model.params_.index),
        "beta": {k: float(v) for k, v in model.params_.to_dict().items()},
        "mu": {k: float(v) for k, v in model._norm_mean.to_dict().items()},
        "baseline_survival": {str(t): sample_step(baseline_survival, t) for t in horizons},
        "baseline_cumulative_hazard": {str(t): sample_step(baseline_hazard, t) for t in horizons},
        "concordance_index": float(model.concordance_index_),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def save_model(model: CoxPHFitter, output_path: str | Path):
    joblib.dump(model, output_path)


def load_formula(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_with_lifelines(model: CoxPHFitter, configs: pd.DataFrame, horizons: list[float]) -> pd.DataFrame:
    survival = model.predict_survival_function(configs, times=horizons)
    result = pd.DataFrame(index=configs.index)
    for t in horizons:
        result[f"P(T<={t})"] = 1.0 - survival.loc[t].values
    result["hazard_ratio"] = model.predict_partial_hazard(configs).values
    return result


def infer_with_formula(configs: pd.DataFrame, formula: dict) -> pd.DataFrame:
    beta = formula["beta"]
    mu = formula["mu"]
    horizons = sorted(formula["baseline_survival"].keys(), key=float)

    lp = np.zeros(len(configs))
    for var, coef in beta.items():
        if var in configs.columns:
            values = configs[var].astype(float).to_numpy()
        else:
            values = np.full(len(configs), mu[var])
        lp += coef * (values - mu[var])

    out = pd.DataFrame(index=configs.index)
    out["linear_predictor"] = lp
    out["hazard_ratio"] = np.exp(lp)
    for t in horizons:
        s0 = formula["baseline_survival"][t]
        out[f"P(T<={t})"] = 1.0 - np.power(s0, np.exp(lp))
    return out.clip(lower=0, upper=1)
