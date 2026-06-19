from __future__ import annotations

import json
from pathlib import Path

from .data import load_config, prepare_dataset
from .metrics import evaluate_brier_and_calibration, evaluate_pit_band, evaluate_time_auc, validate_cox
from .model import export_cox_formula, fit_cox_model, infer_with_formula, infer_with_lifelines, save_model
from .plots import save_brier_plot, save_calibration_plot, save_hazard_ratios, save_pit_band_plot, save_time_auc_plot, save_validation_plot
from .reporting import build_report, write_report


def run_pipeline(config_path: str | Path):
    config = load_config(config_path)
    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    X, T, E, feature_names = prepare_dataset(config)
    model = fit_cox_model(X, T, E, penalizer=float(config["model"].get("penalizer", 0.01)))

    horizons = [float(t) for t in config["model"]["time_horizons"]]
    validation_cfg = config["validation"]

    validation = validate_cox(
        X, T, E,
        penalizer=float(config["model"].get("penalizer", 0.01)),
        n_folds=int(validation_cfg.get("cv_folds", 5)),
        n_boot=int(validation_cfg.get("bootstrap_n_iter", 200)),
        random_state=int(validation_cfg.get("random_state", 42)),
    )
    brier = evaluate_brier_and_calibration(model, X, T, E, horizons)
    pit = evaluate_pit_band(model, X, T, E, band=tuple(validation_cfg.get("pit_band", [0.8, 1.0])))
    auc = evaluate_time_auc(model, X, T, E, horizons, n_times=int(validation_cfg.get("auc_n_times", 20)))

    save_model(model, output_dir / "cox_model.joblib")
    formula = export_cox_formula(model, horizons, output_dir / "cox_embedded.json")

    save_hazard_ratios(model, output_dir)
    save_validation_plot(validation, output_dir)
    save_brier_plot(brier, output_dir)
    save_calibration_plot(brier["calibration"], output_dir)
    save_time_auc_plot(auc, output_dir)
    save_pit_band_plot(pit, output_dir)

    sample = X.head(5).copy()
    infer_with_lifelines(model, sample, horizons).to_csv(output_dir / "inference_lifelines.csv", index=False)
    infer_with_formula(sample, formula).to_csv(output_dir / "inference_formula.csv", index=False)

    report_path = write_report(build_report(validation, brier, auc, pit), output_dir)

    metrics_summary = {
        "n_samples": int(len(X)),
        "n_features": int(len(feature_names)),
        "n_events": int(E.sum()),
        "validation": validation,
        "brier": brier,
        "pit": {k: v for k, v in pit.items() if k != "u_values"},
        "auc": auc,
        "report": str(report_path),
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary
