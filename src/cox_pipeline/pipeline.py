from __future__ import annotations

import json
from pathlib import Path

from .data import load_config, prepare_dataset
from .metrics import compute_cox_residual_diagnostics, evaluate_brier_and_calibration, evaluate_pit_band, evaluate_threshold_success, evaluate_time_auc, optimize_probability_threshold, validate_cox
from .model import export_cox_formula, fit_cox_model, infer_with_formula, infer_with_lifelines, save_model
from .plots import save_brier_plot, save_calibration_plot, save_cox_residual_diagnostics, save_hazard_ratios, save_inference_probability_curves_with_true_time, save_inference_probability_curves_with_true_time_per_example, save_pit_band_plot, save_risk_time_map_eta, save_single_variable_sensitivity_plot, save_survival_vs_kaplan_meier_plot, save_time_auc_plot, save_validation_plot, save_variable_diagnostics
from .reporting import build_report, write_report


def run_pipeline(config_path: str | Path):
    config = load_config(config_path)
    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    X, T, E, feature_names = prepare_dataset(config)
    save_variable_diagnostics(X, output_dir)
    model = fit_cox_model(X, T, E, penalizer=float(config["model"].get("penalizer", 0.01)))
    residual_diagnostics = compute_cox_residual_diagnostics(model, X, T, E)

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
    threshold_cfg = config.get("threshold_evaluation", {})
    threshold_success = evaluate_threshold_success(
        model,
        X,
        T,
        E,
        threshold=float(threshold_cfg.get("threshold", 0.6)),
    )
    threshold_optimization = optimize_probability_threshold(
        model,
        X,
        T,
        E,
        threshold_grid=threshold_cfg.get("grid"),
    )

    save_model(model, output_dir / "cox_model.joblib")
    formula = export_cox_formula(model, horizons, output_dir / "cox_embedded.json")

    save_hazard_ratios(model, output_dir)
    save_survival_vs_kaplan_meier_plot(model, X, T, E, output_dir)
    save_cox_residual_diagnostics(residual_diagnostics, output_dir)
    save_validation_plot(validation, output_dir)
    save_brier_plot(brier, output_dir)
    save_calibration_plot(brier["calibration"], output_dir)
    save_time_auc_plot(auc, output_dir)
    save_pit_band_plot(pit, output_dir)
    save_risk_time_map_eta(model, X, T, E, output_dir)
    save_inference_probability_curves_with_true_time(model, X, T, E, output_dir)
    save_inference_probability_curves_with_true_time_per_example(model, X, T, E, output_dir)

    continuous_features = config.get("data", {}).get("features", {}).get("continuous", [])
    sensitivity_cfg = config.get("sensitivity", {})
    if continuous_features:
        variable_name = sensitivity_cfg.get("variable", continuous_features[0])
        n_curves = int(sensitivity_cfg.get("n_curves", 6))
        fixed_values = sensitivity_cfg.get("fixed_values", {})
        save_single_variable_sensitivity_plot(
            model,
            X,
            output_dir,
            variable_name=variable_name,
            n_curves=n_curves,
            fixed_values=fixed_values,
        )

    sample = X.head(5).copy()
    infer_with_lifelines(model, sample, horizons).to_csv(output_dir / "inference_lifelines.csv", index=False)
    infer_with_formula(sample, formula).to_csv(output_dir / "inference_formula.csv", index=False)

    report_path = write_report(build_report(validation, brier, auc, pit), output_dir)

    metrics_summary = {
        "n_samples": int(len(X)),
        "n_features": int(len(feature_names)),
        "n_events": int(E.sum()),
        "validation": validation,
        "residual_diagnostics": {
            "n_schoenfeld_rows": int(residual_diagnostics["schoenfeld"].shape[0]),
            "n_martingale_rows": int(residual_diagnostics["martingale"].shape[0]),
            "ph_test_rows": int(residual_diagnostics["ph_test"].shape[0]),
        },
        "brier": brier,
        "pit": {k: v for k, v in pit.items() if k != "u_values"},
        "auc": auc,
        "threshold_success": threshold_success,
        "threshold_optimization": threshold_optimization,
        "report": str(report_path),
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary
