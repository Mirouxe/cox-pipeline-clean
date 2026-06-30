from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde
from lifelines import KaplanMeierFitter


def save_variable_diagnostics(X, output_dir: Path):
    X = X.astype(float)
    diag_dir = output_dir / "variable_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for col in X.columns:
        series = X[col].dropna().astype(float)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)] if iqr > 0 else series.iloc[0:0]

        summary_rows.append({
            "variable": col,
            "n": int(series.shape[0]),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "q1": q1,
            "median": float(series.median()),
            "q3": q3,
            "max": float(series.max()),
            "n_outliers_iqr": int(outliers.shape[0]),
            "lower_bound_iqr": float(lower),
            "upper_bound_iqr": float(upper),
        })

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_hist, ax_box = axes

        if series.nunique() <= 10:
            counts = series.value_counts().sort_index()
            ax_hist.bar(counts.index.astype(str), counts.values, color="#4c78a8", alpha=0.85)
            ax_hist.set_ylabel("Effectif")
        else:
            ax_hist.hist(series, bins=20, color="#4c78a8", alpha=0.75, edgecolor="black")
            if not outliers.empty:
                ax_hist.scatter(outliers.values, np.full(outliers.shape[0], 0.02 * max(1, len(series))),
                                color="#d62728", s=18, zorder=5, label="Outliers IQR")
                ax_hist.legend(fontsize=8)
            ax_hist.set_ylabel("Effectif")
        ax_hist.axvline(series.median(), color="black", linestyle="--", linewidth=1, label="Médiane")
        ax_hist.set_title(f"Distribution - {col}")
        ax_hist.set_xlabel(col)

        ax_box.boxplot(series.values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#72b7b2", alpha=0.7),
                       flierprops=dict(marker='o', markerfacecolor='#d62728', markersize=5, markeredgecolor='black'))
        ax_box.set_title(f"Boxplot - {col}")
        ax_box.set_ylabel(col)
        ax_box.set_xticks([])

        fig.suptitle(
            f"{col} | médiane={series.median():.3g} | IQR outliers={outliers.shape[0]}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(diag_dir / f"{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(diag_dir / "variable_summary.csv", index=False)
    return diag_dir


def save_survival_vs_kaplan_meier_plot(model, X, T, E, output_dir: Path):
    X = X.astype(float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)

    km = KaplanMeierFitter().fit(T, E, label="Kaplan-Meier observé")
    times = km.survival_function_.index.values.astype(float)
    surv_pred = model.predict_survival_function(X, times=times)
    mean_surv_pred = surv_pred.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.step(times, km.survival_function_.iloc[:, 0].values, where="post", linewidth=2, color="#1f77b4", label="Kaplan-Meier observé")
    ax.plot(times, mean_surv_pred.values, linewidth=2, color="#d62728", label="Cox prédit moyen")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Survie")
    ax.set_title("Survie prédite par Cox vs Kaplan-Meier observé")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cox_vs_kaplan_meier.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_hazard_ratios(model, output_dir: Path):
    summary = model.summary.sort_values("p", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sig_vars = summary[summary["p"] < 0.05].index.tolist()
    if sig_vars:
        model.plot(columns=sig_vars[:10], ax=ax)
        ax.set_title("Cox PH - Hazard ratios (variables significatives)")
    else:
        model.plot(ax=ax)
        ax.set_title("Cox PH - Hazard ratios")
    plt.tight_layout()
    plt.savefig(output_dir / "cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_validation_plot(validation: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Apparent", "CV", "Corrigé"]
    values = [validation["c_index_apparent"], validation["c_index_cv_mean"], validation["c_index_corrected"]]
    errors = [0.0, validation["c_index_cv_std"], 0.0]
    ax.bar(labels, values, yerr=errors, capsize=6, color=["#bdbdbd", "#4c78a8", "#2ca02c"])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("C-index")
    ax.set_title("Performance discriminante du modèle de Cox")
    plt.tight_layout()
    plt.savefig(output_dir / "cox_validation.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_brier_plot(brier: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(brier["times"], brier["brier_cox"], "-o", ms=4, label=f"Cox (IBS={brier['ibs_cox']:.3f})")
    ax.plot(brier["times"], brier["brier_km"], "--", label=f"KM (IBS={brier['ibs_km']:.3f})")
    ax.axhline(0.25, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Brier score (IPCW)")
    ax.set_title("Erreur de prédiction au cours du temps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cox_brier.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_calibration_plot(calibration: dict, output_dir: Path):
    items = list(calibration.items())
    n = max(1, len(items))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for i, (h, vals) in enumerate(items):
        ax = axes_flat[i]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(vals["predicted"], vals["observed"], "-o", color="#2ca02c")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Risque prédit moyen")
        ax.set_ylabel("Risque observé")
        ax.set_title(f"Calibration à t={h}")
    plt.tight_layout()
    plt.savefig(output_dir / "cox_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_time_auc_plot(auc: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(auc["auc_times"], auc["auc_values"], "-o", color="#9467bd", ms=4, label=f"AUC moyen={auc['auc_mean']:.3f}")
    ax.scatter([float(t) for t in auc["auc_at_horizons"].keys()], list(auc["auc_at_horizons"].values()), color="#d62728")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1)
    ax.axhline(0.8, color="green", linestyle=":", linewidth=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Temps")
    ax.set_ylabel("AUC dépendant du temps")
    ax.set_title("Pouvoir discriminant au cours du temps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cox_time_auc.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_pit_band_plot(pit: dict, output_dir: Path):
    counts = np.asarray(pit["hist_counts"], dtype=float)
    edges = np.linspace(0, 1, len(counts) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    a, b = pit["band"]
    colors = ["#2ca02c" if a <= c <= b else "#4c78a8" for c in centers]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, counts / counts.sum(), width=0.09, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1)
    ax.axvspan(a, b, color="#2ca02c", alpha=0.08)
    ax.set_xlabel("u = P(T ≤ t_réel | x)")
    ax.set_ylabel("Fraction des événements")
    ax.set_title(f"PIT band, fraction observée={pit['fraction_in_band']:.1%}, attendue={pit['expected_fraction']:.0%}")
    plt.tight_layout()
    plt.savefig(output_dir / "cox_pit_band.png", dpi=150, bbox_inches="tight")
    plt.close()


def _build_example_label(subset, row_idx: int, max_features_in_legend: int = 3):
    feature_priority = subset.var().sort_values(ascending=False).index.tolist()
    selected_features = feature_priority[:max_features_in_legend]
    return " | ".join(f"{col}={subset.iloc[row_idx][col]:.3g}" for col in selected_features)


def save_inference_probability_curves_with_true_time(model, X, T, E, output_dir: Path, n_samples: int = 10, max_features_in_legend: int = 3):
    X = X.astype(float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)

    n = min(n_samples, len(X))
    subset = X.iloc[:n]
    surv = model.predict_survival_function(subset)
    times = surv.index.values
    proba = 1.0 - surv.values

    fig, ax = plt.subplots(figsize=(12, 6))
    legend_handles = []
    for i in range(n):
        state = "obs" if E[i] == 1 else "cens"
        feature_desc = _build_example_label(subset, i, max_features_in_legend=max_features_in_legend)
        curve_label = f"Essai {i+1} - {feature_desc}"
        line, = ax.plot(times, proba[:, i], alpha=0.8, linewidth=2)
        color = line.get_color()
        ax.axvline(T[i], color=color, linestyle="--", alpha=0.5, linewidth=1.5)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2,
                                     label=f"{curve_label} ({state})"))

    ax.set_xlabel("Temps")
    ax.set_ylabel("P(apparition avant t)")
    ax.set_title("Courbes de probabilité d'apparition avec temps réel")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(output_dir / "inference_probability_curves_with_true_time.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_inference_probability_curves_with_true_time_per_example(model, X, T, E, output_dir: Path, max_features_in_legend: int = 3):
    X = X.astype(float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)

    per_example_dir = output_dir / "inference_probability_curves_per_example"
    per_example_dir.mkdir(parents=True, exist_ok=True)

    surv = model.predict_survival_function(X)
    times = surv.index.values
    proba = 1.0 - surv.values

    for i in range(len(X)):
        fig, ax = plt.subplots(figsize=(10, 5))
        state = "obs" if E[i] == 1 else "cens"
        feature_desc = _build_example_label(X, i, max_features_in_legend=max_features_in_legend)
        ax.plot(times, proba[:, i], color="#1f77b4", alpha=0.9, linewidth=2.2,
                label=f"Essai {i+1} - {feature_desc} ({state})")
        ax.axvline(T[i], color="#d62728", linestyle="--", linewidth=1.8,
                   label=f"t_réel={T[i]:.2f}")
        p_true = float(np.interp(T[i], times, proba[:, i]))
        ax.scatter([T[i]], [p_true], color="#d62728", s=35, zorder=5)
        ax.set_xlabel("Temps")
        ax.set_ylabel("P(apparition avant t)")
        ax.set_title(f"Courbe d'inférence, essai {i+1}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(per_example_dir / f"example_{i+1:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return per_example_dir


def save_single_variable_sensitivity_plot(model, X, output_dir: Path, variable_name: str, n_curves: int = 6, fixed_values: dict | None = None):
    X = X.astype(float)
    output_dir.mkdir(parents=True, exist_ok=True)

    if variable_name not in X.columns:
        raise ValueError(f"Variable inconnue pour la sensibilité: {variable_name}")

    reference = X.median().to_frame().T
    fixed_values = fixed_values or {}
    for col, value in fixed_values.items():
        if col not in reference.columns:
            raise ValueError(f"Variable fixe inconnue: {col}")
        reference.loc[:, col] = float(value)

    values = np.linspace(float(X[variable_name].quantile(0.05)), float(X[variable_name].quantile(0.95)), n_curves)

    survival = model.predict_survival_function(reference)
    times = survival.index.values

    fig, ax = plt.subplots(figsize=(10, 6))
    for value in values:
        profile = reference.copy()
        profile.loc[:, variable_name] = value
        surv = model.predict_survival_function(profile, times=times)
        prob = 1.0 - surv.values[:, 0]
        ax.plot(times, prob, linewidth=2, label=f"{variable_name}={value:.3g}")

    fixed_values = [f"{col}={reference.iloc[0][col]:.3g}" for col in X.columns if col != variable_name]
    title_suffix = " | ".join(fixed_values[:6])
    if len(fixed_values) > 6:
        title_suffix += " | ..."

    ax.set_xlabel("Temps")
    ax.set_ylabel("P(apparition avant t)")
    ax.set_title(
        f"Sensibilité à {variable_name}\n"
        f"Autres variables fixées: {title_suffix}"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_{variable_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_cox_residual_diagnostics(residuals: dict, output_dir: Path):
    residual_dir = output_dir / "cox_residual_diagnostics"
    residual_dir.mkdir(parents=True, exist_ok=True)

    schoenfeld = residuals["schoenfeld"]
    martingale = residuals["martingale"]
    ph_test = residuals["ph_test"]

    ph_test.to_csv(residual_dir / "proportional_hazard_test.csv", index=False)
    schoenfeld.to_csv(residual_dir / "schoenfeld_residuals.csv", index=False)
    martingale.to_csv(residual_dir / "martingale_residuals.csv", index=False)

    if "T" in martingale.columns and "martingale" in martingale.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(martingale["T"], martingale["martingale"], alpha=0.6, s=18, color="#1f77b4")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Temps observé")
        ax.set_ylabel("Résidu de Martingale")
        ax.set_title("Résidus de Martingale")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(residual_dir / "martingale_residuals.png", dpi=150, bbox_inches="tight")
        plt.close()

    schoenfeld_cols = [col for col in schoenfeld.columns if col not in {"T", "E"}]
    for col in schoenfeld_cols[: min(6, len(schoenfeld_cols))]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(schoenfeld.index, schoenfeld[col], alpha=0.65, s=18, color="#d62728")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Ordre des événements")
        ax.set_ylabel(f"Résidu de Schoenfeld - {col}")
        pval_row = ph_test.loc[ph_test["variable"] == col]
        pval_txt = f", p={float(pval_row['p'].iloc[0]):.3g}" if not pval_row.empty and "p" in pval_row.columns else ""
        ax.set_title(f"Schoenfeld - {col}{pval_txt}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(residual_dir / f"schoenfeld_{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return residual_dir


def save_time_varying_risk_trajectory_plot(trajectory_df, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(trajectory_df["measurement_time"], trajectory_df["event_probability"], "-o", color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("P(apparition avant t)")
    axes[0].set_title("Évolution de la probabilité d'événement")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trajectory_df["measurement_time"], trajectory_df["hazard_ratio"], "-o", color="#d62728", linewidth=2)
    axes[1].set_xlabel("Temps de mesure")
    axes[1].set_ylabel("Hazard ratio")
    axes[1].set_title("Évolution du risque relatif")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_risk_time_map_eta(model, X, T, E, output_dir: Path):
    X = X.astype(float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=bool)

    eta = np.log(model.predict_partial_hazard(X).values)
    coef = model.params_.sort_values()

    contribution_series = coef.sort_values().astype(float)

    bch = model.baseline_cumulative_hazard_
    t_grid = np.linspace(float(bch.index.min()), float(bch.index.max()), 220)
    H0 = np.interp(t_grid, bch.index.values, bch.iloc[:, 0].values)

    lo, hi = float(np.min(eta)), float(np.max(eta))
    pad = 0.05 * max(hi - lo, 1e-6)
    eta_grid = np.linspace(lo - pad, hi + pad, 220)

    se_obs = np.sqrt(np.sum(np.dot(X.values, model.variance_matrix_.values) * X.values, axis=1))
    bw = max(0.30 * np.std(eta), 1e-6)
    W = np.exp(-0.5 * ((eta_grid[:, None] - eta[None, :]) / bw) ** 2)
    se_grid = (W @ se_obs) / (W.sum(axis=1) + 1e-12)

    p5, p95 = np.percentile(eta, [5, 95])
    prob = 1 - np.exp(-np.outer(np.exp(eta_grid), H0))
    AA, TT = np.meshgrid(eta_grid, t_grid)

    kde = gaussian_kde(eta)
    dens = kde(eta_grid)
    dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 2.1], height_ratios=[0.32, 1.05, 4.0], hspace=0.08, wspace=0.22)
    ax_decomp = fig.add_subplot(gs[:, 0])
    ax_strip = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, 1], sharex=ax_strip)
    ax_map = fig.add_subplot(gs[2, 1], sharex=ax_strip)

    colors = ["#d73027" if x > 0 else "#1a9850" for x in contribution_series.values]
    ax_decomp.barh(range(len(contribution_series)), contribution_series.values, color=colors, edgecolor="black", alpha=0.85)
    ax_decomp.axvline(0, color="black", linewidth=1)
    ax_decomp.set_yticks(range(len(contribution_series)))
    ax_decomp.set_yticklabels(contribution_series.index, fontsize=9)
    ax_decomp.set_xlabel("Coefficient β du modèle de Cox", fontsize=9)
    ax_decomp.set_title("Décomposition de η via les coefficients β", fontsize=11)
    ax_decomp.grid(True, axis="x", alpha=0.3)

    ax_strip.imshow(dens_norm[None, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower", extent=[eta_grid.min(), eta_grid.max(), 0, 1])
    ax_strip.axvline(p5, color="black", linestyle=":", linewidth=1)
    ax_strip.axvline(p95, color="black", linestyle=":", linewidth=1)
    ax_strip.set_yticks([])
    ax_strip.tick_params(labelbottom=False)
    ax_strip.set_ylabel("Fiabilité", fontsize=8, rotation=0, ha="right", va="center")

    ax_top.fill_between(eta_grid, dens, color="0.65", alpha=0.55, label="Densité")
    ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False)
    ax_top.axvline(p5, color="black", linestyle=":", linewidth=1)
    ax_top.axvline(p95, color="black", linestyle=":", linewidth=1)
    ax_tw = ax_top.twinx()
    ax_tw.plot(eta_grid, se_grid, color="purple", linewidth=2, label="se(η)")
    ax_tw.fill_between(eta_grid, se_grid, color="purple", alpha=0.08)
    ax_tw.set_ylim(bottom=0)
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_tw.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", ncol=2, framealpha=0.85)

    im = ax_map.pcolormesh(AA, TT, prob.T, cmap="RdYlGn_r", shading="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=[ax_strip, ax_top, ax_map], pad=0.01, fraction=0.04).set_label("P(apparition avant t)")
    cs = ax_map.contour(AA, TT, prob.T, levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors="black", linewidths=0.8, linestyles="--")
    ax_map.clabel(cs, fmt="%.2f", fontsize=8)
    ax_map.scatter(eta[E], T[E], s=13, c="black", edgecolors="white", linewidths=0.3, alpha=0.55, label="Événement")
    ax_map.scatter(eta[~E], T[~E], s=28, marker="x", c="navy", alpha=0.7, label="Censuré")
    ax_map.axvline(p5, color="black", linestyle=":", linewidth=1)
    ax_map.axvline(p95, color="black", linestyle=":", linewidth=1, label="Zone fiable P5–P95")
    ax_map.set_xlabel("η = score de risque global")
    ax_map.set_ylabel("Temps")
    ax_map.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("Cartographie risque × temps — abscisse : η", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / "cox_risk_time_map_eta.png", dpi=150, bbox_inches="tight")
    plt.close()
