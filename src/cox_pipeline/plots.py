from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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


def save_risk_time_map_eta(model, X, T, E, output_dir: Path):
    X = X.astype(float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=bool)

    eta = np.log(model.predict_partial_hazard(X).values)
    coef = model.params_.sort_values()

    contribution = {}
    for var, beta in coef.items():
        unique_values = set(np.unique(X[var].dropna().values))
        if unique_values.issubset({0.0, 1.0}):
            contribution[var] = float(beta)
        else:
            spread = float(X[var].quantile(0.90) - X[var].quantile(0.10))
            contribution[var] = float(beta * spread)
    contribution = np.array([contribution[var] for var in coef.index], dtype=float)
    contribution_series = coef.copy()
    contribution_series.loc[:] = contribution
    contribution_series = contribution_series.sort_values()

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
    ax_decomp.set_xlabel("Contribution à η (continues : β×[P10→P90], dummies : β)", fontsize=9)
    ax_decomp.set_title("Décomposition de η", fontsize=11)
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
