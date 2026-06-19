from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
