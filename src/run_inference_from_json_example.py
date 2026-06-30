from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cox_pipeline.model import load_formula_from_json, infer_with_formula


def main():
    formula_path = ROOT / "outputs" / "cox_embedded.json"
    formula = load_formula_from_json(formula_path)

    # Exemple inédit, hors jeu de données d'origine.
    # Adapter les noms de colonnes si ton modèle a été entraîné avec d'autres features.
    new_example = pd.DataFrame([
        {
            "X1_cont": 0.85,
            "X2_cont": 1.20,
            "X3_cont": -0.40,
            "X4_cont": 0.55,
            "X5_cont": 1.10,
            "X6_cat_B": 1.0,
            "X6_cat_C": 0.0,
            "X7_cat_M2": 1.0,
            "X8_cat_L2": 0.0,
            "X8_cat_L3": 1.0,
            "X8_cat_L4": 0.0,
            "X8_cat_L5": 0.0,
        }
    ], index=["nouvel_exemple"])

    prediction = infer_with_formula(new_example, formula)

    print("Formule chargée depuis :", formula_path)
    print("\nNouvel exemple :")
    print(new_example.to_string())
    print("\nPrédictions :")
    print(prediction.round(4).to_string())

    horizons = sorted(float(t) for t in formula["baseline_survival"].keys())
    probabilities = [float(prediction.loc["nouvel_exemple", f"P(T<={t})"]) for t in horizons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, probabilities, "-o", color="#1f77b4", linewidth=2)
    ax.set_xlabel("Temps")
    ax.set_ylabel("P(apparition avant t)")
    ax.set_title("Probabilité d'apparition en fonction du temps")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figure_path = ROOT / "outputs" / "inference_from_json_example.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\nFigure sauvegardée :", figure_path)


if __name__ == "__main__":
    main()
