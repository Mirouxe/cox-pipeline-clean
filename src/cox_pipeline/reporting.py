from __future__ import annotations

from pathlib import Path


def build_report(validation: dict, brier: dict, auc: dict, pit: dict) -> str:
    return f"""# Rapport d'interprétation du pipeline Cox

## 1. Étapes de calcul
1. Préparation des données, imputation, encodage des catégorielles.
2. Entraînement du modèle de Cox pénalisé.
3. Validation par C-index apparent, validation croisée et correction d'optimisme bootstrap.
4. Évaluation de l'erreur de prédiction par Brier IPCW et IBS.
5. Vérification de la calibration à plusieurs horizons.
6. Évaluation du pouvoir discriminant dans le temps via AUC(t).
7. Analyse PIT pour comparer temps réels et probabilités prédites.
8. Export du modèle `lifelines` et de sa formule explicite JSON.

## 2. Interprétation des sorties

### `cox_hazard_ratios.png`
- Affiche les coefficients exponentiés du Cox.
- `HR > 1` : la variable augmente le risque instantané.
- `HR < 1` : la variable est protectrice.
- Lire en priorité les variables statistiquement crédibles et cohérentes métier.

### `cox_validation.png`
- Compare le C-index apparent, le C-index en validation croisée et le C-index corrigé bootstrap.
- Si l'apparent est nettement plus haut que les deux autres, le modèle est probablement optimiste.
- Un C-index proche de 0.5 signifie une discrimination faible.

### `cox_brier.png`
- Courbe d'erreur de prédiction dans le temps, plus bas = meilleur.
- Le Cox doit idéalement rester sous la référence Kaplan-Meier.
- `IBS` résume globalement la qualité prédictive.
- Le `skill score` positif indique un gain du Cox par rapport à une prédiction sans covariables.

### `cox_calibration.png`
- Compare risque prédit moyen et risque observé.
- Proche de la diagonale = probabilités bien calibrées.
- Au-dessus de la diagonale : le risque réel est plus élevé que prévu.
- En dessous : le modèle surestime le risque.

### `cox_time_auc.png`
- Mesure la séparation cas / non-cas à chaque horizon.
- `0.5` = hasard, `0.8` = bon niveau.
- Une courbe stable et haute indique une discrimination robuste dans le temps.

### `cox_pit_band.png`
- Place chaque temps réel d'événement sur sa courbe de probabilité prédite.
- Si le modèle est bien calibré, les PIT doivent être proches d'une distribution uniforme.
- Pour une bande `[a, b]`, la fraction attendue vaut `b-a`.
- Un excès dans la zone haute suggère des événements prédits trop tard.

## 3. Lecture synthétique des métriques
- C-index apparent : {validation['c_index_apparent']:.3f}
- C-index CV moyen : {validation['c_index_cv_mean']:.3f} ± {validation['c_index_cv_std']:.3f}
- C-index corrigé : {validation['c_index_corrected']:.3f}
- IBS Cox : {brier['ibs_cox']:.3f}
- IBS Kaplan-Meier : {brier['ibs_km']:.3f}
- Skill score : {brier['skill_score']:.1%}
- AUC moyen intégré : {auc['auc_mean']:.3f}
- Fraction PIT dans la bande : {pit['fraction_in_band']:.1%}
- Fraction PIT attendue : {pit['expected_fraction']:.1%}
- p-value D-calibration : {pit['dcalibration_pvalue']:.3f}

## 4. Bonnes pratiques d'interprétation
- Toujours privilégier les métriques hors échantillon pour juger le modèle.
- Ne pas confondre discrimination et calibration.
- Ne pas surinterpréter un beau graphe si la validation externe manque.
- Vérifier que les effets estimés restent plausibles métier.
"""


def write_report(report_text: str, output_dir: Path):
    path = output_dir / "report_interpretation.md"
    path.write_text(report_text, encoding="utf-8")
    return path
