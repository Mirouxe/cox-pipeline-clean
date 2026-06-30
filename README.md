# cox-pipeline-clean

Pipeline propre, claire et concise pour :

- entraîner un modèle de Cox
- l'évaluer proprement
- générer les graphes essentiels
- sauvegarder le modèle et sa formule explicite
- inférer avec ou sans `lifelines`
- produire un rapport d'interprétation

## Contenu

- `src/cox_pipeline/data.py` : chargement et préparation des données
- `src/cox_pipeline/model.py` : entraînement, export, rechargement JSON, inférence, trajectoire de risque
- `src/cox_pipeline/metrics.py` : C-index, Brier IPCW, AUC dépendant du temps
- `src/cox_pipeline/plots.py` : graphes essentiels
- `src/cox_pipeline/reporting.py` : rapport Markdown d'interprétation
- `src/cox_pipeline/pipeline.py` : orchestration complète
- `src/run_pipeline.py` : point d'entrée simple
- `src/run_inference_from_json_example.py` : mini-script d'inférence sur un nouvel exemple
- `examples/example_config.yaml` : configuration exemple

## Sorties

Le pipeline génère notamment :

- `variable_diagnostics/` (un graphe par variable + `variable_summary.csv`)
- `cox_hazard_ratios.png`
- `cox_validation.png`
- `cox_brier.png`
- `cox_calibration.png`
- `cox_time_auc.png`
- `cox_pit_band.png`
- `cox_risk_time_map_eta.png`
- `inference_probability_curves_with_true_time.png`
- `inference_probability_curves_per_example/`
- `sensitivity_<variable>.png`
- `cox_embedded.json`

## Sensibilité mono-variable avec valeurs fixes choisies

Tu peux piloter la variable étudiée et les autres valeurs fixées directement dans la config :

```yaml
sensitivity:
  variable: X1_cont
  n_curves: 6
  fixed_values:
    X2_cont: 0.5
    X3_cont: -0.2
    X6_cat_B: 1.0
    X7_cat_M2: 0.0
```

La figure générée montre 6 courbes en faisant varier seulement `variable`, tandis que les autres colonnes listées dans `fixed_values` sont imposées aux valeurs choisies.

## Évaluation par seuil de probabilité

La pipeline inclut aussi deux fonctions :
- une évaluation pour un seuil fixe, par défaut `0.6`
- une optimisation du seuil pour choisir le seuil le plus élevé parmi ceux qui maximisent le taux de succès

Le critère de succès est :
- pour un individu avec événement observé, on calcule le premier temps où la probabilité prédite dépasse le seuil
- la prédiction est comptée comme un succès si le temps réel de l'événement survient avant ou au moment de ce franchissement

Configuration associée :

```yaml
threshold_evaluation:
  threshold: 0.6
  grid: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

Les résultats sont écrits dans `metrics_summary.json`.
- `cox_model.joblib`
- `report_interpretation.md`
- `metrics_summary.json`

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python src/run_pipeline.py --config examples/example_config.yaml
```

## Format minimal des données

Le fichier source doit contenir :

- une colonne de temps
- une colonne d'événement (1 = événement, 0 = censuré)
- des variables continues
- éventuellement des variables catégorielles

## Recharger la formule JSON pour une inférence explicite

```python
from cox_pipeline.model import load_formula_from_json, infer_with_formula

formula = load_formula_from_json("outputs/cox_embedded.json")
pred = infer_with_formula(configs, formula)
```

Exemple prêt à lancer :

```bash
python3 src/run_inference_from_json_example.py
```

Ce script produit aussi la figure :

```text
outputs/inference_from_json_example.png
```

## Trajectoire de risque avec covariables mesurées dans le temps

Le repo contient aussi une fonction pour appliquer le modèle de Cox à une suite de profils mesurés à différents instants :

```python
from cox_pipeline.model import predict_time_varying_risk_trajectory
from cox_pipeline.plots import save_time_varying_risk_trajectory_plot

trajectory = predict_time_varying_risk_trajectory(model, covariate_history, measurement_times)
save_time_varying_risk_trajectory_plot(trajectory, "outputs/time_varying_risk_trajectory.png")
```

La sortie contient pour chaque instant :
- `measurement_time`
- `hazard_ratio`
- `event_probability`

## Philosophie

Ce repo reprend l'essentiel utile de `MODELE_COX`, en version plus compacte :

- un seul modèle central : Cox PH
- évaluations essentielles, pas de dispersion
- code modulaire, lisible, réutilisable
- inférence explicite portable sans dépendance à `lifelines`
