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
- `src/cox_pipeline/model.py` : entraînement, export, rechargement JSON, inférence
- `src/cox_pipeline/metrics.py` : C-index, Brier IPCW, AUC dépendant du temps
- `src/cox_pipeline/plots.py` : graphes essentiels
- `src/cox_pipeline/reporting.py` : rapport Markdown d'interprétation
- `src/cox_pipeline/pipeline.py` : orchestration complète
- `src/run_pipeline.py` : point d'entrée simple
- `src/run_inference_from_json_example.py` : mini-script d'inférence sur un nouvel exemple
- `examples/example_config.yaml` : configuration exemple

## Sorties

Le pipeline génère notamment :

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

## Philosophie

Ce repo reprend l'essentiel utile de `MODELE_COX`, en version plus compacte :

- un seul modèle central : Cox PH
- évaluations essentielles, pas de dispersion
- code modulaire, lisible, réutilisable
- inférence explicite portable sans dépendance à `lifelines`
