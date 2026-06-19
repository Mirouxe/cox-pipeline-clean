from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict):
    data_cfg = config["data"]
    features_cfg = data_cfg["features"]

    df = pd.read_excel(data_cfg["path"], sheet_name=data_cfg["sheet_name"])

    cont_cols = features_cfg.get("continuous", [])
    cat_cols = features_cfg.get("categorical", [])
    cat_levels_cfg = features_cfg.get("categorical_levels", {}) or {}

    target_time = data_cfg["target_time"]
    target_event = data_cfg["target_event"]

    T = df[target_time].astype(float).to_numpy()
    E = df[target_event].astype(int).to_numpy()

    frames = []

    if cont_cols:
        cont = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(df[cont_cols]),
            columns=cont_cols,
            index=df.index,
        )
        frames.append(cont)

    if cat_cols:
        cat = pd.DataFrame(
            SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols]),
            columns=cat_cols,
            index=df.index,
        )
        encoded = []
        for col in cat_cols:
            levels = cat_levels_cfg.get(col)
            if levels:
                ref = str(levels[0])
                values = cat[col].astype(str)
                for lvl in levels[1:]:
                    encoded.append((values == str(lvl)).astype(float).rename(f"{col}_{lvl}"))
            else:
                encoded.append(pd.get_dummies(cat[col], prefix=col, drop_first=True).astype(float))
        if encoded:
            frames.append(pd.concat(encoded, axis=1))

    X = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    return X.astype(float), T, E, list(X.columns)
