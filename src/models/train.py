"""Phase 4 : recherche d'hyperparamètres sur la validation propre, puis entraînement et sauvegarde des modèles.

Exécution (racine du projet, .venv activé) :
    python -m src.models.train                  # réutilise reports/modeling/search_results.csv s'il existe
    python -m src.models.train --force-search   # relance toute la recherche (≈ 10 min)

Protocole :
- entraînement (fit) sur train uniquement ;
- choix des hyperparamètres sur la validation PROPRE (sans les quasi-doublons), métrique F1 macro ;
- le jeu de test n'est JAMAIS utilisé ici (il est réservé à la phase 5) ;
- le meilleur réglage de chaque famille est ensuite réentraîné sur train + val et sauvegardé dans models/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data.coco_loader import PROJECT_ROOT
from src.features.build_dataset import build_and_save, load_features
from src.models.pipelines import MODELS, SEARCH_SPACES, build_pipeline, build_preprocessor, make_classifier

MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"
SEARCH_CSV = REPORT_DIR / "search_results.csv"
SELECTION_JSON = MODELS_DIR / "selection.json"


# ---------------------------------------------------------------- données
def load_dataset(path: Path | None = None):
    """X (n, 1840), y (noms de classes), meta, configs de prétraitement/descripteurs."""
    path = path or build_and_save()                  # crée le fichier de descripteurs s'il manque
    blocks, y, meta, configs = load_features(path)
    X = np.hstack([blocks[b] for b in configs["features"]["blocks"]])
    return X, y, meta, configs


def split_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    split, clean = meta["split"].to_numpy(), meta["image_eval_keep"].to_numpy(dtype=bool)
    return {"train": split == "train", "val": split == "val", "val_clean": (split == "val") & clean,
            "test": split == "test", "test_clean": (split == "test") & clean}


# ---------------------------------------------------------------- recherche
def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def _evaluate(cfg: dict, clf, Ztr, ytr, Zva, yva) -> dict:
    t0 = time.perf_counter()
    clf.fit(Ztr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_va = clf.predict(Zva)
    predict_s = time.perf_counter() - t0
    return {**cfg, "val_f1": _macro_f1(yva, pred_va), "train_f1": _macro_f1(ytr, clf.predict(Ztr)),
            "fit_s": round(fit_s, 2), "predict_val_s": round(predict_s, 2)}


def search_model(model: str, X, y, tr, va, n_jobs: int = -1, verbose: bool = True) -> list[dict]:
    """Évalue toutes les configurations d'une famille. Le scaler et la PCA sont ajustés une seule fois
    par valeur de `pca` (sur train), puis réutilisés par toutes les configurations qui la partagent."""
    rows = []
    for grid in SEARCH_SPACES[model]:
        configs = [{"model": model, **c} for c in ParameterGrid(grid)]
        if model == "rf":                                   # pas de prétraitement ; parallélisme interne
            rows += [_evaluate(c, make_classifier(c), X[tr], y[tr], X[va], y[va]) for c in configs]
            continue
        for pca in dict.fromkeys(c["pca"] for c in configs):
            pre = Pipeline(build_preprocessor(pca))
            Ztr = pre.fit_transform(X[tr])                  # ajusté sur TRAIN uniquement
            Zva = pre.transform(X[va])
            sub = [c for c in configs if c["pca"] == pca]
            if model == "svm":
                scale = 1.0 / (Ztr.shape[1] * Ztr.var())    # valeur gamma="scale" de scikit-learn
                sub = [{**c, "gamma": c["gamma_mult"] * scale} for c in sub]
            rows += Parallel(n_jobs=n_jobs)(delayed(_evaluate)(c, make_classifier(c), Ztr, y[tr], Zva, y[va])
                                            for c in sub)
        if verbose:
            print(f"  {model} : {len(rows)} configurations évaluées")
    return rows


def run_search(X, y, masks, force: bool = False, path: Path = SEARCH_CSV, verbose: bool = True) -> pd.DataFrame:
    """Recherche complète (les 3 familles) ; résultat mis en cache dans un CSV versionné."""
    if path.exists() and not force:
        return pd.read_csv(path)
    tr, va = masks["train"], masks["val_clean"]
    rows = []
    for model in MODELS:
        t0 = time.time()
        rows += search_model(model, X, y, tr, va, verbose=verbose)
        if verbose:
            print(f"{model} terminé en {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    df["params"] = [json.dumps(params_of(r)) for r in rows]
    df["gap"] = df["train_f1"] - df["val_f1"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def params_of(row) -> dict:
    """Configuration réutilisable par build_pipeline (sans les colonnes de score, sans NaN)."""
    keys = {"knn": ["pca", "n_neighbors", "weights", "metric"], "svm": ["pca", "C", "gamma", "gamma_mult"],
            "rf": ["n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight"]}
    model = row["model"]
    out = {"model": model}
    for k in keys[model]:
        v = row[k]
        out[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
    return out


def best_per_model(results: pd.DataFrame) -> pd.DataFrame:
    """Meilleure configuration de chaque famille (F1 macro sur la validation propre)."""
    idx = results.groupby("model")["val_f1"].idxmax()
    return results.loc[idx].sort_values("val_f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- modèles finaux
def fit_final(params: dict, X, y_enc, probability: bool = True) -> Pipeline:
    return build_pipeline(params, probability=probability).fit(X, y_enc)


def save_artifact(path: Path, pipeline: Pipeline, encoder: LabelEncoder, params: dict, configs: dict,
                  val_f1: float, trained_on: str) -> Path:
    """Tout ce dont l'API a besoin pour prédire de la même façon qu'à l'entraînement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline, "label_encoder": encoder, "classes": list(encoder.classes_), "params": params,
        "preprocess": configs["preprocess"], "features": configs["features"],
        "val_clean_macro_f1": val_f1, "trained_on": trained_on,
        "sklearn_version": sklearn.__version__, "created": datetime.now().isoformat(timespec="seconds"),
    }, path, compress=3)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 : recherche d'hyperparamètres et entraînement final")
    ap.add_argument("--force-search", action="store_true", help="relancer la recherche même si le CSV existe")
    a = ap.parse_args()

    X, y, meta, configs = load_dataset()
    masks = split_masks(meta)
    print(f"train {masks['train'].sum()} régions | validation propre {masks['val_clean'].sum()} | test non utilisé")
    results = run_search(X, y, masks, force=a.force_search)
    best = best_per_model(results)
    print("\nMeilleure configuration par famille (validation propre) :")
    print(best[["model", "val_f1", "train_f1", "params"]].to_string(index=False))

    # Réentraînement sur train + val (toutes les images de val : les doublons ne gênent pas l'apprentissage).
    fit_mask = masks["train"] | masks["val"]
    encoder = LabelEncoder().fit(y)
    y_enc = encoder.transform(y)
    selection = {"selected": best.loc[0, "model"], "metric": "macro F1 sur la validation propre", "models": {}}
    for _, row in best.iterrows():
        params = json.loads(row["params"])
        t0 = time.time()
        pipe = fit_final(params, X[fit_mask], y_enc[fit_mask])
        path = save_artifact(MODELS_DIR / f"{row['model']}.joblib", pipe, encoder, params, configs,
                             float(row["val_f1"]), "train+val")
        size_mb = path.stat().st_size / 1e6
        print(f"{row['model']} réentraîné sur train+val en {time.time() - t0:.0f} s -> {path.relative_to(PROJECT_ROOT)} "
              f"({size_mb:.1f} Mo)")
        selection["models"][row["model"]] = {"file": path.name, "val_clean_macro_f1": round(float(row["val_f1"]), 4),
                                             "params": params, "size_mb": round(size_mb, 1)}
    SELECTION_JSON.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    best.to_csv(REPORT_DIR / "best_per_model.csv", index=False)
    print(f"\nModèle retenu : {selection['selected']} (voir {SELECTION_JSON.relative_to(PROJECT_ROOT)})")


if __name__ == "__main__":
    main()
