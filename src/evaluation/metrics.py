"""Phase 5 : évaluation finale des modèles sauvegardés sur le jeu de test (utilisé une seule fois, ici).

Exécution (racine du projet, .venv activé, après `python -m src.models.train`) :
    python -m src.evaluation.metrics

Deux versions du test sont rapportées :
- test officiel (785 régions) : comparable aux autres travaux sur CarDD ;
- test propre (761 régions)  : sans les images quasi identiques à une image de train/val (score plus honnête).
Les modèles ont été choisis AVANT (phase 4, sur la validation) : ces chiffres ne servent pas à rechoisir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.data.coco_loader import PROJECT_ROOT
from src.models.inference import DamagePredictor
from src.models.train import MODELS_DIR, SELECTION_JSON, load_dataset, split_masks

CLASS_ORDER = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
EVAL_DIR = PROJECT_ROOT / "reports" / "evaluation"
CLEANING_DIR = PROJECT_ROOT / "reports" / "cleaning"


def classification_metrics(y_true, y_pred) -> dict:
    return {"n": int(len(y_true)), "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted")}


def bootstrap_macro_f1(y_true, y_pred, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Intervalle de confiance à 95 % du F1 macro par bootstrap sur les IMAGES (et non les régions) :
    les régions d'une même photo ne sont pas indépendantes, on tire donc des photos entières."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def predict_with(predictor: DamagePredictor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Comme l'API : classe = probabilité maximale. Retourne (classes prédites, probabilités)."""
    proba = predictor.predict_proba_features(X)
    return np.asarray(predictor.classes)[proba.argmax(axis=1)], proba


def region_context(meta: pd.DataFrame) -> pd.DataFrame:
    """Ajoute aux régions les informations utiles à l'analyse des erreurs (taille, angle de vue, ...)."""
    regions = pd.read_csv(CLEANING_DIR / "regions.csv")[["split", "annotation_id", "flags"]]
    images = pd.read_csv(CLEANING_DIR / "image_metadata.csv")[["image_id", "shooting_angle", "complete_or_partial", "color"]]
    out = meta.merge(regions, on=["split", "annotation_id"], how="left").merge(images, on="image_id", how="left")
    out["bbox_rel_area"] = out["bbox_w"] * out["bbox_h"] / (out["width"] * out["height"])
    out["regions_in_image"] = out.groupby(["split", "image_id"])["annotation_id"].transform("size")
    out["tiny_bbox"] = out["flags"].fillna("").str.contains("bbox_minuscule")
    return out


def evaluate_all(write: bool = True) -> dict:
    X, y, meta, _ = load_dataset()
    masks = split_masks(meta)
    selection = json.loads(Path(SELECTION_JSON).read_text(encoding="utf-8"))
    fit_mask = masks["train"] | masks["val"]

    rows, per_class, preds = [], [], {}
    dummy = DummyClassifier(strategy="stratified", random_state=0).fit(X[fit_mask], y[fit_mask])
    candidates = {"baseline": None, **{m: MODELS_DIR / info["file"] for m, info in selection["models"].items()}}
    for name, path in candidates.items():
        if path is None:
            pred_all, proba = dummy.predict(X), None
        else:
            pred_all, proba = predict_with(DamagePredictor.load(path), X)
        preds[name] = (pred_all, proba)
        for subset in ("test", "test_clean"):
            m = masks[subset]
            low, high = bootstrap_macro_f1(y[m], pred_all[m], meta.loc[m, "image_id"])
            rows.append({"model": name, "subset": subset, **classification_metrics(y[m], pred_all[m]),
                         "macro_f1_ci_low": low, "macro_f1_ci_high": high,
                         "selected": name == selection["selected"]})
            per_class.append(pd.Series(f1_score(y[m], pred_all[m], labels=CLASS_ORDER, average=None),
                                       index=CLASS_ORDER, name=(name, subset)))
    metrics = pd.DataFrame(rows)
    per_class = pd.DataFrame(per_class)

    sel = selection["selected"]
    pred_sel, proba_sel = preds[sel]
    test = masks["test"]
    detail = region_context(meta[test].reset_index(drop=True))
    detail["true"], detail["pred"] = y[test], pred_sel[test]
    detail["confidence"] = proba_sel[test].max(axis=1)
    detail["correct"] = detail["true"] == detail["pred"]
    detail["test_clean"] = masks["test_clean"][test]
    cm = pd.DataFrame(confusion_matrix(y[masks["test_clean"]], pred_sel[masks["test_clean"]], labels=CLASS_ORDER),
                      index=CLASS_ORDER, columns=CLASS_ORDER)

    if write:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        metrics.round(4).to_csv(EVAL_DIR / "test_metrics.csv", index=False)
        per_class.round(4).rename_axis(["model", "subset"]).to_csv(EVAL_DIR / "test_per_class_f1.csv")
        cm.to_csv(EVAL_DIR / f"confusion_matrix_test_clean_{sel}.csv")
        detail.drop(columns=["flags"]).to_csv(EVAL_DIR / f"test_predictions_{sel}.csv", index=False)
    return {"metrics": metrics, "per_class": per_class, "detail": detail, "confusion": cm, "selected": sel,
            "X": X, "y": y, "meta": meta, "masks": masks, "preds": preds}


def main() -> None:
    res = evaluate_all()
    cols = ["model", "subset", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]
    print(res["metrics"][cols].round(3).to_string(index=False))
    print(f"\nModèle retenu : {res['selected']} — fichiers écrits dans {EVAL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
