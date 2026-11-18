"""Tests de la phase 4 : pipelines, recherche, sauvegarde et prédiction."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.features.extractors import FeatureConfig, extract_features
from src.features.preprocessing import PreprocessConfig, region_patch
from src.models import train
from src.models.inference import DamagePredictor
from src.models.pipelines import build_pipeline

RNG = np.random.default_rng(0)


def _toy_data(n_per_class=30, d=12):
    """3 classes bien séparées + masques train/val."""
    X = np.vstack([RNG.normal(3 * k, 1, (n_per_class, d)) for k in range(3)])
    y = np.repeat(np.array(["a", "b", "c"]), n_per_class)
    tr = np.tile(np.r_[np.ones(20, bool), np.zeros(n_per_class - 20, bool)], 3)
    return X, y, tr, ~tr


def test_feature_config_survives_json_roundtrip():
    cfg = FeatureConfig()
    assert FeatureConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


@pytest.mark.parametrize("cfg,steps", [
    ({"model": "knn", "pca": 5, "n_neighbors": 3, "weights": "distance", "metric": "cosine"}, ["scaler", "pca", "clf"]),
    ({"model": "svm", "pca": None, "C": 1, "gamma": 0.1}, ["scaler", "clf"]),
    ({"model": "rf", "n_estimators": 10, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None,
      "class_weight": "balanced"}, ["clf"]),
])
def test_pipeline_steps(cfg, steps):
    assert [name for name, _ in build_pipeline(cfg).steps] == steps


def test_svm_is_class_balanced():
    pipe = build_pipeline({"model": "svm", "pca": None, "C": 1, "gamma": 0.1})
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_final_svm_gives_probabilities():
    X, y, tr, _ = _toy_data()
    pipe = train.fit_final({"model": "svm", "pca": None, "C": 1, "gamma": 0.1}, X[tr], y[tr])
    assert pipe.predict_proba(X[:5]).sum(axis=1) == pytest.approx(np.ones(5))


def test_search_finds_separable_classes(monkeypatch):
    monkeypatch.setitem(train.SEARCH_SPACES, "svm", [{"pca": [2, None], "C": [1, 10], "gamma_mult": [1]}])
    X, y, tr, va = _toy_data()
    rows = train.search_model("svm", X, y, tr, va, n_jobs=1, verbose=False)
    assert len(rows) == 4
    assert all(r["val_f1"] > 0.9 and r["gamma"] > 0 for r in rows)


def test_best_per_model_takes_max_val_f1():
    df = pd.DataFrame({"model": ["knn", "knn", "svm"], "val_f1": [0.5, 0.7, 0.6]})
    best = train.best_per_model(df)
    assert list(best["model"]) == ["knn", "svm"] and best.loc[0, "val_f1"] == 0.7


def test_params_of_turns_nan_into_none():
    row = pd.Series({"model": "svm", "pca": np.nan, "C": 10.0, "gamma": 0.01, "gamma_mult": 1.0, "val_f1": 0.8})
    assert train.params_of(row) == {"model": "svm", "pca": None, "C": 10.0, "gamma": 0.01, "gamma_mult": 1.0}


def test_predictor_uses_training_preprocessing(tmp_path):
    """Le prédicteur doit produire exactement les mêmes descripteurs que build_dataset (region_patch)."""
    image = RNG.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    pre, feat = PreprocessConfig(patch_size=64), FeatureConfig()
    bboxes = [(x, y, 40, 30) for x in range(0, 120, 20) for y in range(0, 90, 30)]
    X = np.stack([extract_features(region_patch(image, b, pre), feat) for b in bboxes])
    labels = np.array(["dent", "scratch", "crack"] * (len(bboxes) // 3))
    enc = LabelEncoder().fit(labels)
    params = {"model": "knn", "pca": None, "n_neighbors": 3, "weights": "distance", "metric": "euclidean"}
    pipe = train.fit_final(params, X, enc.transform(labels))
    configs = {"preprocess": json.loads(pre.to_json()), "features": json.loads(json.dumps(feat.to_dict()))}
    path = train.save_artifact(tmp_path / "knn.joblib", pipe, enc, params, configs, 0.5, "test")

    predictor = DamagePredictor.load(path)
    assert np.array_equal(predictor.features_of(image, bboxes[4]), X[4])
    out = predictor.predict(image, bboxes[4])
    assert out["label"] == labels[4]                       # point d'entraînement retrouvé (k-NN pondéré)
    assert sum(t["probability"] for t in out["top"]) == pytest.approx(1)
    assert predictor.predict(image)["label"] in enc.classes_    # bbox=None : image entière
