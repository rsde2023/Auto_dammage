"""Prédiction sur une nouvelle image (utilisé par l'API /predict et par l'évaluation).

Réutilise EXACTEMENT les fonctions de l'entraînement (crop_region, preprocess_region, extract_features)
avec les configurations enregistrées dans le modèle : aucune duplication du prétraitement.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.features.extractors import FeatureConfig, extract_features
from src.features.preprocessing import PreprocessConfig, crop_region, preprocess_region
from src.models.train import MODELS_DIR, SELECTION_JSON


class DamagePredictor:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.pipeline = artifact["pipeline"]
        self.classes = list(artifact["classes"])
        self.preprocess = PreprocessConfig(**artifact["preprocess"])
        self.features = FeatureConfig.from_dict(artifact["features"])

    @classmethod
    def load(cls, path: Path | None = None) -> "DamagePredictor":
        """Charge le modèle indiqué, ou par défaut celui retenu dans models/selection.json."""
        if path is None:
            selection = json.loads(Path(SELECTION_JSON).read_text(encoding="utf-8"))
            path = MODELS_DIR / selection["models"][selection["selected"]]["file"]
        return cls(joblib.load(path))

    def features_of(self, image_rgb: np.ndarray, bbox=None) -> np.ndarray:
        """Vecteur de descripteurs d'une région. bbox=None : l'image entière est la région (photo recadrée)."""
        region = image_rgb if bbox is None else crop_region(image_rgb, bbox, self.preprocess.margin)
        return extract_features(preprocess_region(region, self.preprocess), self.features)

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(np.atleast_2d(X))

    def predict(self, image_rgb: np.ndarray, bbox=None, top_k: int = 3) -> dict:
        """Classe la plus probable + les top_k classes avec leur probabilité estimée."""
        proba = self.predict_proba_features(self.features_of(image_rgb, bbox))[0]
        order = np.argsort(proba)[::-1]
        return {"label": self.classes[order[0]], "confidence": float(proba[order[0]]),
                "top": [{"label": self.classes[i], "probability": float(proba[i])} for i in order[:top_k]]}
