"""API REST AutoDamage ML (FastAPI).

Lancement (racine du projet, .venv activé, après `python -m src.models.train`) :
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

- Documentation interactive : http://127.0.0.1:8000/docs
- Depuis l'émulateur Android, le PC est joignable à l'adresse http://10.0.2.2:8000

Aucun code de Machine Learning ici : l'API décode l'image puis appelle `DamagePredictor`
(src/models/inference.py), qui applique exactement le prétraitement de l'entraînement.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.data.image_io import decode_image_rgb

UNCERTAIN_THRESHOLD = 0.5            # fixé a priori (phase 5) : en dessous, moins d'une chance sur deux
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LABELS_FR = {"dent": "Bosse", "scratch": "Rayure", "crack": "Fissure", "glass shatter": "Vitre brisée",
             "lamp broken": "Phare cassé", "tire flat": "Pneu crevé"}
DISCLAIMER = ("Estimation automatique indicative. Elle ne remplace pas l'avis d'un professionnel "
              "et ne constitue pas une expertise d'assurance.")


class ClassScore(BaseModel):
    label: str
    label_fr: str
    probability: float


class PredictionResponse(BaseModel):
    label: str
    label_fr: str
    confidence: float
    uncertain: bool
    top: list[ClassScore]
    model: str
    image_width: int
    image_height: int
    bbox_used: list[float] | None
    processing_ms: int
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str | None
    classes: list[str]
    uncertain_threshold: float


def _default_loader():
    from src.models.inference import DamagePredictor       # import tardif : les tests peuvent s'en passer
    return DamagePredictor.load()


def _validate_bbox(values: list[float | None], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Les 4 champs ensemble ou aucun ; rectangle de taille positive qui recoupe l'image."""
    given = [v is not None for v in values]
    if not any(given):
        return None
    if not all(given):
        raise HTTPException(422, "bbox incomplète : fournir bbox_x, bbox_y, bbox_w et bbox_h, ou aucun des quatre.")
    x, y, w, h = values
    if w < 1 or h < 1:
        raise HTTPException(422, "bbox_w et bbox_h doivent être d'au moins 1 pixel.")
    if x >= width or y >= height or x + w <= 0 or y + h <= 0:
        raise HTTPException(422, f"La bbox est en dehors de l'image ({width}x{height}).")
    return x, y, w, h


def create_app(loader: Callable[[], object] = _default_loader) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = loader()
            app.state.load_error = None
        except Exception as exc:                            # l'API démarre quand même et l'indique dans /health
            app.state.predictor = None
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="AutoDamage ML", version="1.0.0", lifespan=lifespan,
                  description="Classification d'une zone endommagée de véhicule (6 classes) par Machine Learning classique.")

    def predictor_of(request: Request):
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "Modèle non chargé. Lancer d'abord : python -m src.models.train "
                                     f"({request.app.state.load_error})")
        return p

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        p = request.app.state.predictor
        return HealthResponse(status="ok" if p is not None else "model_missing", model_loaded=p is not None,
                              model=p.artifact["params"]["model"] if p is not None else None,
                              classes=list(p.classes) if p is not None else [],
                              uncertain_threshold=UNCERTAIN_THRESHOLD)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request,
                image: UploadFile = File(..., description="Photo (JPEG ou PNG)"),
                bbox_x: float | None = Form(None), bbox_y: float | None = Form(None),
                bbox_w: float | None = Form(None), bbox_h: float | None = Form(None)):
        """Classe la zone endommagée. Sans bbox, l'image entière est considérée comme la zone (photo déjà recadrée)."""
        predictor = predictor_of(request)
        t0 = time.perf_counter()
        data = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Image trop lourde (maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo).")
        if not data:
            raise HTTPException(400, "Fichier image vide.")
        try:
            img = decode_image_rgb(data, image.filename or "image")
        except ValueError:
            raise HTTPException(400, "Fichier illisible : envoyer une image JPEG ou PNG.")
        h, w = img.shape[:2]
        bbox = _validate_bbox([bbox_x, bbox_y, bbox_w, bbox_h], w, h)
        out = predictor.predict(img, bbox, top_k=3)
        return PredictionResponse(
            label=out["label"], label_fr=LABELS_FR.get(out["label"], out["label"]),
            confidence=round(out["confidence"], 4), uncertain=out["confidence"] < UNCERTAIN_THRESHOLD,
            top=[ClassScore(label=t["label"], label_fr=LABELS_FR.get(t["label"], t["label"]),
                            probability=round(t["probability"], 4)) for t in out["top"]],
            model=predictor.artifact["params"]["model"], image_width=w, image_height=h,
            bbox_used=list(bbox) if bbox else None, processing_ms=int((time.perf_counter() - t0) * 1000),
            disclaimer=DISCLAIMER)

    return app


app = create_app()
