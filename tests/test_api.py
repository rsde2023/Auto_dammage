"""Tests de l'API FastAPI (avec un faux modèle : pas besoin d'avoir entraîné quoi que ce soit)."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import MAX_UPLOAD_BYTES, create_app
from src.models.train import SELECTION_JSON


class FakePredictor:
    classes = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]
    artifact = {"params": {"model": "fake"}}

    def __init__(self, confidence=0.8):
        self.confidence, self.calls = confidence, []

    def predict(self, image, bbox=None, top_k=3):
        self.calls.append((image.shape, bbox))
        rest = (1 - self.confidence) / 2
        return {"label": "dent", "confidence": self.confidence,
                "top": [{"label": "dent", "probability": self.confidence},
                        {"label": "scratch", "probability": rest}, {"label": "crack", "probability": rest}]}


def png_bytes(h=60, w=80):
    ok, buf = cv2.imencode(".png", np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8))
    return buf.tobytes()


@pytest.fixture
def fake():
    return FakePredictor()


@pytest.fixture
def client(fake):
    with TestClient(create_app(loader=lambda: fake)) as c:
        yield c


def post(client, data=None, content=None, name="photo.png", mime="image/png"):
    return client.post("/predict", files={"image": (name, png_bytes() if content is None else content, mime)},
                       data=data or {})


def test_health_reports_loaded_model(client):
    r = client.get("/health").json()
    assert r["model_loaded"] and r["model"] == "fake" and len(r["classes"]) == 6 and r["uncertain_threshold"] == 0.5


def test_predict_whole_image(client, fake):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "dent" and body["label_fr"] == "Bosse" and not body["uncertain"]
    assert [t["label_fr"] for t in body["top"]] == ["Bosse", "Rayure", "Fissure"]
    assert (body["image_width"], body["image_height"]) == (80, 60) and body["bbox_used"] is None
    assert "expertise d'assurance" in body["disclaimer"]
    assert fake.calls == [((60, 80, 3), None)]


def test_predict_with_bbox_passes_it_to_the_model(client, fake):
    r = post(client, data={"bbox_x": 10, "bbox_y": 5, "bbox_w": 30, "bbox_h": 20})
    assert r.status_code == 200 and r.json()["bbox_used"] == [10, 5, 30, 20]
    assert fake.calls[-1][1] == (10, 5, 30, 20)


@pytest.mark.parametrize("data", [{"bbox_x": 1, "bbox_y": 1}, {"bbox_x": 0, "bbox_y": 0, "bbox_w": 0, "bbox_h": 5},
                                  {"bbox_x": 500, "bbox_y": 0, "bbox_w": 10, "bbox_h": 10}])
def test_invalid_bbox_is_rejected(client, data):
    assert post(client, data=data).status_code == 422


def test_unreadable_and_empty_files_are_rejected(client):
    assert post(client, content=b"pas une image", name="x.txt", mime="text/plain").status_code == 400
    assert post(client, content=b"").status_code == 400


def test_too_large_file_is_rejected(client):
    assert post(client, content=b"0" * (MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_low_confidence_is_flagged_uncertain():
    with TestClient(create_app(loader=lambda: FakePredictor(confidence=0.4))) as c:
        assert post(c).json()["uncertain"] is True


def test_missing_model_gives_503_but_api_starts():
    def broken():
        raise FileNotFoundError("models/selection.json")
    with TestClient(create_app(loader=broken)) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert post(c).status_code == 503


@pytest.mark.skipif(not SELECTION_JSON.exists(), reason="modèle non entraîné (python -m src.models.train)")
def test_real_model_end_to_end():
    with TestClient(create_app()) as c:
        body = post(c, content=png_bytes(200, 300)).json()
        classes = c.get("/health").json()["classes"]
    assert body["label"] in classes and abs(sum(t["probability"] for t in body["top"]) - 1) < 0.2
    assert json.loads(SELECTION_JSON.read_text(encoding="utf-8"))["selected"] == body["model"]
