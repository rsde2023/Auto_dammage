"""Prétraitement d'une région endommagée -> patch carré de taille fixe.

C'est LA fonction partagée entre l'entraînement et l'API (/predict) : aucune duplication.
- Entraînement : région = bbox COCO + marge de contexte, découpée dans la photo.
- Application  : région = rectangle recadré par l'utilisateur (il inclut naturellement un peu de contexte).
Dans les deux cas, `preprocess_region` transforme ensuite la région en patch identique.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Valeurs par défaut = configuration retenue en phase 3 (comparée sur la validation propre,
    cf. notebooks/03_feature_engineering.ipynb)."""
    patch_size: int = 128         # côté du patch carré (pixels) ; 64/96 moins bons, 192 pas meilleur
    margin: float = 0.0           # marge autour de la bbox (fraction du plus grand côté) ; 0.05-0.30 dégradent
    resize_mode: str = "stretch"  # "stretch" : déforme au carré ; "pad" : garde les proportions + bords répliqués

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def crop_region(image: np.ndarray, bbox: tuple[float, float, float, float], margin: float) -> np.ndarray:
    """Découpe la bbox COCO [x, y, w, h] élargie d'une marge, sans sortir de l'image."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    pad = margin * max(w, h)
    x0 = int(np.floor(max(0.0, x - pad)))
    y0 = int(np.floor(max(0.0, y - pad)))
    x1 = int(np.ceil(min(w_img, x + w + pad)))
    y1 = int(np.ceil(min(h_img, y + h + pad)))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)          # au moins 1 pixel
    return image[y0:y1, x0:x1]


def preprocess_region(region_rgb: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Région RGB de taille quelconque -> patch RGB uint8 (patch_size x patch_size)."""
    if region_rgb.ndim != 3 or region_rgb.shape[2] != 3:
        raise ValueError(f"Image RGB attendue (H, W, 3), reçu {region_rgb.shape}")
    s = cfg.patch_size
    h, w = region_rgb.shape[:2]
    if cfg.resize_mode == "stretch":
        interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_LINEAR
        return cv2.resize(region_rgb, (s, s), interpolation=interp)
    if cfg.resize_mode != "pad":
        raise ValueError(f"resize_mode inconnu : {cfg.resize_mode}")
    scale = s / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(region_rgb, (nw, nh), interpolation=interp)
    top, left = (s - nh) // 2, (s - nw) // 2
    # Bords RÉPLIQUÉS (et non noirs) : un bord noir créerait de faux contours détectés par HOG.
    return cv2.copyMakeBorder(small, top, s - nh - top, left, s - nw - left, cv2.BORDER_REPLICATE)


def region_patch(image_rgb: np.ndarray, bbox, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Raccourci entraînement : photo complète + bbox -> patch."""
    return preprocess_region(crop_region(image_rgb, bbox, cfg.margin), cfg)
