"""Lecture d'images robuste sous Windows (chemins avec accents/espaces).

`cv2.imread` échoue silencieusement (retourne None) sur certains chemins
non-ASCII sous Windows ; on lit donc les octets puis on décode avec OpenCV.
Cette fonction sera réutilisée à l'identique par l'entraînement et par l'API.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Retourne l'image en RGB (H, W, 3, uint8). Lève une erreur si illisible."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible ou corrompue : {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
