"""Lecture d'images robuste sous Windows (chemins avec accents/espaces).

`cv2.imread` échoue silencieusement (retourne None) sur certains chemins
non-ASCII sous Windows ; on lit donc les octets puis on décode avec OpenCV.
Le même décodage sert à l'entraînement (fichier) et à l'API (octets reçus) :
aucune différence de lecture entre les deux.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_image_rgb(data: bytes | np.ndarray, source: str = "image") -> np.ndarray:
    """Octets d'un fichier image (JPEG, PNG...) -> image RGB (H, W, 3, uint8). Lève ValueError si illisible."""
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise ValueError(f"Image illisible ou corrompue : {source}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Retourne l'image en RGB (H, W, 3, uint8). Lève une erreur si illisible."""
    return decode_image_rgb(np.fromfile(str(path), dtype=np.uint8), str(path))
