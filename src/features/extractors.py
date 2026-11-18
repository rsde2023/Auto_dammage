"""Conversion d'un patch RGB en vecteur numérique (descripteurs « faits main », sans Deep Learning).

Trois familles de descripteurs, calculées séparément pour pouvoir mesurer l'apport de chacune :
- hog     : forme / contours (Histogram of Oriented Gradients)
- color   : histogrammes HSV + moyennes/écarts-types Lab
- texture : LBP (Local Binary Patterns) multi-échelle + statistiques GLCM (Haralick)
Aucune de ces fonctions n'« apprend » quoi que ce soit : elles sont identiques pour chaque image,
donc l'extraction ne peut pas provoquer de fuite de données.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

ALL_BLOCKS = ("hog", "color", "texture")


@dataclass(frozen=True)
class FeatureConfig:
    blocks: tuple[str, ...] = ALL_BLOCKS
    hog_orientations: int = 9
    hog_cells: int = 8                 # nombre de cellules par côté (indépendant de la taille du patch)
    hog_cells_per_block: int = 2
    hue_bins: int = 18
    sat_bins: int = 8
    val_bins: int = 8
    lbp_scales: tuple[tuple[int, int], ...] = ((8, 1), (16, 2))   # (P voisins, rayon R)
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 3)

    def to_dict(self) -> dict:
        return asdict(self)


def hog_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    ppc = gray.shape[0] // cfg.hog_cells
    return hog(gray, orientations=cfg.hog_orientations, pixels_per_cell=(ppc, ppc),
               cells_per_block=(cfg.hog_cells_per_block,) * 2, block_norm="L2-Hys", feature_vector=True)


def color_features(rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)            # H dans [0, 180), S et V dans [0, 256)
    n = hsv.shape[0] * hsv.shape[1]
    hists = [np.histogram(hsv[..., 0], bins=cfg.hue_bins, range=(0, 180))[0],
             np.histogram(hsv[..., 1], bins=cfg.sat_bins, range=(0, 256))[0],
             np.histogram(hsv[..., 2], bins=cfg.val_bins, range=(0, 256))[0]]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) / 255.0
    moments = np.concatenate([lab.mean(axis=0), lab.std(axis=0)])
    return np.concatenate([np.concatenate(hists) / n, moments])


def texture_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    parts = []
    for p, r in cfg.lbp_scales:
        lbp = local_binary_pattern(gray, P=p, R=r, method="uniform")   # valeurs 0..P+1
        parts.append(np.histogram(lbp, bins=p + 2, range=(0, p + 2))[0] / lbp.size)
    q = (gray.astype(np.uint16) * cfg.glcm_levels // 256).astype(np.uint8)
    glcm = graycomatrix(q, distances=list(cfg.glcm_distances), angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=cfg.glcm_levels, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        parts.append(np.nan_to_num(graycoprops(glcm, prop).mean(axis=1)))   # moyenne sur les 4 directions
    return np.concatenate(parts)


def extract_feature_blocks(patch_rgb: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> dict[str, np.ndarray]:
    """Calcule chaque famille de descripteurs demandée dans cfg.blocks (vecteurs float32)."""
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    funcs = {"hog": lambda: hog_features(gray, cfg), "color": lambda: color_features(patch_rgb, cfg),
             "texture": lambda: texture_features(gray, cfg)}
    unknown = set(cfg.blocks) - set(funcs)
    if unknown:
        raise ValueError(f"Blocs inconnus : {unknown}")
    return {b: funcs[b]().astype(np.float32) for b in cfg.blocks}


def extract_features(patch_rgb: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Vecteur final = concaténation des blocs, dans l'ordre de cfg.blocks."""
    blocks = extract_feature_blocks(patch_rgb, cfg)
    return np.concatenate([blocks[b] for b in cfg.blocks])
