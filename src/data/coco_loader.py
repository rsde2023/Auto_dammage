"""Chargement des annotations CarDD (format COCO) en DataFrames pandas.

Ce module ne modifie JAMAIS les fichiers originaux : il les lit uniquement.
Il est réutilisé par l'EDA, le nettoyage et (plus tard) l'extraction des régions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Racine du projet = deux niveaux au-dessus de src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "CarDD_COCO"
SPLITS = ("train", "val", "test")
REQUIRED_TOP_LEVEL_KEYS = ("images", "annotations", "categories")


def annotation_path(split: str, root: Path = DATASET_ROOT) -> Path:
    """Chemin du fichier JSON d'annotations pour un split."""
    return Path(root) / "annotations" / f"instances_{split}2017.json"


def image_dir(split: str, root: Path = DATASET_ROOT) -> Path:
    """Dossier contenant les images d'un split."""
    return Path(root) / f"{split}2017"


def load_coco_json(split: str, root: Path = DATASET_ROOT) -> dict:
    """Lit le JSON COCO d'un split et vérifie les clés obligatoires."""
    if split not in SPLITS:
        raise ValueError(f"Split inconnu : {split!r} (attendu : {SPLITS})")
    with open(annotation_path(split, root), encoding="utf-8") as f:
        coco = json.load(f)
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in coco]
    if missing:
        raise KeyError(f"Clés COCO manquantes dans {split}: {missing}")
    return coco


def coco_to_dataframes(coco: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convertit un dictionnaire COCO en trois DataFrames (images, annotations, catégories).

    Les colonnes de bounding box sont dépliées : bbox = [x, y, largeur, hauteur]
    (convention COCO, coin supérieur gauche + dimensions, en pixels).
    """
    images = pd.DataFrame(coco["images"])
    images["split"] = split

    categories = pd.DataFrame(coco["categories"]).rename(
        columns={"id": "category_id", "name": "category_name"}
    )

    annotations = pd.DataFrame(coco["annotations"])
    annotations["split"] = split
    if not annotations.empty:
        bbox = pd.DataFrame(annotations["bbox"].tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        annotations = pd.concat([annotations.drop(columns=["bbox"]), bbox], axis=1)
        annotations = annotations.merge(
            categories[["category_id", "category_name"]], on="category_id", how="left"
        )
    return images, annotations, categories


def load_all_splits(root: Path = DATASET_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les trois splits et les concatène (colonne `split`).

    Vérifie que les trois fichiers déclarent exactement les mêmes catégories.
    Les `id` d'images et d'annotations ne sont uniques qu'à l'intérieur d'un split :
    toujours joindre sur (split, id).
    """
    all_images, all_annotations, reference_categories = [], [], None
    for split in SPLITS:
        images, annotations, categories = coco_to_dataframes(load_coco_json(split, root), split)
        cats = categories.sort_values("category_id").reset_index(drop=True)
        if reference_categories is None:
            reference_categories = cats
        elif not cats.equals(reference_categories):
            raise ValueError(f"Les catégories du split {split} diffèrent de celles de train")
        all_images.append(images)
        all_annotations.append(annotations)
    return (
        pd.concat(all_images, ignore_index=True),
        pd.concat(all_annotations, ignore_index=True),
        reference_categories,
    )


def add_bbox_features(annotations: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des descripteurs géométriques des bounding boxes (copie, sans effet de bord).

    - bbox_area_px      : largeur × hauteur de la boîte
    - bbox_rel_area     : surface de la boîte / surface de l'image
    - bbox_aspect_ratio : largeur / hauteur
    """
    img = images[["split", "id", "width", "height"]].rename(
        columns={"id": "image_id", "width": "img_w", "height": "img_h"}
    )
    out = annotations.merge(img, on=["split", "image_id"], how="left")
    out["bbox_area_px"] = out["bbox_w"] * out["bbox_h"]
    out["bbox_rel_area"] = out["bbox_area_px"] / (out["img_w"] * out["img_h"])
    out["bbox_aspect_ratio"] = out["bbox_w"] / out["bbox_h"].where(out["bbox_h"] > 0)
    return out


def load_image_info(root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Lit `image_info.xlsx` (métadonnées fournies par les auteurs de CarDD).

    Les noms de colonnes sont normalisés en snake_case pour faciliter leur usage.
    """
    df = pd.read_excel(Path(root) / "annotations" / "image_info.xlsx")
    # Le fichier original contient "complete or partial " (espace final) : on nettoie.
    df.columns = df.columns.str.strip()
    return df.rename(
        columns={
            "file_size (KB)": "file_size_kb",
            "#instances": "n_instances",
            "#categories": "n_categories",
            "shooting angle": "shooting_angle",
            "complete or partial": "complete_or_partial",
        }
    )
