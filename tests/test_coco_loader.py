"""Tests du chargement des annotations COCO.

- Les tests "synthétiques" utilisent un mini-dataset COCO créé dans un dossier temporaire :
  ils tournent partout, même sans le dataset.
- Les tests "réels" vérifient le vrai dataset CarDD et sont ignorés s'il est absent.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.data.coco_loader import (
    DATASET_ROOT,
    SPLITS,
    add_bbox_features,
    coco_to_dataframes,
    load_all_splits,
    load_coco_json,
    load_image_info,
)
from src.data.image_io import read_image_rgb

CATEGORIES = [{"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}]


def _mini_coco(image_id_offset=0):
    return {
        "info": {}, "licenses": [], "categories": CATEGORIES,
        "images": [{"id": 1 + image_id_offset, "width": 100, "height": 50, "file_name": f"{1 + image_id_offset:06d}.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1 + image_id_offset, "category_id": 2, "segmentation": [[0, 0, 10, 0, 10, 5]],
             "area": 25.0, "bbox": [10, 5, 20, 10], "iscrowd": 0},
        ],
    }


@pytest.fixture
def mini_root(tmp_path):
    (tmp_path / "annotations").mkdir()
    for i, split in enumerate(SPLITS):
        with open(tmp_path / "annotations" / f"instances_{split}2017.json", "w", encoding="utf-8") as f:
            json.dump(_mini_coco(image_id_offset=i * 10), f)
    return tmp_path


# ---------- tests synthétiques ----------

def test_coco_to_dataframes_unpacks_bbox_and_category():
    images, anns, cats = coco_to_dataframes(_mini_coco(), "train")
    assert list(cats["category_name"]) == ["dent", "scratch"]
    row = anns.iloc[0]
    assert (row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h) == (10, 5, 20, 10)
    assert row.category_name == "scratch"
    assert row.split == "train" and images.iloc[0].split == "train"


def test_add_bbox_features_computes_relative_area_and_ratio():
    images, anns, _ = coco_to_dataframes(_mini_coco(), "train")
    out = add_bbox_features(anns, images)
    assert out.loc[0, "bbox_area_px"] == 200
    assert out.loc[0, "bbox_rel_area"] == pytest.approx(200 / 5000)
    assert out.loc[0, "bbox_aspect_ratio"] == pytest.approx(2.0)
    assert "img_w" not in anns.columns  # pas d'effet de bord sur l'entrée


def test_zero_height_bbox_gives_nan_ratio_not_crash():
    coco = _mini_coco()
    coco["annotations"][0]["bbox"] = [0, 0, 10, 0]
    images, anns, _ = coco_to_dataframes(coco, "train")
    assert np.isnan(add_bbox_features(anns, images).loc[0, "bbox_aspect_ratio"])


def test_load_all_splits_concatenates(mini_root):
    images, anns, cats = load_all_splits(mini_root)
    assert set(images["split"]) == set(SPLITS)
    assert len(anns) == 3 and len(cats) == 2


def test_inconsistent_categories_raise(mini_root):
    path = mini_root / "annotations" / "instances_val2017.json"
    coco = json.loads(path.read_text(encoding="utf-8"))
    coco["categories"] = [{"id": 1, "name": "other"}]
    path.write_text(json.dumps(coco), encoding="utf-8")
    with pytest.raises(ValueError):
        load_all_splits(mini_root)


def test_missing_key_raises(mini_root):
    path = mini_root / "annotations" / "instances_test2017.json"
    path.write_text(json.dumps({"images": []}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_coco_json("test", mini_root)


def test_unknown_split_raises(mini_root):
    with pytest.raises(ValueError):
        load_coco_json("holdout", mini_root)


def test_read_image_rgb_roundtrip_and_corrupt(tmp_path):
    import cv2
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # rouge en BGR
    path = tmp_path / "rouge é.png"  # chemin non-ASCII volontaire
    ok, buf = cv2.imencode(".png", bgr)
    buf.tofile(str(path))
    rgb = read_image_rgb(path)
    assert rgb.shape == (4, 6, 3) and rgb[0, 0].tolist() == [255, 0, 0]

    bad = tmp_path / "corrompu.jpg"
    bad.write_bytes(b"pas une image")
    with pytest.raises(ValueError):
        read_image_rgb(bad)


# ---------- tests sur le vrai dataset CarDD ----------

real = pytest.mark.skipif(not DATASET_ROOT.exists(), reason="dataset CarDD absent")

EXPECTED_CATEGORIES = {1: "dent", 2: "scratch", 3: "crack", 4: "glass shatter", 5: "lamp broken", 6: "tire flat"}


@real
def test_real_dataset_structure():
    images, anns, cats = load_all_splits()
    assert dict(zip(cats["category_id"], cats["category_name"])) == EXPECTED_CATEGORIES
    assert images.groupby("split").size().to_dict() == {"train": 2816, "val": 810, "test": 374}
    assert len(anns) == 8740
    assert images["file_name"].is_unique  # aucun nom de fichier partagé entre splits
    assert anns["category_name"].notna().all()


@real
def test_real_image_info_matches_json():
    images, _, _ = load_all_splits()
    info = load_image_info()
    assert "complete_or_partial" in info.columns  # espace final du fichier original normalisé
    merged = images.merge(info, on="file_name", suffixes=("_json", "_xlsx"))
    assert len(merged) == len(images) == len(info)
    assert (merged["width_json"] == merged["width_xlsx"]).all()
    assert (merged["height_json"] == merged["height_xlsx"]).all()
