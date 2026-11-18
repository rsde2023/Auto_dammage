"""Tests des contrôles de qualité (données synthétiques : on connaît la bonne réponse)."""
import cv2
import numpy as np
import pandas as pd

from src.data.cleaning import (
    _orb_features,
    apply_cleaning_policy,
    check_annotations,
    check_orphans,
    check_required_fields,
    duplicate_groups,
    orb_ransac_inliers,
)
from src.data.coco_loader import add_bbox_features, coco_to_dataframes

SQUARE = [[10, 10, 30, 10, 30, 30, 10, 30]]


def _coco(anns):
    return {
        "categories": [{"id": 1, "name": "dent"}],
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
                   {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"}],
        "annotations": anns,
    }


def _ann(i, bbox, seg=SQUARE, image_id=1, area=100.0):
    return {"id": i, "image_id": image_id, "category_id": 1, "bbox": bbox, "segmentation": seg, "area": area, "iscrowd": 0}


def _frames(anns):
    images, a, cats = coco_to_dataframes(_coco(anns), "train")
    return images, add_bbox_features(a, images), cats


def test_check_annotations_flags_each_problem():
    images, a, _ = _frames([
        _ann(1, [10, 10, 20, 20]),                 # correcte
        _ann(2, [10, 10, 0, 20]),                  # largeur nulle
        _ann(3, [90, 90, 20, 20], seg=[[90, 90, 110, 90, 110, 110]]),  # dépasse l'image
        _ann(4, [10, 10, 20, 20]),                 # doublon exact de 1
        _ann(5, [10, 10, 5, 40], seg=[[10, 10, 15, 10, 15, 50]]),      # minuscule (5 px)
        _ann(6, [12, 12, 20, 20], seg=[]),         # segmentation vide
        _ann(7, [50, 50, 20, 20]),                 # bbox ne correspond pas au polygone
    ])
    found = check_annotations(a).groupby("check")["id"].apply(set).to_dict()
    assert found["bbox_dimension_nulle"] == {2}
    assert found["bbox_hors_image"] == {3}
    assert found["annotation_dupliquée"] == {4}
    assert found["bbox_minuscule"] == {5}
    assert found["segmentation_vide"] == {6}
    assert found["bbox_incohérente_polygone"] == {2, 7}   # la bbox nulle (2) ne colle pas non plus au polygone
    assert 1 not in set().union(*found.values())


def test_check_orphans_and_required_fields():
    images, a, _ = _frames([_ann(1, [10, 10, 20, 20], image_id=1), _ann(2, [10, 10, 20, 20], image_id=99)])
    orph = check_orphans(images, a)
    assert set(orph.loc[orph["check"] == "annotation_sans_image", "id"]) == {2}
    assert set(orph.loc[orph["check"] == "image_sans_annotation", "id"]) == {2}
    a.loc[0, "segmentation"] = []
    req = check_required_fields(images, a).set_index(["table", "field"])["n_problems"]
    assert req[("annotation", "segmentation")] == 1 and req[("image", "file_name")] == 0


def test_orb_detects_mirrored_and_recolored_copy(tmp_path):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8), (5, 5), 0)
    other = cv2.GaussianBlur(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8), (5, 5), 0)
    copy = cv2.flip(base, 1)[:, :, ::-1]          # miroir + canaux de couleur permutés
    for name, im in (("base", base), ("copy", copy), ("other", other)):
        cv2.imencode(".jpg", im)[1].tofile(str(tmp_path / f"{name}.jpg"))
    fa = _orb_features(tmp_path / "base.jpg")
    assert orb_ransac_inliers(fa, _orb_features(tmp_path / "copy.jpg", flip=True)) >= 25
    assert orb_ransac_inliers(fa, _orb_features(tmp_path / "other.jpg")) < 25


def _pairs(rows):
    cols = ["split_a", "id_a", "file_a", "split_b", "id_b", "file_b", "is_duplicate"]
    return pd.DataFrame(rows, columns=cols)


def test_duplicate_groups_are_transitive():
    pairs = _pairs([("train", 1, "1", "val", 2, "2", True), ("val", 2, "2", "test", 3, "3", True),
                    ("train", 4, "4", "train", 5, "5", False)])
    g = duplicate_groups(pairs)
    assert len(g) == 3 and g["group"].nunique() == 1
    assert g["group_splits"].iloc[0] == "test+train+val"


def test_policy_keeps_train_and_drops_leaking_eval_images():
    images = pd.DataFrame({"split": ["train", "val", "test", "val", "val", "test"],
                           "id": [1, 2, 3, 4, 5, 6], "file_name": list("abcdef")})
    pairs = _pairs([("train", 1, "a", "val", 2, "b", True),    # val 2 fuit depuis train
                    ("train", 1, "a", "test", 3, "c", True),   # test 3 fuit depuis train
                    ("val", 4, "d", "val", 5, "e", True)])     # copie interne à val
    groups = duplicate_groups(pairs)
    ann = pd.DataFrame({"split": images["split"], "id": images["id"], "image_id": images["id"],
                        "category_name": "dent"})
    no_issues = pd.DataFrame(columns=["check", "split", "id"])
    img_dec, ann_dec = apply_cleaning_policy(images, groups, no_issues, ann)
    keep = dict(zip(img_dec["id"], img_dec["eval_keep"]))
    assert keep == {1: True, 2: False, 3: False, 4: True, 5: False, 6: True}
    assert ann_dec["keep"].all()
    assert dict(zip(ann_dec["id"], ann_dec["image_eval_keep"])) == keep


def test_build_region_table_keeps_only_useful_columns():
    from src.data.cleaning import REGION_COLUMNS, build_region_table
    images, a, _ = _frames([_ann(1, [10, 10, 20, 20]), _ann(2, [40, 40, 10, 10], image_id=2)])
    dec = a[["split", "id"]].assign(keep=[True, False], image_eval_keep=True, flags=["", "x"])
    regions = build_region_table(images, a, dec)
    assert list(regions.columns) == REGION_COLUMNS
    assert not {"segmentation", "area", "iscrowd"} & set(regions.columns)
    assert regions.set_index("annotation_id").loc[2, "file_name"] == "b.jpg"
    assert regions["keep"].tolist() == [True, False]


def test_sha256_exact_and_phash_near_duplicates(tmp_path):
    """SHA-256 : seule une copie binaire est un doublon exact.
    pHash : une version recompressée et éclaircie reste proche ; une autre image est loin."""
    from src.data.cleaning import find_exact_duplicates, scan_image_files
    rng = np.random.default_rng(1)
    split_dir = tmp_path / "train2017"
    split_dir.mkdir()
    base = cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (31, 31), 0)
    other = cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (31, 31), 0)
    edited = cv2.convertScaleAbs(base, alpha=1.0, beta=25)                   # éclaircie
    cv2.imencode(".jpg", base)[1].tofile(str(split_dir / "a.jpg"))
    (split_dir / "b.jpg").write_bytes((split_dir / "a.jpg").read_bytes())  # copie binaire exacte
    cv2.imencode(".jpg", edited, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tofile(str(split_dir / "c.jpg"))
    cv2.imencode(".jpg", other)[1].tofile(str(split_dir / "d.jpg"))
    images = pd.DataFrame({"split": "train", "id": [1, 2, 3, 4], "file_name": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]})

    scan = scan_image_files(images, root=tmp_path)
    assert scan["readable"].all() and scan["sha256"].str.len().eq(64).all()
    exact = find_exact_duplicates(scan)
    assert len(exact) == 1 and {exact.loc[0, "file_a"], exact.loc[0, "file_b"]} == {"a.jpg", "b.jpg"}

    h = dict(zip(scan["file_name"], scan["phash"].astype("uint64")))
    dist = lambda x, y: int(np.bitwise_count(np.uint64(h[x]) ^ np.uint64(h[y])))
    assert dist("a.jpg", "c.jpg") <= 10
    assert dist("a.jpg", "d.jpg") > 10
