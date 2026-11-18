"""Construit la matrice X (descripteurs) et les étiquettes y pour toutes les régions de regions.csv.

Exécution (racine du projet, .venv activé) :
    python -m src.features.build_dataset                       # configuration par défaut
    python -m src.features.build_dataset --margin 0.3 --mode stretch --size 64

Sortie : data/processed/features_<tag>.npz (non versionné, régénérable) contenant un tableau par
bloc de descripteurs (hog, color, texture), les étiquettes et les métadonnées de chaque région.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.coco_loader import PROJECT_ROOT, image_dir
from src.data.image_io import read_image_rgb
from src.features.extractors import FeatureConfig, extract_feature_blocks
from src.features.preprocessing import PreprocessConfig, region_patch

REGIONS_CSV = PROJECT_ROOT / "reports" / "cleaning" / "regions.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
META_COLUMNS = ["split", "annotation_id", "image_id", "file_name", "category_name", "image_eval_keep",
                "bbox_w", "bbox_h", "width", "height"]


def load_regions(path: Path = REGIONS_CSV) -> pd.DataFrame:
    """Régions utilisables (keep == True), dans un ordre stable."""
    r = pd.read_csv(path)
    return r[r["keep"]].sort_values(["split", "annotation_id"]).reset_index(drop=True)


def _process_image(split: str, file_name: str, rows: list[tuple[int, tuple]], pre: PreprocessConfig,
                   feat: FeatureConfig) -> list[tuple[int, dict]]:
    img = read_image_rgb(image_dir(split) / file_name)           # chaque photo n'est lue qu'une fois
    return [(idx, extract_feature_blocks(region_patch(img, bbox, pre), feat)) for idx, bbox in rows]


def extract_dataset(regions: pd.DataFrame, pre: PreprocessConfig = PreprocessConfig(),
                    feat: FeatureConfig = FeatureConfig(), n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Retourne {bloc: matrice (n_regions, d_bloc)} dans l'ordre des lignes de `regions`."""
    tasks = []
    for (split, fname), g in regions.groupby(["split", "file_name"], sort=False):
        rows = [(i, (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)) for i, r in zip(g.index, g.itertuples())]
        tasks.append((split, fname, rows))
    results = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_process_image)(s, f, rows, pre, feat) for s, f, rows in tasks)
    flat = dict(item for res in results for item in res)
    return {b: np.stack([flat[i][b] for i in range(len(regions))]) for b in feat.blocks}


def config_tag(pre: PreprocessConfig) -> str:
    return f"s{pre.patch_size}_m{int(round(pre.margin * 100)):02d}_{pre.resize_mode}"


def build_and_save(pre: PreprocessConfig = PreprocessConfig(), feat: FeatureConfig = FeatureConfig(),
                   out_dir: Path = PROCESSED_DIR, force: bool = False, n_jobs: int = -1) -> Path:
    """Extrait les descripteurs et les enregistre ; réutilise le fichier existant sauf si force=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"features_{config_tag(pre)}.npz"
    if path.exists() and not force:
        return path
    regions = load_regions()
    t0 = time.time()
    blocks = extract_dataset(regions, pre, feat, n_jobs)
    np.savez_compressed(
        path, **blocks,
        y=regions["category_name"].to_numpy(dtype=str),
        meta=regions[META_COLUMNS].to_json(orient="split"),
        preprocess=pre.to_json(), features=json.dumps(feat.to_dict()),
    )
    print(f"{len(regions)} régions -> {path.relative_to(PROJECT_ROOT)} en {time.time() - t0:.0f} s "
          f"({', '.join(f'{b}: {m.shape[1]}' for b, m in blocks.items())})")
    return path


def load_features(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame, dict]:
    """Relit un fichier produit par build_and_save : (blocs, y, meta, configs)."""
    with np.load(path, allow_pickle=False) as z:
        blocks = {b: z[b] for b in z.files if b in ("hog", "color", "texture")}
        y = z["y"]
        meta = pd.read_json(__import__("io").StringIO(str(z["meta"])), orient="split")
        configs = {"preprocess": json.loads(str(z["preprocess"])), "features": json.loads(str(z["features"]))}
    return blocks, y, meta, configs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    d = PreprocessConfig()
    ap.add_argument("--size", type=int, default=d.patch_size)
    ap.add_argument("--margin", type=float, default=d.margin)
    ap.add_argument("--mode", choices=["pad", "stretch"], default=d.resize_mode)
    ap.add_argument("--force", action="store_true", help="recalculer même si le fichier existe")
    a = ap.parse_args()
    print(build_and_save(PreprocessConfig(a.size, a.margin, a.mode), force=a.force))


if __name__ == "__main__":
    main()
