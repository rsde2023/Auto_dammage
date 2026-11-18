"""Contrôles de qualité du dataset CarDD (format COCO).

Principe : chaque fonction `check_*` LIT les données et retourne un DataFrame
listant les problèmes trouvés (une ligne = un problème). Rien n'est supprimé
ni modifié sur disque. La décision finale (garder / exclure) est prise par
`apply_cleaning_policy`, de façon explicite et traçable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.coco_loader import DATASET_ROOT, SPLITS, image_dir

REQUIRED_IMAGE_FIELDS = ("id", "file_name", "width", "height")
REQUIRED_ANNOTATION_FIELDS = ("id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "segmentation", "area")


def _issues(df: pd.DataFrame, check: str, cols: list[str], detail) -> pd.DataFrame:
    """Formate un sous-ensemble de lignes problématiques en table d'issues standard."""
    out = df[cols].copy()
    out.insert(0, "check", check)
    out["detail"] = detail if np.isscalar(detail) else list(detail)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- champs / ids

def check_required_fields(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Champs obligatoires absents (colonne manquante) ou nuls (NaN / None / liste vide)."""
    rows = []
    for table, df, fields in (("image", images, REQUIRED_IMAGE_FIELDS),
                              ("annotation", annotations, REQUIRED_ANNOTATION_FIELDS)):
        for field in fields:
            if field not in df.columns:
                rows.append({"table": table, "field": field, "n_problems": len(df), "detail": "colonne absente"})
                continue
            col = df[field]
            is_null = col.isna() | col.map(lambda v: isinstance(v, (list, str)) and len(v) == 0)
            rows.append({"table": table, "field": field, "n_problems": int(is_null.sum()), "detail": "valeurs nulles/vides"})
    return pd.DataFrame(rows)


def check_coco_ids(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Identifiants COCO incohérents : doublons d'id, de file_name, dimensions non positives."""
    parts = []
    dup_img = images[images.duplicated(["split", "id"], keep=False)]
    parts.append(_issues(dup_img, "image_id_dupliqué", ["split", "id", "file_name"], "même id dans un split"))
    dup_name = images[images.duplicated("file_name", keep=False)]
    parts.append(_issues(dup_name, "file_name_dupliqué", ["split", "id", "file_name"], "même fichier référencé 2 fois"))
    dup_ann = annotations[annotations.duplicated(["split", "id"], keep=False)]
    parts.append(_issues(dup_ann, "annotation_id_dupliqué", ["split", "id", "image_id"], "même id dans un split"))
    bad_dim = images[(images["width"] <= 0) | (images["height"] <= 0)]
    parts.append(_issues(bad_dim, "dimensions_non_positives", ["split", "id", "file_name"], "width/height <= 0"))
    return pd.concat(parts, ignore_index=True)


def check_unknown_categories(annotations: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    bad = annotations[~annotations["category_id"].isin(categories["category_id"])]
    return _issues(bad, "catégorie_inconnue", ["split", "id", "image_id", "category_id"], "category_id absent de categories")


def check_orphans(images: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Annotations sans image et images sans annotation (jointure sur (split, id))."""
    img_keys = set(zip(images["split"], images["id"]))
    ann_keys = set(zip(annotations["split"], annotations["image_id"]))
    no_img = annotations[[k not in img_keys for k in zip(annotations["split"], annotations["image_id"])]]
    no_ann = images[[k not in ann_keys for k in zip(images["split"], images["id"])]]
    return pd.concat([
        _issues(no_img, "annotation_sans_image", ["split", "id", "image_id"], "image_id introuvable"),
        _issues(no_ann, "image_sans_annotation", ["split", "id", "file_name"], "aucune annotation"),
    ], ignore_index=True)


# --------------------------------------------------------------------------- annotations / bbox

def _polygon_stats(segmentation) -> tuple[int, int, float, float, float, float]:
    """(nb polygones, nb polygones invalides <3 points ou longueur impaire, xmin, ymin, xmax, ymax)."""
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return 0, 0, np.nan, np.nan, np.nan, np.nan
    n_bad, pts = 0, []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
            n_bad += 1
            continue
        pts.append(np.asarray(poly, dtype=float).reshape(-1, 2))
    if not pts:
        return len(segmentation), n_bad, np.nan, np.nan, np.nan, np.nan
    p = np.vstack(pts)
    return len(segmentation), n_bad, p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def check_annotations(annotations: pd.DataFrame, tol: float = 1.0, tiny_px: float = 16.0) -> pd.DataFrame:
    """Annotations vides et bounding boxes invalides ou suspectes.

    `annotations` doit contenir img_w / img_h (cf. coco_loader.add_bbox_features).
    - bbox_dimension_nulle : largeur ou hauteur <= 0                       (invalide)
    - bbox_hors_image      : dépasse l'image de plus de `tol` px           (corrigeable par recadrage)
    - segmentation_vide    : aucun polygone exploitable                    (info : on utilise la bbox)
    - polygone_invalide    : polygone de moins de 3 points                 (info)
    - bbox_incohérente_polygone : la bbox ne correspond pas aux polygones (écart > 2 px)
    - area_supérieure_bbox : area > largeur × hauteur (impossible)
    - bbox_minuscule       : un côté < `tiny_px` px                         (suspect, peu exploitable)
    - annotation_dupliquée : même image, même classe, même bbox             (doublon)
    """
    a = annotations
    parts = []
    cols = ["split", "id", "image_id", "category_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

    zero = a[(a["bbox_w"] <= 0) | (a["bbox_h"] <= 0)]
    parts.append(_issues(zero, "bbox_dimension_nulle", cols, "w<=0 ou h<=0"))

    x2, y2 = a["bbox_x"] + a["bbox_w"], a["bbox_y"] + a["bbox_h"]
    out = a[(a["bbox_x"] < -tol) | (a["bbox_y"] < -tol) | (x2 > a["img_w"] + tol) | (y2 > a["img_h"] + tol)]
    overflow = np.maximum.reduce([-out["bbox_x"], -out["bbox_y"], (out["bbox_x"] + out["bbox_w"]) - out["img_w"],
                                  (out["bbox_y"] + out["bbox_h"]) - out["img_h"]])
    parts.append(_issues(out, "bbox_hors_image", cols, [f"dépassement max {v:.1f}px" for v in overflow]))

    stats = pd.DataFrame(a["segmentation"].map(_polygon_stats).tolist(), index=a.index,
                         columns=["n_poly", "n_bad_poly", "px0", "py0", "px1", "py1"])
    empty = a[stats["n_poly"] - stats["n_bad_poly"] == 0]
    parts.append(_issues(empty, "segmentation_vide", cols, "aucun polygone valide"))
    badp = a[stats["n_bad_poly"] > 0]
    parts.append(_issues(badp, "polygone_invalide", cols, [f"{n} polygone(s) < 3 points" for n in stats.loc[badp.index, "n_bad_poly"]]))

    ok = stats["px0"].notna()
    gap = pd.concat([(a["bbox_x"] - stats["px0"]).abs(), (a["bbox_y"] - stats["py0"]).abs(),
                     (x2 - stats["px1"]).abs(), (y2 - stats["py1"]).abs()], axis=1).max(axis=1)
    incoh = a[ok & (gap > 2.0)]
    parts.append(_issues(incoh, "bbox_incohérente_polygone", cols, [f"écart max {v:.1f}px" for v in gap[incoh.index]]))

    big_area = a[a["area"] > a["bbox_w"] * a["bbox_h"] * 1.01]
    parts.append(_issues(big_area, "area_supérieure_bbox", cols, "area > w*h"))

    tiny = a[((a["bbox_w"] < tiny_px) | (a["bbox_h"] < tiny_px)) & (a["bbox_w"] > 0) & (a["bbox_h"] > 0)]
    parts.append(_issues(tiny, "bbox_minuscule", cols, [f"{w:.0f}x{h:.0f}px" for w, h in zip(tiny["bbox_w"], tiny["bbox_h"])]))

    key = ["split", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    dup = a[a.duplicated(key, keep="first")]
    parts.append(_issues(dup, "annotation_dupliquée", cols, "doublon exact d'une autre annotation"))

    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- fichiers image

def check_files_on_disk(images: pd.DataFrame, root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Images référencées mais introuvables, et fichiers présents sur disque mais non référencés."""
    parts = []
    exists = [ (image_dir(s, root) / f).is_file() for s, f in zip(images["split"], images["file_name"]) ]
    missing = images[~np.array(exists, dtype=bool)]
    parts.append(_issues(missing, "image_introuvable", ["split", "id", "file_name"], "fichier absent"))
    rows = []
    for split in SPLITS:
        on_disk = {p.name for p in image_dir(split, root).iterdir() if p.is_file()}
        referenced = set(images.loc[images["split"] == split, "file_name"])
        rows += [{"split": split, "id": np.nan, "file_name": f} for f in sorted(on_disk - referenced)]
    extra = pd.DataFrame(rows, columns=["split", "id", "file_name"])
    parts.append(_issues(extra, "fichier_non_référencé", ["split", "id", "file_name"], "présent sur disque, absent du JSON"))
    return pd.concat(parts, ignore_index=True)


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    """Hash perceptuel 'difference hash' (64 bits) : robuste au redimensionnement et à la compression."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _phash(gray: np.ndarray) -> int:
    """Hash perceptuel 'pHash' (64 bits) basé sur la DCT, comme en compression JPEG.

    On garde les 8x8 basses fréquences de la DCT d'une miniature 32x32 (la structure globale
    de l'image) et chaque bit indique si un coefficient est au-dessus de la médiane.
    Robuste à la compression, au redimensionnement et aux retouches de luminosité/contraste.
    """
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])          # on ignore le coefficient DC (luminosité moyenne)
    return int(np.packbits(bits).view(">u8")[0])


def scan_image_files(images: pd.DataFrame, root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Décode ENTIÈREMENT chaque image et calcule : SHA-256 (doublon exact), dHash et pHash (quasi-doublon),
    statut de lecture (corruption), dimensions réelles. Lecture seule. ~1 min pour 4000 images.
    """
    rows = []
    for split, img_id, fname in zip(images["split"], images["id"], images["file_name"]):
        path = image_dir(split, root) / fname
        row = {"split": split, "id": img_id, "file_name": fname, "readable": False,
               "error": None, "sha256": None, "dhash": None, "phash": None, "real_w": np.nan, "real_h": np.nan}
        try:
            data = path.read_bytes()
            row["sha256"] = hashlib.sha256(data).hexdigest()
            with Image.open(path) as im:   # PIL détecte les JPEG tronqués au chargement complet
                im.load()
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError("OpenCV ne peut pas décoder l'image")
            row.update(readable=True, real_h=arr.shape[0], real_w=arr.shape[1], dhash=_dhash(arr), phash=_phash(arr))
        except Exception as e:  # noqa: BLE001 — on veut capturer toute erreur de lecture
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def check_corrupt_images(scan: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Fichiers illisibles, ou dont les dimensions réelles diffèrent du JSON."""
    bad = scan[~scan["readable"]]
    parts = [_issues(bad, "image_corrompue", ["split", "id", "file_name"], bad["error"].tolist())]
    m = scan.merge(images[["split", "id", "width", "height"]], on=["split", "id"])
    dim = m[m["readable"] & ((m["real_w"] != m["width"]) | (m["real_h"] != m["height"]))]
    parts.append(_issues(dim, "dimensions_différentes_json", ["split", "id", "file_name"],
                         [f"réel {w:.0f}x{h:.0f} vs json {W}x{H}" for w, h, W, H in zip(dim.real_w, dim.real_h, dim.width, dim.height)]))
    return pd.concat(parts, ignore_index=True)


def find_exact_duplicates(scan: pd.DataFrame) -> pd.DataFrame:
    """Paires d'images au contenu binaire identique (même empreinte SHA-256)."""
    s = scan.dropna(subset=["sha256"])
    groups = s[s.duplicated("sha256", keep=False)].groupby("sha256")
    rows = []
    for _, g in groups:
        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                rows.append(_pair(recs[i], recs[j], 0, "exact"))
    return pd.DataFrame(rows, columns=_PAIR_COLS)


_PAIR_COLS = ["split_a", "id_a", "file_a", "split_b", "id_b", "file_b", "hamming", "kind", "cross_split"]


def _pair(a: dict, b: dict, dist: int, kind: str) -> dict:
    return {"split_a": a["split"], "id_a": a["id"], "file_a": a["file_name"],
            "split_b": b["split"], "id_b": b["id"], "file_b": b["file_name"],
            "hamming": dist, "kind": kind, "cross_split": a["split"] != b["split"]}


def hash_pairs(scan: pd.DataFrame, column: str = "dhash", max_hamming: int = 12, chunk: int = 512) -> pd.DataFrame:
    """Paires dont les hash perceptuels (`dhash` ou `phash`) sont proches (Hamming <= max_hamming, sur 64 bits).

    Toutes les paires sont comparées, par blocs vectorisés (≈ 8 millions pour 4000 images).
    Un hash proche est un INDICE, pas une preuve : il est confirmé ensuite par ORB + RANSAC.
    """
    s = scan.dropna(subset=[column]).reset_index(drop=True)
    h = s[column].astype("uint64").to_numpy()
    recs = s.to_dict("records")
    rows = []
    for start in range(0, len(h), chunk):
        dist = np.bitwise_count(h[start:start + chunk, None] ^ h[None, :])
        for i, j in zip(*np.nonzero(dist <= max_hamming)):
            i += start
            if j > i:
                rows.append(_pair(recs[i], recs[j], int(dist[i - start, j]), column))
    return pd.DataFrame(rows, columns=_PAIR_COLS)


# --------------------------------------------------------------------------- quasi-doublons
#
# Méthode en 2 étapes (classique en recherche d'images, sans Deep Learning) :
#  1. CANDIDATS (rapide, toutes les paires) : dHash ou pHash proche OU miniatures 32x32 corrélées
#     (niveaux de gris + contours Sobel ; version miroir incluse).
#  2. VÉRIFICATION (précise, candidats seulement) : points clés ORB + RANSAC. Deux photos de la
#     même scène — même recolorée, éclaircie ou recadrée — partagent des dizaines de points
#     cohérents avec une même transformation géométrique ; deux photos différentes presque aucun.

THUMB = 32


def _zrows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def compute_thumbnails(images: pd.DataFrame, root: Path = DATASET_ROOT) -> dict[str, np.ndarray]:
    """Miniatures normalisées (une ligne par image, même ordre que `images`) :
    'gray' (niveaux de gris), 'edge' (amplitude du gradient), 'edge_flip' (miroir horizontal)."""
    gray, edge, edge_flip = [], [], []
    for split, fname in zip(images["split"], images["file_name"]):
        g = cv2.imdecode(np.fromfile(str(image_dir(split, root) / fname), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        t = cv2.resize(g, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = np.hypot(cv2.Sobel(t, cv2.CV_32F, 1, 0), cv2.Sobel(t, cv2.CV_32F, 0, 1))
        gray.append(t.ravel()); edge.append(e.ravel()); edge_flip.append(e[:, ::-1].ravel())
    return {k: _zrows(np.asarray(v)) for k, v in (("gray", gray), ("edge", edge), ("edge_flip", edge_flip))}


def candidate_pairs(images: pd.DataFrame, thumbs: dict[str, np.ndarray], scan: pd.DataFrame,
                    min_gray: float = 0.5, min_edge: float = 0.3, max_hamming: int = 12,
                    max_phash: int = 12) -> pd.DataFrame:
    """Union des candidats : dHash/pHash proches, miniatures corrélées, et meilleur voisin en miroir."""
    n = len(images)
    cg, ce = thumbs["gray"] @ thumbs["gray"].T, thumbs["edge"] @ thumbs["edge"].T
    iu = np.triu_indices(n, 1)
    sel = (cg[iu] >= min_gray) & (ce[iu] >= min_edge)
    cand = {(int(i), int(j)): (False, "miniatures") for i, j in zip(iu[0][sel], iu[1][sel])}

    pos = {(s, i): k for k, (s, i) in enumerate(zip(images["split"], images["id"]))}
    for column, limit in (("dhash", max_hamming), ("phash", max_phash)):
        for r in hash_pairs(scan, column, limit).itertuples():
            i, j = sorted((pos[(r.split_a, r.id_a)], pos[(r.split_b, r.id_b)]))
            cand.setdefault((i, j), (False, column))

    cf = thumbs["edge"] @ thumbs["edge_flip"].T            # image i vs miroir de j
    np.fill_diagonal(cf, -1)
    for i, j in enumerate(cf.argmax(axis=1)):              # meilleur voisin miroir de chaque image
        key = (min(i, int(j)), max(i, int(j)))
        cand.setdefault(key, (True, "miroir"))

    rows = [{"i": i, "j": j, "flip": f, "source": src, "corr_gray": float(cg[i, j]), "corr_edge": float(ce[i, j])}
            for (i, j), (f, src) in cand.items()]
    return pd.DataFrame(rows)


_ORB = None


def _orb_features(path: Path, flip: bool = False):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=1000)
    g = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    g = cv2.equalizeHist(g)                                 # atténue les retouches de luminosité
    if flip:
        g = cv2.flip(g, 1)
    return _ORB.detectAndCompute(g, None)


def orb_ransac_inliers(feat_a, feat_b, ratio: float = 0.75) -> int:
    """Nombre de correspondances ORB cohérentes avec une homographie (RANSAC)."""
    (ka, da), (kb, db) = feat_a, feat_b
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [p[0] for p in matcher.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        return len(good)
    a = np.float32([ka[m.queryIdx].pt for m in good])
    b = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def find_near_duplicates(images: pd.DataFrame, candidates: pd.DataFrame, min_inliers: int = 25,
                         root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Vérifie chaque paire candidate par ORB + RANSAC. Retourne toutes les paires testées
    avec leur nombre d'inliers et `is_duplicate` (inliers >= min_inliers)."""
    paths = [image_dir(s, root) / f for s, f in zip(images["split"], images["file_name"])]
    cache: dict[tuple[int, bool], tuple] = {}

    def feat(k, flip=False):
        if (k, flip) not in cache:
            cache[(k, flip)] = _orb_features(paths[k], flip)
        return cache[(k, flip)]

    out = candidates.copy()
    out["inliers"] = [orb_ransac_inliers(feat(i), feat(j, f)) for i, j, f in zip(out["i"], out["j"], out["flip"])]
    for side, col in (("a", "i"), ("b", "j")):
        out[f"split_{side}"] = images["split"].to_numpy()[out[col]]
        out[f"id_{side}"] = images["id"].to_numpy()[out[col]]
        out[f"file_{side}"] = images["file_name"].to_numpy()[out[col]]
    out["cross_split"] = out["split_a"] != out["split_b"]
    out["is_duplicate"] = out["inliers"] >= min_inliers
    return out.sort_values("inliers", ascending=False, ignore_index=True)


def duplicate_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les paires de doublons en groupes (composantes connexes : si A~B et B~C, {A,B,C}).
    Retourne une ligne par image concernée avec l'identifiant de son groupe."""
    dup = pairs[pairs["is_duplicate"]]
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in dup.itertuples():
        a, b = (r.split_a, r.id_a, r.file_a), (r.split_b, r.id_b, r.file_b)
        parent[find(a)] = find(b)
    rows = [{"split": k[0], "id": k[1], "file_name": k[2], "root": find(k)} for k in list(parent)]
    df = pd.DataFrame(rows, columns=["split", "id", "file_name", "root"])
    if df.empty:
        return df.assign(group=pd.Series(dtype=int), group_splits=pd.Series(dtype=str))
    df["group"] = df.groupby("root").ngroup()
    df["group_splits"] = df.groupby("group")["split"].transform(lambda s: "+".join(sorted(set(s))))
    return df.drop(columns="root").sort_values(["group", "split", "id"], ignore_index=True)


# --------------------------------------------------------------------------- politique de nettoyage

SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
STRONG_COPY_INLIERS = 100   # au-delà : même photo retouchée (couleur, miroir, luminosité)


def apply_cleaning_policy(images: pd.DataFrame, groups: pd.DataFrame, ann_issues: pd.DataFrame,
                          annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique la politique de nettoyage SANS rien supprimer : retourne deux tables de décisions.

    Images (`eval_keep`) — utilisé uniquement pour l'évaluation propre :
      * train : toujours conservé (doublons internes = variantes, effet comparable à une augmentation).
      * val / test : exclu de l'évaluation "propre" si l'image a un quasi-doublon dans un split
        prioritaire (train > val > test), ou si elle est une copie d'une autre image du même split
        (on garde la première par id). Les splits officiels restent inchangés.
    Annotations (`keep`) — utilisé pour construire les régions :
      * invalides (bbox nulle, catégorie inconnue, sans image, doublon exact) -> exclues ;
      * bbox hors image -> conservées, seront recadrées aux bords de l'image ;
      * bbox minuscules -> conservées mais signalées (traitement par marge de contexte en phase 3).
    """
    img = images[["split", "id", "file_name"]].copy()
    g = groups[["split", "id", "group"]] if len(groups) else pd.DataFrame(columns=["split", "id", "group"])
    img = img.merge(g, on=["split", "id"], how="left")
    img["eval_keep"], img["reason"] = True, ""

    for _, members in img.dropna(subset=["group"]).groupby("group"):
        members = members.assign(prio=members["split"].map(SPLIT_PRIORITY)).sort_values(["prio", "id"])
        best = members["prio"].iloc[0]
        first_of_best = members.index[0]
        for idx, r in members.iterrows():
            if r["split"] == "train" or idx == first_of_best:
                continue
            if r["prio"] > best:
                img.loc[idx, ["eval_keep", "reason"]] = [False, f"quasi-doublon d'une image de {members['split'].iloc[0]}"]
            else:
                img.loc[idx, ["eval_keep", "reason"]] = [False, f"copie d'une autre image de {r['split']}"]
    img["group"] = img["group"].astype("Int64")

    ann = annotations[["split", "id", "image_id", "category_name"]].copy()
    ann["keep"], ann["flags"] = True, ""
    exclude = {"bbox_dimension_nulle", "catégorie_inconnue", "annotation_sans_image", "annotation_dupliquée"}
    for check, sub in ann_issues.groupby("check"):
        keys = set(zip(sub["split"], sub["id"]))
        mask = [k in keys for k in zip(ann["split"], ann["id"])]
        ann.loc[mask, "flags"] = ann.loc[mask, "flags"].str.cat([check] * sum(mask), sep=";").str.strip(";")
        if check in exclude:
            ann.loc[mask, "keep"] = False
    eval_keep = dict(zip(zip(img["split"], img["id"]), img["eval_keep"]))
    ann["image_eval_keep"] = [eval_keep.get(k, False) for k in zip(ann["split"], ann["image_id"])]
    return img, ann


# --------------------------------------------------------------------------- sélection des colonnes

# Colonnes conservées dans la table des régions (une ligne = une zone endommagée à classifier).
REGION_COLUMNS = ["split", "annotation_id", "image_id", "file_name", "width", "height",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "category_id", "category_name",
                  "keep", "image_eval_keep", "flags"]

# Champs du dataset NON repris, avec la raison (affiché dans le notebook et le README).
DROPPED_FIELDS = {
    "annotations.segmentation": "contour polygonal pour la segmentation ; l'application ne fournit qu'un rectangle",
    "annotations.area": "surface du polygone, dérivée de la segmentation",
    "annotations.iscrowd": "valeur constante (toujours 0)",
    "annotations.attributes.occluded": "valeur constante (toujours False)",
    "images.license": "valeur constante (toujours 0)",
    "info / licenses": "métadonnées d'auteur (à citer dans le rapport, inutiles au modèle)",
    "xlsx: id, file_name, width, height": "redondants avec les JSON (vérifié identiques)",
    "xlsx: file_size (KB)": "sans lien avec le type de dommage",
    "xlsx: #instances, #categories": "recalculables depuis les JSON",
}

# Métadonnées gardées À PART : jamais utilisées comme variables du modèle (indisponibles dans
# l'application), seulement pour l'analyse des erreurs en phase 5.
METADATA_COLUMNS = ["id", "file_name", "shooting_angle", "complete_or_partial", "color"]


def constant_fields(coco_annotations: pd.DataFrame, images: pd.DataFrame) -> dict:
    """Valeurs distinctes des champs supposés constants (preuve qu'ils n'apportent rien)."""
    return {
        "annotations.iscrowd": sorted(coco_annotations["iscrowd"].unique().tolist()),
        "annotations.attributes.occluded": sorted({str(a.get("occluded")) for a in coco_annotations["attributes"]}),
        "images.license": sorted(images["license"].unique().tolist()),
    }


def build_region_table(images: pd.DataFrame, annotations: pd.DataFrame, ann_decisions: pd.DataFrame) -> pd.DataFrame:
    """Table finale des régions : uniquement les colonnes utiles + décisions de nettoyage."""
    img = images[["split", "id", "file_name", "width", "height"]].rename(columns={"id": "image_id"})
    ann = annotations[["split", "id", "image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "category_id", "category_name"]]
    dec = ann_decisions[["split", "id", "keep", "image_eval_keep", "flags"]]
    out = ann.merge(dec, on=["split", "id"], how="left").merge(img, on=["split", "image_id"], how="left")
    return out.rename(columns={"id": "annotation_id"})[REGION_COLUMNS].reset_index(drop=True)


def build_image_metadata(info: pd.DataFrame) -> pd.DataFrame:
    """Métadonnées descriptives de image_info.xlsx, pour l'analyse des erreurs uniquement."""
    return info[METADATA_COLUMNS].rename(columns={"id": "image_id"})
