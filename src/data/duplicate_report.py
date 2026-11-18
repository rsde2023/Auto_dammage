"""Rapport de doublons entre et dans train2017 / val2017 / test2017 (CarDD).

Exécution (depuis la racine du projet, .venv activé) :
    python -m src.data.duplicate_report

- Doublons EXACTS   : empreinte cryptographique SHA-256 du fichier (identique octet par octet).
- Doublons VISUELS  : hash perceptuel pHash (DCT) + candidats complémentaires (dHash, miniatures,
                      miroir), puis confirmation géométrique ORB + RANSAC.
Lecture seule : aucune image n'est supprimée, déplacée ni modifiée.
Sorties : reports/duplicates/ (rapport Markdown, CSV, planche d'images).
"""
from __future__ import annotations

import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import cleaning as cl  # noqa: E402
from src.data.coco_loader import PROJECT_ROOT, SPLITS, image_dir, load_all_splits  # noqa: E402
from src.data.image_io import read_image_rgb  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "duplicates"
PHASH_NEAR = 10          # distance de Hamming pHash (sur 64 bits) considérée comme "visuellement proche"
MIN_INLIERS = 25         # confirmation ORB + RANSAC (calibré visuellement en phase 2)


def _split_pair(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: " ↔ ".join(sorted([r["split_a"], r["split_b"]], key=SPLITS.index)), axis=1)


def _gallery(pairs: pd.DataFrame, path, title: str, n: int = 12) -> None:
    pairs = pairs.head(n)
    if pairs.empty:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 2.4 * len(pairs)), squeeze=False)
    for k, r in enumerate(pairs.itertuples()):
        for j, (s, f) in enumerate([(r.split_a, r.file_a), (r.split_b, r.file_b)]):
            axes[k, j].imshow(read_image_rgb(image_dir(s) / f))
            axes[k, j].axis("off")
            axes[k, j].set_title(f"{s}/{f}", fontsize=8)
        info = f"pHash d={r.phash_dist}  |  {r.inliers} inliers{'  (miroir)' if r.flip else ''}"
        axes[k, 0].text(1.02, -0.06, info, transform=axes[k, 0].transAxes, fontsize=8, ha="center", color="#52514e")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_report() -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images, _, _ = load_all_splits()

    print("1/4 Empreintes SHA-256, pHash, dHash (décodage complet de chaque image)…")
    scan = cl.scan_image_files(images)

    print("2/4 Doublons exacts (SHA-256)…")
    exact = cl.find_exact_duplicates(scan)
    per_split = scan.groupby("split").agg(images=("sha256", "size"), sha256_uniques=("sha256", "nunique"))
    per_split = per_split.loc[list(SPLITS)]
    all_unique = scan["sha256"].nunique()

    print("3/4 Paires pHash proches + candidats complémentaires…")
    phash_near = cl.hash_pairs(scan, "phash", PHASH_NEAR)
    thumbs = cl.compute_thumbnails(images)
    cand = cl.candidate_pairs(images, thumbs, scan, max_phash=PHASH_NEAR)

    print(f"4/4 Vérification ORB + RANSAC de {len(cand)} paires candidates…")
    pairs = cl.find_near_duplicates(images, cand, min_inliers=MIN_INLIERS)
    ph = scan["phash"].astype("uint64").to_numpy()
    pairs["phash_dist"] = [int(bin(int(ph[i]) ^ int(ph[j])).count("1")) for i, j in zip(pairs["i"], pairs["j"])]
    pairs["type"] = pd.cut(pairs["inliers"], [-1, MIN_INLIERS - 1, cl.STRONG_COPY_INLIERS - 1, 10**9],
                           labels=["différentes", "même scène", "copie retouchée"])
    confirmed = pairs[pairs["is_duplicate"]].copy()
    confirmed["splits"] = _split_pair(confirmed)

    # Paires pHash proches : combien sont confirmées géométriquement ?
    key = lambda df: set(zip(df["split_a"], df["id_a"], df["split_b"], df["id_b"]))
    conf_keys = key(confirmed) | {(c, d, a, b) for a, b, c, d in key(confirmed)}
    phash_near["confirmé_orb"] = [k in conf_keys for k in zip(phash_near["split_a"], phash_near["id_a"],
                                                              phash_near["split_b"], phash_near["id_b"])]
    phash_near["splits"] = _split_pair(phash_near) if len(phash_near) else []

    groups = cl.duplicate_groups(pairs)

    # ---------------- exports
    cols = ["split_a", "file_a", "split_b", "file_b", "phash_dist", "inliers", "flip", "type", "source"]
    exact.to_csv(OUT_DIR / "exact_duplicates_sha256.csv", index=False, encoding="utf-8")
    scan[["split", "id", "file_name", "sha256", "phash", "dhash"]].to_csv(OUT_DIR / "image_fingerprints.csv",
                                                                          index=False, encoding="utf-8")
    phash_near.to_csv(OUT_DIR / "phash_near_pairs.csv", index=False, encoding="utf-8")
    confirmed[cols + ["splits"]].to_csv(OUT_DIR / "visual_duplicates_confirmed.csv", index=False, encoding="utf-8")
    groups.to_csv(OUT_DIR / "visual_duplicate_groups.csv", index=False, encoding="utf-8")
    cross = confirmed[confirmed["cross_split"]].sort_values("inliers", ascending=False)
    _gallery(cross, OUT_DIR / "cross_split_duplicates.png", "Doublons visuels confirmés ENTRE splits", n=12)

    # ---------------- tableaux du rapport
    exact_tbl = per_split.reset_index().assign(doublons_internes=lambda d: d["images"] - d["sha256_uniques"])
    combos = [f"{a} ↔ {b}" for k, a in enumerate(SPLITS) for b in SPLITS[k:]]
    vis_tbl = (confirmed.groupby(["splits", "type"], observed=False).size().unstack(fill_value=0)
               .reindex(combos, fill_value=0))
    vis_tbl = vis_tbl[["copie retouchée", "même scène"]]
    vis_tbl["total"] = vis_tbl.sum(axis=1)
    ph_tbl = (phash_near.groupby("splits")["confirmé_orb"].agg(paires="size", confirmées="sum")
              .reindex(combos, fill_value=0)) if len(phash_near) else pd.DataFrame()

    n_cross_groups = int((groups.drop_duplicates("group")["group_splits"].str.contains(r"\+")).sum()) if len(groups) else 0
    stats = {
        "images": len(images), "sha256_uniques": all_unique, "exact_pairs": len(exact),
        "exact_cross": int(exact["cross_split"].sum()) if len(exact) else 0,
        "phash_near": len(phash_near), "phash_near_confirmed": int(phash_near["confirmé_orb"].sum()) if len(phash_near) else 0,
        "candidates": len(cand), "confirmed": len(confirmed), "confirmed_cross": int(confirmed["cross_split"].sum()),
        "groups": int(groups["group"].nunique()) if len(groups) else 0, "images_in_groups": len(groups),
        "cross_groups": n_cross_groups, "seconds": round(time.time() - t0),
    }

    exact_verdict = ("**Aucun doublon exact** : les {images} fichiers ont {sha256_uniques} empreintes SHA-256 distinctes, "
                     "dans chaque split comme entre les splits.").format(**stats) if stats["exact_pairs"] == 0 else \
        "**{exact_pairs} paires de doublons exacts** trouvées, dont {exact_cross} entre splits (voir `exact_duplicates_sha256.csv`).".format(**stats)

    report = f"""# Rapport de doublons — CarDD (train2017 / val2017 / test2017)

Généré le {datetime.now():%Y-%m-%d %H:%M} par `python -m src.data.duplicate_report` ({stats['seconds']} s).
**Lecture seule : aucune image originale n'a été supprimée, déplacée ni modifiée.**

## 1. Doublons exacts — SHA-256

Chaque fichier est résumé par son empreinte SHA-256 (256 bits). Deux fichiers ont la même empreinte
uniquement s'ils sont identiques octet par octet.

{_md_table(exact_tbl)}

{exact_verdict}

## 2. Doublons visuels — méthode perceptuelle

Une même photo recolorée, éclaircie, recompressée ou retournée a une empreinte SHA-256 totalement différente.
On utilise donc :

1. **pHash** (hash perceptuel par DCT, 64 bits) : deux images visuellement proches ont des hash proches
   (distance de Hamming faible). Seuil « proche » : distance ≤ {PHASH_NEAR}.
2. Candidats complémentaires pour les cas que pHash rate (image en miroir, recadrage) : dHash, miniatures corrélées,
   voisin en miroir.
3. **Confirmation ORB + RANSAC** : nombre de points clés correspondants cohérents avec une même transformation
   géométrique. Doublon confirmé si ≥ {MIN_INLIERS} inliers (« copie retouchée » si ≥ {cl.STRONG_COPY_INLIERS},
   sinon « même scène » : même voiture et même dommage photographiés deux fois).

### 2a. pHash seul

{stats['phash_near']} paires ont une distance pHash ≤ {PHASH_NEAR} ; {stats['phash_near_confirmed']} sont confirmées par ORB + RANSAC.

{_md_table(ph_tbl.reset_index().rename(columns={'splits': 'splits'})) if len(ph_tbl) else '_aucune paire_'}

### 2b. Doublons visuels confirmés (toutes sources de candidats)

{stats['candidates']} paires candidates vérifiées → **{stats['confirmed']} doublons visuels confirmés**,
dont **{stats['confirmed_cross']} entre splits différents**. Ils forment {stats['groups']} groupes
({stats['images_in_groups']} images), dont {stats['cross_groups']} groupes à cheval sur plusieurs splits.

{_md_table(vis_tbl.reset_index().rename(columns={'splits': 'splits'}))}

**pHash seul n'aurait trouvé que {stats['phash_near_confirmed']} de ces {stats['confirmed']} doublons** : les recadrages, les miroirs et les
légers décalages donnent des distances pHash élevées (souvent 20 à 36) alors que l'image est la même.
C'est pourquoi la confirmation géométrique ORB + RANSAC est indispensable.

Exemples entre splits : `cross_split_duplicates.png`.

## 3. Conclusion

- Doublons exacts (SHA-256) : {'aucun' if stats['exact_pairs'] == 0 else stats['exact_pairs']}.
- Doublons visuels : **oui**, y compris {stats['confirmed_cross']} paires entre train, validation et test.
  Ce sont surtout des variantes d'une même photo (teinte, luminosité, miroir) ou des prises successives de la même scène.
- Risque : fuite de données (le modèle serait évalué sur des images presque vues à l'entraînement).
- Traitement retenu (phase 2, `reports/cleaning/image_decisions.csv`) : splits officiels inchangés ; les images de
  validation/test ayant un doublon dans un split prioritaire (train > val > test) sont exclues de l'évaluation « propre ».
- Limites : pHash seul ne détecte ni les miroirs ni les recadrages forts (d'où les candidats complémentaires) ;
  le seuil ORB peut produire quelques faux positifs, examinés visuellement en phase 2.

## Fichiers

| Fichier | Contenu |
|---|---|
| `image_fingerprints.csv` | SHA-256, pHash, dHash de chaque image |
| `exact_duplicates_sha256.csv` | paires de doublons exacts |
| `phash_near_pairs.csv` | paires pHash ≤ {PHASH_NEAR} et confirmation ORB |
| `visual_duplicates_confirmed.csv` | doublons visuels confirmés (distance pHash, inliers, type) |
| `visual_duplicate_groups.csv` | groupes d'images identiques visuellement |
| `cross_split_duplicates.png` | exemples de doublons entre splits |
"""
    (OUT_DIR / "duplicates_report.md").write_text(report, encoding="utf-8")
    print(f"Rapport écrit dans {OUT_DIR.relative_to(PROJECT_ROOT)} ({stats['seconds']} s)")
    return stats


if __name__ == "__main__":
    for k, v in build_report().items():
        print(f"  {k}: {v}")
