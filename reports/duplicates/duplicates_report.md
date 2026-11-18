# Rapport de doublons — CarDD (train2017 / val2017 / test2017)

Généré le 2026-11-18 13:07 par `python -m src.data.duplicate_report` (213 s).
**Lecture seule : aucune image originale n'a été supprimée, déplacée ni modifiée.**

## 1. Doublons exacts — SHA-256

Chaque fichier est résumé par son empreinte SHA-256 (256 bits). Deux fichiers ont la même empreinte
uniquement s'ils sont identiques octet par octet.

| split | images | sha256_uniques | doublons_internes |
|---|---|---|---|
| train | 2816 | 2816 | 0 |
| val | 810 | 810 | 0 |
| test | 374 | 374 | 0 |

**Aucun doublon exact** : les 4000 fichiers ont 4000 empreintes SHA-256 distinctes, dans chaque split comme entre les splits.

## 2. Doublons visuels — méthode perceptuelle

Une même photo recolorée, éclaircie, recompressée ou retournée a une empreinte SHA-256 totalement différente.
On utilise donc :

1. **pHash** (hash perceptuel par DCT, 64 bits) : deux images visuellement proches ont des hash proches
   (distance de Hamming faible). Seuil « proche » : distance ≤ 10.
2. Candidats complémentaires pour les cas que pHash rate (image en miroir, recadrage) : dHash, miniatures corrélées,
   voisin en miroir.
3. **Confirmation ORB + RANSAC** : nombre de points clés correspondants cohérents avec une même transformation
   géométrique. Doublon confirmé si ≥ 25 inliers (« copie retouchée » si ≥ 100,
   sinon « même scène » : même voiture et même dommage photographiés deux fois).

### 2a. pHash seul

20 paires ont une distance pHash ≤ 10 ; 9 sont confirmées par ORB + RANSAC.

| splits | paires | confirmées |
|---|---|---|
| train ↔ train | 12 | 4 |
| train ↔ val | 7 | 5 |
| train ↔ test | 1 | 0 |
| val ↔ val | 0 | 0 |
| val ↔ test | 0 | 0 |
| test ↔ test | 0 | 0 |

### 2b. Doublons visuels confirmés (toutes sources de candidats)

8320 paires candidates vérifiées → **106 doublons visuels confirmés**,
dont **40 entre splits différents**. Ils forment 87 groupes
(186 images), dont 36 groupes à cheval sur plusieurs splits.

| splits | copie retouchée | même scène | total |
|---|---|---|---|
| train ↔ train | 29 | 28 | 57 |
| train ↔ val | 18 | 11 | 29 |
| train ↔ test | 7 | 2 | 9 |
| val ↔ val | 7 | 2 | 9 |
| val ↔ test | 0 | 2 | 2 |
| test ↔ test | 0 | 0 | 0 |

**pHash seul n'aurait trouvé que 9 de ces 106 doublons** : les recadrages, les miroirs et les
légers décalages donnent des distances pHash élevées (souvent 20 à 36) alors que l'image est la même.
C'est pourquoi la confirmation géométrique ORB + RANSAC est indispensable.

Exemples entre splits : `cross_split_duplicates.png`.

## 3. Conclusion

- Doublons exacts (SHA-256) : aucun.
- Doublons visuels : **oui**, y compris 40 paires entre train, validation et test.
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
| `phash_near_pairs.csv` | paires pHash ≤ 10 et confirmation ORB |
| `visual_duplicates_confirmed.csv` | doublons visuels confirmés (distance pHash, inliers, type) |
| `visual_duplicate_groups.csv` | groupes d'images identiques visuellement |
| `cross_split_duplicates.png` | exemples de doublons entre splits |
