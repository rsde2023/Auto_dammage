# Clarifications — AutoDamage ML

Document de préparation à la présentation : rôle de chaque fichier, structure du dataset original, technologies utilisées,
et réponses aux questions probables du jury. Tous les chiffres viennent des notebooks et des fichiers de `reports/`.

---

## 1. Le projet en une phrase

À partir d'une photo de voiture et d'un rectangle entourant un dommage, le système prédit le **type de dommage** parmi 6 classes
(bosse, rayure, fissure, vitre brisée, phare cassé, pneu crevé), avec du **Machine Learning classique** (pas de Deep Learning),
puis l'expose via une **API FastAPI** utilisée par une **application Android**.

```
photo + rectangle ─► découpe ─► patch 128×128 ─► 1840 descripteurs (HOG + couleur + texture)
                 ─► StandardScaler ─► PCA (400) ─► SVM RBF ─► classe + probabilités
```

---

## 2. Rôle de chaque fichier

### Racine

| Fichier | Rôle |
|---|---|
| `README.md` | guide d'installation et d'utilisation, notes importantes, méthodologie et résultats |
| `clarifications.md` | ce document (préparation de la présentation) |
| `requirements.txt` | bibliothèques Python avec leurs versions exactes |
| `pytest.ini` | configuration des tests (dossier `tests/`, racine du projet dans le chemin Python) |
| `.gitignore` | fichiers exclus de Git : dataset, `.venv`, descripteurs, modèles, secrets, build Android |
| `scripts/envoyer_photos.bat` | copie des photos du PC dans la galerie de l'émulateur Android |

### `src/data/` — lecture et nettoyage des données

| Fichier | Rôle |
|---|---|
| `coco_loader.py` | lit les 3 fichiers JSON COCO et `image_info.xlsx`, les transforme en tableaux pandas (une ligne par image, une ligne par annotation) |
| `image_io.py` | lit une image (fichier ou octets reçus par l'API) en RGB avec OpenCV ; le même code sert à l'entraînement et à l'API |
| `cleaning.py` | contrôles qualité (bbox hors image, polygones invalides, doublons d'annotation...), empreintes SHA-256 / pHash / dHash, détection des quasi-doublons (ORB + RANSAC), politique de nettoyage, construction de `regions.csv` |
| `duplicate_report.py` | génère le rapport de doublons dans `reports/duplicates/` (`python -m src.data.duplicate_report`) |

### `src/features/` — de l'image au vecteur

| Fichier | Rôle |
|---|---|
| `preprocessing.py` | découpe la zone (bbox) et la redimensionne en patch 128×128 ; **fonction partagée entre l'entraînement et l'API** |
| `extractors.py` | calcule les descripteurs : HOG (forme), histogrammes HSV + statistiques Lab (couleur), LBP + GLCM (texture) |
| `build_dataset.py` | calcule les descripteurs des 8740 régions et les enregistre dans `data/processed/` (`python -m src.features.build_dataset`) |

### `src/models/` — entraînement et prédiction

| Fichier | Rôle |
|---|---|
| `pipelines.py` | construit les pipelines scikit-learn (scaler → PCA → classifieur) et définit les grilles d'hyperparamètres |
| `train.py` | recherche des hyperparamètres sur la validation, réentraînement sur train + val, sauvegarde dans `models/` (`python -m src.models.train`) |
| `inference.py` | classe `DamagePredictor` : charge le modèle et prédit une nouvelle image avec exactement le prétraitement de l'entraînement |

### `src/evaluation/` et `src/api/`

| Fichier | Rôle |
|---|---|
| `evaluation/metrics.py` | évaluation finale sur le test : accuracy, F1 macro, intervalle de confiance par bootstrap, matrice de confusion (`python -m src.evaluation.metrics`) |
| `api/main.py` | API FastAPI : `GET /health` et `POST /predict` ; validation des entrées et messages d'erreur |

### `notebooks/` — démarche expliquée et figures

| Notebook | Contenu |
|---|---|
| `01_eda.ipynb` | exploration : nombre d'images et d'annotations, classes, tailles, exemples |
| `02_data_cleaning.ipynb` | contrôle qualité, doublons exacts et visuels, décisions de nettoyage, choix des colonnes |
| `03_feature_engineering.ipynb` | conversion des régions en vecteurs, choix mesurés (taille, marge, descripteurs), encodages et normalisation |
| `04_modeling.ipynb` | baselines, recherche d'hyperparamètres KNN / SVM / Random Forest, comparaison, courbe d'apprentissage |
| `05_evaluation.ipynb` | évaluation finale sur le test, fiabilité des probabilités, analyse des erreurs |

### `tests/` — tests automatiques (58 au total, `python -m pytest`)

| Fichier | Ce qu'il vérifie |
|---|---|
| `test_coco_loader.py` | lecture des JSON COCO et de l'Excel |
| `test_cleaning.py` | contrôles qualité, empreintes, quasi-doublons, politique de nettoyage |
| `test_features.py` | découpe, taille du patch, dimensions des descripteurs (1764 / 40 / 36), absence de NaN |
| `test_models.py` | pipelines, recherche, sauvegarde, prédiction identique à l'entraînement |
| `test_evaluation.py` | métriques et intervalle de confiance |
| `test_api.py` | réponses de l'API, entrées invalides, modèle absent |

### `reports/` — résultats légers versionnés

| Dossier | Contenu |
|---|---|
| `reports/cleaning/` | `regions.csv` (table de travail : une ligne = une zone endommagée), décisions par image et par annotation, problèmes détectés, métadonnées des images |
| `reports/duplicates/` | rapport des doublons exacts (SHA-256) et visuels (pHash, ORB) |
| `reports/modeling/` | `search_results.csv` (les 174 réglages testés), `best_per_model.csv` |
| `reports/evaluation/` | scores du test, F1 par classe, matrice de confusion, prédictions du SVM |

### Fichiers non versionnés (recréés par des commandes)

| Dossier | Contenu | Recréé par |
|---|---|---|
| `dataset/CarDD_COCO/` | dataset original (≈ 3 Go) | copie manuelle |
| `data/processed/` | descripteurs (`.npz`) | `python -m src.features.build_dataset` |
| `models/` | `svm.joblib`, `rf.joblib`, `knn.joblib`, `selection.json` | `python -m src.models.train` |

### `android/` — application mobile (Java + XML)

| Fichier / dossier | Rôle |
|---|---|
| `settings.gradle`, `build.gradle`, `app/build.gradle`, `gradlew(.bat)`, `gradle/wrapper/` | configuration Gradle et dépendances (Material, Navigation, Room, Retrofit) |
| `app/src/main/AndroidManifest.xml` | permissions (Internet), activité principale, partage de fichier avec l'appareil photo |
| `ui/MainActivity.java` | activité unique : barre d'outils, navigation du bas, réglage de l'adresse du serveur |
| `ui/analyze/AnalyzeFragment.java` | écran 1 : choisir ou prendre une photo, encadrer la zone, lancer l'analyse |
| `ui/analyze/AnalyzeViewModel.java`, `AnalyzeState.java` | état de l'écran (photo, cadre, chargement, erreur), conservé lors des rotations |
| `ui/widget/CropOverlayView.java` | vue de recadrage : cadre déplaçable et redimensionnable |
| `ui/result/ResultFragment.java`, `ResultViewModel.java` | écran 2 : classe, confiance, 3 classes probables, bandeau « incertain », avertissement |
| `ui/history/HistoryFragment.java`, `HistoryAdapter.java`, `HistoryViewModel.java` | écran 3 : historique des analyses |
| `data/repository/PredictionRepository.java` | **seul point d'accès aux données** : appelle l'API et enregistre dans Room |
| `data/api/*.java` | Retrofit : interface de l'API (`ApiService`), client HTTP (`ApiClient`), objets JSON |
| `data/db/*.java` | Room : base locale (`AppDatabase`), requêtes (`PredictionDao`), table (`PredictionEntity`) |
| `data/settings/ServerSettings.java` | adresse de l'API (par défaut `http://10.0.2.2:8000/` pour l'émulateur) |
| `util/*.java` | outils : conversion du cadre en pixels, images (rotation EXIF, JPEG), messages d'erreur, URL, JSON |
| `app/src/test/.../UtilsTest.java` | 6 tests unitaires Java |
| `res/layout/`, `res/values/`, `res/navigation/`, `res/menu/`, `res/drawable/` | écrans XML, textes en français, couleurs et thème Material 3, navigation, icônes |

---

## 3. Le dataset original : CarDD

**CarDD** (*Car Damage Detection*, Université des sciences et technologies de Chine, 2023) est un dataset public conçu à l'origine pour la
**détection** et la **segmentation** de dommages automobiles par Deep Learning. Nous l'utilisons pour de la **classification de zones**.

### 3.1 Organisation des fichiers

```text
dataset/CarDD_COCO/
├── annotations/
│   ├── instances_train2017.json   annotations du train (format COCO)
│   ├── instances_val2017.json     annotations de la validation
│   ├── instances_test2017.json    annotations du test
│   └── image_info.xlsx            une ligne par image : taille, nombre de dommages, angle de vue, couleur...
├── train2017/                     2816 photos .jpg
├── val2017/                       810 photos .jpg
└── test2017/                      374 photos .jpg
```

Le nom « 2017 » vient de la convention du format COCO, pas de la date des photos.

### 3.2 Chiffres clés

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| Images | 2816 (70,4 %) | 810 (20,3 %) | 374 (9,4 %) | **4000** |
| Annotations (zones endommagées) | 6211 | 1744 | 785 | **8740** |

- Une photo contient en moyenne **2,2 dommages** (de 1 à 13).
- Les photos font en général **1000 × 667** ou **1000 × 750** pixels.

### 3.3 Les 6 classes et leur répartition (nombre de zones)

| Classe (anglais) | Traduction | Train | Val | Test | Total | Part |
|---|---|---|---|---|---|---|
| scratch | rayure | 2560 | 728 | 307 | 3595 | 41 % |
| dent | bosse | 1806 | 501 | 236 | 2543 | 29 % |
| crack | fissure | 651 | 177 | 70 | 898 | 10 % |
| lamp broken | phare cassé | 494 | 141 | 69 | 704 | 8 % |
| glass shatter | vitre brisée | 475 | 135 | 71 | 681 | 8 % |
| tire flat | pneu crevé | 225 | 62 | 32 | 319 | 3,6 % |

Rapport de déséquilibre : 3595 / 319 ≈ **11,3** entre la classe la plus fréquente et la plus rare.

### 3.4 Le format COCO (fichiers JSON)

Chaque fichier JSON contient 5 parties : `info`, `licenses`, `categories`, `images`, `annotations`.

```json
"images":      [{"id": 13, "file_name": "000013.jpg", "width": 1000, "height": 685}]
"categories":  [{"id": 1, "name": "dent"}, {"id": 2, "name": "scratch"}, {"id": 3, "name": "crack"},
                {"id": 4, "name": "glass shatter"}, {"id": 5, "name": "lamp broken"}, {"id": 6, "name": "tire flat"}]
"annotations": [{"id": 1, "image_id": 13, "category_id": 3,
                 "bbox": [454.28, 227.22, 67.98, 30.59],
                 "segmentation": [[x1, y1, x2, y2, ...]],
                 "area": 1332.0, "iscrowd": 0}]
```

| Champ | Signification | Utilisé ? |
|---|---|---|
| `image_id` | photo à laquelle appartient le dommage | oui (pour retrouver la photo) |
| `category_id` | classe du dommage → **la cible y** | oui |
| `bbox` | rectangle `[x, y, largeur, hauteur]` en pixels (x, y = coin haut gauche) | oui (pour découper la zone) |
| `segmentation` | contour exact (polygone) du dommage | non : sert à la segmentation, pas à la classification |
| `area` | surface du polygone en pixels | non (contrôle qualité seulement) |
| `iscrowd` | annotation de groupe (format COCO) | non : toujours 0 |

### 3.5 Le fichier `image_info.xlsx`

4000 lignes, une par image : `id`, `file_name`, `width`, `height`, `file_size (KB)`, `#instances` (nombre de dommages),
`#categories` (nombre de classes différentes), `shooting angle` (side / front / rear / inside), `complete or partial`
(voiture entière ou partielle), `color` (couleur de la voiture).

- Valeurs manquantes : `shooting angle` 440, `color` 344. Elles ont été **laissées telles quelles**, car ces colonnes ne servent qu'à l'analyse des erreurs.
- Elles ne sont **pas des entrées du modèle** : dans l'application, l'utilisateur ne fournit pas ces informations.

### 3.6 Ce que nous en faisons

- **Une ligne de données = une zone endommagée** (une annotation), et non une photo. On découpe la bbox dans la photo, on la transforme en vecteur, et la cible est sa classe.
- **Le dataset original n'est jamais modifié** : le nettoyage produit des tables de décision dans `reports/cleaning/`.

---

## 4. Technologies et bibliothèques

### Python (Machine Learning, API)

| Bibliothèque | Version | Rôle dans le projet |
|---|---|---|
| Python | 3.11 | langage |
| pandas | 3.1.1 | tableaux de données (annotations, résultats) |
| NumPy | 2.4.6 | calcul numérique, matrices de descripteurs |
| scikit-learn | 1.9.1 | modèles (KNN, SVM, Random Forest), StandardScaler, PCA, LabelEncoder, métriques |
| scikit-image | 0.26.0 | descripteurs HOG, LBP, GLCM |
| OpenCV (`opencv-python`) | 5.1.0.95 | lecture des images, redimensionnement, conversions de couleurs (HSV, Lab), ORB + RANSAC pour les quasi-doublons |
| joblib | 1.6.0 | sauvegarde des modèles, calcul parallèle |
| matplotlib, seaborn | 3.11.2 / 0.13.2 | graphiques des notebooks |
| openpyxl | 3.1.5 | lecture de `image_info.xlsx` |
| Pillow | 13.0.0 | contrôle des images (lisibilité, dimensions) |
| Jupyter (nbformat, nbconvert, ipykernel) | — | notebooks |
| pytest | 9.2.0 | tests automatiques |
| FastAPI | 0.164.0 | API REST |
| uvicorn | 0.54.0 | serveur qui fait tourner l'API |
| python-multipart | 0.0.34 | réception des fichiers envoyés à l'API |
| httpx | 0.28.1 | client HTTP utilisé par les tests de l'API |

### Android

| Technologie | Rôle |
|---|---|
| Java 17 + XML | langage et écrans |
| Gradle 8.11.1 + Android Gradle Plugin 8.10.1 | compilation (minSdk 26 = Android 8, targetSdk 35) |
| Material Components 1.12.0 (Material Design 3) | boutons, cartes, barre de navigation, thème clair / sombre |
| Navigation 2.8.5 | passage entre les fragments |
| Lifecycle 2.8.7 (ViewModel, LiveData) | architecture MVVM |
| Retrofit 2.11.0 + Gson + OkHttp 4.12.0 | appels HTTP à l'API |
| Room 2.6.1 | base de données locale (historique) |
| ExifInterface 1.3.7 | remise à l'endroit des photos prises au téléphone |
| JUnit 4.13.2 | tests unitaires |

### Interdits par le cahier des charges (non utilisés)

YOLO, CNN, TensorFlow, PyTorch, Mask R-CNN, modèles pré-entraînés de Deep Learning.

---

## 5. Questions probables du jury et réponses

### Données

**Q1. Quel dataset avez-vous utilisé et combien de données ?**
CarDD : **4000 photos** et **8740 zones endommagées annotées**, réparties en 6 classes. Chaque zone est un exemple
d'apprentissage, donc nous avons 8740 exemples.

**Q2. Comment avez-vous divisé les données en train / validation / test ?**
Nous avons gardé la **division officielle de CarDD** :

| | Images | Zones (exemples) | Rôle |
|---|---|---|---|
| Train | 2816 | 6211 | apprentissage |
| Validation | 810 | 1744 (1666 après nettoyage) | choix des hyperparamètres et du modèle |
| Test | 374 | 785 (761 après nettoyage) | évaluation finale, une seule fois |

Trois raisons :
- c'est la division des auteurs, ce qui rend les résultats **comparables** aux autres travaux ;
- elle est faite **par image** : toutes les zones d'une même photo sont dans le même ensemble, sinon le modèle verrait au test des morceaux de photos déjà vues ;
- les proportions de classes sont semblables dans les trois ensembles.

**Q3. Pourquoi trois ensembles et pas deux ?**
- Le **train** sert à apprendre.
- La **validation** sert à choisir : hyperparamètres, descripteurs, modèle.
- Le **test** mesure le résultat final sur des données jamais utilisées pour un choix.

Si l'on choisissait sur le test, le score final serait trop optimiste, car on aurait « triché » en choisissant ce qui marche sur ces données précises.

**Q4. Pourquoi ne pas avoir fait de validation croisée ?**
La validation officielle est grande (1666 zones) et séparée par image. Une validation croisée aurait mélangé des photos presque identiques entre les plis (voir Q6), ce qui aurait gonflé les scores.

**Q5. Qu'avez-vous nettoyé ?**
Rien n'a été supprimé du dataset original. On a **contrôlé** les données et noté les décisions dans `reports/cleaning/`.

| Contrôle | Résultat |
|---|---|
| fichiers manquants ou corrompus | aucun |
| doublons exacts (empreinte SHA-256) | aucun sur 4000 images |
| bbox hors image, polygones invalides, annotations en double | contrôlés et signalés dans `issues.csv` |
| bbox très petites (< 16 px) | 96, gardées mais signalées |
| quasi-doublons visuels | **106 paires**, dont **40 entre ensembles différents** (186 images, 87 groupes) |

Les 8740 annotations sont toutes conservées pour l'entraînement.

**Q6. Comment avez-vous trouvé les images presque identiques et qu'en avez-vous fait ?**
- **Repérage des paires candidates** : empreintes perceptuelles (dHash, pHash) et comparaison de miniatures.
- **Vérification** : points d'intérêt **ORB** et homographie **RANSAC**, à partir de 25 correspondances géométriquement cohérentes.

On a trouvé 40 paires train ↔ val/test, c'est-à-dire la même voiture photographiée deux fois. Si une photo du test est presque identique à une photo du train, le test devient trop facile.

Ces images sont donc **exclues de l'évaluation « propre »**, sans être supprimées : 36 images de validation et 10 du test. D'où la « validation propre » (774 images, 1666 zones) et le « test propre » (364 images, 761 zones).

**Q7. Quelles colonnes avez-vous gardées ou retirées ?**
- **Gardées** : la bbox (pour découper la zone) et la classe (la cible).
- **Retirées** : la segmentation (sert à la segmentation, pas à la classification), `area`, `iscrowd` (toujours 0), les licences.
- **Gardées à part** : les colonnes de `image_info.xlsx` (angle, couleur...), pour l'analyse des erreurs seulement ; l'utilisateur de l'application ne les fournit pas.

**Q8. Les classes sont déséquilibrées : comment l'avez-vous géré ?**
*scratch* représente 41 % des zones et *tire flat* 3,6 %. Trois mesures :
1. `class_weight="balanced"` dans le SVM et le Random Forest : une erreur sur une classe rare coûte plus cher ;
2. le **F1 macro** comme métrique de choix, qui donne le même poids aux 6 classes ;
3. le score **par classe** est toujours rapporté.

Nous n'avons pas dupliqué ni supprimé d'exemples (sur-échantillonnage ou sous-échantillonnage) : ce n'était pas nécessaire avec les poids de classes.

### Transformation des images

**Q9. Comment une image devient-elle un vecteur ?**
On découpe la bbox et on la redimensionne en **128 × 128 pixels**, puis on calcule trois familles de descripteurs « faits main » :

| Famille | Ce qu'elle décrit | Nombre de valeurs |
|---|---|---|
| **HOG** (histogrammes de gradients orientés) | forme, contours | 1764 |
| **Couleur** : histogrammes HSV + moyennes et écarts-types Lab | couleurs | 40 |
| **Texture** : LBP + GLCM | surface lisse, rayée, fissurée, éclatée | 36 |
| **Total** | | **1840** |

**Q10. Pourquoi 128 pixels, sans marge, en déformant au carré ?**
Ces trois choix ont été **mesurés sur la validation** (notebook 03, F1 macro moyen de deux classifieurs de test) :

| Choix | Valeurs testées | Retenu |
|---|---|---|
| Taille du patch | 64 → 0,668 ; 96 → 0,701 ; **128 → 0,722** ; 192 → 0,720 | 128 (192 coûte plus sans gain) |
| Marge autour de la bbox | **0 % → 0,645** ; 15 % → 0,619 ; 30 % → 0,595 | aucune |
| Redimensionnement | **déformation au carré → 0,722** ; garder les proportions → 0,683 | déformation |

Conséquence pour l'application : l'utilisateur doit encadrer le dommage **au plus près**.

**Q11. Les trois familles de descripteurs sont-elles toutes utiles ?**
Oui, c'est mesuré : HOG seul donne 0,622, HOG + texture 0,690, les trois ensemble **0,722**. La couleur est faible seule mais améliore encore le résultat.

**Q12. Avez-vous utilisé un encodage (Label Encoding, One-Hot...) et une normalisation ?**

| Technique | Utilisée ? | Pourquoi |
|---|---|---|
| Label Encoding | oui, pour la cible | 6 noms de classes → entiers 0 à 5, même correspondance dans l'API |
| One-Hot / Leave-One-Out | non | aucune variable d'entrée catégorielle |
| StandardScaler | oui | échelles très différentes entre descripteurs ; indispensable pour SVM, KNN et PCA |
| MinMaxScaler | non | sensible aux valeurs extrêmes, pas meilleur |
| PCA | oui (400 composantes pour le SVM) | réduit 1840 → 400 dimensions, même score et entraînement 6 fois plus rapide |
| Discrétisation | non | les descripteurs sont déjà des histogrammes |

Le scaler et la PCA sont **ajustés sur le train uniquement**, dans un pipeline scikit-learn, pour éviter toute fuite de données.

### Modèle

**Q13. Quels modèles avez-vous comparés ?**
**KNN, SVM et Random Forest**, comme demandé, et deux baselines (classe majoritaire et hasard stratifié). **174 réglages** ont été testés sur la validation propre : 96 KNN, 62 SVM, 16 Random Forest.

**Q14. Quel est le modèle retenu et pourquoi ?**
Un **SVM à noyau RBF** (après StandardScaler et PCA à 400 composantes). Il a le meilleur F1 macro en validation (0,807, contre 0,738 pour le Random Forest et 0,659 pour le KNN), et il est **le meilleur sur chacune des 6 classes**.

Le SVM convient bien ici :
- beaucoup de variables (1840) pour relativement peu d'exemples (6211) ;
- le noyau RBF sépare des classes qui ne le sont pas par une simple droite ;
- `class_weight` gère le déséquilibre.

**Q15. Quels sont ses hyperparamètres ?**

| Paramètre | Valeur | Signification |
|---|---|---|
| noyau | RBF (gaussien) | frontières courbes |
| C | 10 | compromis entre marge large et erreurs d'entraînement (testé : 1, 3, 10, 30, 100) |
| gamma | « scale » ≈ 0,00059 | portée d'influence de chaque exemple (testé : 0,5×, 1×, 2× la valeur « scale ») |
| PCA | 400 composantes | testé : 50, 100, 200, 400 et sans PCA (0,803, équivalent mais plus lent) |
| class_weight | balanced | compense le déséquilibre des classes |
| probabilités | `probability=True` (méthode de libsvm) | probabilités calibrées pour l'application |

Le meilleur C et le meilleur gamma sont **au milieu** de leurs grilles : la recherche n'était pas trop étroite.

**Q16. Et les meilleurs réglages des autres modèles ?**
- **Random Forest** : 300 arbres, `max_features = sqrt`, profondeur illimitée, `class_weight = balanced` → 0,738.
- **KNN** : PCA 100, k = 5, distance cosinus, vote uniforme → 0,659. Le KNN souffre de la grande dimension (« malédiction de la dimension »).

**Q17. Pourquoi pas de Deep Learning ? Ce serait plus performant.**
C'est **interdit par le cahier des charges**, car le projet porte sur le Machine Learning classique. Un réseau de neurones ferait sûrement mieux, ce que nous reconnaissons comme une limite.

L'intérêt de notre approche :
- chaque étape est explicable (descripteurs interprétables) ;
- l'entraînement prend quelques minutes sur un PC portable, sans carte graphique ;
- le modèle est léger (15 Mo) et une prédiction prend environ 70 ms.

### Résultats

**Q18. Quelle est l'accuracy du modèle ?**
Sur le **test** (jamais utilisé pour un choix) :

| Mesure | Test officiel (785 zones) | Test propre (761 zones) |
|---|---|---|
| **Accuracy** | **78,3 %** | **77,9 %** |
| **F1 macro** | **0,776** | **0,774** (intervalle de confiance à 95 % : 0,734 – 0,807) |
| Balanced accuracy | 0,751 | 0,748 |

Pour comparaison :
- prédire toujours *scratch* donne 39 % d'accuracy sur le test, mais un F1 macro de 0,1 ;
- le hasard stratifié donne un F1 macro de 0,17.

Notre modèle fait donc environ **4,5 fois mieux** que le hasard en F1 macro.

**Q19. Pourquoi présentez-vous surtout le F1 macro plutôt que l'accuracy ?**
Avec des classes déséquilibrées, l'accuracy est trompeuse : un modèle qui répond toujours *scratch* aurait 41 % d'accuracy en ignorant 5 classes sur 6.

Le F1 macro calcule le F1 (compromis entre précision et rappel) de chaque classe, puis en fait la moyenne : chaque classe compte autant.

**Q20. Quels sont les résultats par classe ?** (SVM, test propre, F1)

| tire flat | glass shatter | scratch | lamp broken | dent | crack |
|---|---|---|---|---|---|
| 0,897 | 0,823 | 0,802 | 0,791 | 0,773 | **0,557** |

**Q21. Quelles sont les principales erreurs ?**
- **crack** (fissure) est la classe la plus difficile : moins d'une fissure sur deux est reconnue (rappel 0,46), la plupart sont prises pour des rayures, car ce sont deux traits fins et allongés.
- **dent ↔ scratch** est la confusion la plus fréquente en nombre, car beaucoup de bosses portent aussi des rayures.

**Q22. Le score de test est plus bas que celui de validation (0,774 contre 0,807) : est-ce grave ?**
Non :
- c'est attendu, car choisir le meilleur réglage parmi 174 sur la validation donne toujours un léger optimisme ;
- l'écart reste dans l'intervalle de confiance ;
- l'ordre des modèles est le même sur le test (SVM > Random Forest > KNN), ce qui confirme le choix.

**Q23. Y a-t-il du surapprentissage ?**
Oui en partie : le score sur le train est presque parfait (≈ 1,0) contre 0,81 en validation.

Mais la **courbe d'apprentissage** montre que le score de validation monte encore avec plus de données : 0,61 avec 10 % du train, 0,81 avec 100 %. **Plus de données améliorerait le modèle.** C'est pourquoi le modèle final est réentraîné sur train + validation, une fois les hyperparamètres choisis.

**Q24. Que signifie la « confiance » affichée ? Peut-on s'y fier ?**
C'est la probabilité estimée de la classe prédite. Elle est **bien calibrée** sur le test :
- quand le modèle annonce plus de 90 %, il a raison dans 96 % des cas ;
- sous 50 %, il n'a raison que dans 46 % des cas.

L'application affiche donc « **résultat incertain** » sous 0,5. Ce seuil a été fixé a priori, pas optimisé sur le test.

La méthode de probabilité (libsvm) a été choisie sur la validation : elle est bien mieux calibrée (erreur de calibration 0,03 contre 0,08) que `CalibratedClassifierCV`, qui donnait presque toujours ≈ 0,7.

**Q25. Pourquoi un intervalle de confiance, et comment est-il calculé ?**
Le test ne contient que 761 zones : un autre échantillon aurait donné un score un peu différent.

On fait un **bootstrap sur les images** : on retire 1000 fois au hasard (avec remise) 364 photos, puis on recalcule le F1. On tire des photos entières, car les zones d'une même photo ne sont pas indépendantes.

### Application et API

**Q26. Comment fonctionne l'application ?**
1. L'utilisateur prend ou choisit une photo et encadre le dommage.
2. Le téléphone envoie la photo (réduite à 1280 px) et le rectangle à l'API.
3. Le serveur découpe la zone, calcule les 1840 descripteurs et applique le SVM.
4. Il renvoie la classe, la confiance, les 3 classes les plus probables et un avertissement.
5. Le résultat est enregistré dans l'historique du téléphone (Room).

**Q27. Comment garantissez-vous que l'API fait le même prétraitement qu'à l'entraînement ?**
L'API appelle **les mêmes fonctions Python** (`crop_region`, `preprocess_region`, `extract_features`) avec la configuration enregistrée dans le modèle. Il n'y a **aucune copie du code**.

Un test automatique vérifie que les descripteurs calculés par l'API sont identiques à ceux de l'entraînement.

**Q28. Pourquoi le modèle tourne-t-il sur un serveur et pas dans le téléphone ?**
- Le cahier des charges demande une API.
- Les bibliothèques (scikit-learn, scikit-image, OpenCV) sont en Python et ne tournent pas directement sous Android.
- On peut améliorer le modèle sans mettre à jour l'application.

**Q29. Quelle architecture pour l'application Android ?**
**MVVM** avec un Repository :

| Couche | Rôle |
|---|---|
| Activity / Fragments | affichage uniquement, aucun calcul ni appel réseau |
| ViewModels | gardent l'état, y compris lors d'une rotation de l'écran |
| Repository | seul point d'accès aux données |
| Retrofit | appels à l'API |
| Room | historique local |

**Q30. Pourquoi l'adresse `10.0.2.2` ?**
Dans l'émulateur Android, `10.0.2.2` désigne le PC qui fait tourner l'émulateur ; `127.0.0.1` désignerait l'émulateur lui-même. Sur un vrai téléphone, on saisit l'adresse IP du PC dans le menu de l'application.

### Limites et perspectives

**Q31. Que se passe-t-il avec une photo qui n'est pas un dommage de voiture (une personne, par exemple) ?**
Le modèle répond quand même l'une des 6 classes : c'est un classifieur « à ensemble fermé », qui n'a jamais vu d'exemple « autre chose ».

Nous avons testé des détecteurs de nouveauté non supervisés (distance aux k plus proches voisins, Mahalanobis, Isolation Forest, One-Class SVM). Ils **ne font pas mieux que le hasard** avec nos descripteurs, car un morceau de photo quelconque a des contours et des textures semblables à ceux d'un dommage.

La confiance est un peu plus basse sur ces images (médiane 0,62 contre 0,82), mais cela ne suffit pas à les rejeter.

Piste testée : un deuxième classifieur « zone endommagée / voiture intacte » rejette 80 % des voitures intactes, mais seulement environ un quart des portraits. Il faudrait des exemples négatifs variés (personnes, objets) pour aller plus loin.

**Q32. Quelles sont les autres limites ?**
- Les scores sont mesurés avec les rectangles des annotateurs ; dans l'application, un recadrage approximatif dégrade le résultat.
- *crack* reste mal reconnu.
- Une zone qui contient deux dommages (bosse rayée) ne reçoit qu'une étiquette.
- **Une prédiction n'est pas une expertise d'assurance** : c'est une aide indicative, à confirmer par un professionnel. L'application l'affiche à chaque résultat.

**Q33. Comment amélioreriez-vous le projet ?**
- Plus de données, puisque la courbe d'apprentissage monte encore.
- Une classe « pas de dommage » avec des exemples négatifs.
- Des descripteurs plus fins pour les fissures (multi-échelle).
- Détecter automatiquement la zone au lieu de la faire encadrer.
- Sans la contrainte du cahier des charges : un réseau de neurones.

**Q34. Comment avez-vous vérifié que votre code fonctionne ?**
- **58 tests automatiques Python** (`pytest`) et **6 tests unitaires Java**.
- La compilation Android passe sans erreur, et lint (l'outil d'analyse d'Android) ne trouve aucune erreur.
- Un essai de bout en bout dans l'émulateur : photo → recadrage → API → résultat → historique.

---

## 6. Chiffres à retenir

| Élément | Valeur |
|---|---|
| Images / zones annotées | 4000 / 8740 |
| Division (images) | train 2816 · val 810 · test 374 |
| Division (zones) | train 6211 · val 1744 (propre 1666) · test 785 (propre 761) |
| Classes | 6 (scratch 41 % … tire flat 3,6 %) |
| Vecteur par zone | 1840 valeurs (HOG 1764 + couleur 40 + texture 36), patch 128×128 |
| Modèle | StandardScaler → PCA 400 → SVM RBF (C = 10, gamma = scale, class_weight = balanced) |
| Réglages testés | 174 (KNN 96, SVM 62, Random Forest 16) |
| F1 macro validation | SVM 0,807 · RF 0,738 · KNN 0,659 · hasard 0,163 |
| Test (propre) | accuracy 77,9 %, F1 macro 0,774 [0,734 – 0,807] |
| Classe la plus facile / difficile | tire flat 0,90 / crack 0,56 |
| Seuil « incertain » | confiance < 0,5 |
| Temps de prédiction | ≈ 70 ms par zone ; modèle de 15 Mo |
