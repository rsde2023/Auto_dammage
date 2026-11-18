
# AutoDamage ML

Classification du type de dommage automobile (Machine Learning classique) à partir du dataset [CarDD](https://cardd-ustc.github.io/).

## Installation (Windows + VS Code)

Prérequis : **Python 3.11** et **Git** installés.

### 1. Récupérer le projet

```bat
git clone https://github.com/VOTRE_COMPTE/autodamage-ml.git
cd autodamage-ml
```

### 2. Ajouter le dataset

Le dataset n'est pas sur GitHub (trop volumineux). Le copier dans le projet pour obtenir :

```text
dataset/CarDD_COCO/
├── annotations/      (image_info.xlsx + 3 fichiers instances_*.json)
├── train2017/
├── val2017/
└── test2017/
```

### 3. Créer l'environnement Python (une seule fois)

```bat
py -3.11 -m venv .venv
```

### 4. Activer l'environnement (à chaque nouveau terminal)

- **Invite de commandes (cmd)** — invite `C:\...>` :
  ```bat
  .venv\Scripts\activate.bat
  ```
- **PowerShell** — invite `PS C:\...>` :
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  Si PowerShell refuse : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayer.
- **Git Bash** — invite `MINGW64 ~/...$` :
  ```bash
  source .venv/Scripts/activate
  ```

L'invite doit commencer par `(.venv)`. Sans activation, on peut toujours appeler directement
`.venv\Scripts\python.exe` (cmd/PowerShell) ou `.venv/Scripts/python.exe` (Git Bash) à la place de `python`.

### 5. Installer les dépendances (une seule fois)

```bat
python -m pip install -r requirements.txt
```

### 6. Vérifier que tout fonctionne

```bat
python -m pytest -v
```

Résultat attendu : `34 passed`.

Pour lancer un seul fichier de tests (plus rapide quand on travaille sur une partie précise) :

```bat
python -m pytest tests/test_coco_loader.py -v
python -m pytest tests/test_cleaning.py -v
python -m pytest tests/test_features.py -v
```

| Fichier | Ce qu'il vérifie | Résultat attendu |
|---|---|---|
| `test_coco_loader.py` | lecture des JSON COCO et de `image_info.xlsx` | `10 passed` |
| `test_cleaning.py` | contrôles qualité, SHA-256/pHash, quasi-doublons, politique de nettoyage | `7 passed` |
| `test_features.py` | recadrage, patch 128×128, dimensions HOG/couleur/texture (1764/40/36), pas de NaN | `17 passed` |

Pour régénérer le rapport de doublons (SHA-256 + pHash, environ 5 minutes) dans `reports/duplicates/` :

```bat
python -m src.data.duplicate_report
```

### 7. Ouvrir les notebooks

Dans VS Code : `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`,
puis ouvrir les notebooks dans l'ordre et choisir le noyau `.venv` :

| Notebook | Contenu | Durée d'exécution |
|---|---|---|
| `01_eda.ipynb` | exploration du dataset | ≈ 1 min |
| `02_data_cleaning.ipynb` | contrôle qualité, quasi-doublons, décisions de nettoyage | ≈ 5 min |
| `03_feature_engineering.ipynb` | conversion des régions en vecteurs, choix mesurés des descripteurs | ≈ 15 min |

### 8. Calculer les descripteurs (phase 3)

Les descripteurs ne sont pas sur GitHub (fichiers de plusieurs centaines de Mo). Pour les régénérer avec la configuration retenue
(≈ 30 s) dans `data/processed/features_s128_m00_stretch.npz` :

```bat
python -m src.features.build_dataset
```

Options : `--size 64 --margin 0.15 --mode pad` pour tester une autre configuration, `--force` pour recalculer un fichier existant.

## Notes importantes

- **Terminal** : la commande d'activation dépend du terminal (cmd, PowerShell ou Git Bash, voir étape 4). Si `python` ne trouve pas un module
  (`No module named pytest`…), c'est presque toujours que le `.venv` n'est pas activé.
- **Copier-coller dans Git Bash** : si la commande apparaît entourée de `[200~ … ~`, la retaper ou coller avec clic droit → *Paste*.
- **Le dataset original n'est jamais modifié.** Le nettoyage produit des tables de décisions dans `reports/cleaning/` ;
  la table de travail est `reports/cleaning/regions.csv` (une ligne = une zone endommagée).
- **Valeurs manquantes de `image_info.xlsx`** (`shooting angle` : 440, `color` : 344) : laissées telles quelles, car ces colonnes ne sont pas
  des entrées du modèle ; elles servent seulement à l'analyse des erreurs.
- **Le test n'est jamais utilisé pour faire un choix** (prétraitement, descripteurs, hyperparamètres) : tous les choix se font sur la validation propre.
- **Recadrage dans l'application** : le modèle est entraîné sur la zone exacte du dommage (sans marge) ; l'utilisateur doit recadrer **au plus près** du dommage.
- **Fichiers régénérables non versionnés** : `data/processed/` (descripteurs) et, plus tard, `models/` (modèles entraînés).

## Données utilisées

CarDD a été conçu pour la **détection** et la **segmentation** (Deep Learning). Notre projet fait de la **classification de zones** :
chaque dommage annoté est découpé selon sa bounding box, puis classé. Seules les informations utiles à cette tâche sont reprises dans
`reports/cleaning/regions.csv`. Les fichiers originaux ne sont jamais modifiés.

### Champs conservés

| Champ | Rôle |
|---|---|
| `category_id`, `category_name` | étiquette à prédire (6 classes) |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | découper la zone endommagée dans la photo (la taille de la bbox n'est pas une variable du modèle : dans l'application, c'est l'utilisateur qui recadre) |
| `image_id`, `file_name`, `split` | retrouver la photo et respecter le découpage officiel train / val / test |
| `width`, `height` | empêcher la zone découpée de dépasser l'image |
| `annotation_id` | retrouver une zone mal classée lors de l'analyse des erreurs |
| `keep`, `image_eval_keep`, `flags` | décisions du nettoyage (phase 2) |

### Champs non repris

| Champ | Raison |
|---|---|
| `segmentation` | contour exact pour la segmentation ; l'application ne fournit qu'un rectangle |
| `area` | surface du polygone, dérivée de la segmentation |
| `iscrowd` | toujours 0 dans tout le dataset |
| `attributes.occluded` | toujours `False` dans tout le dataset |
| `license` (images) | toujours 0 dans tout le dataset |
| `info`, `licenses` | auteurs et source du dataset (cités dans le rapport, inutiles au modèle) |
| xlsx : `id`, `file_name`, `width`, `height` | redondants avec les JSON (vérifiés identiques) |
| xlsx : `file_size (KB)` | sans lien avec le type de dommage |
| xlsx : `#instances`, `#categories` | recalculables depuis les JSON |

### Champs gardés à part

`shooting angle`, `complete or partial` et `color` (fichier `image_info.xlsx`) sont enregistrés dans `reports/cleaning/image_metadata.csv`.
Ils ne sont **jamais** des variables du modèle, car l'utilisateur ne les fournit pas dans l'application. Ils servent uniquement à analyser
les erreurs (par exemple : le modèle se trompe-t-il davantage sur les voitures noires ?).


## Méthodologie

### Division train / validation / test

Nous conservons la **division officielle** fournie par les auteurs de CarDD (dossiers `train2017/`, `val2017/`, `test2017/`) :
elle rend nos résultats comparables aux autres travaux et évite d'ajuster la division à notre avantage.

| Split | Photos | Régions | Régions « propres » | Rôle |
|---|---|---|---|---|
| train | 2816 (70 %) | 6211 | 6211 | apprentissage |
| val | 810 (20 %) | 1744 | 1666 | comparer les modèles et choisir les hyperparamètres |
| test | 374 (10 %) | 785 | 761 | score final, utilisé **une seule fois** à la fin |

- La division se fait **par photo** : toutes les zones d'une même photo sont dans le même split
  (sinon le modèle reconnaîtrait la photo plutôt que le dommage).
- Les 6 classes ont les **mêmes proportions** dans les trois splits : les scores de validation et de test sont représentatifs.
- **Aucun doublon exact** (SHA-256), mais **40 paires de photos quasi identiques entre splits** (même photo recolorée,
  en miroir ou éclaircie). Elles sont exclues de l'évaluation « propre » sans modifier la division officielle
  (`image_eval_keep` dans `reports/cleaning/regions.csv`). Le score final sera donné sur le test officiel **et** sur le test propre.
- Le test ne sert **jamais** à choisir un modèle ou un réglage : sinon son score deviendrait optimiste.

### Déséquilibre des classes

Dans le train, *scratch* représente 41 % des régions (2560) et *tire flat* 3,6 % (225), soit un rapport d'environ 11.
Ce n'est pas une erreur de division (les proportions sont identiques dans les trois splits) : c'est la fréquence réelle des dommages dans le dataset.

| Risque | Parade |
|---|---|
| Le modèle privilégie *scratch* | `class_weight="balanced"` pour SVM et Random Forest (comparé avec/sans) ; vote pondéré par la distance testé pour KNN |
| Un score trompeur : répondre toujours « scratch » donne déjà 39 % d'accuracy sur le test | métrique principale = **F1 macro** (même poids pour chaque classe), F1 par classe et matrice de confusion |
| Ne pas savoir si un modèle est réellement utile | **baseline** `DummyClassifier` (classe majoritaire) que nos modèles doivent nettement dépasser |

Nous ne rééquilibrons pas en supprimant des *scratch* (perte d'information) ni en dupliquant les *tire flat* (risque de mémorisation) :
la pondération des classes rééquilibre l'apprentissage sans perdre ni inventer de données.

Le score sur le **train** n'est jamais utilisé pour juger le modèle, mais il est comparé au score de validation pour
détecter le **surapprentissage** (train très élevé et validation faible = le modèle a mémorisé les photos).

### Conversion des images en vecteurs (phase 3)

```
photo ─► découpe de la bbox (sans marge) ─► patch 128×128 ─► HOG (1764) + couleur (40) + texture (36) ─► 1840 valeurs
```

| Choix | Valeur retenue | Pourquoi (mesuré sur la validation propre) |
|---|---|---|
| Marge autour de la bbox | aucune | 5 à 30 % de contexte dégradent le F1 macro |
| Taille du patch | 128 × 128 | meilleure que 64 et 96 ; 192 n'apporte rien et coûte 2 × plus |
| Redimensionnement | déformation au carré (*stretch*) | meilleur que garder les proportions (*pad*) à 128 px |
| HOG (forme) | gardé | meilleur bloc seul |
| Texture (LBP + GLCM) | gardée | améliore nettement HOG |
| Couleur (HSV + Lab) | gardée | faible seule, mais améliore encore HOG + texture |
| Label Encoding | pour la cible | 6 classes → 0..5, même correspondance dans l'API |
| One-Hot / Leave-One-Out | non | aucune variable d'entrée catégorielle |
| StandardScaler | oui, dans le pipeline | échelles très différentes ; ajusté sur le train uniquement |
| MinMaxScaler | non | sensible aux valeurs extrêmes, pas meilleur |
| PCA | testée en phase 4 | aide fortement KNN (malédiction de la dimension) |
| Discrétisation | non | variables déjà sous forme d'histogrammes |
