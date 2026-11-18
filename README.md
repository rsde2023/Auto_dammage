
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

L'invite doit commencer par `(.venv)`.

### 5. Installer les dépendances (une seule fois)

```bat
python -m pip install -r requirements.txt
```

### 6. Vérifier que tout fonctionne

```bat
python -m pytest -v
```

Résultat attendu : `17 passed`.

Pour régénérer le rapport de doublons (SHA-256 + pHash, environ 5 minutes) dans `reports/duplicates/` :

```bat
python -m src.data.duplicate_report
```

### 7. Ouvrir les notebooks

Dans VS Code : `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`,
puis ouvrir les notebooks dans l'ordre (`notebooks/01_eda.ipynb`, `notebooks/02_data_cleaning.ipynb`) et choisir le noyau `.venv`.

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

