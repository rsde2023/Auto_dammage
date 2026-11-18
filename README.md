
# AutoDamage ML

Classification du type de dommage automobile (Machine Learning classique) à partir du dataset [CarDD](https://cardd-ustc.github.io/).

## Guide de démarrage après le clone

> [!IMPORTANT]
> **Le dataset, les descripteurs et les modèles ne sont PAS sur GitHub** (trop volumineux, exclus par `.gitignore`).
> Après le clone, il faut **d'abord copier le dataset CarDD original**, puis **lancer 3 commandes** qui recalculent tout le reste.
> Rien n'est à télécharger à la main en dehors du dataset : les bibliothèques s'installent avec `requirements.txt`.

### Ce qui est sur GitHub et ce qu'il faut recréer

| Élément | Sur GitHub ? | Comment l'obtenir |
|---|---|---|
| Code (`src/`, `tests/`, `android/`, `scripts/`), notebooks, README | oui | `git clone` |
| Résultats légers (`reports/cleaning/`, `reports/duplicates/`, `reports/modeling/`, `reports/evaluation/`) | oui | `git clone` |
| **Dataset CarDD** (`dataset/CarDD_COCO/`, ≈ 3 Go) | **non** | **le copier à la main** (étape B) |
| Environnement Python (`.venv/`) | non | étapes C et D |
| Descripteurs (`data/processed/`, ≈ 50 Mo) | non | commande `build_dataset` (étape F) |
| Modèles entraînés (`models/`) | non | commande `train` (étape F) |
| Build Android (`android/app/build/`, `local.properties`) | non | créés automatiquement par Android Studio / Gradle |

### Prérequis à installer une seule fois sur le PC

| Logiciel | Pour quoi | Vérifier dans cmd |
|---|---|---|
| **Python 3.11** (cocher *Add python.exe to PATH*) | tout le projet | `py -3.11 --version` |
| **Git** | cloner le projet | `git --version` |
| **VS Code** + extensions *Python* et *Jupyter* | éditer, notebooks | — |
| **Android Studio** (avec un émulateur, par exemple Pixel 6) | application Android uniquement | — |

### Les étapes, dans l'ordre (terminal **cmd**, invite `C:\...>`)

> [!TIP]
> Toutes les commandes se lancent **à la racine du projet**. En PowerShell (`PS C:\...>`), seule l'activation change :
> `.\.venv\Scripts\Activate.ps1` au lieu de `.venv\Scripts\activate.bat`.

**A. Cloner le projet**
```bat
git clone https://github.com/rsde2023/Auto_dammage.git
cd Auto_dammage
```

**B. Copier le dataset original (obligatoire, avant toute commande)**

Récupérer le dossier `CarDD_COCO` de l'archive CarDD (`CarDD_release.zip` → `CarDD_release\CarDD_COCO`, à demander à ton binôme
ou sur le site officiel de CarDD) et le copier **tel quel, sans rien modifier**, pour obtenir exactement :
```text
Auto_dammage/
└── dataset/
    └── CarDD_COCO/
        ├── annotations/   image_info.xlsx, instances_train2017.json, instances_val2017.json, instances_test2017.json
        ├── train2017/     2816 images
        ├── val2017/       810 images
        └── test2017/      374 images
```

**C. Créer l'environnement Python (une seule fois)**
```bat
py -3.11 -m venv .venv
```

**D. Activer l'environnement (à chaque nouveau terminal) et installer les bibliothèques (une seule fois)**
```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
L'invite doit commencer par `(.venv)`. Bibliothèques installées (versions figées dans `requirements.txt`) : numpy, pandas,
scikit-learn **1.9.1** (ne pas mettre à jour), scikit-image, opencv-python, joblib, matplotlib, seaborn, openpyxl, pillow,
jupyter, pytest, fastapi, uvicorn, python-multipart, httpx.

**E. Vérifier l'installation**
```bat
python -m pytest -q
```
Résultat attendu à ce stade : `57 passed, 1 skipped` (le test ignoré a besoin du modèle, créé à l'étape F).
Si on obtient `3 skipped`, le dataset n'est pas au bon endroit : revoir l'étape B.

**F. Recréer les descripteurs, les modèles et l'évaluation (≈ 5 minutes au total)**
```bat
python -m src.features.build_dataset
python -m src.models.train
python -m src.evaluation.metrics
```

| Commande | Durée | Produit |
|---|---|---|
| `python -m src.features.build_dataset` | ≈ 30 s | `data/processed/features_s128_m00_stretch.npz` |
| `python -m src.models.train` | ≈ 2 min | `models/svm.joblib`, `rf.joblib`, `knn.joblib`, `selection.json` |
| `python -m src.evaluation.metrics` | ≈ 1 min | tableau des scores du test (SVM : F1 macro ≈ 0,77) |

Puis `python -m pytest -q` doit afficher `58 passed`.

**G. Lancer l'API, puis l'application** (deux terminaux à laisser ouverts, détails aux étapes 11 et 12 plus bas)
```bat
REM Terminal 1 : API
.venv\Scripts\activate.bat
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

REM Terminal 2 : émulateur
"%LOCALAPPDATA%\Android\Sdk\emulator\emulator.exe" -avd Pixel_6

REM Terminal 3 (une fois Android démarré) : installer l'application
cd android
gradlew.bat installDebug
```
Ou, plus simple : Android Studio → **Open** → dossier `android/` → **Run ▶**.
Photos de test dans l'émulateur : `scripts\envoyer_photos.bat dataset\CarDD_COCO\test2017\000040.jpg`.

> [!NOTE]
> **Facultatif** : rouvrir les notebooks (étape 7) et relancer le rapport de doublons (`python -m src.data.duplicate_report`, ≈ 5 min).
> Leurs résultats sont déjà sur GitHub, il n'est pas nécessaire de les relancer pour utiliser le projet.

## Détail des étapes (Windows + VS Code)

Prérequis : **Python 3.11** et **Git** installés.

### 1. Récupérer le projet

```bat
git clone https://github.com/rsde2023/Auto_dammage.git
cd Auto_dammage
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

Résultat attendu : `58 passed` une fois les modèles entraînés (étape 9) ; avant : `57 passed, 1 skipped` ;
sans dataset : `55 passed, 3 skipped`.

Pour lancer un seul fichier de tests (plus rapide quand on travaille sur une partie précise) :

```bat
python -m pytest tests/test_coco_loader.py -v
python -m pytest tests/test_cleaning.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_evaluation.py -v
python -m pytest tests/test_api.py -v
```

| Fichier | Ce qu'il vérifie | Résultat attendu |
|---|---|---|
| `test_coco_loader.py` | lecture des JSON COCO et de `image_info.xlsx` | `10 passed` |
| `test_evaluation.py` | métriques (F1 macro, accuracy) et intervalle de confiance par bootstrap | `3 passed` |
| `test_api.py` | API : réponse, bbox invalide, fichier illisible ou trop lourd, modèle absent, seuil « incertain » (+ test réel si le modèle est entraîné) | `11 passed` |
| `test_cleaning.py` | contrôles qualité, SHA-256/pHash, quasi-doublons, politique de nettoyage | `7 passed` |
| `test_features.py` | recadrage, patch 128×128, dimensions HOG/couleur/texture (1764/40/36), pas de NaN | `17 passed` |
| `test_models.py` | pipelines, recherche d'hyperparamètres, sauvegarde du modèle, prédiction avec le même prétraitement qu'à l'entraînement | `10 passed` |

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
| `04_modeling.ipynb` | KNN / SVM / Random Forest : réglage, comparaison, modèle retenu (après l'étape 9) | ≈ 3 min |
| `05_evaluation.ipynb` | évaluation finale sur le test, fiabilité des probabilités, analyse des erreurs (après l'étape 10) | ≈ 3 min |

### 8. Calculer les descripteurs (phase 3)

Les descripteurs ne sont pas sur GitHub (fichiers de plusieurs centaines de Mo). Pour les régénérer avec la configuration retenue
(≈ 30 s) dans `data/processed/features_s128_m00_stretch.npz` :

```bat
python -m src.features.build_dataset
```

Options : `--size 64 --margin 0.15 --mode pad` pour tester une autre configuration, `--force` pour recalculer un fichier existant.

### 9. Entraîner les modèles (phase 4)

Les modèles ne sont pas sur GitHub (fichiers volumineux). Cette commande règle les hyperparamètres de KNN, SVM et Random Forest
sur la validation propre (résultats dans `reports/modeling/search_results.csv`), réentraîne le meilleur réglage de chaque famille
sur train + val et l'enregistre dans `models/` (`knn.joblib`, `svm.joblib`, `rf.joblib`, et `selection.json` qui indique le modèle retenu) :

```bat
python -m src.models.train
```

Si `reports/modeling/search_results.csv` existe déjà (il est sur GitHub), la recherche n'est pas relancée : seuls les modèles finaux
sont réentraînés (quelques minutes). `--force-search` relance toute la recherche (≈ 10 à 15 min).

### 10. Évaluer sur le jeu de test (phase 5)

Évalue une seule fois les modèles de `models/` sur le test (officiel et propre) et écrit les résultats dans `reports/evaluation/`
(scores avec intervalles de confiance, F1 par classe, matrice de confusion, prédictions du modèle retenu) :

```bat
python -m src.evaluation.metrics
```

### 11. Lancer l'API (phase 6)

Après l'étape 9 (le modèle doit exister dans `models/`), dans un terminal où le `.venv` est activé :

```bat
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Laisser ce terminal ouvert pendant l'utilisation de l'application.
- Documentation interactive (pour tester sans téléphone) : http://127.0.0.1:8000/docs
- Vérification rapide : http://127.0.0.1:8000/health doit afficher `"model_loaded": true`.
- `--host 0.0.0.0` rend l'API joignable depuis un vrai téléphone sur le même Wi-Fi ; Windows peut demander d'autoriser Python dans le pare-feu (accepter pour les réseaux privés).

Test en ligne de commande avec une photo du test (Git Bash) :

```bash
curl -X POST http://127.0.0.1:8000/predict -F "image=@dataset/CarDD_COCO/test2017/000040.jpg" -F bbox_x=392.88 -F bbox_y=0 -F bbox_w=423.34 -F bbox_h=415.28
```

Résultat attendu : `"label_fr":"Phare cassé"` avec une confiance d'environ 0,95.

### 12. Lancer l'application Android (phase 6)

1. Android Studio → **Open** → choisir le dossier `android/` (pas la racine du projet). Attendre la fin de la synchronisation Gradle
   (la première fois, Android Studio télécharge les bibliothèques ; il crée lui-même `local.properties`).
2. Démarrer un émulateur (Device Manager, par exemple Pixel 6, Android 8 minimum) puis **Run ▶**.
3. L'API de l'étape 11 doit tourner sur le PC : l'émulateur la joint à l'adresse `http://10.0.2.2:8000/` (réglage par défaut).
4. **Mettre des photos du PC dans l'émulateur** (émulateur démarré), dans cmd à la racine du projet :

   ```bat
   scripts\envoyer_photos.bat "C:\chemin\vers\photo.jpg"
   scripts\envoyer_photos.bat "C:\chemin\vers\un_dossier"
   ```

   Les photos sont copiées dans `Pictures/AutoDamage` et indexées par Android. Dans l'application : *Choisir dans la galerie* →
   rubrique **Récent**, ou onglet **Albums** → *AutoDamage*. Autre méthode : glisser-déposer un `.jpg` sur la fenêtre de l'émulateur
   (il arrive dans **Albums → Download**). « Prendre une photo » utilise une caméra simulée (scène 3D, pas de vraie voiture).
5. **Vrai téléphone** : même réseau Wi-Fi que le PC, puis menu ⋮ → *Adresse du serveur* → `http://IP-du-PC:8000/`
   (IP visible avec `ipconfig` dans cmd, ligne « Adresse IPv4 ») → *Tester* → *Enregistrer*.

**Depuis VS Code, sans Android Studio** (ouvrir un nouveau terminal avec le **+** du panneau Terminal).
Regarder l'invite : `C:\...>` = **cmd**, `PS C:\...>` = **PowerShell** — les commandes diffèrent.

| Terminal | cmd (`C:\...>`) | PowerShell (`PS C:\...>`) | À laisser ouvert ? |
|---|---|---|---|
| 1 — API | `.venv\Scripts\activate.bat` puis `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` | `.\.venv\Scripts\Activate.ps1` puis la même commande `uvicorn` | oui |
| 2 — émulateur | `"%LOCALAPPDATA%\Android\Sdk\emulator\emulator.exe" -avd Pixel_6` | `& "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -avd Pixel_6` | oui |
| 3 — installation | `cd android` puis `gradlew.bat installDebug` | `cd android` puis `.\gradlew.bat installDebug` | non |

Attendre l'écran d'accueil d'Android dans l'émulateur avant le terminal 3, puis toucher l'icône **AutoDamage ML** (bleue, voiture blanche)
dans la liste des applications (glisser vers le haut depuis le bas de l'écran). `emulator.exe -list-avds` affiche les noms d'émulateurs disponibles.

En ligne de commande (sans Android Studio), depuis `android/` :

```bat
gradlew.bat assembleDebug testDebugUnitTest
```

Résultat attendu : `BUILD SUCCESSFUL`, 6 tests unitaires Java réussis, APK dans `android/app/build/outputs/apk/debug/app-debug.apk`.

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
- **Hyperparamètres** : choisis uniquement sur la validation propre (F1 macro) ; le test reste intact pour la phase 5.
  Le modèle final est ensuite réentraîné sur train + val avec ces hyperparamètres.
- **Probabilités du SVM** : on utilise `SVC(probability=True)` (méthode de libsvm), choisie sur la validation car bien mieux calibrée
  que `CalibratedClassifierCV`, dont les confiances valaient presque toutes ≈ 0,7 (notebook 05, section 3).
  Cette option est dépréciée depuis scikit-learn 1.9 et sera supprimée en 1.11 : **ne pas mettre à jour scikit-learn** (version figée à 1.9.1).
- **Seuil « résultat incertain »** : confiance < 0,5 (fixé a priori). Sur le test, 91 % des régions sont au-dessus (81 % de réussite),
  et sous le seuil la réussite n'est que de 46 %.
- **Fichiers régénérables non versionnés** : `data/processed/` (descripteurs) et `models/` (modèles entraînés) : à régénérer avec les étapes 8 et 9.
- **Le test a été utilisé une seule fois** (phase 5), avec des modèles déjà choisis : ses scores ne doivent pas servir à modifier le modèle.
- **Une prédiction n'est pas une expertise d'assurance** : l'API renvoie et l'application affiche cet avertissement à chaque résultat.
- **Limite : le modèle répond toujours l'une des 6 classes**, même pour une photo sans voiture (personne, objet) ou une voiture intacte :
  c'est un classifieur à ensemble fermé, il n'a jamais vu d'exemple « autre chose ». Mesures (images de démonstration de scikit-image,
  dont un portrait, et zones de voitures intactes de CarDD) : les détecteurs de nouveauté non supervisés (distance aux k plus proches voisins,
  Mahalanobis, Isolation Forest, One-Class SVM) ne font pas mieux que le hasard (AUROC ≈ 0,5) avec nos descripteurs HOG/couleur/texture ;
  la confiance du SVM est un peu plus basse sur ces images (médiane 0,62 sur le portrait contre 0,82 sur les vrais dommages)
  mais seules 23 % passent sous le seuil « incertain ».
- **Aucun code ML dans l'application Android ni dans l'API** : l'API appelle `src/models/inference.py`, qui réutilise exactement
  le prétraitement de l'entraînement (même découpe, même patch 128×128, mêmes descripteurs).
- **Émulateur ↔ PC** : `10.0.2.2` est l'adresse du PC vue depuis l'émulateur ; `127.0.0.1` dans l'émulateur désigne l'émulateur lui-même.
- **HTTP en clair** autorisé dans l'application (`network_security_config.xml`) car l'API de développement n'a pas HTTPS ; une mise en production l'exigerait.
- **Nombres envoyés à l'API** : toujours avec un point décimal (`Locale.US`) ; avec un téléphone en français, `12,5` serait refusé.
- **Émulateur lent / « System UI isn't responding »** : choisir *Wait*, ou redémarrer l'émulateur (Device Manager → *Cold Boot Now*).

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

### Modélisation (phase 4)

Entraînement sur train, choix des hyperparamètres sur la validation propre (F1 macro), test jamais utilisé
(détails : `notebooks/04_modeling.ipynb`, résultats complets : `reports/modeling/search_results.csv`, 174 réglages testés).

| Modèle | Meilleur réglage | F1 macro (val propre) |
|---|---|---|
| **SVM RBF (retenu)** | StandardScaler → PCA 400 → SVM, C = 10, gamma = scale, class_weight = balanced | **0,807** |
| Random Forest | 300 arbres, max_features = sqrt, sans limite de profondeur, class_weight = balanced | 0,738 |
| KNN | StandardScaler → PCA 100 → k = 5, distance cosinus | 0,659 |
| Baseline (hasard stratifié) | — | 0,163 |

- Le SVM est le meilleur sur chacune des 6 classes. Classe la plus difficile : *crack* (F1 0,61), souvent confondue avec *scratch*.
- Surapprentissage (score train ≈ 1,0) mais la courbe d'apprentissage monte encore : le modèle final est réentraîné sur **train + val**.
- Le déséquilibre est géré par `class_weight="balanced"` (SVM, RF) et par le choix du F1 macro.
- Le modèle sauvegardé (`models/svm.joblib`) contient le pipeline, le LabelEncoder et les configurations de prétraitement :
  l'API appellera `src/models/inference.py`, qui réutilise exactement le prétraitement de l'entraînement.

### Évaluation finale (phase 5)

Modèles choisis en phase 4, réentraînés sur train + val, évalués **une seule fois** sur le test
(détails : `notebooks/05_evaluation.ipynb`, chiffres : `reports/evaluation/`).

| Modèle | F1 macro test officiel (785) | F1 macro test propre (761) [IC 95 %] | Accuracy test propre |
|---|---|---|---|
| **SVM RBF (retenu)** | **0,776** | **0,774** [0,734 – 0,807] | 0,779 |
| Random Forest | 0,701 | 0,696 [0,657 – 0,732] | 0,689 |
| KNN | 0,659 | 0,655 [0,616 – 0,691] | 0,656 |
| Baseline (hasard stratifié) | 0,170 | 0,171 | 0,271 |

- Intervalle de confiance par bootstrap sur les **images** (les régions d'une même photo ne sont pas indépendantes).
- Score de test proche de celui de validation (0,807) : pas de surestimation importante.
- F1 par classe (SVM, test propre) : tire flat 0,90, glass shatter 0,82, scratch 0,80, lamp broken 0,79, dent 0,77, **crack 0,56**
  (souvent pris pour scratch).
- Probabilités fiables : à plus de 0,9 de confiance, 96 % de réussite.
- Limite principale : le test utilise les bbox des annotateurs ; dans l'application, la qualité du recadrage par l'utilisateur comptera.

### API et application Android (phase 6)

```
Application Android ──(photo + cadre, HTTP multipart)──► API FastAPI ──► DamagePredictor (même prétraitement qu'à l'entraînement) ──► SVM
        ▲                                                    │
        └────────────(classe, confiance, top 3, avertissement, JSON)┘
```

**API** (`src/api/main.py`)

| Route | Rôle |
|---|---|
| `GET /health` | état du serveur, modèle chargé, classes |
| `POST /predict` | `image` (fichier) + `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` (facultatifs, pixels) → classe, confiance, `uncertain`, top 3, avertissement |

Erreurs gérées : 400 (fichier illisible ou vide), 413 (plus de 10 Mo), 422 (cadre incomplet ou hors de l'image), 503 (modèle non entraîné).

**Application Android** (`android/`, Java + XML, Material Design 3, minSdk 26)

| Couche | Classes |
|---|---|
| Activity / Fragments (affichage uniquement) | `MainActivity`, `AnalyzeFragment`, `ResultFragment`, `HistoryFragment`, `CropOverlayView` (cadre de recadrage) |
| ViewModel (état, survit aux rotations) | `AnalyzeViewModel`, `ResultViewModel`, `HistoryViewModel` |
| Repository (seul accès aux données) | `PredictionRepository` |
| Retrofit (réseau) | `ApiService`, `ApiClient` |
| Room (historique local) | `AppDatabase`, `PredictionDao`, `PredictionEntity` |

Parcours : prendre ou choisir une photo → encadrer le dommage au plus près → *Analyser* → résultat (classe, confiance, 3 classes probables,
bandeau « résultat incertain » sous 0,5, avertissement) → enregistré dans l'historique (consultable hors connexion).

Le téléphone envoie la photo réduite à 1280 px maximum et le cadre en pixels ; c'est le **serveur** qui découpe la zone,
avec la même fonction que pour l'entraînement.
