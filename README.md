
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

Résultat attendu : `10 passed`.

### 7. Ouvrir les notebooks

Dans VS Code : `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`,
puis ouvrir `notebooks/01_eda.ipynb` et choisir le noyau `.venv`.

# Auto_dammage
