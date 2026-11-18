"""Construction des pipelines scikit-learn (KNN, SVM, Random Forest) à partir d'un dictionnaire d'hyperparamètres.

Un pipeline enchaîne : StandardScaler -> PCA (optionnelle) -> classifieur.
Toutes les étapes sont ajustées (fit) sur les données d'entraînement uniquement : le scaler et la PCA
n'« apprennent » jamais rien de la validation ou du test (pas de fuite de données).
Le Random Forest n'a ni scaler ni PCA : un arbre compare une variable à un seuil, l'échelle ne change rien,
et la PCA mélangerait les descripteurs en perdant leur sens.
"""
from __future__ import annotations

import warnings

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("knn", "svm", "rf")
SEED = 0

# Grilles explorées sur la validation propre (une liste de grilles = plusieurs sous-grilles).
# Pour le SVM, gamma est exprimé en multiple de la valeur « scale » de scikit-learn,
# calculée sur les données transformées : gamma = gamma_mult / (n_variables * variance).
SEARCH_SPACES: dict[str, list[dict]] = {
    "knn": [{"pca": [20, 50, 100, None], "n_neighbors": [1, 3, 5, 9, 15, 25],
             "weights": ["uniform", "distance"], "metric": ["euclidean", "cosine"]}],
    "svm": [{"pca": [50, 100, 200, 400], "C": [1, 3, 10, 30, 100], "gamma_mult": [0.5, 1, 2]},
            {"pca": [None], "C": [10, 30], "gamma_mult": [1]}],          # sans PCA : 10x plus lent, pour vérifier
    "rf": [{"n_estimators": [300], "max_features": ["sqrt", "log2"], "min_samples_leaf": [1, 3],
            "max_depth": [None, 20], "class_weight": ["balanced", "balanced_subsample"]}],
}


def build_preprocessor(pca: int | None) -> list[tuple[str, object]]:
    """Étapes communes KNN/SVM : standardisation puis PCA éventuelle."""
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=int(pca), random_state=SEED)))
    return steps


def make_classifier(cfg: dict, probability: bool = False):
    """Classifieur seul. `probability=True` (SVM) active l'estimation des probabilités (modèle final, API)."""
    m = cfg["model"]
    if m == "knn":
        return KNeighborsClassifier(n_neighbors=int(cfg["n_neighbors"]), weights=cfg["weights"],
                                    metric=cfg["metric"], n_jobs=1)
    if m == "svm":
        # class_weight="balanced" : chaque classe pèse autant (tire flat 3,6 % vs scratch 41 %).
        kwargs = dict(kernel="rbf", C=float(cfg["C"]), gamma=float(cfg["gamma"]), class_weight="balanced",
                      random_state=SEED)
        if not probability:
            return SVC(**kwargs)
        # Probabilités : méthode de libsvm (Platt sur chacun des 15 duels entre classes + couplage par paires,
        # validation croisée interne à 5 plis sur les données d'entraînement).
        # Comparée sur la validation (notebook 05, section 3) à CalibratedClassifierCV (sigmoïde, température) : bien mieux
        # calibrée (log-loss 0,58 contre 0,71 ; ECE 0,03 contre 0,08). CalibratedClassifierCV calibre le score
        # « un contre tous » du SVC, qui n'est presque qu'un nombre de votes : toutes les confiances valent ≈ 0,7.
        # `probability` est déprécié depuis scikit-learn 1.9 (supprimé en 1.11) ; la version est figée à 1.9.1.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)
            return SVC(probability=True, **kwargs)
    if m == "rf":
        return RandomForestClassifier(n_estimators=int(cfg["n_estimators"]), max_features=cfg["max_features"],
                                      min_samples_leaf=int(cfg["min_samples_leaf"]),
                                      max_depth=None if cfg["max_depth"] is None else int(cfg["max_depth"]),
                                      class_weight=cfg["class_weight"], n_jobs=-1, random_state=SEED)
    raise ValueError(f"Modèle inconnu : {m}")


def build_pipeline(cfg: dict, probability: bool = False) -> Pipeline:
    """Pipeline complet (non entraîné) correspondant à une configuration de la recherche."""
    if cfg["model"] == "rf":
        return Pipeline([("clf", make_classifier(cfg))])
    return Pipeline(build_preprocessor(cfg.get("pca")) + [("clf", make_classifier(cfg, probability))])
