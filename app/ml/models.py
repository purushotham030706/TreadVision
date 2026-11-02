"""
TreadVision: Machine Learning Model Factory
Provides Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbors (KNN).
"""
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_model(model_name: str = "rf", **kwargs):
    """
    Instantiate specified classification model with calibrated hyper-parameters.
    """
    model_name = model_name.lower()

    if model_name in ("svm", "svc"):
        # Support Vector Machine with Radial Basis Function kernel
        c = kwargs.get("C", 10.0)
        gamma = kwargs.get("gamma", "scale")
        base_svc = SVC(C=c, gamma=gamma, kernel="rbf", random_state=42)
        try:
            return CalibratedClassifierCV(base_svc, ensemble=False)
        except TypeError:
            return SVC(C=c, gamma=gamma, kernel="rbf", probability=True, random_state=42)

    elif model_name in ("rf", "random_forest"):
        # Random Forest with ensemble of decision trees
        n_estimators = kwargs.get("n_estimators", 150)
        max_depth = kwargs.get("max_depth", 10)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            random_state=42
        )

    elif model_name in ("knn", "kneighbors"):
        # K-Nearest Neighbors with distance weighting
        n_neighbors = kwargs.get("n_neighbors", 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Use 'svm', 'rf', or 'knn'.")
