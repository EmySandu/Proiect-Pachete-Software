"""Scalare feature-uri numerice — StandardScaler / MinMaxScaler (PLAN Decizia 5).

Disciplina anti-leakage: fit pe TRAIN, transform pe TEST (un fitted_scaler reutilizabil).
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_features(
    X: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
    fitted_scaler=None,
):
    """Scalează coloanele specificate; returnează (X_scaled, scaler).

    Pattern adaptat din Seminar/S5/K-means/kmeans.py (StandardScaler.fit_transform).

    - method='standard' → media 0, std 1 (recomandat K-means + LogReg)
    - method='minmax' → [0, 1] (recomandat când distribuția nu e gaussian)
    - fitted_scaler=None → fit+transform pe X (folosit la train)
    - fitted_scaler=<scaler> → doar transform (folosit la test → anti-leakage)
    """
    X = X.copy()
    if fitted_scaler is None:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Metoda '{method}' nesuportată; folosește 'standard' sau 'minmax'."
            )
        X[columns] = scaler.fit_transform(X[columns])
        return X, scaler
    else:
        X[columns] = fitted_scaler.transform(X[columns])
        return X, fitted_scaler
