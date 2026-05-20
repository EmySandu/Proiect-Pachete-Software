"""Modele ML: K-means (PLAN Tab 8), LogReg (Tab 9.A), OLS (Tab 9.B).

Fiecare run_* întoarce un DICT bogat (model + predicții + metrici + diagnostice)
ca să fie consumat din app.py fără re-calcule.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    KMEANS_K_DEFAULT,
    LOGREG_THRESHOLD_DEFAULT,
    RANDOM_STATE,
    TEST_SIZE,
)


# ====================================================================
#  K-MEANS — Segmentare comportamentală (Tab 8)
# ====================================================================


def kmeans_diagnostics(
    X_scaled: np.ndarray, k_range=range(2, 11), n_init: int = 3
) -> dict:
    """Calculează inerția (Elbow) + silhouette pentru o gamă de K.

    Pattern adaptat din Seminar/S5/K-means/kmeans.py (loop k cu wcss/silhouette).
    n_init redus la 3 default (pentru UI responsiv); pentru fitul final K-ales,
    run_kmeans folosește n_init=10 (mai stabil).
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=n_init, random_state=RANDOM_STATE
        )
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return {"k_range": list(k_range), "inertias": inertias, "silhouettes": silhouettes}


def run_kmeans(
    X_scaled: np.ndarray,
    feature_names: list[str],
    scaler: StandardScaler = None,
    k: int = KMEANS_K_DEFAULT,
    df_original: pd.DataFrame = None,
    n_init: int = 5,
) -> dict:
    """Fit K-means pe feature-uri scalate, returnează labels + profile cluster.

    Pattern adaptat din Seminar/S5/K-means/kmeans.py (KMeans + silhouette_score).

    - centroids_original: centroizii revertiți la scala originală (interpretabili business)
    - profiles: tabel cu medii feature + volum + cancel_rate per cluster
    - algorithm='elkan' = optimizare cu inegalitatea triunghiulară, mai rapid pentru low-dim
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        algorithm="elkan",
        random_state=RANDOM_STATE,
    )
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    # Centroizii pe scala originală (inverse transform) — pentru interpretare business
    if scaler is not None:
        centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    else:
        centroids_orig = km.cluster_centers_

    profiles = pd.DataFrame(centroids_orig, columns=feature_names)
    profiles["cluster"] = range(k)

    if df_original is not None:
        df_with_labels = df_original.copy()
        df_with_labels["cluster"] = labels
        profiles["volume"] = df_with_labels.groupby("cluster").size().values
        if "is_canceled" in df_with_labels.columns:
            profiles["cancel_rate"] = (
                df_with_labels.groupby("cluster")["is_canceled"].mean().values
            )

    return {
        "model": km,
        "labels": labels,
        "silhouette": sil,
        "inertia": km.inertia_,
        "centroids_scaled": km.cluster_centers_,
        "centroids_original": centroids_orig,
        "profiles": profiles,
        "k": k,
    }


# ====================================================================
#  LOGISTIC REGRESSION — Predicție anulare (Tab 9.A)
# ====================================================================


def run_logreg(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = LOGREG_THRESHOLD_DEFAULT,
    class_weight: str = "balanced",
) -> dict:
    """Antrenează LogReg pe X→y; întoarce model + predicții + metrici complete.

    Pattern adaptat din Seminar/S5/K-means/Seminar Python4_*.docx (LogReg + train_test_split).

    - stratify=y (PLAN sec. 9 — păstrează ratele clasei în train/test)
    - class_weight='balanced' (PLAN Decizia 9: cost asimetric anulare > fals-pozitiv)
    - StandardScaler fit pe train, transform pe test (anti-leakage)
    - threshold custom (PLAN Decizia 10: 0.4 vs default 0.5)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling fit pe train, transform pe test (anti-leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_train_s, y_train)

    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    coefs = pd.DataFrame({"feature": X.columns, "coef": model.coef_[0]}).sort_values(
        "coef", key=abs, ascending=False
    )

    return {
        "model": model,
        "scaler": scaler,
        "X_test_scaled": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "coefs": coefs,
        "threshold": threshold,
    }


# ====================================================================
#  OLS MULTIPLĂ — Predicție ADR (Tab 9.B)
# ====================================================================


def run_ols(formula: str, df: pd.DataFrame, test_frac: float = TEST_SIZE) -> dict:
    """OLS via statsmodels formula API + metrici pe test set.

    Pattern: smf.ols din Seminar/S5/K-means/Seminar Python4_*.docx (predat la seminar);
    metrici (MAE/RMSE/R²) adaptate din Seminar/S5/Regresie/housing_california.py.

    Folosim formula API pentru că dă acces la p-values, F-stat, intervale de încredere
    (necesare pentru Decizia 11 — selecție full vs redus).
    """
    df_train = df.sample(frac=1 - test_frac, random_state=RANDOM_STATE)
    df_test = df.drop(df_train.index)

    model = smf.ols(formula=formula, data=df_train).fit()

    # Predict pe test
    y_test_pred = model.predict(df_test)
    target_col = formula.split("~")[0].strip()
    y_test_true = df_test[target_col]

    # Drop NaN-uri create de predict (dacă test conține valori cat. nevăzute în train)
    valid = ~(y_test_pred.isna() | y_test_true.isna())
    y_test_pred = y_test_pred[valid]
    y_test_true = y_test_true[valid]

    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    r2_test = r2_score(y_test_true, y_test_pred)

    # Reziduuri pe TRAIN (pentru diagnostic heteroscedasticity — Decizia 12)
    residuals_train = model.resid
    fitted_train = model.fittedvalues

    return {
        "model": model,
        "summary_text": str(model.summary()),
        "r_squared": model.rsquared,
        "r_squared_adj": model.rsquared_adj,
        "f_pvalue": model.f_pvalue,
        "params": model.params,
        "pvalues": model.pvalues,
        "residuals_train": residuals_train,
        "fitted_train": fitted_train,
        "mae_test": mae,
        "rmse_test": rmse,
        "r2_test": r2_test,
        "n_train": len(df_train),
        "n_test_valid": int(valid.sum()),
    }
