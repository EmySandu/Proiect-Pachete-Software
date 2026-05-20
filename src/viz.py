"""Helpere de vizualizare reutilizabile — toate întorc un matplotlib Figure
(consumat din app.py via st.pyplot(fig)).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_missing(df: pd.DataFrame) -> plt.Figure:
    """Bar chart doar pentru coloanele care AU NaN, cu adnotare count + procent."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    total_rows = len(df)

    if missing.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.text(
            0.5,
            0.5,
            "✅ Niciun NaN detectat în dataset",
            ha="center",
            va="center",
            fontsize=12,
            color="#2ca02c",
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    pct = (missing / total_rows * 100).round(2)
    height = max(1.8, 0.55 * len(missing) + 0.8)
    fig, ax = plt.subplots(figsize=(8, height))
    bars = ax.barh(missing.index, missing.values, color="#d62728", edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Număr valori lipsă (NaN)")
    ax.set_title(
        f"Coloane cu valori lipsă: {len(missing)} din {df.shape[1]} "
        f"(dataset: {total_rows:,} rânduri)".replace(",", "."),
        fontsize=11,
    )

    for bar, p in zip(bars, pct.values):
        x = bar.get_width()
        label = f"  {int(x):,} ({p:.2f}%)".replace(",", ".")
        ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)

    ax.margins(x=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_box(df: pd.DataFrame, col: str) -> plt.Figure:
    """Boxplot pentru o coloană numerică — folosit la identificare outliers."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    sns.boxplot(x=df[col], color="#1f77b4", ax=ax)
    ax.set_title(f"Boxplot — {col}")
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, labels: list[str] = None) -> plt.Figure:
    """Heatmap matrice de confuzie pentru LogReg.

    Pattern adaptat din Seminar/S5/K-means/Seminar Python4_*.docx (confusion_matrix display).
    """
    if labels is None:
        labels = ["Onorat", "Anulat"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Real")
    ax.set_title("Matrice de confuzie")
    fig.tight_layout()
    return fig


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    """Curba ROC + linia random baseline."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"LogReg (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curba ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(fitted: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    """Scatter reziduuri vs fitted — diagnostic heteroscedasticity OLS (Decizia 12)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(fitted, residuals, alpha=0.3, s=10, color="#2ca02c")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Reziduuri")
    ax.set_title("Reziduuri vs Fitted (homoscedasticity check)")
    fig.tight_layout()
    return fig


def plot_elbow_silhouette(
    k_range: list, inertias: list, silhouettes: list
) -> plt.Figure:
    """2 subplots: Elbow (inertia) + Silhouette — selecție K (PLAN Decizia 7).

    Pattern adaptat din Seminar/S5/K-means/kmeans.py (loop k + plot wcss/silhouette).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_range, inertias, marker="o", color="#d62728")
    axes[0].set_title("Elbow Method (Inertia / WCSS)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_range, silhouettes, marker="o", color="#2ca02c")
    axes[1].set_title("Silhouette Score per K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    x_label: str,
    y_label: str,
    centroids_2d: np.ndarray = None,
) -> plt.Figure:
    """Scatter 2D al clusterelor (când utilizatorul alege 2 axe din UI)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("tab10", n_colors=n_clusters)
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=8,
            alpha=0.5,
            color=palette[c],
            label=f"Cluster {c}",
        )
    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            s=200,
            marker="X",
            color="black",
            label="Centroizi",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Clustere K-means (proiecție 2D)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_coef_bar(coefs: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Top N coeficienți LogReg după valoarea absolută — interpretabilitate."""
    top = coefs.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top["coef"]]
    ax.barh(top["feature"], top["coef"], color=colors)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Coeficient standardizat")
    ax.set_title(f"Top {top_n} predictori (după |coef|)")
    fig.tight_layout()
    return fig
