"""Analiză geografică — agregare pe țară + choropleth (Tab 7).

Funcționalitate NEPREZENTĂ în seminar — folosim geopandas conform PLAN sec. 5.7.
"""

import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ASSETS_DIR, SHAPEFILE_PATH

# Natural Earth — CDN stabil (110m = low resolution, suficient pt choropleth)
NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


def download_shapefile(target_dir: Path = ASSETS_DIR) -> Path:
    """Descarcă Natural Earth 110m countries în assets/. Idempotent."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if SHAPEFILE_PATH.exists():
        return SHAPEFILE_PATH

    zip_path = target_dir / "ne_110m_admin_0_countries.zip"
    urllib.request.urlretrieve(NATURAL_EARTH_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    zip_path.unlink(missing_ok=True)
    assert SHAPEFILE_PATH.exists(), f"Shapefile lipsă post-extract: {SHAPEFILE_PATH}"
    return SHAPEFILE_PATH


def aggregate_country_stats(df: pd.DataFrame, min_volume: int = 0) -> pd.DataFrame:
    """Agregă statistici per țară de origine (volum, rata anulare, ADR mediu).

    PLAN Decizia 6: pentru hărți rate (anulare, ADR) → filtru min_volume = 100
    (stabilitate statistică); pentru harta volum → toate țările (min_volume = 0).
    """
    stats = (
        df.groupby("country")
        .agg(
            volume=("is_canceled", "count"),
            cancel_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean"),
        )
        .reset_index()
    )
    if min_volume > 0:
        stats = stats[stats["volume"] >= min_volume]
    return stats


def prepare_world_map(country_stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """Merge shapefile Natural Earth cu statisticile pe ISO_A3 (cod 3 litere)."""
    if not SHAPEFILE_PATH.exists():
        download_shapefile()
    world = gpd.read_file(SHAPEFILE_PATH)
    # Hotel Booking Demand folosește deja cod ISO-3 (PRT, FRA, GBR, ...)
    merged = world.merge(
        country_stats, left_on="ISO_A3", right_on="country", how="left"
    )
    return merged


def plot_choropleth(
    world_data: gpd.GeoDataFrame,
    metric: str = "volume",
    cmap: str = "OrRd",
    title: str = None,
) -> plt.Figure:
    """Choropleth pe metricul ales (volume | cancel_rate | avg_adr)."""
    if title is None:
        title = f"{metric} per țară de origine"
    fig, ax = plt.subplots(figsize=(13, 6.5))
    world_data.plot(
        column=metric,
        cmap=cmap,
        linewidth=0.4,
        edgecolor="gray",
        ax=ax,
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "Fără date"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig
