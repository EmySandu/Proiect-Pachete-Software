"""Constante globale ale proiectului — un singur loc pentru reproducibilitate."""

from pathlib import Path

# Reproducibilitate (folosit în toate locurile cu randomness)
RANDOM_STATE = 42

# Paths (rezolvate față de rădăcina proiectului)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "Dataset" / "hotel_bookings.csv"
PROCESSED_DIR = ROOT_DIR / "Dataset" / "processed"
ASSETS_DIR = ROOT_DIR / "assets"
SHAPEFILE_PATH = ASSETS_DIR / "ne_110m_admin_0_countries.shp"

# Split train/test
TEST_SIZE = 0.2

# Decizii pre-luate (PLAN sec. 7) — schimbabile via UI, dar default-uri aici
LOGREG_THRESHOLD_DEFAULT = 0.4  # Decizia 10
KMEANS_K_DEFAULT = 4  # Decizia 7
KMEANS_K_RANGE = range(2, 11)
ADR_IQR_MULTIPLIER = 1.5  # Decizia 2 (Tukey standard)

# Coloane periculoase
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]
PII_COLUMNS = ["name", "email", "phone-number", "credit_card"]

# Feature set pentru K-means (PLAN Decizia 8: behavior-focused)
NUMERIC_FEATURES_KMEANS = ["lead_time", "adr", "total_nights", "total_guests"]

# Mapping luni (Ianuarie → Decembrie ca string în CSV)
MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

# Ordinal deposit (PLAN Decizia 4: commitment financiar crescător)
DEPOSIT_ORDER = {"No Deposit": 0, "Refundable": 1, "Non Refund": 2}
