"""Încărcare CSV cu validare structură."""

import pandas as pd
import streamlit as st

from src.config import DATASET_PATH


@st.cache_data
def load_raw() -> pd.DataFrame:
    """Încarcă Hotel Booking Demand din Dataset/hotel_bookings.csv.

    Pattern adaptat din Seminar/S5/Regresie/housing_california.py (read_csv + shape check).
    Nu drop nimic — păstrăm coloanele leakage/PII vizibile pentru Tab 2 (calitatea datelor).
    Drop-ul se face explicit în cleaning.py.
    """
    df = pd.read_csv(DATASET_PATH)
    # Sanity check (PLAN sec. 2.2)
    assert df.shape == (119390, 32), (
        f"Shape neașteptat: {df.shape} (așteptat: (119390, 32))"
    )
    return df


def load_raw_uncached() -> pd.DataFrame:
    """Identic cu load_raw dar fără decorator — pentru scripting fără Streamlit."""
    df = pd.read_csv(DATASET_PATH)
    assert df.shape == (119390, 32), f"Shape neașteptat: {df.shape}"
    return df
