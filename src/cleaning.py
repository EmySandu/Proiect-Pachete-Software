"""Curățare dataset — imputare missing, outliers, filtrare rânduri fără sens business.

Toate funcțiile întorc un df NOU (.copy()) — NU modifică in-place.
"""

import pandas as pd

from src.config import (
    ADR_IQR_MULTIPLIER,
    LEAKAGE_COLUMNS,
    MONTH_MAP,
    PII_COLUMNS,
)


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop reservation_status + reservation_status_date.

    Decizia leakage (PLAN sec. 3.4): reservation_status ia valori
    {Check-Out, Canceled, No-Show} = targetul mascat → accuracy 100% (cheat).
    """
    return df.drop(columns=LEAKAGE_COLUMNS, errors="ignore").copy()


def drop_pii_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PII (name/email/phone/credit_card) dacă există în această versiune CSV."""
    return df.drop(columns=PII_COLUMNS, errors="ignore").copy()


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Tratare valori lipsă per coloană (PLAN Decizia 1).

    - children (4 NaN, 0.003%): DROP rânduri — impact neglijabil
    - country (488 NaN, 0.41%): IMPUTARE cu modă ("PRT") — păstrează realitate business
    - agent (16K NaN, 14%): FILL cu 0 — semantic ("no agent" = booking direct)
    - company (112K NaN, 94%): DROP coloană — prea sparse pentru a fi utilă

    Pattern adaptat din Seminar/S5/Regresie/housing_california.py (fillna per coloană).
    """
    df = df.copy()

    if "company" in df.columns:
        df = df.drop(columns="company")

    if "children" in df.columns:
        df = df.dropna(subset=["children"])

    if "country" in df.columns:
        country_mode = df["country"].mode().iloc[0]
        df["country"] = df["country"].fillna(country_mode)

    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna(0)

    return df


def cap_outliers_adr(
    df: pd.DataFrame,
    method: str = "iqr",
    multiplier: float = ADR_IQR_MULTIPLIER,
    domain_threshold: float = 1000.0,
) -> pd.DataFrame:
    """Filtrare outliers ADR — 3 strategii (PLAN Decizia 2).

    - method='iqr' + multiplier=1.5 → Tukey standard (default, ~94% date păstrate)
    - method='iqr' + multiplier=3.0 → conservator (~98% date păstrate)
    - method='domain' + domain_threshold=1000 → ADR < 1000€ (~99.5% date păstrate)
    - method='none' → nu elimină nimic (pentru comparație)

    ADR = 0 păstrat în toate cazurile (complimentary stay, nu outlier numeric).
    """
    df = df.copy()
    if method == "none":
        return df
    if method == "domain":
        return df[df["adr"] < domain_threshold].copy()
    if method == "iqr":
        q1, q3 = df["adr"].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + multiplier * iqr
        return df[df["adr"] <= upper].copy()
    raise ValueError(f"method '{method}' nesuportat; folosește 'iqr' | 'domain' | 'none'")


def filter_zero_business_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rânduri fără sens business (PLAN Decizia 3).

    - adults+children+babies == 0 → booking fără oaspeți
    - stays_in_weekend_nights + stays_in_week_nights == 0 → booking fără cazare
    """
    df = df.copy()
    has_guests = (df["adults"] + df["children"] + df["babies"]) > 0
    has_stay = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) > 0
    return df[has_guests & has_stay].copy()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă feature-uri derivate folosite downstream.

    - total_nights: sumă weekend + week (folosit în K-means + LogReg)
    - total_guests: sumă adults + children + babies (folosit în K-means + LogReg)
    - month_num: mapare ianuarie→1, ..., decembrie→12 (folosit pt cyclic encoding)
    """
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["month_num"] = df["arrival_date_month"].map(MONTH_MAP).astype(int)
    return df


def clean_pipeline(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    outlier_multiplier: float = ADR_IQR_MULTIPLIER,
    outlier_domain_threshold: float = 1000.0,
) -> pd.DataFrame:
    """Pipeline complet de curățare — apel unic din app.py.

    Ordine importantă: leakage/PII drop ÎNAINTE de imputare (evită contaminare),
    outliers DUPĂ imputare (NaN-ul ar fi exclus oricum din quantile), zero-rows
    DUPĂ outliers (rândurile filtrate nu mai contează pentru IQR).
    """
    df = drop_leakage_columns(df)
    df = drop_pii_if_present(df)
    df = handle_missing(df)
    df = cap_outliers_adr(
        df,
        method=outlier_method,
        multiplier=outlier_multiplier,
        domain_threshold=outlier_domain_threshold,
    )
    df = filter_zero_business_rows(df)
    df = add_derived_features(df)
    return df
