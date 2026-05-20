"""Codificare variabile categorice — 5 strategii per cardinalitate (PLAN Decizia 4).

Fiecare funcție întoarce un df NOU (.copy()) cu coloanele encoded ADĂUGATE și
coloanele originale DROPATE (excepție: country_freq păstrează 'country' până la
final pentru analiză geografică).
"""

import numpy as np
import pandas as pd

from src.config import DEPOSIT_ORDER


# Coloane nominale fără ordine → One-Hot (PLAN Decizia 4)
ONEHOT_COLUMNS = [
    "meal",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "reserved_room_type",
    "assigned_room_type",
]


def encode_binary_hotel(df: pd.DataFrame) -> pd.DataFrame:
    """Binary manual: hotel = Resort/City → is_resort ∈ {0,1}.

    Pattern adaptat din Seminar/S5/K-means/kmeans.py (Genre.map({'Male':0,'Female':1})).
    Cardinalitate 2 → binary e cel mai eficient (1 coloană vs 2 one-hot).
    """
    df = df.copy()
    df["is_resort"] = (df["hotel"] == "Resort Hotel").astype(int)
    return df.drop(columns=["hotel"])


def encode_onehot(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """One-Hot pentru variabile nominale fără ordine.

    Pattern adaptat din Seminar/S5/Regresie/housing_california.py (pd.get_dummies).
    drop_first=True elimină coloana de referință (evită multicoliniaritate perfectă).
    """
    if columns is None:
        columns = ONEHOT_COLUMNS
    columns = [c for c in columns if c in df.columns]
    return pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)


def encode_ordinal_deposit(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal pentru deposit_type — ordine business: No Deposit < Refundable < Non Refund.

    Spre deosebire de meal/market_segment, deposit_type ARE ordine (commitment
    financiar crescător), deci ordinal e mai informativ decât one-hot.
    """
    df = df.copy()
    df["deposit_type_ord"] = df["deposit_type"].map(DEPOSIT_ORDER)
    return df.drop(columns=["deposit_type"])


def encode_frequency_country(
    df: pd.DataFrame, keep_original: bool = True
) -> pd.DataFrame:
    """Frequency encoding pentru country (~178 valori unice).

    One-Hot ar genera 178 coloane sparse → curse of dimensionality pentru LR/OLS.
    Frequency păstrează informația de prevalență într-o singură coloană numerică.
    keep_original=True: păstrăm 'country' string pentru analiza geografică (Tab 7).
    """
    df = df.copy()
    freq_map = df["country"].value_counts(normalize=True).to_dict()
    df["country_freq"] = df["country"].map(freq_map)
    if not keep_original:
        df = df.drop(columns=["country"])
    return df


def encode_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encoding (sin/cos) pentru lună — păstrează ciclicitatea calendaristică.

    Decembrie (12) e adiacent cu Ianuarie (1) în realitate; encoding numeric naiv
    ar pune distanță 11 între ele. sin/cos pe 2π/12 face decembrie-ianuarie ≈ vecini.
    Necesită coloana 'month_num' (creată de cleaning.add_derived_features).
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    # 'arrival_date_month' (string) nu mai e necesar după cyclic
    return df.drop(columns=["arrival_date_month"])


def encode_pipeline(df: pd.DataFrame, keep_country_string: bool = True) -> pd.DataFrame:
    """Pipeline complet de encoding — apel unic din app.py.

    Ordine: binary → onehot → ordinal → frequency → cyclic.
    """
    df = encode_binary_hotel(df)
    df = encode_onehot(df)
    df = encode_ordinal_deposit(df)
    df = encode_frequency_country(df, keep_original=keep_country_string)
    df = encode_cyclic_month(df)
    return df
