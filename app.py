"""Hotel Booking Analytics — aplicația Streamlit principală.

Acoperă cele 11 facilități cerute de proiect. Logica grea este delegată
modulelor din src/; aici rămâne doar orchestrarea UI.

Pattern navigare adaptat din Seminar/Seminar1/s1.py (st.sidebar.radio pentru
navigare verticală + filtre globale dedesubt).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.cleaning import clean_pipeline
from src.config import (
    ADR_IQR_MULTIPLIER,
    KMEANS_K_DEFAULT,
    LOGREG_THRESHOLD_DEFAULT,
    NUMERIC_FEATURES_KMEANS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_raw
from src.encoding import encode_pipeline
from src.modeling import (
    kmeans_diagnostics,
    run_logreg,
    run_ols,
)
from src.scaling import scale_features

st.set_page_config(
    page_title="Analiza pieței hoteliere",
    layout="wide",
    initial_sidebar_state="expanded",
)



FRIENDLY_NAMES = {
    # Variabile ML (folosite în S4-S10)
    "lead_time": "Zile între rezervare și sosire",
    "adr": "Preț mediu pe noapte (€)",
    "is_canceled": "Anulare",
    "total_nights": "Nopți totale",
    "total_guests": "Total oaspeți",
    "is_resort": "Hotel Resort (1=da)",
    "deposit_type_ord": "Tip depozit (ordinal)",
    "previous_cancellations": "Anulări anterioare",
    "previous_bookings_not_canceled": "Rezervări anterioare onorate",
    "booking_changes": "Modificări la rezervare",
    "required_car_parking_spaces": "Locuri parcare cerute",
    "total_of_special_requests": "Cereri speciale",
    "country_freq": "Frecvență țară origine",
    "month_sin": "Sezonalitate (sin)",
    "month_cos": "Sezonalitate (cos)",
    "is_repeated_guest": "Client fidel",
    # Variabile descriptive originale (folosite în S1, S6, S7)
    "hotel": "Hotel",
    "country": "Țara (ISO-3)",
    "arrival_date_month": "Luna sosirii",
    "arrival_date_year": "Anul sosirii",
    "arrival_date_week_number": "Săptămâna sosirii",
    "arrival_date_day_of_month": "Ziua sosirii",
    "market_segment": "Segment de piață",
    "deposit_type": "Tip depozit",
    "distribution_channel": "Canal de distribuție",
    "customer_type": "Tip client",
    "reserved_room_type": "Tip cameră rezervată",
    "assigned_room_type": "Tip cameră alocată",
    "meal": "Masă inclusă",
    "agent": "Cod agent",
    "company": "Cod companie",
    "adults": "Adulți",
    "children": "Copii",
    "babies": "Bebeluși",
    "stays_in_weekend_nights": "Nopți weekend",
    "stays_in_week_nights": "Nopți săptămână",
    "days_in_waiting_list": "Zile pe lista de așteptare",
    "month_num": "Luna (1-12)",
    "volum": "Volum rezervări",
    "rezervari": "Număr rezervări",
    "rata_anulare": "Rata anulărilor",
    "rata_anulare_pct": "Rata anulărilor (%)",
    "cancel_rate": "Rata anulărilor (%)",
    "pret_mediu": "Preț mediu (€)",
    "pret_mediu_noapte": "Preț mediu / noapte (€)",
    "anulate": "Anulate",
    "onorate": "Onorate",
    "Stare": "Stare rezervare",
}


def pretty(name: str) -> str:
    """Returnează denumirea prietenoasă pentru UI; fallback la numele original."""
    return FRIENDLY_NAMES.get(name, name)



def px_box_horizontal(values, title: str, color: str = "#1f77b4") -> go.Figure:
    """Boxplot orizontal interactiv (inlocuieste plot_box matplotlib)."""
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=values,
            name="",
            marker_color=color,
            boxpoints="outliers",
            jitter=0.3,
            pointpos=0,
            line=dict(width=1.5),
        )
    )
    fig.update_layout(
        title=title,
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        yaxis=dict(showticklabels=False),
    )
    return fig


def px_confusion(cm: np.ndarray, labels: list[str] = None) -> go.Figure:
    """Heatmap interactiv pentru matrice de confuzie."""
    if labels is None:
        labels = ["Onorat", "Anulat"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predictie", y="Real", color="Numar cazuri"),
        title="Matrice de confuzie",
    )
    fig.update_layout(height=380, margin=dict(t=50, l=40, r=20, b=40))
    return fig


def px_residuals(fitted: np.ndarray, residuals: np.ndarray, title: str = "Reziduuri vs valori prezise") -> go.Figure:
    """Scatter interactiv reziduuri vs fitted, cu linia 0."""
    n = len(fitted)
    if n > 5000:
        idx = np.random.RandomState(RANDOM_STATE).choice(n, 5000, replace=False)
        fitted = np.asarray(fitted)[idx]
        residuals = np.asarray(residuals)[idx]
    fig = px.scatter(
        x=fitted,
        y=residuals,
        opacity=0.45,
        labels={"x": "Valori prezise (fitted)", "y": "Reziduuri"},
        title=title,
        height=380,
        color_discrete_sequence=["#2ca02c"],
    )
    fig.update_traces(marker=dict(size=5))
    fig.add_hline(y=0, line_color="red", line_dash="dash", line_width=1.5)
    fig.update_layout(margin=dict(t=50, l=40, r=20, b=40))
    return fig


def px_coef_bar(coefs: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bar chart orizontal interactiv pentru top coeficienti."""
    top = coefs.head(top_n).iloc[::-1].copy()
    top["feature_display"] = top["feature"].map(lambda f: FRIENDLY_NAMES.get(f, f))
    top["effect"] = top["coef"].apply(
        lambda v: "Crește anularea" if v > 0 else "Scade anularea"
    )
    fig = px.bar(
        top,
        x="coef",
        y="feature_display",
        orientation="h",
        color="effect",
        color_discrete_map={
            "Crește anularea": "#d62728",
            "Scade anularea": "#1f77b4",
        },
        labels={"coef": "Coeficient standardizat", "feature_display": "Variabilă"},
        title=f"Top {top_n} predictori după magnitudine",
        height=420,
    )
    fig.add_vline(x=0, line_color="gray", line_width=0.5)
    fig.update_layout(legend_title_text="Efect", margin=dict(t=50, l=40, r=20, b=40))
    return fig


def px_missing(df: pd.DataFrame) -> go.Figure:
    """Bar chart Plotly cu coloanele care AU NaN, cu adnotare count + procent."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    total_rows = len(df)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Niciun NaN detectat in dataset",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#2ca02c"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10))
        return fig
    pct = (missing / total_rows * 100).round(2)
    text_labels = [
        f"{int(v):,} ({p:.2f}%)".replace(",", ".")
        for v, p in zip(missing.values, pct.values)
    ]
    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        text=text_labels,
        labels={"x": "Numar valori lipsa (NaN)", "y": ""},
        title=(
            f"Coloane cu valori lipsa: {len(missing)} din {df.shape[1]} "
            f"(dataset: {total_rows:,} randuri)".replace(",", ".")
        ),
        color=missing.values,
        color_continuous_scale="Reds",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(220, 50 * len(missing) + 80),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=50, l=10, r=80, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig



@st.cache_data
def get_raw() -> pd.DataFrame:
    return load_raw()


@st.cache_data
def get_clean(
    outlier_method: str = "iqr",
    outlier_multiplier: float = ADR_IQR_MULTIPLIER,
    outlier_domain_threshold: float = 1000.0,
) -> pd.DataFrame:
    df = get_raw()
    return clean_pipeline(
        df,
        outlier_method=outlier_method,
        outlier_multiplier=outlier_multiplier,
        outlier_domain_threshold=outlier_domain_threshold,
    )


@st.cache_data
def get_encoded(
    outlier_method: str = "iqr",
    outlier_multiplier: float = ADR_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Cleaning + encoding complet (păstrăm `country` ca string pentru geo)."""
    df = get_clean(outlier_method=outlier_method, outlier_multiplier=outlier_multiplier)
    return encode_pipeline(df, keep_country_string=True)


@st.cache_data
def get_features_for_modeling(
    outlier_method: str = "iqr", outlier_multiplier: float = ADR_IQR_MULTIPLIER
) -> pd.DataFrame:
    """Versiunea fără coloana `country` text — pentru fit ML (toate numerice)."""
    df = get_encoded(outlier_method, outlier_multiplier)
    return df.drop(columns=["country"], errors="ignore")


@st.cache_data
def get_world_geojson() -> tuple[dict, list]:
    """Încarcă shapefile-ul Natural Earth via geopandas și-l convertește în GeoJSON.

    Astfel geopandas e folosit efectiv în pipeline (cerință proiect, facilitate #2):
    citire shapefile + manipulare GeoDataFrame + export. Plotly preia GeoJSON-ul.
    Returnează (geojson, lista_ISO_A3_disponibile).
    """
    import json

    import geopandas as gpd

    from src.geo import SHAPEFILE_PATH, download_shapefile

    if not SHAPEFILE_PATH.exists():
        download_shapefile()
    gdf = gpd.read_file(SHAPEFILE_PATH)[["ISO_A3", "NAME", "geometry"]]
    geojson = json.loads(gdf.to_json())
    iso_codes = gdf["ISO_A3"].tolist()
    return geojson, iso_codes


@st.cache_data
def get_kmeans_diag(outlier_method: str = "iqr", sample_size: int = 20000) -> dict:
    """K-means diagnostics pe subsample (UI rapid)."""
    df = get_features_for_modeling(outlier_method)
    X = df[NUMERIC_FEATURES_KMEANS]
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=RANDOM_STATE)
    X_scaled, _ = scale_features(X, NUMERIC_FEATURES_KMEANS, method="standard")
    return kmeans_diagnostics(X_scaled.values, k_range=range(2, 11), n_init=3)


@st.cache_data
def get_kmeans_fit(k: int, outlier_method: str = "iqr", sample_size: int = 30000) -> dict:
    """Fit K-means pe subesantion 30k randuri pentru UI rapid; predict pe tot setul.

    Motivatie performance: K-means pe 114k randuri × n_init × O(k·iter·d) e lent la
    fiecare schimbare de K din slider. Fitul pe 30k pastreaza calitatea (centroizii
    sunt aproape identici cu fitul pe full dataset) iar predictiile finale acopera
    intregul set prin .predict() (foarte rapid, O(n·k·d)).
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.metrics import silhouette_samples as _silh_samples
    from sklearn.metrics import silhouette_score as _silh

    df = get_features_for_modeling(outlier_method)
    X = df[NUMERIC_FEATURES_KMEANS]
    X_scaled, scaler = scale_features(X, NUMERIC_FEATURES_KMEANS, method="standard")

    train_idx = X_scaled.sample(
        min(sample_size, len(X_scaled)), random_state=RANDOM_STATE
    ).index
    X_train = X_scaled.loc[train_idx].values

    km = _KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        algorithm="elkan",
        random_state=RANDOM_STATE,
    )
    km.fit(X_train)

    labels_all = km.predict(X_scaled.values)
    train_labels = km.predict(X_train)
    sil_global = _silh(X_train, train_labels, sample_size=min(10000, len(X_train)))

    sil_sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), min(10000, len(X_train)), replace=False
    )
    X_silh = X_train[sil_sample_idx]
    labels_silh = train_labels[sil_sample_idx]
    silh_per_sample = _silh_samples(X_silh, labels_silh)
    sil_per_cluster = {
        c: float(silh_per_sample[labels_silh == c].mean()) for c in range(k)
    }

    centroids_orig = scaler.inverse_transform(km.cluster_centers_)

    profiles = pd.DataFrame(centroids_orig, columns=NUMERIC_FEATURES_KMEANS)
    profiles["cluster"] = range(k)
    df_lbl = df.reset_index(drop=True).copy()
    df_lbl["cluster"] = labels_all
    profiles["volume"] = (
        df_lbl.groupby("cluster").size().reindex(range(k), fill_value=0).values
    )
    if "is_canceled" in df_lbl.columns:
        profiles["cancel_rate"] = (
            df_lbl.groupby("cluster")["is_canceled"]
            .mean()
            .reindex(range(k), fill_value=0)
            .values
        )
    profiles["silhouette"] = [round(sil_per_cluster[c], 3) for c in range(k)]

    return {
        "model": km,
        "labels": labels_all,
        "silhouette": sil_global,
        "silhouette_per_cluster": sil_per_cluster,
        "inertia": km.inertia_,
        "centroids_scaled": km.cluster_centers_,
        "centroids_original": centroids_orig,
        "profiles": profiles,
        "k": k,
        "train_size": len(X_train),
    }


LOGREG_FEATURES = [
    "lead_time",
    "adr",
    "total_nights",
    "total_guests",
    "is_resort",
    "deposit_type_ord",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "country_freq",
    "month_sin",
    "month_cos",
    "is_repeated_guest",
]


@st.cache_data
def get_logreg_fit(threshold: float, outlier_method: str = "iqr") -> dict:
    df = get_features_for_modeling(outlier_method)
    feat = [c for c in LOGREG_FEATURES if c in df.columns]
    return run_logreg(df[feat], df["is_canceled"], threshold=threshold)


OLS_FORMULA_BASE = (
    "{target} ~ lead_time + total_nights + total_guests + is_resort + "
    "deposit_type_ord + required_car_parking_spaces + total_of_special_requests + "
    "is_repeated_guest + booking_changes + month_sin + month_cos + country_freq"
)


@st.cache_data
def get_ols_fit(log_target: bool = False, outlier_method: str = "iqr") -> dict:
    df = get_features_for_modeling(outlier_method).copy()
    if log_target:
        df["adr_log"] = np.log(df["adr"] + 1)
        formula = OLS_FORMULA_BASE.format(target="adr_log")
    else:
        formula = OLS_FORMULA_BASE.format(target="adr")
    result = run_ols(formula, df)

    if log_target:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        df_train = df.sample(frac=1 - TEST_SIZE, random_state=RANDOM_STATE)
        df_test = df.drop(df_train.index)
        log_predictions = result["model"].predict(df_test)
        eur_predictions = np.exp(log_predictions) - 1
        eur_true = df_test["adr"]
        valid = ~(eur_predictions.isna() | eur_true.isna())
        result["mae_test_eur"] = float(
            mean_absolute_error(eur_true[valid], eur_predictions[valid])
        )
        result["rmse_test_eur"] = float(
            np.sqrt(mean_squared_error(eur_true[valid], eur_predictions[valid]))
        )
    else:
        result["mae_test_eur"] = result["mae_test"]
        result["rmse_test_eur"] = result["rmse_test"]
    return result





def apply_global_filters(
    df: pd.DataFrame,
    hotel: str,
    seasons: list[str],
    lead_range: tuple[int, int],
) -> pd.DataFrame:
    df = df.copy()
    if hotel != "Toate" and "hotel" in df.columns:
        df = df[df["hotel"] == hotel]
    if seasons:
        season_map = {
            "Iarnă": ["December", "January", "February"],
            "Primăvară": ["March", "April", "May"],
            "Vară": ["June", "July", "August"],
            "Toamnă": ["September", "October", "November"],
        }
        months = [m for s in seasons for m in season_map[s]]
        if "arrival_date_month" in df.columns:
            df = df[df["arrival_date_month"].isin(months)]
    if "lead_time" in df.columns:
        df = df[(df["lead_time"] >= lead_range[0]) & (df["lead_time"] <= lead_range[1])]
    return df


st.sidebar.title("Navigare proiect")
SECTIUNI = [
    "1. Prezentare",
    "2. Calitatea datelor",
    "3. Curățare",
    "4. Codificare",
    "5. Scalare",
    "6. Statistici & Group-by",
    "7. Analiză geografică",
    "8. Segmentare K-means",
    "9. Predicții (LogReg + OLS)",
]
sectiune = st.sidebar.radio("Alege secțiunea:", SECTIUNI, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.title("Filtre globale")
st.sidebar.caption("Aplicate doar în secțiunile 1, 6, 7 (descriptive/vizualizare).")

filter_hotel = st.sidebar.selectbox("Hotel", ["Toate", "Resort Hotel", "City Hotel"])
filter_seasons = st.sidebar.multiselect(
    "Anotimp",
    ["Iarnă", "Primăvară", "Vară", "Toamnă"],
    default=[],
    help="Empty = toate anotimpurile",
)
filter_lead = st.sidebar.slider(
    "Zile între rezervare și sosire",
    0,
    737,
    (0, 737),
    step=10,
    help="Engleză: 'lead time'. Cu cât valoarea e mai mare, cu atât clientul a rezervat mai în avans.",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Curățare** configurabilă în secțiunea 3. **Modelele** (secț. 8, 9) rulează pe "
    "dataset complet curățat, NU filtrat — pentru reproducibilitate."
)


st.markdown(
    """
    <h1 style='color:#3498DB; margin-bottom:1rem;'>
      Analiza pieței hoteliere
    </h1>
    """,
    unsafe_allow_html=True,
)

df_raw = get_raw()
outlier_method = st.session_state.get("outlier_method", "iqr")
outlier_multiplier = st.session_state.get("outlier_multiplier", ADR_IQR_MULTIPLIER)
df_clean = get_clean(
    outlier_method=outlier_method,
    outlier_multiplier=outlier_multiplier,
)



if sectiune == SECTIUNI[0]:
    df_view = apply_global_filters(df_raw, filter_hotel, filter_seasons, filter_lead)

    st.header("Prezentare proiect")
    st.markdown(
        """
        **Obiectiv:** fundamentarea cantitativă a deciziilor strategice ale unui
        investitor din industria hotelieră privind intrarea pe noi piețe europene.

        **Patru obiective analitice** structurează aplicația:

        1. **Modelarea probabilității de anulare** a unei rezervări — clasificare binară pe `is_canceled` (sec. 9, regresie logistică).
        2. **Determinanții prețului mediu pe noapte** (`adr`) — regresie liniară multiplă (sec. 9, OLS).
        3. **Segmentarea comportamentală a clientelei** după profilul rezervării — clustering nesupervizat (sec. 8, K-means).
        4. **Distribuția spațială a cererii** și identificarea piețelor-sursă cu potențial de extindere — analiză geografică (sec. 7, geopandas).
        """
    )

    st.subheader("Indicatori cheie (după aplicarea filtrelor globale)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total rezervări",
        f"{len(df_view):,}".replace(",", "."),
        help="Număr total de rezervări din dataset (după aplicarea filtrelor din sidebar).",
    )
    c2.metric(
        "Rata anulărilor",
        f"{df_view['is_canceled'].mean() * 100:.2f}%",
        help="Procentul rezervărilor anulate de client înainte de check-in.",
    )
    c3.metric(
        "Preț mediu pe noapte",
        f"{df_view['adr'].mean():.2f} €",
        help="ADR (Average Daily Rate) — venitul mediu generat de o rezervare per noapte ocupată.",
    )
    c4.metric(
        "Țări de origine",
        f"{df_view['country'].nunique()}",
        help="Număr de țări distincte din care provin clienții (cod ISO din 3 litere).",
    )

    if len(df_view) < len(df_raw):
        st.info(
            f"Filtrele globale au redus de la {len(df_raw):,} la {len(df_view):,} rânduri "
            f"({len(df_view) / len(df_raw) * 100:.1f}% din total).".replace(",", ".")
        )

    st.subheader("Distribuția rezervărilor pe ani și hoteluri")
    yearly = (
        df_view.groupby(["arrival_date_year", "hotel"])
        .size()
        .reset_index(name="rezervari")
    )
    fig_year = px.bar(
        yearly,
        x="arrival_date_year",
        y="rezervari",
        color="hotel",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={
            "arrival_date_year": "Anul sosirii",
            "rezervari": "Număr rezervări",
            "hotel": "Hotel",
        },
        title="Volum rezervări pe an, separat pe tipul de hotel",
        height=380,
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("Sezonalitatea rezervărilor — onorate vs anulate pe lună")
    MONTH_ORDER_S1 = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    monthly = (
        df_view.groupby("arrival_date_month")["is_canceled"]
        .agg(total="count", anulate="sum")
        .reset_index()
    )
    monthly["onorate"] = monthly["total"] - monthly["anulate"]
    monthly["arrival_date_month"] = pd.Categorical(
        monthly["arrival_date_month"], categories=MONTH_ORDER_S1, ordered=True
    )
    monthly = monthly.sort_values("arrival_date_month")
    fig_season = go.Figure()
    fig_season.add_trace(
        go.Scatter(
            x=monthly["arrival_date_month"],
            y=monthly["onorate"],
            mode="lines+markers",
            name="Onorate",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.15)",
        )
    )
    fig_season.add_trace(
        go.Scatter(
            x=monthly["arrival_date_month"],
            y=monthly["anulate"],
            mode="lines+markers",
            name="Anulate",
            line=dict(color="#d62728", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.15)",
        )
    )
    fig_season.update_layout(
        height=400,
        title="Volum lunar (cumulat pe toți anii) — onorate vs anulate",
        xaxis_title="Luna sosirii",
        yaxis_title="Număr rezervări",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=70),
    )
    st.plotly_chart(fig_season, use_container_width=True)

    st.subheader("Preview date (primele 20 rânduri, după filtre)")
    st.dataframe(df_view.head(20), use_container_width=True, height=350)

    with st.expander("Schema completă (tip date per coloană)"):
        schema = pd.DataFrame({
            "Coloană": df_raw.columns,
            "Tip": [str(df_raw[c].dtype) for c in df_raw.columns],
            "Non-null": [df_raw[c].notna().sum() for c in df_raw.columns],
            "Unice": [df_raw[c].nunique() for c in df_raw.columns],
        })
        st.dataframe(schema, use_container_width=True, height=500)



elif sectiune == SECTIUNI[1]:
    st.header("Calitatea datelor — Audit înainte de tratare")
    st.markdown(
        "Această secțiune **identifică** problemele de calitate: valori lipsă, valori "
        "extreme, scurgeri de informație (leakage). Tratarea efectivă se face în secțiunea 3."
    )

    st.subheader("1. Valori lipsă pe coloană")
    missing_total = int(df_raw.isna().sum().sum())
    missing_cols = int((df_raw.isna().sum() > 0).sum())
    st.caption(
        f"Total: **{missing_total:,} valori lipsă** distribuite în **{missing_cols} coloane** "
        f"(din {df_raw.shape[1]} coloane). Numerele și procentele sunt adnotate pe grafic.".replace(",", ".")
    )
    st.plotly_chart(px_missing(df_raw), use_container_width=True)

    st.subheader("2. Distribuția variabilelor categorice")
    pie_col1, pie_col2, pie_col3 = st.columns(3)
    with pie_col1:
        hotel_counts = df_raw["hotel"].value_counts()
        fig_h = px.pie(
            values=hotel_counts.values,
            names=hotel_counts.index,
            title="Tip de hotel",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_h.update_traces(textinfo="percent+label", textposition="inside")
        fig_h.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)
    with pie_col2:
        cancel_counts = df_raw["is_canceled"].value_counts().rename({0: "Onorate", 1: "Anulate"})
        fig_c = px.pie(
            values=cancel_counts.values,
            names=cancel_counts.index,
            title="Rezervări onorate vs anulate",
            color_discrete_sequence=["#2ca02c", "#d62728"],
            hole=0.4,
        )
        fig_c.update_traces(textinfo="percent+label", textposition="inside")
        fig_c.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)
    with pie_col3:
        ms_counts = df_raw["market_segment"].value_counts().sort_values(ascending=True)
        pct_ms = (ms_counts / ms_counts.sum() * 100).round(1)
        fig_m = px.bar(
            x=ms_counts.values,
            y=ms_counts.index,
            orientation="h",
            text=[f"{p}%" for p in pct_ms.values],
            color=ms_counts.values,
            color_continuous_scale="Tealgrn",
            labels={"x": "Număr rezervări", "y": "Segment de piață"},
            title="Segment de piață (ordonat după volum)",
        )
        fig_m.update_traces(textposition="outside")
        fig_m.update_layout(
            height=320, margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("3. Distribuții ale variabilelor numerice cheie")
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    with hist_col1:
        df_adr_plot = df_raw[df_raw["adr"] < df_raw["adr"].quantile(0.99)]
        fig_a = px.histogram(
            df_adr_plot, x="adr", nbins=50,
            title="Preț pe noapte ADR (€) — fără percentila 99",
            color_discrete_sequence=["#1f77b4"],
            labels={"adr": "ADR (€)"},
        )
        fig_a.update_layout(height=320, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_a, use_container_width=True)
    with hist_col2:
        fig_l = px.histogram(
            df_raw, x="lead_time", nbins=50,
            title="Zile între rezervare și sosire",
            color_discrete_sequence=["#ff7f0e"],
            labels={"lead_time": "Zile"},
        )
        fig_l.update_layout(height=320, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_l, use_container_width=True)
    with hist_col3:
        df_tmp = df_raw.copy()
        df_tmp["total_guests"] = (
            df_tmp["adults"] + df_tmp["children"].fillna(0) + df_tmp["babies"]
        )
        fig_g = px.histogram(
            df_tmp[df_tmp["total_guests"] <= 6], x="total_guests", nbins=7,
            title="Număr total de oaspeți",
            color_discrete_sequence=["#2ca02c"],
            labels={"total_guests": "Oaspeți"},
        )
        fig_g.update_layout(height=320, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("4. Distribuție bivariată — `lead_time` × `adr` colorat după anulare")
    bivar_sample = df_raw.sample(min(8000, len(df_raw)), random_state=RANDOM_STATE).copy()
    bivar_sample = bivar_sample[bivar_sample["adr"] < bivar_sample["adr"].quantile(0.99)]
    bivar_sample["Stare"] = bivar_sample["is_canceled"].map({0: "Onorat", 1: "Anulat"})
    fig_biv = px.scatter(
        bivar_sample,
        x="lead_time",
        y="adr",
        color="Stare",
        color_discrete_map={"Onorat": "#2ca02c", "Anulat": "#d62728"},
        opacity=0.4,
        title="Scatter lead_time × adr (eșantion 8.000 rezervări, fără top 1% preț)",
        labels={"lead_time": "Zile între rezervare și sosire", "adr": "Preț pe noapte (€)"},
        height=460,
    )
    fig_biv.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_biv, use_container_width=True)
    st.caption(
        "Anulările (roșu) se concentrează la `lead_time` > 50 zile; "
        "lipsa unor clustere distincte pe planul 2D motivează folosirea celor 4 dimensiuni în S8."
    )

    st.subheader("5. Coloane care divulgă răspunsul (de eliminat înainte de modelare)")
    leak_cols = ["reservation_status", "reservation_status_date"]
    st.warning(
        "**Data leakage:** `reservation_status` conține rezultatul final "
        "{Check-Out, Canceled, No-Show} — eliminare obligatorie înainte de antrenare."
    )
    if all(c in df_raw.columns for c in leak_cols):
        st.dataframe(df_raw[leak_cols].head(10), use_container_width=True, height=200)

    st.subheader("6. Statistici descriptive")
    with st.expander("Variabile numerice (df.describe())"):
        st.dataframe(df_raw.describe(), use_container_width=True)
    with st.expander("Variabile categorice (df.describe(include='object'))"):
        st.dataframe(df_raw.describe(include="object"), use_container_width=True)



elif sectiune == SECTIUNI[2]:
    st.header("Curățare — Tratarea problemelor identificate")
    st.markdown(
        "Pipeline în 5 pași: eliminare leakage/PII → imputare valori lipsă → "
        "filtrare outlieri ADR → eliminare rânduri fără sens business → "
        "derivare 3 features (`total_nights`, `total_guests`, `month_num`)."
    )

    st.subheader("Strategia de curățare a datelor")
    decisions = pd.DataFrame({
        "Variabilă sau problemă": [
            "`reservation_status` + `reservation_status_date`",
            "`company`",
            "`agent`",
            "`country`",
            "`children`",
            "Valori ADR extreme (> Q3 + 1.5×IQR)",
            "Rezervări fără oaspeți (`adults`+`children`+`babies` = 0)",
            "Rezervări fără nopți de cazare (`stays_*` = 0)",
        ],
        "Acțiune": [
            "Eliminăm coloanele",
            "Eliminăm coloana",
            "Completăm valorile lipsă cu 0",
            "Înlocuim valorile lipsă cu cea mai frecventă valoare (PRT)",
            "Eliminăm cele 4 rânduri afectate",
            "Eliminăm rândurile (regula 1.5×IQR)",
            "Eliminăm rândurile",
            "Eliminăm rândurile",
        ],
        "Justificare": [
            "Conțin starea finală a rezervării — divulgă răspunsul (data leakage)",
            "Aproximativ 94% din valori lipsesc — coloana e prea goală pentru a fi utilă",
            "Lipsa unui agent = rezervare directă (înțeles real, nu valoare lipsă)",
            "Portugalia (PRT) e clar dominantă în date; păstrăm acest tipar",
            "Impact neglijabil — 0,003% din înregistrări",
            "Prețuri pe noapte de peste ~220€ sunt implauzibile pentru un hotel de 3-4",
            "Înregistrări neutilizabile pentru analiză comportamentală",
            "Înregistrări neutilizabile pentru analiză comportamentală",
        ],
    })
    st.dataframe(
        decisions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Variabila sau problema": st.column_config.TextColumn(width="medium"),
            "Actiune": st.column_config.TextColumn(width="medium"),
            "Justificare": st.column_config.TextColumn(width="large"),
        },
    )

    with st.expander("Regula 1.5×IQR (Tukey)"):
        st.markdown(
            "Valoare extremă = observație dincolo de **Q3 + 1.5·IQR** sau sub **Q1 − 1.5·IQR**, "
            "unde IQR = Q3 − Q1. Pentru ADR: Q3 ≈ 126€, IQR ≈ 65€, prag superior ≈ 224€."
        )

    st.subheader("Tratarea valorilor extreme ADR — alege strategia")
    col_l, col_r = st.columns([1, 2])
    with col_l:
        method_label = st.radio(
            "Algoritm",
            options=[
                "Regula 1.5×IQR (Tukey, standard)",
                "Regula 3.0×IQR (conservatoare)",
                "Prag fix (preț < 1.000€)",
                "Niciun filtru",
            ],
            index=0,
            help="Strategia implicită recomandată este 1.5×IQR. Schimbă pentru a vedea impactul vizual.",
        )
        method_map = {
            "Regula 1.5×IQR (Tukey, standard)": ("iqr", 1.5),
            "Regula 3.0×IQR (conservatoare)": ("iqr", 3.0),
            "Prag fix (preț < 1.000€)": ("domain", None),
            "Niciun filtru": ("none", None),
        }
        method, mult = method_map[method_label]
        st.session_state["outlier_method"] = method
        if mult is not None:
            st.session_state["outlier_multiplier"] = mult

    df_clean_local = get_clean(
        outlier_method=method,
        outlier_multiplier=mult if mult is not None else ADR_IQR_MULTIPLIER,
    )

    with col_r:
        st.metric("Rânduri raw", f"{len(df_raw):,}".replace(",", "."))
        st.metric(
            "Rânduri după pipeline",
            f"{len(df_clean_local):,}".replace(",", "."),
            delta=f"-{len(df_raw) - len(df_clean_local):,} ({(1 - len(df_clean_local) / len(df_raw)) * 100:.1f}%)".replace(",", "."),
        )
        st.metric("NaN rămase", int(df_clean_local.isna().sum().sum()))

    st.subheader("Comparație ADR — înainte vs după")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Înainte (raw)**")
        st.plotly_chart(
            px_box_horizontal(df_raw["adr"], "ADR — înainte (raw)"),
            use_container_width=True,
        )
        st.caption(
            f"Max: {df_raw['adr'].max():.0f}€ · Median: {df_raw['adr'].median():.0f}€"
        )
    with col_b2:
        st.markdown(f"**După — {method_label}**")
        st.plotly_chart(
            px_box_horizontal(df_clean_local["adr"], f"ADR — după ({method_label})", color="#2ca02c"),
            use_container_width=True,
        )
        st.caption(
            f"Max: {df_clean_local['adr'].max():.0f}€ · Median: {df_clean_local['adr'].median():.0f}€"
        )

    st.success(
        f"Curățare aplicată cu succes. Dataset rezultat: **{len(df_clean_local):,} rânduri × "
        f"{df_clean_local.shape[1]} coloane** (cu 3 variabile derivate adăugate: "
        "`total_nights` = nopți totale, `total_guests` = total oaspeți, `month_num` = lună 1-12).".replace(",", ".")
    )

    with st.expander("Preview dataset curățat (primele 15 rânduri)"):
        st.dataframe(df_clean_local.head(15), use_container_width=True)


elif sectiune == SECTIUNI[3]:
    st.header("Codificare date categorice")
    st.markdown(
        "Cinci strategii, alese după **cardinalitate** și **ordonabilitate** "
        "(prezența unei ordini intrinseci între valori)."
    )

    encoding_table = pd.DataFrame({
        "Strategie": [
            "Binary manual",
            "One-Hot Encoding",
            "Ordinal manual",
            "Frequency Encoding",
            "Cyclic (sin/cos)",
        ],
        "Aplicată pe": [
            "`hotel` (2 valori)",
            "`meal`, `market_segment`, `distribution_channel`, `customer_type`, `reserved_room_type`, `assigned_room_type`",
            "`deposit_type` (3 valori cu ordine)",
            "`country` (~178 valori)",
            "`arrival_date_month` (12 valori)",
        ],
        "De ce această strategie?": [
            "Cardinalitate 2 → o coloană e suficientă (`is_resort`). One-Hot ar fi redundant.",
            "Nominal, fără ordine. `LabelEncoder` ar introduce ordine artificială (HB nu e 2× BB).",
            "EXISTĂ ordine business: No Deposit < Refundable < Non Refund (commitment crescător).",
            "One-Hot pe 178 valori = 178 coloane sparse → curse of dimensionality. Frequency păstrează prevalența într-o singură coloană.",
            "Decembrie e ADIACENT cu Ianuarie în realitate; numeric naiv ar pune distanță 11. `sin/cos(2π·m/12)` păstrează ciclicitatea.",
        ],
    })
    st.dataframe(
        encoding_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strategie": st.column_config.TextColumn(width="small"),
            "Aplicat pe": st.column_config.TextColumn(width="medium"),
            "De ce aceasta strategie?": st.column_config.TextColumn(width="large"),
        },
    )

    df_clean_local = get_clean(outlier_method=outlier_method, outlier_multiplier=outlier_multiplier)
    df_enc_local = get_encoded(outlier_method=outlier_method, outlier_multiplier=outlier_multiplier)

    st.subheader("Inspectează transformarea pentru o variabilă")
    var = st.selectbox(
        "Alege variabila:",
        ["hotel", "meal", "deposit_type", "country", "arrival_date_month"],
    )

    col_v1, col_v2 = st.columns([1, 1.2])

    with col_v1:
        st.markdown("**Înainte (categorical raw)**")
        if var in df_clean_local.columns:
            counts = df_clean_local[var].value_counts().head(10)
            st.dataframe(
                pd.DataFrame({"valoare": counts.index, "frecvență": counts.values}),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
        else:
            st.caption(f"Coloana `{var}` deja eliminată în pipeline (e.g. dropped după encoding).")

    with col_v2:
        st.markdown("**După encoding (coloane numerice rezultate)**")
        if var == "hotel":
            res_cols = ["is_resort"]
        elif var == "deposit_type":
            res_cols = [c for c in df_enc_local.columns if c.startswith("deposit_type")]
        elif var == "country":
            res_cols = ["country_freq"]
        elif var == "arrival_date_month":
            res_cols = ["month_sin", "month_cos", "month_num"]
        else:
            res_cols = [c for c in df_enc_local.columns if c.startswith(f"{var}_")]
        if res_cols:
            preview = df_enc_local[res_cols].head(15)
            st.dataframe(preview, use_container_width=True, height=320)
            st.caption(
                f"Coloane generate: **{len(res_cols)}** "
                f"({', '.join(f'`{c}`' for c in res_cols[:5])}"
                + (f", ... +{len(res_cols) - 5}" if len(res_cols) > 5 else "")
                + ")"
            )
        else:
            st.caption("Nu am identificat coloane rezultate (verifică pipeline).")

    st.subheader("Rezumat global")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Coloane înainte (după cleaning)", df_clean_local.shape[1])
    col_s2.metric("Coloane după encoding", df_enc_local.shape[1])
    col_s3.metric(
        "Variabile numerice noi",
        df_enc_local.select_dtypes(include="number").shape[1]
        - df_clean_local.select_dtypes(include="number").shape[1],
    )




elif sectiune == SECTIUNI[4]:
    st.header("Scalare — uniformizarea scărilor")
    st.markdown(
        "Algoritmii bazați pe distanță (K-means) și pe gradient descent (regresie logistică) "
        "sunt sensibili la scara variabilelor. OLS nu necesită scalare."
    )

    df_enc_local = get_features_for_modeling(outlier_method=outlier_method, outlier_multiplier=outlier_multiplier)

    st.subheader("Compară scalele înainte / după")
    method_choice = st.radio(
        "Algoritm scalare:",
        ["StandardScaler (Z-score: μ=0, σ=1)", "MinMaxScaler (interval [0, 1])"],
        index=0,
        horizontal=True,
        help="StandardScaler e recomandat pentru K-means + LogReg (distribuție centrată); "
        "MinMaxScaler e mai potrivit când distribuția nu e gaussiană.",
    )
    method_key = "standard" if method_choice.startswith("Standard") else "minmax"

    cols_to_scale = ["lead_time", "adr", "total_nights", "total_guests"]
    X_raw = df_enc_local[cols_to_scale].copy()
    X_scaled, _ = scale_features(X_raw, cols_to_scale, method=method_key)

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("**Înainte de scalare** — statistici per coloană")
        stats_before = (
            X_raw.describe()
            .T[["mean", "std", "min", "max"]]
            .round(2)
            .rename(index=FRIENDLY_NAMES, columns={"mean": "Medie", "std": "Dev. std", "min": "Min", "max": "Max"})
        )
        st.dataframe(stats_before, use_container_width=True)
        df_long_before = X_raw.rename(columns=FRIENDLY_NAMES).melt(
            var_name="Variabilă", value_name="Valoare"
        )
        fig_b = px.box(
            df_long_before,
            x="Variabilă",
            y="Valoare",
            color="Variabilă",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points=False,
            title="Distribuții (scara originală)",
            height=360,
        )
        fig_b.update_layout(showlegend=False, margin=dict(t=50, l=40, r=20, b=40))
        st.plotly_chart(fig_b, use_container_width=True)
    with col_n2:
        st.markdown(f"**După scalare ({method_choice})**")
        stats_after = (
            X_scaled.describe()
            .T[["mean", "std", "min", "max"]]
            .round(2)
            .rename(index=FRIENDLY_NAMES, columns={"mean": "Medie", "std": "Dev. std", "min": "Min", "max": "Max"})
        )
        st.dataframe(stats_after, use_container_width=True)
        df_long_after = X_scaled.rename(columns=FRIENDLY_NAMES).melt(
            var_name="Variabilă", value_name="Valoare"
        )
        fig_a = px.box(
            df_long_after,
            x="Variabilă",
            y="Valoare",
            color="Variabilă",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points=False,
            title=f"Distribuții ({method_choice})",
            height=360,
        )
        fig_a.update_layout(showlegend=False, margin=dict(t=50, l=40, r=20, b=40))
        st.plotly_chart(fig_a, use_container_width=True)

    st.subheader("Efectul scalării asupra K-means")
    st.markdown(
        "Pe `lead_time` (0–737) vs `adr` (0–200+), fără scalare clusterele se formează "
        "aproape exclusiv pe `lead_time`; cu scalare, ambele dimensiuni contribuie egal."
    )

    from sklearn.cluster import KMeans as _KMeans

    sample = df_enc_local[["lead_time", "adr"]].sample(5000, random_state=RANDOM_STATE)
    sample_scaled, _ = scale_features(sample, ["lead_time", "adr"], method="standard")

    km_raw = _KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(sample)
    km_scaled = _KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10).fit(sample_scaled)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Fără scalare** — K-means dominat de `lead_time`")
        df_raw_clust = sample.copy()
        df_raw_clust["Cluster"] = km_raw.labels_.astype(str)
        fig1 = px.scatter(
            df_raw_clust,
            x="lead_time",
            y="adr",
            color="Cluster",
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="Clusterele se taie vertical (după lead_time)",
            labels={
                "lead_time": "Zile între rezervare și sosire (0-737)",
                "adr": "Preț pe noapte ADR (0-200+ €)",
            },
            height=380,
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_d2:
        st.markdown("**Cu StandardScaler** — clusterele țin cont de ambele axe")
        df_s_clust = sample_scaled.copy()
        df_s_clust["Cluster"] = km_scaled.labels_.astype(str)
        fig2 = px.scatter(
            df_s_clust,
            x="lead_time",
            y="adr",
            color="Cluster",
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="Clusterele se formează pe ambele axe",
            labels={
                "lead_time": "Zile între rezervare și sosire (standardizate)",
                "adr": "Preț pe noapte ADR (standardizat)",
            },
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)



elif sectiune == SECTIUNI[5]:
    df_view = apply_global_filters(df_clean, filter_hotel, filter_seasons, filter_lead)

    st.header("Statistici descriptive & analize de grup")
    st.markdown("`groupby().agg()`, `pivot_table()` și `apply(lambda)`. Filtrele globale se aplică.")

    if len(df_view) < len(df_clean):
        st.info(
            f"Filtrele globale au redus la **{len(df_view):,}** rânduri "
            f"({len(df_view) / len(df_clean) * 100:.1f}% din curățat).".replace(",", ".")
        )

    analiza = st.selectbox(
        "Alege dimensiunea analizei:",
        [
            "Hotel × Lună — rata anulărilor",
            "Segment × Tip depozit — preț mediu (pivot_table)",
            "Top 10 țări — volum, rata anulărilor, preț mediu",
            "Funcție de grup (apply) — clienți care rezervă mai târziu decât mediana",
            "Matrice de corelație între variabilele numerice",
            "Sunburst ierarhic — Hotel → Segment → Stare rezervare",
        ],
    )

    MONTH_ORDER = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    if analiza.startswith("Hotel × Lună"):
        st.subheader("Rata anulărilor pe Hotel × Lună")
        st.code(
            "df.groupby(['hotel', 'arrival_date_month'])['is_canceled'].agg(['mean', 'count'])",
            language="python",
        )
        gby = (
            df_view.groupby(["hotel", "arrival_date_month"])["is_canceled"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "rata_anulare", "count": "volum"})
        )
        gby["arrival_date_month"] = pd.Categorical(
            gby["arrival_date_month"], categories=MONTH_ORDER, ordered=True
        )
        gby = gby.sort_values(["hotel", "arrival_date_month"]).reset_index(drop=True)
        st.dataframe(
            gby.rename(columns=FRIENDLY_NAMES),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        st.subheader("Vizualizare — heatmap interactiv")
        pivot = gby.pivot(index="arrival_date_month", columns="hotel", values="rata_anulare")
        pivot = pivot.reindex(MONTH_ORDER) * 100
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            labels=dict(x="Hotel", y="Luna", color="Rata anulărilor (%)"),
            text_auto=".1f",
            title="Rata anulărilor (%) per hotel × lună",
        )
        fig.update_layout(height=520, coloraxis_colorbar=dict(title="Rata (%)"))
        st.plotly_chart(fig, use_container_width=True)

    elif analiza.startswith("Segment × Tip depozit"):
        st.subheader("Preț mediu pe noapte — Segment × Tip depozit")
        st.code(
            "df.pivot_table(values='adr', index='market_segment', columns='deposit_type', aggfunc='mean')",
            language="python",
        )
        pvt = df_view.pivot_table(
            values="adr",
            index="market_segment",
            columns="deposit_type",
            aggfunc="mean",
        )
        fig = px.imshow(
            pvt.values,
            x=pvt.columns.tolist(),
            y=pvt.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(x="Tip depozit", y="Segment de piață", color="Preț mediu (€)"),
            text_auto=".1f",
            title="Preț mediu pe noapte (€) — segment de piață × tip depozit",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    elif analiza.startswith("Top 10 țări"):
        st.subheader("Top 10 țări după volum de rezervări")
        st.code(
            "df.groupby('country').agg(volum=('is_canceled','count'), rata_anulare=('is_canceled','mean'), pret_mediu=('adr','mean'))",
            language="python",
        )
        top = (
            df_view.groupby("country")
            .agg(
                volum=("is_canceled", "count"),
                rata_anulare=("is_canceled", "mean"),
                pret_mediu_noapte=("adr", "mean"),
            )
            .sort_values("volum", ascending=False)
            .head(10)
            .reset_index()
        )
        top["rata_anulare"] = (top["rata_anulare"] * 100).round(2)
        top["pret_mediu_noapte"] = top["pret_mediu_noapte"].round(2)
        st.dataframe(
            top.rename(columns=FRIENDLY_NAMES),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            top.sort_values("volum"),
            x="volum",
            y="country",
            orientation="h",
            color="rata_anulare",
            color_continuous_scale="RdYlGn_r",
            labels={
                "volum": "Volum rezervări",
                "country": "Țară",
                "rata_anulare": "Rata anulărilor (%)",
            },
            title="Top 10 țări — volum și rata anulărilor (intensitate culoare)",
            height=460,
            text_auto=True,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    elif analiza.startswith("Matrice de corelație"):
        st.subheader("Matrice de corelație Pearson")
        st.markdown(
            "Verificăm multicoliniaritatea între predictori înainte de regresie."
        )
        numeric_cols = [
            "lead_time", "adr", "total_nights", "total_guests", "is_canceled",
            "previous_cancellations", "previous_bookings_not_canceled",
            "booking_changes", "required_car_parking_spaces", "total_of_special_requests",
        ]
        numeric_cols = [c for c in numeric_cols if c in df_view.columns]
        corr = df_view[numeric_cols].corr().round(2)
        fig_corr = px.imshow(
            corr.values,
            x=[pretty(c) for c in corr.columns],
            y=[pretty(c) for c in corr.index],
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto",
            title="Coeficienți Pearson între variabilele numerice principale",
        )
        fig_corr.update_layout(height=560, margin=dict(t=60))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Corelații negative: `is_canceled` ↔ `total_of_special_requests`, "
            "`required_car_parking_spaces`. Pozitivă: `lead_time` ↔ `is_canceled`. "
            "Corelațiile dintre predictori sunt slabe — multicoliniaritate redusă."
        )

    elif analiza.startswith("Sunburst"):
        st.subheader("Sunburst ierarhic — Hotel → Segment → Stare rezervare")
        st.markdown(
            "Trei nivele ierarhice; mărimea sectorului = volumul de rezervări."
        )
        sb_df = df_view.copy()
        sb_df["Stare"] = sb_df["is_canceled"].map({0: "Onorat", 1: "Anulat"})
        sb_counts = (
            sb_df.groupby(["hotel", "market_segment", "Stare"])
            .size()
            .reset_index(name="volum")
            .rename(columns={
                "hotel": "Hotel",
                "market_segment": "Segment de piață",
                "volum": "Volum rezervări",
            })
        )
        fig_sb = px.sunburst(
            sb_counts,
            path=["Hotel", "Segment de piață", "Stare"],
            values="Volum rezervări",
            color="Stare",
            color_discrete_map={"Onorat": "#2ca02c", "Anulat": "#d62728"},
            title="Distribuția rezervărilor pe Hotel × Segment × Stare",
            height=620,
        )
        fig_sb.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_sb, use_container_width=True)
        st.caption(
            "Anulări concentrate pe Online TA și Groups; rate joase pe Direct și Corporate."
        )

    else:  # apply
        st.subheader("Funcție de grup cu `apply` + lambda — volatilitate preț per segment")
        st.code(
            "df.groupby('market_segment').apply(\n"
            "    lambda x: pd.Series({\n"
            "        'preț mediu (€)': x['adr'].mean(),\n"
            "        'volatilitate (CV %)': x['adr'].std() / x['adr'].mean() * 100,\n"
            "        '% cu cereri speciale': (x['total_of_special_requests'] > 0).mean() * 100,\n"
            "    })\n"
            ")",
            language="python",
        )
        st.markdown(
            "Trei agregări per segment într-o singură funcție lambda: preț mediu, "
            "coeficient de variație (CV = std/mean) și ponderea clienților cu cereri speciale."
        )
        result = (
            df_view.groupby("market_segment")
            .apply(
                lambda x: pd.Series({
                    "preț_mediu_eur": x["adr"].mean(),
                    "volatilitate_cv_pct": (x["adr"].std() / x["adr"].mean() * 100)
                    if x["adr"].mean() > 0 else 0,
                    "pct_cu_cereri_speciale": (x["total_of_special_requests"] > 0).mean() * 100,
                    "volum": len(x),
                }),
                include_groups=False,
            )
            .round(2)
            .reset_index()
            .sort_values("volum", ascending=False)
        )
        result_display = result.rename(columns={
            "market_segment": "Segment de piață",
            "preț_mediu_eur": "Preț mediu (€)",
            "volatilitate_cv_pct": "Volatilitate preț (CV %)",
            "pct_cu_cereri_speciale": "% clienți cu cereri speciale",
            "volum": "Volum rezervări",
        })
        st.dataframe(result_display, use_container_width=True, hide_index=True)
        st.caption(
            "CV > 40% = pricing eterogen; CV < 20% = pachete standardizate."
        )



elif sectiune == SECTIUNI[6]:
    st.header("Analiză geografică — piețe-sursă")
    st.markdown(
        "Țara de origine (cod ISO-3) per rezervare. Filtrele globale se aplică."
    )

    df_view = apply_global_filters(df_clean, filter_hotel, filter_seasons, filter_lead)
    EXCLUDED_TERRITORIES = ["ATA", "ATF", "BVT", "HMD", "SGS", "UMI", "IOT", "PCN", "TMP"]
    n_excluded = int(df_view["country"].isin(EXCLUDED_TERRITORIES).sum())
    df_view = df_view[~df_view["country"].isin(EXCLUDED_TERRITORIES)]
    if n_excluded > 0:
        st.caption(
            f"ℹ Au fost excluse **{n_excluded} rânduri** cu coduri ISO pentru teritorii "
            f"fără populație stabilă ({', '.join(EXCLUDED_TERRITORIES)}) — probabile erori "
            "de introducere a datelor."
        )
    country_stats_full = (
        df_view.groupby("country")
        .agg(
            volum=("is_canceled", "count"),
            rata_anulare=("is_canceled", "mean"),
            pret_mediu=("adr", "mean"),
        )
        .reset_index()
    )

    metric_choice = st.radio(
        "Metrica afișată pe hartă:",
        ["Volum rezervări", "Rata anulărilor (%)", "Preț mediu pe noapte (€)"],
        horizontal=True,
    )
    metric_map = {
        "Volum rezervări": ("volum", "OrRd"),
        "Rata anulărilor (%)": ("rata_anulare", "RdYlGn_r"),
        "Preț mediu pe noapte (€)": ("pret_mediu", "Viridis"),
    }
    col_name, cmap = metric_map[metric_choice]

    if col_name == "volum":
        plot_data = country_stats_full.copy()
        st.caption(
            f"Toate cele **{len(plot_data)} țări** sunt afișate (volum total)."
        )
    else:
        plot_data = country_stats_full[country_stats_full["volum"] >= 100].copy()
        st.caption(
            f"Filtru aplicat: doar țări cu **cel puțin 100 de rezervări** "
            f"({len(plot_data)} țări) — pentru stabilitate statistică, ratele calculate "
            "pe puține observații sunt nefiabile."
        )

    plot_values = plot_data[col_name].copy()
    if col_name == "rata_anulare":
        plot_values = plot_values * 100
    plot_data_for_map = plot_data.copy()
    plot_data_for_map["display_value"] = plot_values.values

    geo_warning = None
    try:
        world_geojson, world_iso_codes = get_world_geojson()
    except Exception as err:
        geo_warning = str(err)
        world_geojson, world_iso_codes = None, None

    if world_geojson is not None:
        fig_map = px.choropleth(
            plot_data_for_map,
            geojson=world_geojson,
            locations="country",
            featureidkey="properties.ISO_A3",
            color="display_value",
            hover_name="country",
            hover_data={
                "country": False,
                "display_value": False,
                "volum": ":,",
                "rata_anulare": ":.2%",
                "pret_mediu": ":.2f €",
            },
            color_continuous_scale=cmap,
            labels={"display_value": metric_choice},
            title=f"{metric_choice} pe țări de origine "
            "(geometrii încărcate cu geopandas din shapefile Natural Earth)",
        )
    else:
        fig_map = px.choropleth(
            plot_data_for_map,
            locations="country",
            locationmode="ISO-3",
            color="display_value",
            hover_name="country",
            color_continuous_scale=cmap,
            labels={"display_value": metric_choice},
            title=f"{metric_choice} pe țări de origine (fallback — geopandas indisponibil)",
        )

    fig_map.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=metric_choice),
    )
    fig_map.update_geos(
        showframe=False,
        showcoastlines=True,
        showcountries=True,
        countrycolor="white",
        coastlinecolor="#888",
        landcolor="#f5f5f5",
        oceancolor="#eef6fb",
        showocean=True,
        projection_type="natural earth",
        lataxis_range=[-58, 85],  # taie Antarctica
    )
    st.plotly_chart(fig_map, use_container_width=True)
    if geo_warning:
        st.warning(
            f"Shapefile geopandas indisponibil ({geo_warning}). "
            "Folosesc geometriile încorporate Plotly ca fallback."
        )

    st.subheader("Matrice strategică — volum × preț × rată anulări")
    st.markdown(
        "X = volum (scară logaritmică); Y = preț mediu pe noapte; culoare = rata anulărilor. "
        "Țintele de extindere: cadranul dreapta-sus, verde."
    )
    matrix_data = country_stats_full[country_stats_full["volum"] >= 100].copy()
    matrix_data["rata_anulare_pct"] = (matrix_data["rata_anulare"] * 100).round(2)
    median_volum = matrix_data["volum"].median()
    median_pret = matrix_data["pret_mediu"].median()

    fig_mat = px.scatter(
        matrix_data,
        x="volum",
        y="pret_mediu",
        color="rata_anulare_pct",
        size="volum",
        hover_name="country",
        color_continuous_scale="RdYlGn_r",
        labels={
            "volum": "Volum rezervări",
            "pret_mediu": "Preț mediu pe noapte (€)",
            "rata_anulare_pct": "Rata anulărilor (%)",
        },
        size_max=42,
        height=560,
    )
    fig_mat.add_hline(y=median_pret, line_dash="dot", line_color="gray", opacity=0.5,
                      annotation_text="median preț", annotation_position="bottom right")
    fig_mat.add_vline(x=median_volum, line_dash="dot", line_color="gray", opacity=0.5,
                      annotation_text="median volum", annotation_position="top left")
    fig_mat.update_xaxes(type="log", title="Volum rezervări (scară logaritmică)")
    st.plotly_chart(fig_mat, use_container_width=True)

    st.subheader("Top 10 țări după volum")
    top10 = country_stats_full.sort_values("volum", ascending=False).head(10).copy()
    top10["rata_anulare"] = (top10["rata_anulare"] * 100).round(2)
    top10["pret_mediu"] = top10["pret_mediu"].round(2)
    top10 = top10.rename(columns={
        "country": "Țară (ISO-3)",
        "volum": "Volum",
        "rata_anulare": "Rata anulărilor (%)",
        "pret_mediu": "Preț mediu / noapte (€)",
    })
    st.dataframe(top10, use_container_width=True, hide_index=True)



elif sectiune == SECTIUNI[7]:
    st.header("Segmentare K-means — profile comportamentale de clienți")
    st.markdown(
        "Segmentare nesupervizată pe profilul rezervării (`lead_time`, `adr`, "
        "`total_nights`, `total_guests`). Selectarea K prin metoda Elbow (inerție) "
        "și scor Silhouette pe K ∈ [2, 10]."
    )

    st.subheader("Selectarea K — Elbow + Silhouette (interactiv)")
    with st.spinner("Calculez diagnosticele K-means pe subeșantion (20.000 rânduri)…"):
        diag = get_kmeans_diag(outlier_method=outlier_method)

    _k_arr = np.array(diag["k_range"])
    _inert = np.array(diag["inertias"])
    _coords = np.column_stack((_k_arr, (_inert - _inert.min()) / (_inert.max() - _inert.min())))
    _line_start, _line_end = _coords[0], _coords[-1]
    _line_vec = _line_end - _line_start
    _line_norm = _line_vec / np.linalg.norm(_line_vec)
    _point_vecs = _coords - _line_start
    _projs = _point_vecs @ _line_norm
    _distances = np.linalg.norm(_point_vecs - _projs[:, None] * _line_norm, axis=1)
    elbow_k = int(_k_arr[np.argmax(_distances)])
    best_sil_k = int(_k_arr[np.argmax(diag["silhouettes"])])

    fig_diag = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Metoda Elbow — Inerție (WCSS) per K",
            "Metoda Silhouette — scor per K",
        ),
    )
    fig_diag.add_trace(
        go.Scatter(
            x=list(diag["k_range"]),
            y=diag["inertias"],
            mode="lines+markers+text",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=12),
            text=[f"K={k}" for k in diag["k_range"]],
            textposition="top center",
            textfont=dict(size=11, color="#5d5d5d"),
            name="Inerție",
            hovertemplate="K = %{x}<br>Inerție = %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_diag.add_vline(
        x=elbow_k,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Elbow detectat: K = {elbow_k}",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig_diag.add_trace(
        go.Scatter(
            x=list(diag["k_range"]),
            y=diag["silhouettes"],
            mode="lines+markers+text",
            line=dict(color="#2ca02c", width=2.5),
            marker=dict(size=12),
            text=[f"K={k}" for k in diag["k_range"]],
            textposition="top center",
            textfont=dict(size=11),
            name="Silhouette",
            hovertemplate="K = %{x}<br>Silhouette = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_diag.add_vline(
        x=best_sil_k,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Maxim: K = {best_sil_k}",
        annotation_position="top right",
        row=1,
        col=2,
    )
    fig_diag.update_xaxes(title_text="K (număr clustere)", row=1, col=1, dtick=1)
    fig_diag.update_xaxes(title_text="K (număr clustere)", row=1, col=2, dtick=1)
    fig_diag.update_yaxes(title_text="Inerție (WCSS)", row=1, col=1)
    fig_diag.update_yaxes(title_text="Silhouette score", row=1, col=2)
    fig_diag.update_layout(height=420, showlegend=False, margin=dict(t=60, l=40, r=20, b=40))
    st.plotly_chart(fig_diag, use_container_width=True)
    st.caption(
        f"Elbow detectat la **K = {elbow_k}**; Silhouette maxim la **K = {best_sil_k}**. "
        f"K = 4 ales ca echilibru între granularitate și interpretabilitate."
    )

    k = st.slider(
        "Alege K (numărul de clustere):", 2, 10, value=KMEANS_K_DEFAULT, step=1
    )

    with st.spinner(f"Antrenez K-means cu K={k}…"):
        km_res = get_kmeans_fit(k=k, outlier_method=outlier_method)

    st.subheader(f"Profile clustere (K={k})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", f"{km_res['silhouette']:.3f}")
    c2.metric("Inertie (WCSS)", f"{km_res['inertia']:.0f}")
    c3.metric("Rânduri folosite", f"{len(km_res['labels']):,}".replace(",", "."))

    profiles = km_res["profiles"].copy()
    if "cancel_rate" in profiles.columns:
        profiles["cancel_rate"] = (profiles["cancel_rate"] * 100).round(2)
    for col in NUMERIC_FEATURES_KMEANS:
        profiles[col] = profiles[col].round(2)

    means = profiles[NUMERIC_FEATURES_KMEANS].mean()

    def _label_cluster(row: pd.Series) -> tuple[str, str]:
        """Compune o denumire și recomandare bazate pe rang relativ în setul de clustere."""
        lead = row["lead_time"]
        adr_val = row["adr"]
        nights = row["total_nights"]
        guests = row["total_guests"]

        traits = []  
        recoms = []  


        if lead > means["lead_time"] * 1.4:
            traits.append("planificare în avans")
            recoms.append(
                "rezervările sunt făcute cu mult timp înainte — comunicare periodică "
                "(reminders 30/14/7 zile) și ofertă early-bird pentru următorul sejur"
            )
        elif lead < means["lead_time"] * 0.6:
            traits.append("last-minute")
            recoms.append(
                "risc mai mare de anulare datorită deciziei rapide — depozit "
                "non-refundabil obligatoriu, comunicare scurtă și directă"
            )

        if adr_val > means["adr"] * 1.2:
            traits.append("preț ridicat")
            recoms.append(
                "segment premium — program loialitate, servicii personalizate, "
                "comunicare prin canal direct (nu Online TA)"
            )
        elif adr_val < means["adr"] * 0.8:
            traits.append("preț scăzut")
            recoms.append(
                "segment buget — oferte upgrade plătit la check-in, evitarea reducerilor "
                "suplimentare care erodează marjele"
            )

        if nights > means["total_nights"] * 1.3:
            traits.append("sejur prelungit")
            recoms.append(
                "ședere lungă — pachete tarifare pe pachet (nu pe noapte), incluzând "
                "activități locale și servicii suplimentare"
            )
        elif nights < means["total_nights"] * 0.7:
            traits.append("sejur scurt")
            recoms.append(
                "ședere scurtă — ofertă upsell pentru noapte adițională la check-in, "
                "servicii rapide pentru clientul business"
            )

        if guests > means["total_guests"] * 1.2:
            traits.append("grup mare")
            recoms.append(
                "rezervare pentru grup — pachete cu activități comune, depozit "
                "non-refundabil pentru rezervări mari"
            )
        elif guests < means["total_guests"] * 0.8:
            traits.append("solo sau cuplu")
            recoms.append(
                "rezervare individuală sau în doi — oferte specifice (city break, "
                "spa, weekend romantic)"
            )

        if not traits:
            nume = "Profil mediu"
            recom = (
                "Acest cluster nu are trăsături distinctive marcate față de medie — "
                "reprezintă clientul-tipic al hotelului. Strategie: comunicare standard, "
                "fără diferențiere specială."
            )
        else:
            nume = " · ".join(t.capitalize() for t in traits)
            recom = "Strategie diferențiată: " + "; ".join(recoms) + "."

        return nume, recom

    profiles[["denumire", "recomandare"]] = profiles.apply(
        lambda r: pd.Series(_label_cluster(r)), axis=1
    )

    display_cols_order = (
        ["cluster", "denumire", "volume", "cancel_rate"]
        + NUMERIC_FEATURES_KMEANS
        + ["recomandare"]
    )
    display_cols_order = [c for c in display_cols_order if c in profiles.columns]
    profiles_display = profiles[display_cols_order].rename(
        columns={
            "cluster": "Cluster",
            "denumire": "Denumire",
            "volume": "Volum",
            "cancel_rate": "Rata anulărilor (%)",
            "lead_time": "Zile până la sosire",
            "adr": "Preț mediu / noapte (€)",
            "total_nights": "Nopți totale",
            "total_guests": "Total oaspeți",
            "recomandare": "Recomandare strategică",
        }
    )
    st.dataframe(
        profiles_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Denumire": st.column_config.TextColumn(width="medium"),
            "Recomandare strategica": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Centroizii sunt revertiți pe scara originală (`inverse_transform`); "
        "denumirea și recomandarea sunt derivate din pragul față de media populației de clustere."
    )

    st.subheader("Vizualizare 2D — perechi de axe originale (perspectivă alternativă)")
    col_x, col_y = st.columns(2)
    with col_x:
        ax_x = st.selectbox(
            "Axa X:",
            NUMERIC_FEATURES_KMEANS,
            index=0,
            format_func=pretty,
        )
    with col_y:
        ax_y = st.selectbox(
            "Axa Y:",
            NUMERIC_FEATURES_KMEANS,
            index=1,
            format_func=pretty,
        )
    if ax_x != ax_y:
        df_enc_local = get_features_for_modeling(outlier_method=outlier_method)
        X_xy = df_enc_local[[ax_x, ax_y]].reset_index(drop=True)
        sample_idx = X_xy.sample(min(8000, len(X_xy)), random_state=RANDOM_STATE).index
        sample_df = X_xy.iloc[sample_idx].copy()
        sample_df["Cluster"] = pd.Series(km_res["labels"]).iloc[sample_idx].astype(str).values

        fig = px.scatter(
            sample_df,
            x=ax_x,
            y=ax_y,
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.65,
            title=f"Clustere K-means proiectate pe {pretty(ax_x)} × {pretty(ax_y)}",
            labels={ax_x: pretty(ax_x), ax_y: pretty(ax_y)},
            height=520,
        )
        # Centroizi
        centroids_df = pd.DataFrame(
            km_res["centroids_original"], columns=NUMERIC_FEATURES_KMEANS
        )
        fig.add_trace(
            go.Scatter(
                x=centroids_df[ax_x],
                y=centroids_df[ax_y],
                mode="markers+text",
                marker=dict(symbol="x", size=20, color="#d62728", line=dict(width=3, color="#d62728")),
                text=[f"<b>C{c}</b>" for c in range(len(centroids_df))],
                textposition="top center",
                textfont=dict(size=14, color="#d62728"),
                name="Centroizi",
                hovertemplate="Centroid<br>"
                + f"{pretty(ax_x)}: %{{x:.2f}}<br>{pretty(ax_y)}: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.update_traces(marker=dict(size=6), selector=dict(mode="markers", name=None))
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Alege axe diferite pentru scatter 2D.")




elif sectiune == SECTIUNI[8]:
    st.header("Predicții — Regresie logistică + OLS")
    sub_a, sub_b = st.tabs(["Anulare (LogReg)", "Preț (OLS)"])

    # ----------- 9.A — LogReg --------------
    with sub_a:
        st.subheader("Modelarea probabilității de anulare — `is_canceled`")
        st.markdown(
            "Train/test 80/20 stratificat, StandardScaler fit pe train, "
            "`class_weight='balanced'` (cost asimetric), prag de decizie ajustabil."
        )

        threshold = st.slider(
            "Prag de clasificare (probabilitatea minimă pentru a marca *va anula*):",
            0.10,
            0.90,
            LOGREG_THRESHOLD_DEFAULT,
            step=0.05,
            help="Default 0.4: mai sensibil decât 0.5 — capturăm mai multe anulări reale, "
            "cu prețul unor false pozitive în plus.",
        )

        with st.spinner(f"Antrenez LogReg cu prag = {threshold}…"):
            lr = get_logreg_fit(threshold=threshold, outlier_method=outlier_method)

        cls = lr["classification_report"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC AUC", f"{lr['roc_auc']:.3f}", help="Aria sub curba ROC. >0.8 = model bun.")
        c2.metric("Acuratețe", f"{cls['accuracy'] * 100:.1f}%")
        c3.metric(
            "Recall anulări",
            f"{cls['1']['recall'] * 100:.1f}%",
            help="Din toate anulările reale, câte am detectat.",
        )
        c4.metric(
            "Precizie anulări",
            f"{cls['1']['precision'] * 100:.1f}%",
            help="Din toate cele marcate ca anulare, câte sunt corecte.",
        )

        st.subheader("Matrice de confuzie și curba ROC")
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.plotly_chart(px_confusion(lr["confusion_matrix"]), use_container_width=True)
        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=lr["fpr"],
                    y=lr["tpr"],
                    mode="lines",
                    name=f"LogReg (AUC = {lr['roc_auc']:.3f})",
                    line=dict(color="#1f77b4", width=2.5),
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Clasificator aleator (AUC = 0.5)",
                    line=dict(color="gray", dash="dash", width=1.5),
                    hoverinfo="skip",
                )
            )
            fig_roc.update_layout(
                title="Curba ROC",
                xaxis_title="Rata falsurilor pozitive (FPR)",
                yaxis_title="Rata adevăraților pozitivi (TPR)",
                height=400,
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("AUC > 0.8 indică un model bun discriminativ; diagonala = clasificator aleator.")

        st.subheader("Top 10 variabile cu cel mai mare impact (coeficienți standardizați)")
        st.plotly_chart(px_coef_bar(lr["coefs"], top_n=10), use_container_width=True)

        st.subheader("Interpretare business — ce ne spun coeficienții?")
        coef_interpretations = {
            "deposit_type_ord": (
                "Depozitul non-refundabil este **factorul cel mai puternic anti-anulare**. "
                "Strategia: încurajează rezervările cu depozit (reducere mică pentru depozit nerambursabil)."
            ),
            "lead_time": (
                "Cu cât clientul rezervă mai în avans, cu atât **crește riscul** ca, până la sosire, "
                "planurile lui să se schimbe. Strategie: confirmări periodice prin email, reminder cu 14/7 zile înainte."
            ),
            "country_freq": (
                "Țările cu volum mare de rezervări au comportament diferit față de cele rare — "
                "tipic, clienții din piețele dominante (Portugalia, Spania) au pattern-uri locale diferite."
            ),
            "previous_cancellations": (
                "**Cea mai bună predicție = trecutul.** Un client care a anulat înainte va anula din nou. "
                "Strategie: marchează acești clienți pentru contact proactiv sau cer depozit non-refundabil."
            ),
            "required_car_parking_spaces": (
                "Clienții care cer parcare sunt **mult mai puțin probabili să anuleze** — semnal de "
                "intenție concretă (au planificat transportul). Variabilă predictivă valoroasă."
            ),
            "total_of_special_requests": (
                "Clienții cu cereri speciale (pat suplimentar, etaj înalt, etc.) au investit timp în "
                "rezervare → **anulare mai puțin probabilă**. Indică implicare emoțională în călătorie."
            ),
            "booking_changes": (
                "Modificările făcute la rezervare arată **angajament** — clientul interacționează "
                "activ cu rezervarea, deci mai puțin probabil să anuleze."
            ),
            "is_repeated_guest": (
                "Clienții fideli (au mai stat la noi) anulează mai rar — segment-țintă pentru "
                "programe de loialitate."
            ),
            "adr": (
                "Prețul plătit influențează anularea, dar direcția depinde de segment "
                "(prețuri mari pot însemna client premium, fidel — sau o ofertă specială cu risc)."
            ),
            "is_resort": (
                "Hotelul resort vs city are comportament de anulare diferit — resort: rezervări "
                "vacanță (planuri schimbabile); city: business (mai stabil)."
            ),
            "total_nights": "Sejururile lungi au pattern de anulare diferit de cele scurte.",
            "total_guests": "Grupurile mari planifică mai serios — mai puține anulări.",
            "month_sin": "Componenta sinusoidală a sezonalității lunare.",
            "month_cos": "Componenta cosinusoidală a sezonalității lunare.",
            "previous_bookings_not_canceled": (
                "Istoricul de rezervări onorate este predictor puternic anti-anulare."
            ),
        }

        top_coefs = lr["coefs"].head(8).copy()
        top_coefs["interpretare"] = top_coefs["feature"].map(
            lambda f: coef_interpretations.get(f, "Variabilă derivată; interpretare contextuală.")
        )
        top_coefs["direcție"] = top_coefs["coef"].apply(
            lambda v: "Crește anularea" if v > 0 else "Scade anularea"
        )
        top_coefs["feature"] = top_coefs["feature"].map(lambda f: FRIENDLY_NAMES.get(f, f))
        top_coefs = top_coefs.rename(
            columns={"feature": "Variabilă", "coef": "Coeficient", "direcție": "Direcție"}
        )
        top_coefs["Coeficient"] = top_coefs["Coeficient"].round(3)
        st.dataframe(
            top_coefs[["Variabilă", "Coeficient", "Direcție", "interpretare"]].rename(
                columns={"interpretare": "Interpretare business + recomandare"}
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Variabilă": st.column_config.TextColumn(width="small"),
                "Coeficient": st.column_config.NumberColumn(width="small"),
                "Direcție": st.column_config.TextColumn(width="small"),
                "Interpretare business + recomandare": st.column_config.TextColumn(width="large"),
            },
        )

    with sub_b:
        st.subheader("Determinanții prețului mediu pe noapte (`adr`)")
        st.markdown(
            "OLS prin `statsmodels.formula.api`. Două variante: `adr` brut "
            "și `log(adr + 1)` (recomandat dacă reziduurile sunt heteroscedastice)."
        )

        with st.spinner("Antrenez OLS (raw + log)…"):
            ols_raw = get_ols_fit(log_target=False, outlier_method=outlier_method)
            ols_log = get_ols_fit(log_target=True, outlier_method=outlier_method)

        st.subheader("Comparație: target brut vs target log-transformat")
        compare_df = pd.DataFrame({
            "Metrică": ["R²", "R² ajustat", "MAE (test)", "RMSE (test)", "n train", "n test valid"],
            "Model A — adr (raw)": [
                f"{ols_raw['r_squared']:.4f}",
                f"{ols_raw['r_squared_adj']:.4f}",
                f"{ols_raw['mae_test']:.2f}",
                f"{ols_raw['rmse_test']:.2f}",
                f"{ols_raw['n_train']:,}".replace(",", "."),
                f"{ols_raw['n_test_valid']:,}".replace(",", "."),
            ],
            "Model B — log(adr+1) [back-transform pe €]": [
                f"{ols_log['r_squared']:.4f}",
                f"{ols_log['r_squared_adj']:.4f}",
                f"{ols_log['mae_test_eur']:.2f}",
                f"{ols_log['rmse_test_eur']:.2f}",
                f"{ols_log['n_train']:,}".replace(",", "."),
                f"{ols_log['n_test_valid']:,}".replace(",", "."),
            ],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.info(
            "MAE/RMSE pentru Model B sunt back-transformate (`exp(pred) − 1`) la scara €, "
            "pentru comparație directă cu Model A."
        )

        st.subheader("Diagnostic reziduuri — comparație side-by-side")
        col_resid_a, col_resid_b = st.columns(2)
        with col_resid_a:
            st.markdown("**Model A — `adr` (raw)**")
            st.plotly_chart(
                px_residuals(
                    ols_raw["fitted_train"].values,
                    ols_raw["residuals_train"].values,
                    title="Model A: reziduuri vs fitted (raw)",
                ),
                use_container_width=True,
            )
        with col_resid_b:
            st.markdown("**Model B — `log(adr+1)`**")
            st.plotly_chart(
                px_residuals(
                    ols_log["fitted_train"].values,
                    ols_log["residuals_train"].values,
                    title="Model B: reziduuri vs fitted (log)",
                ),
                use_container_width=True,
            )

        st.caption(
            "Reziduuri ideale: bandă orizontală, varianță constantă în jurul lui 0. "
            "Lățire spre dreapta = heteroscedasticitate (favorabil log transform)."
        )

        st.subheader("Detalii model — alege varianta")
        which = st.radio(
            "Vezi detaliile pentru:",
            ["Model A — adr (raw)", "Model B — log(adr+1)"],
            horizontal=True,
        )
        chosen = ols_raw if which.startswith("Model A") else ols_log

        with st.expander("Sumarul complet `statsmodels.OLS.summary()`"):
            st.text(chosen["summary_text"])

        st.subheader("Coeficienți semnificativi (p < 0.05)")
        sig = pd.DataFrame({
            "Variabilă": [FRIENDLY_NAMES.get(v, v) for v in chosen["params"].index],
            "Coeficient": chosen["params"].values.round(4),
            "p-value": chosen["pvalues"].values.round(4),
        })
        sig = sig[sig["p-value"] < 0.05].sort_values("p-value").reset_index(drop=True)
        st.dataframe(sig, use_container_width=True, hide_index=True)
        st.caption(
            f"Din **{len(chosen['params'])}** variabile totale, **{len(sig)}** sunt "
            "semnificative statistic (p < 0,05). Acestea sunt variabilele pe care le-am "
            "păstra într-un model redus (Decizia 11 din metodologie)."
        )
