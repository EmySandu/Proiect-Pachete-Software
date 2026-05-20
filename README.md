# Hotel Booking Analytics

Aplicație Streamlit pentru analiza setului **Hotel Booking Demand** (Antonio, Almeida & Nunes, 2019) — 119.390 rezervări, două hoteluri din Portugalia (City Hotel + Resort Hotel). Acoperă pipeline-ul complet al unui analist de date: calitatea datelor, curățare, codificare, scalare, EDA, vizualizare geografică, segmentare K-means, regresie logistică pentru anulări și regresie liniară multiplă OLS pentru preț.

Componenta **SAS** — 10 facilități pe același dataset (PROC IMPORT, PROC FORMAT, DATA step, PROC SQL, ARRAY, PROC FREQ, PROC REPORT, PROC LOGISTIC, PROC SGPLOT, PROC SGPANEL) — este documentată în `Proiect Pachete Software.pdf` cu suport vizual în `SAS/`.

## Rulare

Necesită Python 3.12.

Cu **uv** (recomandat):

```bash
uv sync
uv run streamlit run app.py
```

Cu **pip**:

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / macOS / Git Bash
pip install -e .
streamlit run app.py
```

Aplicația pornește pe `http://localhost:8501`.

## Structura proiectului

- `app.py` — aplicația Streamlit (9 secțiuni navigate prin sidebar)
- `src/` — module: `cleaning`, `encoding`, `scaling`, `modeling`, `geo`, `viz`, `data_loader`, `config`
- `Dataset/hotel_bookings.csv` — dataset principal (Kaggle)
- `assets/` — shapefile Natural Earth 110m pentru harta din secțiunea 7
- `SAS/` — capturi din SAS On Demand + cod SAS (`COD_SAS_FACILITATI.docx`)
- `Proiect Pachete Software.pdf` — documentația completă (Python + SAS) cu cele 20 facilități

## Cele 9 secțiuni ale aplicației

1. **Prezentare** — KPI și sezonalitate
2. **Calitatea datelor** — valori lipsă, distribuții, identificare leakage
3. **Curățare** — imputare, outliers ADR, filtrare rânduri fără sens business
4. **Codificare** — 5 strategii (binary, one-hot, ordinal, frequency, cyclic sin/cos)
5. **Scalare** — StandardScaler vs MinMaxScaler
6. **Statistici & Group-by** — pivot tables, corelație, sunburst, apply+lambda
7. **Analiză geografică** — choropleth pe volum / rata anulare / ADR mediu
8. **Segmentare K-means** — Elbow + Silhouette, profile cluster
9. **Predicții** — Logistic Regression (`is_canceled`) + OLS multiplă (`adr`)

## Dataset

[Hotel Booking Demand pe Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — Antonio, N., Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.
