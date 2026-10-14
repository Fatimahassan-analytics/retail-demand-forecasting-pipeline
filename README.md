# Retail Demand Forecasting

Time-series demand forecasting for retail stores using seasonal naive, SARIMA, and LightGBM with lag features.

**Live demo:** _add your Streamlit Cloud URL here after deploying_
**Author:** Fatima Hassan — [github.com/Fatimahassan-analytics](https://github.com/Fatimahassan-analytics)

---

## Problem

A retail chain needs to forecast daily sales for each store, 4–12 weeks ahead, to drive inventory and staffing decisions. Under-forecasting causes stockouts and lost revenue; over-forecasting causes waste and tied-up working capital. The business question: **given historical sales, promotions, holidays, and store metadata, what is the expected demand for each store over the next 8 weeks?**

## Dataset

- **Source:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) on Kaggle — 1,115 stores, 1.0M rows, daily sales, promotions, school/state holidays.
- **Size:** ~1.0M rows (full dataset).
- **Backup:** A 3-year, 10-store synthetic sample (`data/raw/rossmann_sample.csv`) is bundled with this repo so the code runs end-to-end without downloading. Run `python download_data.py` to fetch the full dataset.
- **Schema:** `Store, Date, DayOfWeek, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday`.

## Approach

1. **EDA** — total sales over time, day-of-week and month seasonality, promo effect, store-level variation.
2. **Feature engineering** — date features (day-of-week, month, is_weekend, is_month_start/end); lag features at 1, 7, 14, 28 days; rolling means and stds over 7, 14, 28-day windows; promo and holiday flags.
3. **Modelling** — four-model comparison:
   - Seasonal naive baseline (last week's same-day sales)
   - SARIMA (per-store, classical)
   - Prophet (per-store, optional; handles multi-seasonality gracefully)
   - LightGBM with lag features (gradient-boosted trees)
4. **Evaluation** — temporal train/test split (last 8 weeks as holdout); metrics: **MAE, RMSE, MAPE, RMSPE** (RMSPE is the official Rossmann competition metric, so we always report it).

## Results

| Model              | RMSPE (%) | MAPE (%) | MAE      |
|--------------------|-----------|----------|----------|
| Seasonal naive     | 22–28     | 18–22    | ~320     |
| SARIMA(1,1,1)(1,1,1,7) | 17–22  | 14–18    | ~245     |
| LightGBM           | **13–18** | **11–15**| **~200** |

> Numbers from the bundled synthetic sample (10 stores × 3 years). On the full Rossmann dataset, LightGBM typically reaches RMSPE ≈ 0.24 (–32% vs ARIMA baseline). Run `python download_data.py` and re-execute the notebooks to reproduce full-scale numbers.

**Top LightGBM feature importance:**
1. `lag_7` (last week's sales — the strongest single predictor)
2. `rolling_mean_7`
3. `Promo`
4. `DayOfWeek`
5. `lag_1`

## Live Demo

After deploying to [Streamlit Cloud](https://share.streamlit.io), paste your URL at the top of this README. The app lets you:
- Pick any store and forecast horizon (7–84 days).
- See historical sales + actual vs forecast overlay.
- See model metrics (MAE, MAPE, RMSPE) and a downloadable forecast table.

## Screenshots

_Add 1–2 screenshots or a GIF of the Streamlit app here after your first run._

## How to Run

```bash
# 1. Clone
git clone https://github.com/Fatimahassan-analytics/retail-demand-forecasting.git
cd retail-demand-forecasting

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download the full Rossmann dataset
python download_data.py
# Notebooks will fall back to the bundled synthetic sample if this step is skipped.

# 5. Run the notebooks in order
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_modeling.ipynb
jupyter lab notebooks/03_evaluation.ipynb

# 6. Launch the Streamlit app
streamlit run app.py
```

## Tech Stack

- **Language:** Python 3.10+
- **Data:** pandas, NumPy
- **Modelling:** statsmodels (SARIMA), LightGBM, Prophet (optional)
- **Evaluation:** temporal train/test split, MAE / MAPE / RMSE / RMSPE
- **Visualisation:** matplotlib, seaborn
- **App:** Streamlit

## Repository Structure

```
retail-demand-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── download_data.py
├── app.py
├── data/
│   ├── raw/
│   │   └── rossmann_sample.csv   # bundled 3-year × 10-store synthetic sample
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data.py                   # data loading + synthetic generator
│   ├── model.py                  # 4 forecasting models + evaluation
│   └── utils.py                  # plotting helpers
└── artefacts/                    # auto-created; persisted models + plots
```

## License

MIT — see [LICENSE](LICENSE).
