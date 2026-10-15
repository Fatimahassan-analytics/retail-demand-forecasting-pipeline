"""Retail Demand Forecasting — data loading and sample generation.

Generates a synthetic daily-store-sales dataset that mirrors the schema and
seasonal behaviour of the Rossmann Store Sales dataset, so the notebooks
run end-to-end without requiring an external Kaggle download.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_PATH = RAW_DIR / "rossmann_sample.csv"
FULL_PATH = RAW_DIR / "rossmann_train.csv"

RANDOM_SEED = 42
N_STORES = 10
N_DAYS = 365 * 3   # 3 years of daily data per store


def generate_sample_data(n_stores: int = N_STORES, n_days: int = N_DAYS,
                         seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic daily-store-sales dataset with realistic seasonality.

    The data has:
    - Linear upward trend over the 3-year window
    - Day-of-week seasonality (Sun = lowest)
    - Monthly seasonality (Christmas peak)
    - Promo lift on random days (~30% of days)
    - State holidays (closed stores → 0 sales)
    - Store-level random intercept (some stores larger than others)
    """
    rng = np.random.default_rng(seed)
    rows = []
    start_date = pd.Timestamp("2022-01-01")

    store_base = rng.normal(loc=6000, scale=1500, size=n_stores)
    store_trend = rng.normal(loc=3.0, scale=1.0, size=n_stores)

    for store_idx in range(n_stores):
        for day_idx in range(n_days):
            date = start_date + pd.Timedelta(days=day_idx)
            dow = date.dayofweek
            month = date.month

            # Closed Sundays and Christmas day
            open_today = 1
            if dow == 6:           # Sunday
                open_today = 0
            if month == 12 and date.day == 25:
                open_today = 0

            if open_today == 0:
                rows.append({
                    "Store": store_idx + 1,
                    "Date": date.strftime("%Y-%m-%d"),
                    "DayOfWeek": dow + 1,
                    "Sales": 0,
                    "Customers": 0,
                    "Open": 0,
                    "Promo": 0,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                })
                continue

            # Seasonal components
            dow_effect = np.array([0, 1.10, 1.05, 1.05, 1.10, 1.20, 0.0, 0.0])[dow]
            month_effect = 1 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12)
            if month == 12:
                month_effect *= 1.18    # Christmas lift
            if month in [6, 7]:
                month_effect *= 1.05    # Summer lift

            promo = int(rng.random() < 0.30)
            promo_lift = 1.20 if promo else 1.0

            state_holiday = "0"
            if month == 12 and date.day in [24, 26, 31]:
                state_holiday = "a"

            school_holiday = int(rng.random() < 0.10)

            base = store_base[store_idx]
            trend = store_trend[store_idx] * day_idx
            noise = rng.normal(loc=0, scale=400)
            customers = max(0, int(base * 0.10 * dow_effect * month_effect * promo_lift + noise * 0.3))
            sales = max(0, int((base + trend) * dow_effect * month_effect * promo_lift + noise))

            rows.append({
                "Store": store_idx + 1,
                "Date": date.strftime("%Y-%m-%d"),
                "DayOfWeek": dow + 1,
                "Sales": sales,
                "Customers": customers,
                "Open": 1,
                "Promo": promo,
                "StateHoliday": state_holiday,
                "SchoolHoliday": school_holiday,
            })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    """Load sales data: prefer full Rossmann CSV, fall back to bundled sample."""
    if FULL_PATH.exists():
        df = pd.read_csv(FULL_PATH, parse_dates=["Date"])
    elif SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])
    else:
        df = generate_sample_data()
        df.to_csv(SAMPLE_PATH, index=False)
    return df


def save_processed(df: pd.DataFrame, name: str = "processed.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = generate_sample_data()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample dataset: {SAMPLE_PATH} ({len(df)} rows)")
    print(f"Stores: {df.Store.nunique()}  |  Date range: {df.Date.min()} → {df.Date.max()}")
    print(f"Mean daily sales (open stores): {df[df.Open==1].Sales.mean():.0f}")
    print(df.head())
