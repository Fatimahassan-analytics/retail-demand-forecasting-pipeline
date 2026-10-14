"""Streamlit app — Retail Demand Forecasting.

Run locally:
    streamlit run app.py

The app lets the user pick a store and see:
- Historical daily sales
- A LightGBM 8-week-ahead forecast with confidence bands
- Model performance summary
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_data, generate_sample_data, RAW_DIR  # noqa: E402
from model import (  # noqa: E402
    temporal_split, add_date_features, add_lag_features,
    train_lightgbm, lightgbm_forecast, evaluate_forecast,
    FEATURE_COLS_LGB, save_model, load_model, RANDOM_SEED,
)

MODELS_DIR = Path(__file__).parent / "artefacts"
CACHED_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"


@st.cache_resource(show_spinner="Training LightGBM model on first run ...")
def get_or_train_model():
    if CACHED_MODEL_PATH.exists():
        return load_model("lightgbm_model", MODELS_DIR)
    df = load_data()
    df = add_date_features(df)
    df = add_lag_features(df)
    train, _ = temporal_split(df)
    model = train_lightgbm(train)
    save_model(model, "lightgbm_model", MODELS_DIR)
    return model


def main():
    st.set_page_config(page_title="Retail Demand Forecasting", page_icon=":chart_with_upwards_trend:",
                       layout="wide")
    st.title("Retail Demand Forecasting")
    st.caption("8-week-ahead demand forecast per store — LightGBM with lag features.")

    model = get_or_train_model()
    df = load_data()
    df = add_date_features(df)

    st.sidebar.header("Forecast parameters")
    stores = sorted(df["Store"].unique())
    selected_store = st.sidebar.selectbox("Store", stores, index=0)
    horizon_days = st.sidebar.slider("Forecast horizon (days)", 7, 84, 56, step=7)

    # Build the forecast: take last `horizon_days` of the selected store and re-predict
    store_df = df[df["Store"] == selected_store].sort_values("Date").copy()
    full_with_lags = add_lag_features(store_df)
    train_part, test_part = temporal_split(full_with_lags, test_days=horizon_days)

    if len(test_part) == 0:
        st.warning(f"Not enough history for store {selected_store} for an 8-week holdout.")
        return

    preds = model.predict(test_part[FEATURE_COLS_LGB])
    metrics = evaluate_forecast(test_part["Sales"], preds)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.0f}")
    col2.metric("MAPE", f"{metrics['mape']:.1f}%")
    col3.metric("RMSPE", f"{metrics['rmspe']:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train_part["Date"].tail(120), train_part["Sales"].tail(120),
            label="Recent history", color="#374952", linewidth=1.4)
    ax.plot(test_part["Date"], test_part["Sales"],
            label="Actual (test)", color="#216b91", linewidth=1.8)
    ax.plot(test_part["Date"], preds,
            label="LightGBM forecast", color="#b05b3e", linestyle="--", linewidth=1.8)
    ax.set_title(f"Store {selected_store} — {horizon_days}-day forecast vs actual")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("Recent forecast table")
    out_df = test_part[["Date", "Sales"]].copy()
    out_df["Forecast"] = preds.round(0).astype(int)
    out_df["Abs_error"] = (out_df["Sales"] - out_df["Forecast"]).abs().round(0)
    st.dataframe(out_df.head(30), use_container_width=True)


if __name__ == "__main__":
    main()
