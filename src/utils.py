"""Shared utilities for the Demand Forecasting project."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")


def save_fig(name: str, fig_dir: Path = Path("artefacts")) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def plot_total_sales_over_time(df: pd.DataFrame, date_col="Date", target="Sales"):
    daily = df.groupby(date_col)[target].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily.values, linewidth=0.8, color="#216b91")
    ax.set_title("Total daily sales across all stores")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    return ax


def plot_seasonality(df: pd.DataFrame, target="Sales"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df[df[target] > 0], x="DayOfWeek", y=target, ax=axes[0], palette="pastel")
    axes[0].set_title("Sales by day of week")
    sns.boxplot(data=df[df[target] > 0], x="Month", y=target, ax=axes[1], palette="pastel")
    axes[1].set_title("Sales by month")
    return axes


def plot_promo_effect(df: pd.DataFrame, target="Sales"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df[df[target] > 0], x="Promo", y=target, ax=ax, palette="Set2")
    ax.set_title("Promo effect on sales (open stores only)")
    return ax


def plot_forecast_vs_actual(test: pd.DataFrame, pred_series: pd.Series,
                            store: int = 1, target="Sales"):
    test_store = test[test["Store"] == store].copy()
    pred_store = pred_series.loc[test_store.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test_store["Date"], test_store[target], label="Actual", linewidth=1.5, color="#374952")
    ax.plot(test_store["Date"], pred_store, label="Forecast", linewidth=1.5, color="#b05b3e", linestyle="--")
    ax.set_title(f"Store {store} — Forecast vs Actual (test set)")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.legend()
    return ax


def plot_model_comparison(metrics: dict):
    """metrics: {model_name: {mae, rmse, mape, rmspe}}"""
    df = pd.DataFrame(metrics).T.reset_index().melt(id_vars="index", var_name="metric")
    df.columns = ["model", "metric", "value"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["rmspe", "mae"]):
        sub = df[df["metric"] == metric]
        sns.barplot(data=sub, x="model", y="value", ax=ax, palette="viridis")
        ax.set_title(f"Model comparison — {metric.upper()}")
        ax.set_xlabel(""); ax.set_ylabel(metric.upper())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    return axes
