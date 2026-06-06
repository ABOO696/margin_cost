"""Static matplotlib chart: price + cost lines (upper) and balance bars (lower)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_margin_cost(
    df: pd.DataFrame,
    title: str = "Margin / Short Average Cost",
    date_col: str = "date",
    close_col: str = "close",
    margin_cost_col: str = "avg_margin_cost",
    short_cost_col: str = "avg_short_cost",
    margin_balance_col: str = "margin_balance",
    short_balance_col: str = "short_balance",
) -> Figure:
    """Two-panel figure: price + cost lines on top, balances on the bottom."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    fig, (ax_price, ax_bal) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_price.set_title(title, fontsize=14, fontweight="bold")
    _plot_price_panel(ax_price, data, date_col, close_col, margin_cost_col, short_cost_col)
    _plot_balance_panel(ax_bal, data, date_col, margin_balance_col, short_balance_col)
    fig.tight_layout()
    return fig


def _plot_price_panel(
    ax: Axes,
    data: pd.DataFrame,
    date_col: str,
    close_col: str,
    margin_cost_col: str,
    short_cost_col: str,
) -> None:
    dates = data[date_col]
    close = data[close_col]

    ax.plot(dates, close, color="#1f1f1f", linewidth=1.6, label="Close")

    if margin_cost_col in data.columns:
        mc = data[margin_cost_col]
        ax.plot(dates, mc, color="#d62728", linewidth=1.4,
                linestyle="--", label="Margin Long Cost")
        ax.fill_between(dates, close, mc, where=(close < mc),
                        color="#d62728", alpha=0.12, interpolate=True)

    if short_cost_col in data.columns:
        ax.plot(dates, data[short_cost_col], color="#2ca02c",
                linewidth=1.4, linestyle="--", label="Margin Short Cost")

    ax.set_ylabel("Price (TWD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_balance_panel(
    ax: Axes,
    data: pd.DataFrame,
    date_col: str,
    margin_balance_col: str,
    short_balance_col: str,
) -> None:
    dates = data[date_col]

    if margin_balance_col in data.columns:
        ax.bar(dates, data[margin_balance_col], color="#d62728",
               alpha=0.5, label="Margin Balance")
    if short_balance_col in data.columns:
        ax.bar(dates, data[short_balance_col], color="#2ca02c",
               alpha=0.5, label="Short Balance")

    ax.set_ylabel("Balance (shares)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
