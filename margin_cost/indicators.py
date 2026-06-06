"""Deviation and profit/loss status indicators for margin and institutional costs."""
from __future__ import annotations

import pandas as pd

_INF = float("inf")
_LONG_LABELS  = ["trapped", "neutral", "profit"]   # long:  price < cost = trapped
_SHORT_LABELS = ["profit",  "neutral", "trapped"]  # short: price > cost = trapped


def _classify(deviation: pd.Series, labels: list[str]) -> pd.Categorical:
    return pd.cut(deviation, bins=[-_INF, -0.05, 0.05, _INF], labels=labels)


def add_cost_deviation(
    df: pd.DataFrame,
    price_col: str = "close",
    cost_col: str = "avg_margin_cost",
) -> pd.DataFrame:
    """Add margin long deviation pct and status (trapped / neutral / profit)."""
    result = df.copy()
    dev = (result[price_col] - result[cost_col]) / result[cost_col]
    result["margin_deviation_pct"] = dev
    result["margin_status"] = _classify(dev, _LONG_LABELS)
    return result


def add_short_deviation(
    df: pd.DataFrame,
    price_col: str = "close",
    cost_col: str = "avg_short_cost",
) -> pd.DataFrame:
    """Add margin short deviation pct and status (direction reversed)."""
    result = df.copy()
    dev = (result[price_col] - result[cost_col]) / result[cost_col]
    result["short_deviation_pct"] = dev
    result["short_status"] = _classify(dev, _SHORT_LABELS)
    return result


def add_institutional_status(
    df: pd.DataFrame,
    price_col: str = "close",
    cost_cols: tuple[str, ...] = ("foreign_cost", "trust_cost", "dealer_cost"),
) -> pd.DataFrame:
    """Add deviation pct and status for each institutional cost column."""
    result = df.copy()
    for col in cost_cols:
        if col not in result.columns:
            continue
        dev = (result[price_col] - result[col]) / result[col]
        result[f"{col}_dev_pct"] = dev
        result[f"{col}_status"] = _classify(dev, _LONG_LABELS)
    return result
