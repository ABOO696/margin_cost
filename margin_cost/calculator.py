"""Institutional-grade margin long / short average cost calculator.

Supports three methods:
    moving_average  - forward cumulative weighted average (default, good for trends).
    back_calculate  - LIFO back-calculation from current balance (no warm-up error).
    fifo_lifo_blend - 70% FIFO + 30% LIFO blend (default weights, configurable).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from margin_cost.core import (
    adjust_cost_for_dividends,
    back_calculate_cost,
    blend_fifo_lifo,
    moving_average_cost,
)


@dataclass(frozen=True)
class MarginCostConfig:
    date_col: str = "date"
    price_col: str = "avg_price"           # VWAP from Trading_money / Trading_Volume
    high_col: str = "high"                 # renamed from max by data.py
    low_col: str = "low"                   # renamed from min by data.py
    balance_col: str = "margin_balance"
    short_balance_col: str = "short_balance"
    buy_vol_col: str = "margin_buy"        # gross daily buy volume
    short_vol_col: str = "short_sell"      # gross daily short-sell volume
    margin_cost_col: str = "avg_margin_cost"
    short_cost_col: str = "avg_short_cost"
    adjust_dividends: bool = True
    method: str = "moving_average"         # "moving_average" | "back_calculate" | "fifo_lifo_blend"
    fifo_weight: float = 0.80              # backtest-optimised: 80% FIFO + 20% LIFO


class MarginCostCalculator:
    """Compute average cost for margin long and short positions.

    Methods
    -------
    moving_average  (default)
        Forward cumulative VWAP-weighted average.
        Best for long-term cost trend visualisation.

    back_calculate  (LIFO)
        Walks backward from today balance. No warm-up error.
        Best for estimating the current trapped/breakeven price.

    fifo_lifo_blend  (70% FIFO + 30% LIFO by default)
        FIFO: oldest lots sold first -> remaining = most recent buys.
        LIFO: newest lots sold first -> remaining = oldest buys.
        Blend reflects mixed holding behaviour of actual margin traders.
        Extra output columns: avg_margin_cost_fifo, avg_margin_cost_lifo.
    """

    def __init__(self, config: MarginCostConfig | None = None) -> None:
        self._cfg = config or MarginCostConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with cost, upper-band and lower-band columns added."""
        cfg = self._cfg
        if cfg.price_col not in df.columns:
            raise ValueError(f"Missing price column: {cfg.price_col}")
        if cfg.balance_col not in df.columns:
            raise ValueError(f"Missing margin balance column: {cfg.balance_col}")

        data   = df.sort_values(cfg.date_col).reset_index(drop=True)
        prices = data[cfg.price_col].to_numpy(dtype=float)
        highs  = data[cfg.high_col].to_numpy(dtype=float) if cfg.high_col in data.columns else None
        lows   = data[cfg.low_col].to_numpy(dtype=float)  if cfg.low_col  in data.columns else None
        result = data.copy()

        dividends = self._extract_dividends(data, cfg)

        # --- Margin long ---
        margin_bal = data[cfg.balance_col].to_numpy(dtype=float)
        buy_vols   = self._buy_vols(data.get(cfg.buy_vol_col), margin_bal)
        bands      = self._calc(cfg, prices, margin_bal, buy_vols, highs, lows)

        cost = bands["cost"]
        if cfg.adjust_dividends and not dividends.empty:
            cost = adjust_cost_for_dividends(cost, data[cfg.date_col], dividends)

        result[cfg.margin_cost_col]            = cost
        result[f"{cfg.margin_cost_col}_upper"] = bands["upper"]
        result[f"{cfg.margin_cost_col}_lower"] = bands["lower"]

        # Extra FIFO and LIFO component columns (only for blend method)
        if cfg.method == "fifo_lifo_blend":
            result[f"{cfg.margin_cost_col}_fifo"] = bands.get("fifo")
            result[f"{cfg.margin_cost_col}_lifo"] = bands.get("lifo")

        # --- Margin short (optional) ---
        if cfg.short_balance_col in data.columns:
            short_bal  = data[cfg.short_balance_col].to_numpy(dtype=float)
            short_vols = self._buy_vols(data.get(cfg.short_vol_col), short_bal)
            s_bands    = self._calc(cfg, prices, short_bal, short_vols, highs, lows)
            s_cost     = s_bands["cost"]
            if cfg.adjust_dividends and not dividends.empty:
                s_cost = adjust_cost_for_dividends(s_cost, data[cfg.date_col], dividends)

            result[cfg.short_cost_col]            = s_cost
            result[f"{cfg.short_cost_col}_upper"] = s_bands["upper"]
            result[f"{cfg.short_cost_col}_lower"] = s_bands["lower"]
            if cfg.method == "fifo_lifo_blend":
                result[f"{cfg.short_cost_col}_fifo"] = s_bands.get("fifo")
                result[f"{cfg.short_cost_col}_lifo"] = s_bands.get("lifo")

        return result

    def _calc(self, cfg, prices, balances, buy_vols, highs, lows) -> dict:
        if cfg.method == "back_calculate":
            return back_calculate_cost(prices, buy_vols, balances, highs, lows)
        elif cfg.method == "fifo_lifo_blend":
            return blend_fifo_lifo(
                prices, buy_vols, balances, highs, lows,
                fifo_weight=cfg.fifo_weight,
            )
        else:
            return moving_average_cost(prices, balances, highs, lows)

    @staticmethod
    def _buy_vols(series, balances):
        import numpy as np
        if series is not None:
            return series.to_numpy(dtype=float)
        return balances.copy()

    @staticmethod
    def _extract_dividends(data: pd.DataFrame, cfg: MarginCostConfig) -> pd.DataFrame:
        needed = {"cash_dividend", "stock_dividend_ratio"}
        if not needed.issubset(data.columns):
            return pd.DataFrame()
        div = data[data["cash_dividend"] + data["stock_dividend_ratio"] > 0][
            [cfg.date_col, "cash_dividend", "stock_dividend_ratio"]
        ].rename(columns={cfg.date_col: "date"})
        return div.reset_index(drop=True)
