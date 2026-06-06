"""Institutional investor average cost calculator.

Three methods available:
    cumulative      - forward moving weighted average (good for long-term trend).
    rolling         - N-day buy-weighted window.
    foreign_precise - TSE-anchored back-calculation for foreign investors only.
                      Uses exact ForeignInvestmentShares as the holding anchor
                      and foreign-specific buy price instead of market VWAP.
                      This is the highest-accuracy method available from public data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from margin_cost.core import (
    adjust_cost_for_dividends,
    foreign_precise_cost,
    moving_average_cost,
)


@dataclass(frozen=True)
class InstitutionalCostConfig:
    date_col: str = "date"
    price_col: str = "avg_price"
    high_col: str = "high"
    low_col: str = "low"
    method: str = "foreign_precise"     # "cumulative" | "rolling" | "foreign_precise"
    rolling_window: int = 20
    adjust_dividends: bool = True
    # gross buy volume columns (used by foreign_precise)
    foreign_buy_col: str = "foreign_buy"
    foreign_holding_col: str = "foreign_holding_shares"

    # --- foreign_precise tuning parameters ---
    # Estimated average cost of pre-history foreign positions.
    # None = use forward MA as gap fill (default).
    # Set to a known long-run average entry price to reduce MA warm-up error.
    # Example: gap_anchor_price=600.0 for TSMC if foreign avg cost is ~600 TWD.
    gap_anchor_price: float | None = None

    # Multiplicative adjustment on daily VWAP for foreign buy price.
    # Foreign tends to execute near open/close; 0.99 = assume 1% below VWAP.
    # Range: 0.97 ~ 1.03. Default 1.0 = use VWAP as-is.
    buy_price_adjustment: float = 1.0

    # Exponential half-life (trading days) for LIFO decay weighting.
    # None = pure LIFO (default). 250 = 1-year half-life.
    # Smaller value = more weight on recent buys.
    lifo_decay_halflife_days: int | None = None

    net_to_cost: Mapping[str, str] = field(default_factory=lambda: {
        "foreign_net": "foreign_cost",
        "trust_net":   "trust_cost",
        "dealer_net":  "dealer_cost",
    })
    net_to_buy: Mapping[str, str] = field(default_factory=lambda: {
        "foreign_net": "foreign_buy",
        "trust_net":   "trust_buy",
        "dealer_net":  "dealer_buy",
    })


class InstitutionalCostCalculator:
    """Compute average holding cost for institutional investors.

    Method: foreign_precise (default)
    -----------------------------------
    For foreign investors, uses:
        - exact TSE-reported holding shares as the position anchor
        - foreign-specific buy price  = (foreign buy amount proxy) / (foreign buy volume)
          Since FinMind does not provide foreign buy amount directly, we derive it as:
              foreign_buy_price[t] = avg_price[t] * (1 + daily_return_adjustment)
          approximated by: when foreign_buy > 0, price = VWAP * scaling_factor
          Best approximation: use VWAP on buy days (foreign trades near market price).
        - LIFO back-calculation from today's exact holding to reconstruct cost

    This eliminates the warm-up drift of the forward cumulative method.
    """

    def __init__(self, config: InstitutionalCostConfig | None = None) -> None:
        self._cfg = config or InstitutionalCostConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg    = self._cfg
        data   = df.sort_values(cfg.date_col).reset_index(drop=True)
        result = data.copy()

        prices = data[cfg.price_col].to_numpy(dtype=float)
        highs  = data[cfg.high_col].to_numpy(dtype=float) if cfg.high_col in data.columns else None
        lows   = data[cfg.low_col].to_numpy(dtype=float)  if cfg.low_col  in data.columns else None
        dividends = self._extract_dividends(data, cfg)

        for net_col, cost_col in cfg.net_to_cost.items():
            if net_col not in data.columns:
                continue

            net = data[net_col].to_numpy(dtype=float)

            # --- foreign_precise: only applied to foreign investors ---
            if cfg.method == "foreign_precise" and net_col == "foreign_net":
                result = self._compute_foreign_precise(
                    result, data, cfg, prices, highs, lows,
                    cost_col, dividends
                )

            elif cfg.method == "rolling":
                buy_col = cfg.net_to_buy.get(net_col)
                buy_vols = (
                    data[buy_col].to_numpy(dtype=float)
                    if buy_col and buy_col in data.columns
                    else np.where(net > 0, net, 0.0)
                )
                result[cost_col] = self._rolling_weighted_cost(prices, buy_vols, cfg.rolling_window)
                if highs is not None:
                    result[f"{cost_col}_upper"] = self._rolling_weighted_cost(highs, buy_vols, cfg.rolling_window)
                if lows is not None:
                    result[f"{cost_col}_lower"] = self._rolling_weighted_cost(lows, buy_vols, cfg.rolling_window)

            else:  # cumulative (or foreign_precise fallback for non-foreign)
                balances = np.cumsum(net)
                bands    = moving_average_cost(prices, balances, highs, lows)
                cost_arr = bands["cost"]
                if cfg.adjust_dividends and not dividends.empty:
                    cost_arr = adjust_cost_for_dividends(cost_arr, data[cfg.date_col], dividends)
                result[cost_col]            = cost_arr
                result[f"{cost_col}_upper"] = bands["upper"]
                result[f"{cost_col}_lower"] = bands["lower"]

        return result

    def _compute_foreign_precise(
        self, result, data, cfg, prices, highs, lows, cost_col, dividends
    ):
        """TSE-anchored LIFO back-calculation for foreign investors."""
        holding_col = cfg.foreign_holding_col
        buy_col     = cfg.foreign_buy_col

        if holding_col not in data.columns:
            # Fallback to cumulative if exact holding not available
            net      = data["foreign_net"].to_numpy(dtype=float)
            balances = np.cumsum(net)
            bands    = moving_average_cost(prices, balances, highs, lows)
            result[cost_col]            = bands["cost"]
            result[f"{cost_col}_upper"] = bands["upper"]
            result[f"{cost_col}_lower"] = bands["lower"]
            return result

        exact_holdings = data[holding_col].to_numpy(dtype=float)
        buy_volumes    = (
            data[buy_col].to_numpy(dtype=float)
            if buy_col in data.columns
            else np.where(data["foreign_net"].to_numpy(dtype=float) > 0,
                          data["foreign_net"].to_numpy(dtype=float), 0.0)
        )

        # Foreign-specific buy price: on buy days, foreign transacts near VWAP.
        # We use VWAP as the best single-price proxy for foreign buy price.
        # (If broker-level data becomes available, replace prices here.)
        bands    = foreign_precise_cost(
                prices, buy_volumes, exact_holdings, highs, lows,
                gap_anchor_price=cfg.gap_anchor_price,
                buy_price_adjustment=cfg.buy_price_adjustment,
                lifo_decay_halflife_days=cfg.lifo_decay_halflife_days,
            )
        cost_arr = bands["cost"]

        if cfg.adjust_dividends and not dividends.empty:
            cost_arr = adjust_cost_for_dividends(cost_arr, data[cfg.date_col], dividends)

        result[cost_col]            = cost_arr
        result[f"{cost_col}_upper"] = bands["upper"]
        result[f"{cost_col}_lower"] = bands["lower"]
        return result

    @staticmethod
    def _rolling_weighted_cost(
        prices: np.ndarray, buy_vols: np.ndarray, window: int
    ) -> np.ndarray:
        buy    = np.where(buy_vols > 0, buy_vols, 0.0)
        amount = buy * prices
        buy_sum = pd.Series(buy).rolling(window, min_periods=1).sum().to_numpy()
        amt_sum = pd.Series(amount).rolling(window, min_periods=1).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(buy_sum > 0, amt_sum / buy_sum, np.nan)

    @staticmethod
    def _extract_dividends(data: pd.DataFrame, cfg: InstitutionalCostConfig) -> pd.DataFrame:
        needed = {"cash_dividend", "stock_dividend_ratio"}
        if not needed.issubset(data.columns):
            return pd.DataFrame()
        div = data[data["cash_dividend"] + data["stock_dividend_ratio"] > 0][
            [cfg.date_col, "cash_dividend", "stock_dividend_ratio"]
        ].rename(columns={cfg.date_col: "date"})
        return div.reset_index(drop=True)
