"""FinMind data provider."""
from __future__ import annotations

import pandas as pd
from FinMind.data import DataLoader

from margin_cost.core import compute_vwap
from margin_cost.providers.base import DataProvider


class FinMindProvider(DataProvider):
    """Fetch data via FinMind open API."""

    def __init__(self, token: str | None = None) -> None:
        self._api = DataLoader()
        if token:
            self._api.login_by_token(api_token=token)

    def stock_price(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._api.taiwan_stock_daily(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        df["avg_price"] = compute_vwap(df)
        df = df.rename(columns={"max": "high", "min": "low"})
        return df[["date", "close", "high", "low", "avg_price",
                   "Trading_Volume", "Trading_money"]].copy()

    def margin_balance(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        df = df.rename(columns={
            "MarginPurchaseTodayBalance": "margin_balance",
            "ShortSaleTodayBalance":      "short_balance",
            "MarginPurchaseBuy":          "margin_buy",
            "ShortSaleSell":              "short_sell",
        })
        for col in ["margin_buy", "short_sell"]:
            if col not in df.columns:
                df[col] = 0.0
        return df[["date", "margin_balance", "short_balance",
                   "margin_buy", "short_sell"]].copy()

    def institutional_net(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._api.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        df["net"] = df["buy"] - df["sell"]
        net_piv = df.pivot_table(index="date", columns="name", values="net",  aggfunc="sum").fillna(0.0)
        buy_piv = df.pivot_table(index="date", columns="name", values="buy",  aggfunc="sum").fillna(0.0)

        def _net(c): return net_piv[c] if c in net_piv.columns else pd.Series(0.0, index=net_piv.index)
        def _buy(c): return buy_piv[c] if c in buy_piv.columns else pd.Series(0.0, index=buy_piv.index)

        return pd.DataFrame({
            "date":        net_piv.index,
            "foreign_net": _net("Foreign_Investor") + _net("Foreign_Dealer_Self"),
            "foreign_buy": _buy("Foreign_Investor") + _buy("Foreign_Dealer_Self"),
            "trust_net":   _net("Investment_Trust"),
            "trust_buy":   _buy("Investment_Trust"),
            "dealer_net":  _net("Dealer_self") + _net("Dealer_Hedging"),
            "dealer_buy":  _buy("Dealer_self") + _buy("Dealer_Hedging"),
        }).reset_index(drop=True)

    def foreign_shareholding(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self._api.taiwan_stock_shareholding(
                stock_id=stock_id, start_date=start_date, end_date=end_date
            )
        except Exception:
            return pd.DataFrame(columns=["date", "foreign_holding_shares"])
        if df.empty:
            return pd.DataFrame(columns=["date", "foreign_holding_shares"])
        out = pd.DataFrame({
            "date":                   df["date"].astype(str),
            "foreign_holding_shares": pd.to_numeric(df["ForeignInvestmentShares"], errors="coerce").fillna(0.0),
        })
        return out[out["date"].between(start_date, end_date)].reset_index(drop=True)

    def dividends(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["date", "cash_dividend", "stock_dividend_ratio"])
        try:
            div = self._api.taiwan_stock_dividend_result(
                stock_id=stock_id, start_date=start_date, end_date=end_date
            )
        except Exception:
            return empty
        if div.empty:
            return empty
        date_col  = next((c for c in div.columns if "ExDividend" in c or "date" in c.lower()), None)
        cash_col  = next((c for c in div.columns if "Cash" in c), None)
        stock_col = next((c for c in div.columns if "Stock" in c and "Dividend" in c), None)
        if not date_col:
            return empty
        out = pd.DataFrame()
        out["date"]                 = pd.to_datetime(div[date_col]).dt.strftime("%Y-%m-%d")
        out["cash_dividend"]        = pd.to_numeric(div[cash_col],  errors="coerce").fillna(0.0) if cash_col  else 0.0
        out["stock_dividend_ratio"] = pd.to_numeric(div[stock_col], errors="coerce").fillna(0.0) / 10.0        if stock_col else 0.0
        return out[out["date"].between(start_date, end_date)].reset_index(drop=True)
