from __future__ import annotations

import pandas as pd

from margin_cost.providers import get_provider
from margin_cost.providers.base import DataProvider


def _build_provider(source: str, **kwargs) -> DataProvider:
    return get_provider(source, **kwargs)


def load_full_dataset(
    stock_id: str,
    start_date: str,
    end_date: str,
    source: str = "finmind",
    **provider_kwargs,
) -> pd.DataFrame:
    """Load and merge all required data into a single DataFrame.

    Parameters
    ----------
    stock_id  : Taiwan stock ID, e.g. "2330"
    start_date: YYYY-MM-DD
    end_date  : YYYY-MM-DD
    source    : "finmind" (default) | "goodinfo"
    provider_kwargs : passed to the provider constructor.
        FinMind  -> token="<your_token>"
        GoodInfo -> delay_range=(4.0, 8.0)
    """
    prov = _build_provider(source, **provider_kwargs)

    price        = prov.stock_price(stock_id, start_date, end_date)
    margin       = prov.margin_balance(stock_id, start_date, end_date)
    inst         = prov.institutional_net(stock_id, start_date, end_date)
    shareholding = prov.foreign_shareholding(stock_id, start_date, end_date)
    dividends    = prov.dividends(stock_id, start_date, end_date)

    merged = price.merge(margin, on="date", how="inner")

    flow_cols = ["foreign_net", "foreign_buy", "trust_net", "trust_buy",
                 "dealer_net", "dealer_buy"]
    merged = merged.merge(inst, on="date", how="left")
    merged[flow_cols] = merged[flow_cols].fillna(0.0)

    merged = merged.merge(shareholding, on="date", how="left")
    merged["foreign_holding_shares"] = merged["foreign_holding_shares"].ffill().fillna(0.0)

    merged = merged.merge(dividends, on="date", how="left")
    merged["cash_dividend"]        = merged["cash_dividend"].fillna(0.0)
    merged["stock_dividend_ratio"] = merged["stock_dividend_ratio"].fillna(0.0)

    return merged.sort_values("date").reset_index(drop=True)


def compute_longrun_avg_price(
    stock_id: str,
    years: int = 7,
    end_date: str | None = None,
    source: str = "finmind",
    **provider_kwargs,
) -> float:
    """Return the volume-weighted average price over a long historical window.

    Used as gap_anchor_price for foreign_precise cost calculation.
    """
    from datetime import date, timedelta

    ed = end_date or date.today().strftime("%Y-%m-%d")
    sd = (date.fromisoformat(ed) - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    prov  = _build_provider(source, **provider_kwargs)
    price = prov.stock_price(stock_id, sd, ed)
    if price.empty:
        return float("nan")

    total_money  = pd.to_numeric(price["Trading_money"],  errors="coerce").sum()
    total_volume = pd.to_numeric(price["Trading_Volume"], errors="coerce").sum()
    if total_volume <= 0:
        return float(price["close"].mean())

    return total_money / total_volume
