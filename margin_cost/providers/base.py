"""Data provider abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Common interface for all data sources.

    Every provider must return DataFrames with the same column schema
    so that calculators are completely source-agnostic.

    Required columns per method
    ---------------------------
    stock_price()
        date, close, high, low, avg_price,
        Trading_Volume, Trading_money

    margin_balance()
        date, margin_balance, short_balance,
        margin_buy, short_sell

    institutional_net()
        date, foreign_net, foreign_buy,
        trust_net,   trust_buy,
        dealer_net,  dealer_buy

    foreign_shareholding()
        date, foreign_holding_shares

    dividends()
        date, cash_dividend, stock_dividend_ratio
    """

    @abstractmethod
    def stock_price(
        self, stock_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame: ...

    @abstractmethod
    def margin_balance(
        self, stock_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame: ...

    @abstractmethod
    def institutional_net(
        self, stock_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame: ...

    @abstractmethod
    def foreign_shareholding(
        self, stock_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame: ...

    @abstractmethod
    def dividends(
        self, stock_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame: ...
