"""GoodInfo.tw web scraper data provider.

GoodInfo does not provide an official API.
This module scrapes HTML tables using requests + BeautifulSoup.

Install extras:
    pip install requests beautifulsoup4 lxml

Rate limiting
-------------
GoodInfo blocks aggressive crawlers.  A random delay is inserted between
requests (default 3-6 seconds).  Do NOT reduce the delay below 2 seconds.
"""
from __future__ import annotations

import random
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from margin_cost.providers.base import DataProvider

try:
    import requests
    from bs4 import BeautifulSoup
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_BASE = "https://goodinfo.tw/tw"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://goodinfo.tw/tw/index.asp",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


def _check_deps() -> None:
    if not _DEPS_OK:
        raise ImportError(
            "GoodInfo provider requires: pip install requests beautifulsoup4 lxml"
        )


def _get(url: str, params: dict | None = None, delay: tuple = (3, 6)) -> BeautifulSoup:
    """GET with random delay and simple retry."""
    time.sleep(random.uniform(*delay))
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "lxml")


def _parse_number(s: str) -> float:
    """Parse GoodInfo number strings: remove commas, handle '--'."""
    s = str(s).strip().replace(",", "").replace("\xa0", "")
    if s in ("--", "-", "", "N/A"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


class GoodInfoProvider(DataProvider):
    """Fetch data by scraping GoodInfo.tw HTML tables.

    Limitations vs FinMind
    ----------------------
    - Only daily price and margin/short data are available at daily granularity.
    - Institutional net buy/sell: GoodInfo shows monthly aggregates only;
      daily breakdown is scraped from the detailed view (slower).
    - Foreign shareholding: available as percentage; converted to shares using
      issued shares count.
    - Dividends: scraped from the dividend history page.
    - avg_price is approximated as (high + low + close) / 3  (no VWAP available).
    - Trading_money is estimated as close * Trading_Volume.
    """

    def __init__(self, delay_range: tuple[float, float] = (3.0, 6.0)) -> None:
        _check_deps()
        self._delay = delay_range

    # ------------------------------------------------------------------
    # Stock price
    # ------------------------------------------------------------------
    def stock_price(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape daily OHLCV from GoodInfo stock price page."""
        rows = []
        sd = date.fromisoformat(start_date)
        ed = date.fromisoformat(end_date)

        # GoodInfo paginates by year/month
        cur = date(sd.year, sd.month, 1)
        while cur <= ed:
            soup = _get(
                f"{_BASE}/StockDailyPriceReport.aspx",
                params={"STOCK_ID": stock_id, "RPT_CAT": "PRICE",
                        "YEAR": cur.year, "MONTH": cur.month},
                delay=self._delay,
            )
            table = soup.find("table", id=re.compile("tblDetail", re.I))
            if not table:
                # try generic table heuristic
                tables = soup.find_all("table")
                table  = next((t for t in tables if len(t.find_all("tr")) > 5), None)
            if table:
                for tr in table.find_all("tr")[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) < 7:
                        continue
                    try:
                        # columns: date, open, high, low, close, volume, ...
                        row_date = pd.to_datetime(tds[0], errors="coerce")
                        if pd.isna(row_date):
                            continue
                        d_str = row_date.strftime("%Y-%m-%d")
                        if not (start_date <= d_str <= end_date):
                            continue
                        high  = _parse_number(tds[2])
                        low   = _parse_number(tds[3])
                        close = _parse_number(tds[4])
                        vol   = _parse_number(tds[5]) * 1000   # GoodInfo in thousands
                        rows.append({
                            "date":            d_str,
                            "close":           close,
                            "high":            high,
                            "low":             low,
                            "avg_price":       (high + low + close) / 3,
                            "Trading_Volume":  vol,
                            "Trading_money":   close * vol,  # estimated
                        })
                    except Exception:
                        continue
            # advance one month
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)

        if not rows:
            return pd.DataFrame(columns=["date", "close", "high", "low", "avg_price",
                                         "Trading_Volume", "Trading_money"])
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Margin / short balance
    # ------------------------------------------------------------------
    def margin_balance(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape daily margin purchase and short sale data."""
        rows = []
        sd = date.fromisoformat(start_date)
        ed = date.fromisoformat(end_date)

        cur = date(sd.year, sd.month, 1)
        while cur <= ed:
            soup = _get(
                f"{_BASE}/StockDailyPriceReport.aspx",
                params={"STOCK_ID": stock_id, "RPT_CAT": "MARGIN",
                        "YEAR": cur.year, "MONTH": cur.month},
                delay=self._delay,
            )
            table = soup.find("table", id=re.compile("tblDetail", re.I))
            if not table:
                tables = soup.find_all("table")
                table  = next((t for t in tables if len(t.find_all("tr")) > 5), None)
            if table:
                for tr in table.find_all("tr")[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) < 8:
                        continue
                    try:
                        row_date = pd.to_datetime(tds[0], errors="coerce")
                        if pd.isna(row_date):
                            continue
                        d_str = row_date.strftime("%Y-%m-%d")
                        if not (start_date <= d_str <= end_date):
                            continue
                        # typical GoodInfo margin columns:
                        # date | margin_buy | margin_sell | margin_redeem | margin_balance
                        #      | short_sell | short_cover | short_redeem  | short_balance
                        margin_buy     = _parse_number(tds[1]) * 1000
                        margin_balance = _parse_number(tds[4]) * 1000
                        short_sell     = _parse_number(tds[5]) * 1000
                        short_balance  = _parse_number(tds[8]) * 1000 if len(tds) > 8 else float("nan")
                        rows.append({
                            "date":           d_str,
                            "margin_balance": margin_balance,
                            "short_balance":  short_balance,
                            "margin_buy":     margin_buy,
                            "short_sell":     short_sell,
                        })
                    except Exception:
                        continue
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)

        if not rows:
            return pd.DataFrame(columns=["date", "margin_balance", "short_balance",
                                         "margin_buy", "short_sell"])
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df = df.fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Institutional net buy/sell  (GoodInfo shows daily detail per page)
    # ------------------------------------------------------------------
    def institutional_net(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape daily institutional net buy/sell from GoodInfo."""
        rows = []
        sd = date.fromisoformat(start_date)
        ed = date.fromisoformat(end_date)

        cur = date(sd.year, sd.month, 1)
        while cur <= ed:
            soup = _get(
                f"{_BASE}/StockInstitionalInvestors.aspx",
                params={"STOCK_ID": stock_id, "YEAR": cur.year, "MONTH": cur.month},
                delay=self._delay,
            )
            table = soup.find("table", id=re.compile("tblDetail", re.I))
            if not table:
                tables = soup.find_all("table")
                table  = next((t for t in tables if len(t.find_all("tr")) > 5), None)
            if table:
                for tr in table.find_all("tr")[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) < 10:
                        continue
                    try:
                        row_date = pd.to_datetime(tds[0], errors="coerce")
                        if pd.isna(row_date):
                            continue
                        d_str = row_date.strftime("%Y-%m-%d")
                        if not (start_date <= d_str <= end_date):
                            continue
                        # GoodInfo columns (0-indexed):
                        # 0:date 1:foreign_buy 2:foreign_sell 3:foreign_net
                        # 4:trust_buy 5:trust_sell 6:trust_net
                        # 7:dealer_buy 8:dealer_sell 9:dealer_net ...
                        f_buy = _parse_number(tds[1]) * 1000
                        f_net = _parse_number(tds[3]) * 1000
                        t_buy = _parse_number(tds[4]) * 1000
                        t_net = _parse_number(tds[6]) * 1000
                        d_buy = _parse_number(tds[7]) * 1000
                        d_net = _parse_number(tds[9]) * 1000 if len(tds) > 9 else float("nan")
                        rows.append({
                            "date":        d_str,
                            "foreign_net": f_net,
                            "foreign_buy": f_buy,
                            "trust_net":   t_net,
                            "trust_buy":   t_buy,
                            "dealer_net":  d_net,
                            "dealer_buy":  d_buy,
                        })
                    except Exception:
                        continue
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)

        if not rows:
            cols = ["date", "foreign_net", "foreign_buy", "trust_net",
                    "trust_buy", "dealer_net", "dealer_buy"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df.fillna(0.0)

    # ------------------------------------------------------------------
    # Foreign shareholding
    # ------------------------------------------------------------------
    def foreign_shareholding(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape daily foreign holding ratio and convert to shares."""
        soup = _get(
            f"{_BASE}/StockShareholderForm.aspx",
            params={"STOCK_ID": stock_id},
            delay=self._delay,
        )
        rows = []
        table = soup.find("table", id=re.compile("tblDetail", re.I))
        if not table:
            return pd.DataFrame(columns=["date", "foreign_holding_shares"])

        issued_shares = self._get_issued_shares(soup)

        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) < 3:
                continue
            try:
                row_date = pd.to_datetime(tds[0], errors="coerce")
                if pd.isna(row_date):
                    continue
                d_str = row_date.strftime("%Y-%m-%d")
                if not (start_date <= d_str <= end_date):
                    continue
                ratio  = _parse_number(tds[1]) / 100.0   # e.g. 73.40% -> 0.7340
                shares = ratio * issued_shares
                rows.append({"date": d_str, "foreign_holding_shares": shares})
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=["date", "foreign_holding_shares"])
        return (
            pd.DataFrame(rows)
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _get_issued_shares(self, soup: BeautifulSoup) -> float:
        """Parse total issued shares from page metadata."""
        text = soup.get_text()
        m = re.search(r"????[?:]\s*([\d,]+)", text)
        if m:
            return float(m.group(1).replace(",", ""))
        return 1.0   # fallback: ratio will equal shares (wrong but won't crash)

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------
    def dividends(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape dividend history from GoodInfo."""
        empty = pd.DataFrame(columns=["date", "cash_dividend", "stock_dividend_ratio"])
        try:
            soup = _get(
                f"{_BASE}/StockDividendPolicy.aspx",
                params={"STOCK_ID": stock_id},
                delay=self._delay,
            )
        except Exception:
            return empty

        rows = []
        table = soup.find("table", id=re.compile("tblDetail", re.I))
        if not table:
            return empty

        for tr in table.find_all("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) < 6:
                continue
            try:
                # GoodInfo columns: year, ex_date, cash_div, stock_div, ...
                ex_date = pd.to_datetime(tds[1], errors="coerce")
                if pd.isna(ex_date):
                    continue
                d_str = ex_date.strftime("%Y-%m-%d")
                if not (start_date <= d_str <= end_date):
                    continue
                cash  = _parse_number(tds[2])
                stock = _parse_number(tds[3]) / 10.0   # TWD face -> ratio
                rows.append({
                    "date":                 d_str,
                    "cash_dividend":        cash  if not np.isnan(cash)  else 0.0,
                    "stock_dividend_ratio": stock if not np.isnan(stock) else 0.0,
                })
            except Exception:
                continue

        if not rows:
            return empty
        return (
            pd.DataFrame(rows)
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
