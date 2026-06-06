"""Institutional-grade moving weighted average cost engine.



Improvements over the basic model:

  1. VWAP price  : uses turnover/volume instead of (H+L+C)/3.

  2. Dividend adj: adjusts cost basis on ex-dividend dates (cash + stock).

  3. Confidence  : returns upper/lower band built from intraday high/low.

"""

from __future__ import annotations



import numpy as np

import pandas as pd





# ---------------------------------------------------------------------------

# Price helpers

# ---------------------------------------------------------------------------



def compute_vwap(df: pd.DataFrame) -> pd.Series:

    """Daily VWAP = Trading_money / Trading_Volume (shares).



    Falls back to (max + min + close) / 3 when volume is zero.

    Accepts FinMind column names: Trading_Volume, Trading_money, max, min, close.

    """

    vol = df["Trading_Volume"].replace(0, np.nan)

    vwap = df["Trading_money"] / vol

    fallback = (df["max"] + df["min"] + df["close"]) / 3

    return vwap.fillna(fallback)





# ---------------------------------------------------------------------------

# Dividend cost-basis adjustment

# ---------------------------------------------------------------------------



def adjust_cost_for_dividends(

    cost_series: np.ndarray,

    dates: pd.Series,

    dividends: pd.DataFrame,

) -> np.ndarray:

    """Reduce cost basis on each ex-dividend date.



    Parameters

    ----------

    cost_series : running average cost array (will be copied, not mutated).

    dates       : date series aligned with cost_series.

    dividends   : DataFrame with columns [date, cash_dividend, stock_dividend_ratio].

                  cash_dividend        - TWD per share paid out.

                  stock_dividend_ratio - new shares per existing share (e.g. 0.1 = 10%).

    """

    if dividends.empty:

        return cost_series.copy()



    out = cost_series.copy()

    date_index = pd.Series(range(len(dates)), index=pd.to_datetime(dates).values)



    for _, row in dividends.iterrows():

        ex_date = pd.Timestamp(row["date"])

        if ex_date not in date_index.index:

            continue

        idx = date_index[ex_date]



        # From ex-date onwards the cost basis is adjusted downward:

        #   new_cost = (old_cost - cash_dividend) / (1 + stock_dividend_ratio)

        cash  = float(row.get("cash_dividend", 0.0))

        ratio = float(row.get("stock_dividend_ratio", 0.0))

        out[idx:] = (out[idx:] - cash) / (1.0 + ratio)



    return out





# ---------------------------------------------------------------------------

# Core engine

# ---------------------------------------------------------------------------



def moving_average_cost(

    prices: np.ndarray,

    balances: np.ndarray,

    high_prices: np.ndarray | None = None,

    low_prices: np.ndarray | None = None,

) -> dict[str, np.ndarray]:

    """Institutional-grade moving-weighted-average cost.



    Parameters

    ----------

    prices    : daily reference price (VWAP recommended).

    balances  : daily balance / position array, same length as prices.

    high_prices, low_prices : optional intraday high/low for confidence band.



    Returns

    -------

    dict with keys:

        "cost"  - centre estimate (moving weighted average).

        "upper" - upper confidence bound (built from daily high).

        "lower" - lower confidence bound (built from daily low).

    """

    if len(prices) != len(balances):

        raise ValueError("prices and balances must have the same length.")



    use_band = high_prices is not None and low_prices is not None

    n = len(balances)



    cost_out  = np.full(n, np.nan)

    upper_out = np.full(n, np.nan)

    lower_out = np.full(n, np.nan)



    # Centre estimate state

    total_cost = 0.0

    total_qty  = 0.0

    prev_bal   = 0.0



    # Band state (same logic applied to high / low prices)

    total_high = 0.0

    total_low  = 0.0



    for i in range(n):

        bal   = float(balances[i])

        delta = bal - prev_bal

        price = float(prices[i])



        if delta > 0:

            total_cost += delta * price

            total_qty  += delta

            if use_band:

                total_high += delta * float(high_prices[i])

                total_low  += delta * float(low_prices[i])



        elif delta < 0 and total_qty > 0:

            avg_c = total_cost / total_qty

            reduce = min(-delta, total_qty)

            total_cost -= reduce * avg_c

            total_qty  -= reduce

            if use_band:

                avg_h = total_high / (total_qty + reduce)

                avg_l = total_low  / (total_qty + reduce)

                total_high -= reduce * avg_h

                total_low  -= reduce * avg_l



        if total_qty > 0:

            cost_out[i]  = total_cost / total_qty

            if use_band:

                upper_out[i] = total_high / total_qty

                lower_out[i] = total_low  / total_qty

        else:

            cost_out[i] = upper_out[i] = lower_out[i] = np.nan



        prev_bal = bal



    return {"cost": cost_out, "upper": upper_out, "lower": lower_out}





# ---------------------------------------------------------------------------

# Back-calculation engine

# ---------------------------------------------------------------------------



def back_calculate_cost(

    prices: np.ndarray,

    buy_volumes: np.ndarray,

    balances: np.ndarray,

    high_prices: np.ndarray | None = None,

    low_prices: np.ndarray | None = None,

) -> dict[str, np.ndarray]:

    """Margin balance back-calculation cost (???????).



    For each day T, the method answers:

        "The current balance on day T was built by the most recent N buy days.

         What is the weighted-average price of those buy days?"



    Algorithm

    ---------

    For each day T (from oldest to newest):

        1. Take the known balance on day T.

        2. Walk backward from day T, accumulating buy_volume until the

           cumulative buy matches the current balance (LIFO assumption).

        3. The weighted-average price of those buy days is the cost estimate.



    This eliminates the warm-up / starting-point error of the forward

    moving-average method because the anchor is today's known balance,

    not an unknown historical starting position.



    Parameters

    ----------

    prices      : daily reference price array (VWAP recommended).

    buy_volumes : daily gross buy volume (only buying days count; 0 on sell days).

    balances    : daily margin balance array.

    high_prices, low_prices : optional intraday high/low for confidence band.



    Returns

    -------

    dict with keys: "cost", "upper", "lower"

    """

    if not (len(prices) == len(buy_volumes) == len(balances)):

        raise ValueError("prices, buy_volumes and balances must have the same length.")



    use_band = high_prices is not None and low_prices is not None

    n = len(balances)



    cost_out  = np.full(n, np.nan)

    upper_out = np.full(n, np.nan)

    lower_out = np.full(n, np.nan)



    for t in range(n):

        target_qty = float(balances[t])

        if target_qty <= 0:

            continue



        # Walk backward from day t accumulating buy lots until we fill target_qty

        remaining    = target_qty

        cost_sum     = 0.0

        high_sum     = 0.0

        low_sum      = 0.0



        for k in range(t, -1, -1):

            buy_qty = float(buy_volumes[k])

            if buy_qty <= 0:

                continue



            take = min(buy_qty, remaining)

            cost_sum += take * float(prices[k])

            if use_band:

                high_sum += take * float(high_prices[k])

                low_sum  += take * float(low_prices[k])

            remaining -= take



            if remaining <= 0:

                break



        filled = target_qty - remaining   # may be < target if history is too short

        if filled > 0:

            cost_out[t]  = cost_sum / filled

            if use_band:

                upper_out[t] = high_sum / filled

                lower_out[t] = low_sum  / filled



    return {"cost": cost_out, "upper": upper_out, "lower": lower_out}





# ---------------------------------------------------------------------------

# Foreign investor precision engine  (hybrid: back-calc + forward fill)

# ---------------------------------------------------------------------------



def foreign_precise_cost(

    prices: np.ndarray,

    buy_volumes: np.ndarray,

    exact_holdings: np.ndarray,

    high_prices: np.ndarray | None = None,

    low_prices: np.ndarray | None = None,

    gap_anchor_price: float | None = None,

    buy_price_adjustment: float = 1.0,

    lifo_decay_halflife_days: int | None = None,

) -> dict[str, np.ndarray]:

    """Hybrid foreign investor cost: TSE-anchored LIFO back-calc + forward MA fill.



    Tunable parameters to reduce error

    ------------------------------------

    gap_anchor_price : float | None

        Estimated average cost for pre-history positions (shares held before

        the data window starts).  When None (default), the forward MA is used

        to fill the gap - MA itself has a warm-up error.

        Best value: long-run average price of the stock over 5-10 years, or a

        known analyst estimate of foreign average entry price.

        Example: gap_anchor_price=800.0



    buy_price_adjustment : float  (default 1.0 = no adjustment)

        Multiplicative factor applied to the daily price before it is used as

        the foreign buy price.  Foreign institutions often execute near the

        open or closing auction; if they systematically buy below VWAP, set

        this slightly below 1.0.

        Range: typically 0.97 ~ 1.03.

        Example: buy_price_adjustment=0.99  (assume foreign buys 1% below VWAP)



    lifo_decay_halflife_days : int | None  (default None = pure LIFO)

        Exponential half-life (trading days) applied to buy lots when walking

        backward.  Pure LIFO gives full weight to all historical buy records

        equally, but in reality long-tenured foreign positions are partly

        recycled.  A decay factor reduces the effective contribution of very

        old buy lots, making the estimate more sensitive to recent activity.

        Smaller value = faster decay = more weight on recent buys.

        Example: lifo_decay_halflife_days=250  (1-year half-life)



    Algorithm (three-step)

    -----------------------

    Step 1 - Forward MA  : compute running MA cost as fallback gap-fill.

    Step 2 - LIFO walk   : from day T walk backward, applying optional decay,

                           accumulating (price * adj * weight) until

                           covered_qty == exact_holdings[T].

    Step 3 - Blend       : blend back-calc cost (covered) with gap cost

                           (gap_anchor_price or MA) for the uncovered portion.

    """

    if not (len(prices) == len(buy_volumes) == len(exact_holdings)):

        raise ValueError("All arrays must have the same length.")



    use_band = high_prices is not None and low_prices is not None

    n = len(exact_holdings)



    # Step 1: forward MA cost - used as gap fill when gap_anchor_price is None

    ma_bands = moving_average_cost(prices, exact_holdings, high_prices, low_prices)

    ma_cost  = ma_bands["cost"]

    ma_upper = ma_bands["upper"]

    ma_lower = ma_bands["lower"]



    # Pre-compute LIFO decay weights: weight[lag] = 0.5 ^ (lag / halflife)

    if lifo_decay_halflife_days is not None and lifo_decay_halflife_days > 0:

        decay_fn = lambda lag: 0.5 ** (lag / lifo_decay_halflife_days)

    else:

        decay_fn = lambda lag: 1.0   # pure LIFO, no decay



    cost_out  = np.full(n, np.nan)

    upper_out = np.full(n, np.nan)

    lower_out = np.full(n, np.nan)



    for t in range(n):

        target = float(exact_holdings[t])

        if target <= 0:

            continue



        # Step 2: LIFO walk backward with optional decay

        remaining    = target

        cost_wsum    = 0.0   # sum of (weight * qty * price)

        cost_wqty    = 0.0   # sum of (weight * qty)   - denominator

        high_wsum    = 0.0

        low_wsum     = 0.0



        for k in range(t, -1, -1):

            bvol = float(buy_volumes[k])

            if bvol <= 0:

                continue



            lag    = t - k

            weight = decay_fn(lag)

            take   = min(bvol, remaining)



            adj_price = float(prices[k]) * buy_price_adjustment

            cost_wsum += weight * take * adj_price

            cost_wqty += weight * take



            if use_band:

                high_wsum += weight * take * float(high_prices[k]) * buy_price_adjustment

                low_wsum  += weight * take * float(low_prices[k])  * buy_price_adjustment



            remaining -= take

            if remaining <= 0:

                break



        covered_qty = target - remaining

        gap_qty     = remaining



        # Determine gap fill price

        if gap_anchor_price is not None:

            _gap_cost  = gap_anchor_price

            _gap_upper = gap_anchor_price

            _gap_lower = gap_anchor_price

        else:

            _gap_cost  = ma_cost[t]  if not np.isnan(ma_cost[t])  else np.nan

            _gap_upper = ma_upper[t] if use_band and not np.isnan(ma_upper[t]) else _gap_cost

            _gap_lower = ma_lower[t] if use_band and not np.isnan(ma_lower[t]) else _gap_cost



        # Step 3: blend covered (back-calc) + gap portions

        if covered_qty <= 0:

            cost_out[t]  = _gap_cost

            if use_band:

                upper_out[t] = _gap_upper

                lower_out[t] = _gap_lower

            continue



        bc_cost  = cost_wsum / cost_wqty if cost_wqty > 0 else np.nan

        bc_upper = high_wsum / cost_wqty if use_band and cost_wqty > 0 else np.nan

        bc_lower = low_wsum  / cost_wqty if use_band and cost_wqty > 0 else np.nan



        if gap_qty > 0 and not np.isnan(_gap_cost):

            cost_out[t]  = (bc_cost  * covered_qty + _gap_cost  * gap_qty) / target

            if use_band:

                upper_out[t] = (bc_upper * covered_qty + _gap_upper * gap_qty) / target

                lower_out[t] = (bc_lower * covered_qty + _gap_lower * gap_qty) / target

        else:

            cost_out[t]  = bc_cost

            if use_band:

                upper_out[t] = bc_upper

                lower_out[t] = bc_lower



    return {"cost": cost_out, "upper": upper_out, "lower": lower_out}



    return {"cost": cost_out, "upper": upper_out, "lower": lower_out}



# ---------------------------------------------------------------------------
# FIFO cost engine
# ---------------------------------------------------------------------------

def fifo_cost(
    prices: np.ndarray,
    buy_volumes: np.ndarray,
    balances: np.ndarray,
    high_prices: np.ndarray | None = None,
    low_prices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """FIFO (First-In First-Out) margin cost.

    For each day T, assume the oldest buy lots are sold first.
    The remaining position therefore consists of the MOST RECENT buy lots
    that exactly fill the current balance -- which is the opposite of LIFO.

    Algorithm
    ---------
    Walk FORWARD from day 0, accumulating buy_volumes into a deque.
    On sell days, consume from the FRONT of the deque (oldest first).
    The remaining deque lots weighted average = FIFO cost on day T.

    Parameters
    ----------
    prices      : daily reference price (VWAP recommended).
    buy_volumes : daily gross buy volume.
    balances    : daily margin balance.
    high_prices, low_prices : optional for confidence band.
    """
    from collections import deque

    if not (len(prices) == len(buy_volumes) == len(balances)):
        raise ValueError("All arrays must have the same length.")

    use_band = high_prices is not None and low_prices is not None
    n = len(balances)

    cost_out  = np.full(n, np.nan)
    upper_out = np.full(n, np.nan)
    lower_out = np.full(n, np.nan)

    # Each entry: [qty, price, high, low]
    lot_queue: deque = deque()
    prev_bal = 0.0

    for i in range(n):
        bal   = float(balances[i])
        delta = bal - prev_bal

        if delta > 0:
            bvol = float(buy_volumes[i]) if buy_volumes[i] > 0 else delta
            # Push new buy lot to the BACK of queue
            lot_queue.append([
                delta,
                float(prices[i]),
                float(high_prices[i]) if use_band else 0.0,
                float(low_prices[i])  if use_band else 0.0,
            ])

        elif delta < 0:
            # Consume from FRONT of queue (oldest lots sold first)
            to_sell = -delta
            while to_sell > 0 and lot_queue:
                front = lot_queue[0]
                if front[0] <= to_sell:
                    to_sell -= front[0]
                    lot_queue.popleft()
                else:
                    front[0] -= to_sell
                    to_sell = 0

        # Compute weighted average of remaining lots
        total_qty  = sum(lot[0] for lot in lot_queue)
        if total_qty > 0:
            cost_out[i]  = sum(lot[0] * lot[1] for lot in lot_queue) / total_qty
            if use_band:
                upper_out[i] = sum(lot[0] * lot[2] for lot in lot_queue) / total_qty
                lower_out[i] = sum(lot[0] * lot[3] for lot in lot_queue) / total_qty

        prev_bal = bal

    return {"cost": cost_out, "upper": upper_out, "lower": lower_out}


# ---------------------------------------------------------------------------
# FIFO / LIFO blend
# ---------------------------------------------------------------------------

def blend_fifo_lifo(
    prices: np.ndarray,
    buy_volumes: np.ndarray,
    balances: np.ndarray,
    high_prices: np.ndarray | None = None,
    low_prices: np.ndarray | None = None,
    fifo_weight: float = 0.80,
) -> dict[str, np.ndarray]:
    """Weighted blend of FIFO and LIFO (back-calculate) margin cost.

    Default: 70% FIFO + 30% LIFO.

    Rationale
    ---------
    Pure LIFO reflects the most recently-entered trapped positions.
    Pure FIFO reflects the oldest (likely lower-cost) positions still held.
    A 70/30 blend acknowledges that most margin traders do not strictly
    follow either rule -- they mix short-term punts with carry-over positions.

    Parameters
    ----------
    fifo_weight : weight for FIFO component (0.0~1.0). LIFO weight = 1 - fifo_weight.
    """
    lifo_weight = 1.0 - fifo_weight

    fifo_bands = fifo_cost(prices, buy_volumes, balances, high_prices, low_prices)
    lifo_bands = back_calculate_cost(prices, buy_volumes, balances, high_prices, low_prices)

    def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full(len(a), np.nan)
        for i in range(len(a)):
            if not np.isnan(a[i]) and not np.isnan(b[i]):
                out[i] = fifo_weight * a[i] + lifo_weight * b[i]
            elif not np.isnan(a[i]):
                out[i] = a[i]
            elif not np.isnan(b[i]):
                out[i] = b[i]
        return out

    return {
        "cost":  blend(fifo_bands["cost"],  lifo_bands["cost"]),
        "upper": blend(fifo_bands["upper"], lifo_bands["upper"]),
        "lower": blend(fifo_bands["lower"], lifo_bands["lower"]),
        "fifo":  fifo_bands["cost"],
        "lifo":  lifo_bands["cost"],
    }
