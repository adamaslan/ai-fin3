"""
percent_change_utils.py

Reusable utilities to compute percent changes for price time-series over many horizons.

Features:
- Support human-friendly horizon strings (e.g. '5m','15m','1h','4h','1d','1w','1M','1y', 'all').
- Uses pandas merge_asof to find past prices for irregular timestamps.
- Works with pd.Series (DatetimeIndex) or pd.DataFrame with a chosen column.
- Returns a DataFrame where each column is the percent-change for a horizon.

Design notes:
- For monthly/yearly horizons, uses pandas DateOffset so subtraction is calendar-aware.
- "all" horizon computes pct change vs the first available price.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

Horizon = Union[str, pd.Timedelta, pd.DateOffset]


DEFAULT_HORIZONS: List[str] = [
    "1d",
    "2d",
    "3d",
    "1w",
    "2w",
    "1M",
    "3M",
    "6M",
    "1y",
    "2y",
    "3y",
    "5y",
    "all",
]


def _to_offset(h: Horizon) -> Union[pd.Timedelta, pd.DateOffset, str]:
    """Parse horizon into pd.Timedelta or pd.DateOffset.

    Accepts strings like '5m','15min','1h','1d','1w','1M','3M','1y','all'.
    Returns 'all' sentinel for that label.
    """
    if isinstance(h, (pd.Timedelta, pd.DateOffset)):
        return h
    if not isinstance(h, str):
        raise ValueError("horizon must be a str or pandas offset/timedelta")
    s = h.strip().lower()
    if s == "all":
        return "all"

    m = re.match(r"^(\d+)\s*(m|min|minute|minutes|h|hour|hours|d|day|days|w|week|weeks|mo|month|months|mth|y|yr|year|years|M)$", s)
    if not m:
        # support plain '1M' or '3M' uppercase months
        m2 = re.match(r"^(\d+)\s*([mMhHdDwWyY])$", h.strip())
        if m2:
            qty = int(m2.group(1))
            unit = m2.group(2).lower()
        else:
            raise ValueError(f"Unrecognized horizon: {h}")
    else:
        qty = int(m.group(1))
        unit = m.group(2).lower()

    # map units
    if unit in ("m", "min", "minute", "minutes"):
        return pd.Timedelta(minutes=qty)
    if unit in ("h", "hour", "hours"):
        return pd.Timedelta(hours=qty)
    if unit in ("d", "day", "days"):
        return pd.Timedelta(days=qty)
    if unit in ("w", "week", "weeks"):
        return pd.Timedelta(weeks=qty)
    if unit in ("mo", "month", "months", "mth") or unit == "m":
        # careful: 'm' was minutes earlier; handled above. This branch is defensive.
        return pd.DateOffset(months=qty)
    if unit in ("M",):
        return pd.DateOffset(months=qty)
    if unit in ("y", "yr", "year", "years"):
        return pd.DateOffset(years=qty)

    raise ValueError(f"Unsupported horizon unit: {unit} for {h}")


def _normalize_series(prices: Union[pd.Series, pd.DataFrame], column: Optional[str] = None) -> pd.Series:
    """Return a pd.Series of prices indexed by a DatetimeIndex.

    - If a DataFrame is given, `column` must be provided (or the first numeric column will be used).
    - Ensures index is DatetimeIndex, sorted ascending.
    """
    if isinstance(prices, pd.DataFrame):
        if column is None:
            # pick first numeric column
            numeric = prices.select_dtypes(include=["number"]).columns
            if len(numeric) == 0:
                raise ValueError("DataFrame provided but no numeric column found; pass `column`")
            column = numeric[0]
        s = prices[column].copy()
    elif isinstance(prices, pd.Series):
        s = prices.copy()
    else:
        raise ValueError("prices must be a pandas Series or DataFrame")

    if not isinstance(s.index, pd.DatetimeIndex):
        # try to find a datetime column and set as index
        if isinstance(prices, pd.DataFrame) and "time" in prices.columns:
            df = prices.copy()
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
            s = df[column].copy()
        else:
            raise ValueError("prices must have a DatetimeIndex or a 'time' column with datetimes")

    s = s.sort_index()
    # drop NA prices
    s = s[~s.index.duplicated(keep="first")]
    return s


def percent_changes(
    prices: Union[pd.Series, pd.DataFrame],
    horizons: Optional[Iterable[Horizon]] = None,
    column: Optional[str] = None,
    as_pct: bool = True,
) -> pd.DataFrame:
    """Compute percent changes for the supplied horizons.

    Parameters
    - prices: pd.Series (DatetimeIndex) or pd.DataFrame with a datetime index or a 'time' column.
    - horizons: iterable of horizons (strings like '5m','1h','1M','1y' or pd.Timedelta/pd.DateOffset). If None, DEFAULT_HORIZONS is used.
    - column: when prices is a DataFrame, the column name to use. If None it will pick the first numeric column.
    - as_pct: if True returns percentage (e.g., 1.5 means +1.5%), else returns fractional change (0.015).

    Returns a DataFrame indexed like the input prices with columns named like 'pct_5m', 'pct_1h', 'pct_1M', 'pct_all'.

    Behavior notes:
    - Uses pandas.merge_asof to find the past price at or before the target time (t - horizon). This handles irregular timestamps.
    - For 'all' horizon the baseline is the first non-null price.
    - Missing baseline values produce NaN.
    """
    s = _normalize_series(prices, column=column)
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    # Prepare right DataFrame for merge_asof
    # reset_index will produce a dataframe where the first column is the index
    # (datetime) column. We don't assume it's named 'time' — derive and rename
    # it to a canonical 'time' column so later code can rely on that name.
    df_right = s.reset_index()
    time_col_right = df_right.columns[0]
    # second column holds the price values (may be 0 if series name is None)
    price_col_right = df_right.columns[1]
    df_right = df_right.rename(columns={time_col_right: "time", price_col_right: "price"})
    df_right = df_right.sort_values("time")

    # Prepare left DataFrame (now/current prices)
    df_left = s.reset_index()
    time_col_left = df_left.columns[0]
    price_col_left = df_left.columns[1]
    df_left = df_left.rename(columns={time_col_left: "time", price_col_left: "price_now"})
    df_left = df_left.sort_values("time")

    out = pd.DataFrame(index=df_left["time"])
    out.index.name = df_left["time"].name

    first_price = None
    if "all" in [str(h).lower() for h in horizons]:
        # find first valid price
        fp = s.dropna()
        first_price = fp.iloc[0] if len(fp) > 0 else None

    for h in horizons:
        label = str(h)
        offset = _to_offset(h)
        colname = f"pct_{label}"

        if isinstance(offset, str) and offset == "all":
            if first_price is None:
                out[colname] = pd.NA
            else:
                out[colname] = (df_left["price_now"] / first_price) - 1.0
        else:
            # compute target times (t - offset). For DateOffset we need elementwise subtraction.
            if isinstance(offset, pd.DateOffset):
                df_left["target_time"] = df_left["time"].apply(lambda t: t - offset)
            else:
                # Timedelta is vectorized
                df_left["target_time"] = df_left["time"] - offset

            # merge_asof: find price at or before target_time
            merged = pd.merge_asof(
                df_left.sort_values("target_time"),
                df_right.sort_values("time"),
                left_on="target_time",
                right_on="time",
                direction="backward",
            )

            past_price = merged["price"].values
            now_price = merged["price_now"].values
            
            # compute percent change, handling inf/nan explicitly
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = (now_price / past_price) - 1.0
                # replace inf/-inf with nan
                pct = np.where(np.isinf(pct), np.nan, pct)

            out[colname] = pct

    # convert to DataFrame indexed by original DatetimeIndex
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if as_pct:
        out = out * 100.0

    return out


def fetch_prices_yf(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
    column: str = "Close",
    threads: bool = False,
    progress: bool = False,
    auto_adjust: bool = True,
) -> pd.Series:
    """Fetch historical prices using yfinance and return a pd.Series of prices.

    - Attempts to import yfinance locally. If yfinance is not installed, raises ImportError with a clear message.
    - Picks `column` if present, otherwise tries common alternatives ('Close','Adj Close').
    - Ensures the returned Series has a DatetimeIndex and a sensible name.

    Example:
        s = fetch_prices_yf('AAPL', period='5d', interval='1m', column='Close')
    """
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover - import-time failure
        raise ImportError(
            "yfinance is required for fetch_prices_yf but could not be imported. "
            "Install it with `pip install yfinance` or pass a pd.Series/DataFrame directly to percent_changes."
        ) from e

    # Use yf.download which returns a DataFrame
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        threads=threads,
        progress=progress,
        auto_adjust=auto_adjust,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Prefer requested column, else try common fallbacks
    if column in df.columns:
        s = df[column].copy()
    else:
        fallbacks = ["Close", "Adj Close", "close", "adjclose", "adj_close"]
        found = False
        for c in fallbacks:
            if c in df.columns:
                s = df[c].copy()
                found = True
                break
        if not found:
            raise KeyError(f"Requested column '{column}' not found in yfinance data. Available columns: {list(df.columns)}")

    # Name the series consistently
    s.name = s.name or column.lower()

    return s


if __name__ == "__main__":
    # demo: try yfinance first, fall back to synthetic data if unavailable
    print("Running demo of percent_change_utils...")
    try:
        # try to fetch recent AAPL minute data if yfinance is installed
        s = fetch_prices_yf("AAPL", period="5d", interval="1m", column="Close")
        if s.empty:
            raise RuntimeError("yfinance returned empty dataset")
        prices = s
        print("Fetched data using yfinance; sample:")
        print(prices.head())
    except Exception as e:  # pragma: no cover - runtime fallback
        import random

        print(f"yfinance fetch failed ({e}); using synthetic data instead.")
        rng = pd.date_range("2023-01-01 09:30", periods=60 * 6, freq="min")  # 6 hours of minute bars
        vals = []
        v = 100.0
        for _ in range(len(rng)):
            v += random.gauss(0, 0.1)
            vals.append(v)
        prices = pd.Series(vals, index=rng, name="close")

    # Use all default horizons (1d through all-time)
    pct = percent_changes(prices)  # uses DEFAULT_HORIZONS
    print("\nPercent changes across standard horizons:")
    print(pct.head().round(4))
