from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path, date_col="Date", close_col="Close") -> pd.Series:
    df = pd.read_csv(path)
    lookup = {str(c).lower(): c for c in df.columns}
    dcol = lookup.get(date_col.lower(), date_col)
    ccol = lookup.get(close_col.lower(), close_col)
    if dcol not in df or ccol not in df:
        raise ValueError(f"CSV needs {date_col!r} and {close_col!r}; got {list(df)}")
    idx = pd.to_datetime(df[dcol], utc=True, errors="raise").dt.tz_localize(None)
    s = pd.Series(
        pd.to_numeric(df[ccol], errors="raise").values, index=idx, name="close"
    )
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if len(s) < 300 or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("CSV must contain at least 300 finite, positive closes")
    return s


def download_yahoo(ticker: str, start="2014-01-01", end=None) -> pd.Series:
    return download_yahoo_universe([ticker], start, end).iloc[:, 0].rename("close")


def download_yahoo_universe(
    tickers: list[str], start="1998-01-01", end=None
) -> pd.DataFrame:
    """Adjusted-close research panel. Yahoo is not an institutional data source."""
    import yfinance as yf

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df.empty:
        raise RuntimeError("No Yahoo data; use a vendor-vetted wide CSV")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if len(tickers) == 1:
        close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.reindex(columns=tickers).sort_index()
    missing = [c for c in tickers if close[c].dropna().empty]
    if missing:
        raise RuntimeError(f"No Yahoo history for {missing}")
    # Common history avoids silently treating pre-inception assets as cash.
    close = close[~close.index.duplicated(keep="last")].dropna()
    if len(close) < 500 or not np.isfinite(close.to_numpy()).all():
        raise RuntimeError("Insufficient or non-finite common Yahoo history")
    if (close <= 0).any().any():
        raise RuntimeError("Yahoo returned non-positive adjusted closes")
    return close
