"""Utility to compute handcrafted memory features."""
from __future__ import annotations

import pandas as pd


def add_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["rolling_mean"] = df["close"].rolling(10).mean()
    df["rolling_std"] = df["close"].rolling(10).std()
    df["momentum"] = df["close"].diff(10)
    df["volume_mean"] = df["volume"].rolling(10).mean()
    df["volume_std"] = df["volume"].rolling(10).std()
    df["high_low"] = df["high"] - df["low"]
    df["open_close"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["open"]
    df["cumulative_return"] = df["return"].cumsum()
    df = df.dropna()
    return df
