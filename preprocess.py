"""Data preprocessing module for XAU/USD 5-minute dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def preprocess_raw_data(raw_path: Path, save_dir: Path) -> Path:
    """Clean raw parquet data and save the processed version.

    Args:
        raw_path: Path to the raw parquet file.
        save_dir: Directory to save the cleaned file.

    Returns:
        Path to the cleaned parquet file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    df = pd.read_parquet(raw_path)
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)

    # Remove weekend gaps (Fri 21:00 to Sun 21:00 UTC)
    df = df[~((df.index.weekday == 4) & (df.index.hour >= 21))]
    df = df[~((df.index.weekday == 5) | (df.index.weekday == 6))]
    df = df[~((df.index.weekday == 0) & (df.index.hour < 21))]

    # Detect extreme spikes
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    atr = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    price_change = df["typical_price"].diff().abs()
    extreme_mask = price_change > (10 * atr)
    df = df[~extreme_mask]

    df = df.ffill()

    save_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = save_dir / "xauusd_5m_clean.parquet"
    df.to_parquet(cleaned_path)
    logger.info("Saved cleaned data to %s", cleaned_path)
    return cleaned_path


if __name__ == "__main__":
    RAW_PATH = Path("data/raw/XAUUSD-5m-2022-Present.parquet")
    SAVE_DIR = Path("data/pre-processed")
    preprocess_raw_data(RAW_PATH, SAVE_DIR)
