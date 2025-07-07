"""Trading environment for XAU/USD 5-minute data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


class TradingEnv:
    """Gym-style environment for 5-minute trading."""

    def __init__(self, data: pd.DataFrame, seq_len: int, leverage: float = 1.0) -> None:
        self.data = data.reset_index(drop=True)
        self.seq_len = seq_len
        self.leverage = leverage
        self.cost = 0.0001
        self.step_idx = 0
        self.logger = logging.getLogger(__name__)

    def reset(self) -> np.ndarray:
        self.step_idx = self.seq_len
        state = self._get_state()
        return state

    def _get_state(self) -> np.ndarray:
        window = self.data.iloc[self.step_idx - self.seq_len : self.step_idx]
        return window.values.astype(np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        if self.step_idx >= len(self.data) - 1:
            raise StopIteration("End of data")
        price_now = self.data.loc[self.step_idx, "close"]
        price_next = self.data.loc[self.step_idx + 1, "close"]
        pnl = (price_next - price_now) / price_now * action * self.leverage
        reward = pnl - self.cost
        self.step_idx += 1
        done = self.step_idx >= len(self.data) - 1
        info = {"pnl": pnl}
        next_state = self._get_state()
        return next_state, float(reward), done, info

    def __len__(self) -> int:
        return len(self.data) - self.seq_len
