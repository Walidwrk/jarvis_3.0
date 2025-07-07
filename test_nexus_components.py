"""Pytest suite for Nexus components."""
from __future__ import annotations

import pytest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.forecast_encoder import ForecastEncoder
from models.sac_agent import SACAgent, SACConfig
from environment import TradingEnv
from preprocess import preprocess_raw_data


def test_preprocess_produces_data(tmp_path: Path) -> None:
    raw = Path("data/raw/XAUUSD-5m-2022-Present.parquet")
    if not raw.exists():
        pytest.skip("Raw data not available")
    out = preprocess_raw_data(raw, tmp_path)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 200_000


def test_encoder_forward_shape() -> None:
    model = ForecastEncoder()
    x = torch.randn(4, 60, 5)
    out = model(x)
    assert out["forecast_state"].shape[1] == 128 + 1 + 1


def test_sac_action_range() -> None:
    agent = SACAgent(10, 1, SACConfig())
    action = agent.act(np.random.randn(10))
    assert -1.0 <= action <= 1.0


def test_env_single_episode() -> None:
    df = pd.DataFrame({"open": [1, 1.1, 1.2, 1.3], "high": [1, 1.1, 1.2, 1.3], "low": [1, 1.1, 1.2, 1.3], "close": [1, 1.1, 1.2, 1.3], "volume": [1, 1, 1, 1]})
    env = TradingEnv(df, seq_len=2)
    _ = env.reset()
    _, reward, done, _info = env.step(0.5)
    assert isinstance(reward, float)


def test_encoder_not_neutral() -> None:
    model = ForecastEncoder()
    x = torch.randn(16, 60, 5)
    out = model(x)
    probs = out["direction_logits"].softmax(dim=1).max(dim=1)[0]
    assert probs.mean().item() > 0.4
