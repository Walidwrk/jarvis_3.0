"""Hyper-parameters for Nexus 2.0."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 42
    seq_len: int = 60
    horizon: int = 12
    rl_epochs: int = 3
    batch_size: int = 64
    device: str = "cpu"

