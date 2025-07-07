"""Forecast encoder architecture with Conv1D, GRU, and Transformer."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class ForecastEncoder(nn.Module):
    """Combines CNN, GRU, and Transformer blocks."""

    def __init__(self, input_dim: int = 5, gru_hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.gru = nn.GRU(32, gru_hidden, batch_first=True, num_layers=2)
        self.pos_enc = PositionalEncoding(gru_hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=gru_hidden, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dir_head = nn.Linear(gru_hidden, 3)
        self.vol_head = nn.Linear(gru_hidden, 1)
        self.conf_head = nn.Linear(gru_hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, seq, feat = x.shape
        x = x.permute(0, 2, 1)  # (B, F, S)
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)  # (B, S, F)
        gru_out, _ = self.gru(x)
        x = self.pos_enc(gru_out)
        x = self.transformer(x)
        last = x[:, -1, :]
        direction_logits = self.dir_head(last)
        volatility = self.vol_head(last).squeeze(-1)
        confidence = torch.sigmoid(self.conf_head(last)).squeeze(-1)
        forecast_state = torch.cat([last, volatility.unsqueeze(-1), confidence.unsqueeze(-1)], dim=1)
        return {
            "direction_logits": direction_logits,
            "volatility": volatility,
            "confidence": confidence,
            "forecast_state": forecast_state,
        }
