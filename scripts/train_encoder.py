"""Supervised training for the forecast encoder."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.forecast_encoder import ForecastEncoder
from models.memory_features import add_memory_features


def load_dataset(path: Path, seq_len: int, horizon: int) -> Tuple[TensorDataset, TensorDataset]:
    df = pd.read_parquet(path)
    df = add_memory_features(df)
    targets = np.sign(df["close"].shift(-horizon) - df["close"])
    targets = targets.replace({-1.0: 0.0, 0.0: 1.0, 1.0: 2.0})
    df = df.iloc[:-horizon]
    data = torch.tensor(df.values, dtype=torch.float32)
    labels = torch.tensor(targets.iloc[:-horizon].values, dtype=torch.long)
    sequences = data.unfold(0, seq_len, 1).permute(0, 2, 1)
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels[seq_len - 1 :], test_size=0.2, random_state=42)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    return train_ds, val_ds


def train_encoder(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    torch.manual_seed(42)
    raw_path = Path("data/pre-processed/xauusd_5m_clean.parquet")
    train_ds, val_ds = load_dataset(raw_path, args.seq_len, args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = ForecastEncoder(input_dim=train_ds.tensors[0].shape[-1])
    criterion = nn.CrossEntropyLoss()
    brier = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            out = model(X)
            loss = criterion(out["direction_logits"], y) + 0.2 * brier(out["confidence"], (y != 1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X)
                preds = out["direction_logits"].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        logger.info("Epoch %d validation accuracy: %.4f", epoch, acc)
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/encoder_pretrained.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    train_encoder(args)
