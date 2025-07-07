"""Reinforcement learning training pipeline for Nexus 2.0."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import TensorDataset
import numpy as np
import torch

from config import TrainingConfig
from environment import TradingEnv, load_data
from models.forecast_encoder import ForecastEncoder
from models.memory_features import add_memory_features
from models.sac_agent import SACAgent, SACConfig, set_seed


def create_dataset(df: np.ndarray, seq_len: int) -> TensorDataset:
    data = torch.tensor(df, dtype=torch.float32)
    sequences = data.unfold(0, seq_len, 1).permute(0, 2, 1)
    return TensorDataset(sequences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TrainingConfig.rl_epochs)
    parser.add_argument("--fast-debug", action="store_true")
    parser.add_argument("--joint-finetune", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    set_seed(TrainingConfig.seed)
    raw = Path("data/pre-processed/xauusd_5m_clean.parquet")
    df = load_data(raw)
    df = add_memory_features(df)
    dataset = create_dataset(df.values, TrainingConfig.seq_len)
    env = TradingEnv(df, TrainingConfig.seq_len)

    encoder = ForecastEncoder(input_dim=dataset.tensors[0].shape[-1])
    encoder.load_state_dict(torch.load("models/encoder_pretrained.pt"))
    for param in encoder.parameters():
        param.requires_grad = False
    # forecast_state = last hidden (128) + volatility + confidence = 130 dims
    state_dim = encoder.gru.hidden_size + 2  # last hidden + vol/conf
    agent = SACAgent(state_dim, 1, SACConfig(device=TrainingConfig.device))

    steps_per_episode = 800 if args.fast_debug else len(env)
    step_counter = 0

    for epoch in range(1, args.epochs + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done and step_counter < steps_per_episode:
            with torch.no_grad():
                enc_out = encoder(torch.tensor(state[None, :, :], dtype=torch.float32))
            action = agent.act(enc_out["forecast_state"].numpy().squeeze())
            next_state, reward, done, info = env.step(action)
            next_enc_out = encoder(torch.tensor(next_state[None, :, :], dtype=torch.float32))
            agent.buffer.push((enc_out["forecast_state"].numpy(), action, reward, next_enc_out["forecast_state"].numpy(), done))
            agent.update()
            episode_reward += reward
            step_counter += 1
            state = next_state
            if step_counter % 500 == 0:
                logger.info("Step %d reward %.4f", step_counter, episode_reward)
        torch.save(agent.actor.state_dict(), f"models/checkpoints/epoch_{epoch:02d}.pt")
        logger.info("Epoch %d finished with reward %.4f", epoch, episode_reward)

    if args.joint_finetune:
        for param in encoder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(list(agent.actor.parameters()) + list(encoder.parameters()), lr=1e-4)
        for _ in range(args.epochs):
            state = env.reset()
            done = False
            while not done:
                enc_out = encoder(torch.tensor(state[None, :, :], dtype=torch.float32))
                action = agent.act(enc_out["forecast_state"].numpy().squeeze())
                next_state, reward, done, _ = env.step(action)
                next_enc_out = encoder(torch.tensor(next_state[None, :, :], dtype=torch.float32))
                agent.buffer.push((enc_out["forecast_state"].numpy(), action, reward, next_enc_out["forecast_state"].numpy(), done))
                agent.update()
                # hybrid loss placeholder
                optimizer.zero_grad()
                loss = torch.tensor(0.0)
                loss.backward()
                optimizer.step()
                state = next_state
        logger.info("Joint fine-tune completed")


if __name__ == "__main__":
    main()
