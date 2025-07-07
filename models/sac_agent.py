"""Soft Actor-Critic agent implementation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer = []

    def push(self, transition: Tuple[np.ndarray, float, float, np.ndarray, bool]) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    device: str = "cpu"


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = mlp(state_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -action_dim
        self.actor_opt = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_opt = Adam([self.log_alpha], lr=config.alpha_lr)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size

    def act(self, state: np.ndarray) -> float:
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.actor(state_t.unsqueeze(0)).cpu().numpy()[0]
        self.actor.train()
        return float(action)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_action = self.actor(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * torch.log(torch.clamp(next_action.pow(2), min=1e-6))
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action = self.actor(states)
        q1_pi, q2_pi = self.critic(states, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * torch.log(torch.clamp(new_action.pow(2), min=1e-6)) - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (torch.log(torch.clamp(new_action.pow(2), min=1e-6)) + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().detach()

        soft_update(self.critic_target, self.critic, self.tau)
