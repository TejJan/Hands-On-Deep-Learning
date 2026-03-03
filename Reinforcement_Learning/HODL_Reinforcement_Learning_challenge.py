import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import flappy_bird_gymnasium

import collections
from collections import deque
from gymnasium import spaces
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LidarStackWrapper(gym.Wrapper):
    def __init__(self, env, k: int = 4, max_range: float | None = None):
        super().__init__(env)
        self.k = k
        self.buffer = collections.deque(maxlen=k)

        orig_space = env.observation_space
        assert isinstance(orig_space, spaces.Box), "expected box observation space"

        self.orig_dim = orig_space.shape[0]

        if max_range is not None:
            self.max_range = float(max_range)
        else:
            if np.isfinite(orig_space.high).all():
                self.max_range = float(orig_space.high.max())
            else:
                self.max_range = 250.0

        low = np.zeros(self.k * self.orig_dim, dtype=np.float32)
        high = np.ones(self.k * self.orig_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.clip(obs, 0.0, self.max_range)
        obs = obs / self.max_range
        return obs

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        self.buffer.clear()
        for _ in range(self.k):
            self.buffer.append(obs)
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        self.buffer.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs = np.stack(obs, axis=0)
        next_obs = np.stack(next_obs, axis=0)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return obs, actions, rewards, next_obs, dones


def compute_double_dqn_loss(agent, batch, gamma: float):
    obs, actions, rewards, next_obs, dones = batch

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = agent.online(obs_t)
    q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = agent.online(next_obs_t)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_q_target = agent.target(next_obs_t)
        next_q_sa = next_q_target.gather(1, next_actions).squeeze(1)
        next_q_sa = next_q_sa * (1.0 - dones_t)
        targets = rewards_t + gamma * next_q_sa

    loss = F.smooth_l1_loss(q_sa, targets)
    return loss


HYPERPARAMS = {
    "batch_size": 64,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 30_000,
    "capacity": 100_000,
    "min_buffer": 3_000,
    "target_update_interval": 1_000,
    "steps_total": 60_000,
    "updates_per_step": 1,
    "lr": 1e-4,
}


def apply_wrappers(env):
    # apply the wrappers used by your agent
    # env = tensorwrapper()
    if isinstance(env, LidarStackWrapper):
        return env
    return LidarStackWrapper(env, k=4)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 2, hidden_sizes=(256, 256)):
        super().__init__()
        self.n_actions = n_actions
        self.online = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=hidden_sizes).to(device)
        self.target = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def train_mode(self):
        self.online.train()
        self.target.eval()

    def eval_mode(self):
        self.online.eval()
        self.target.eval()

    @torch.no_grad()
    def act(self, observation, eps: float = 0.0):
        # this method can also have some keyword arguments
        if eps > 0.0 and random.random() < eps:
            return random.randrange(self.n_actions)
        if isinstance(observation, np.ndarray):
            obs_t = torch.from_numpy(observation).float().to(device)
        else:
            obs_t = observation.to(device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        q_vals = self.online(obs_t)
        action = int(torch.argmax(q_vals, dim=1).item())
        return action


def init_model(train_env):
    # initialize your agent
    base_obs_dim = train_env.observation_space.shape[0]
    k = 4
    obs_dim = base_obs_dim * k
    n_actions = train_env.action_space.n
    agent = Agent(obs_dim=obs_dim, n_actions=n_actions)
    return agent


def train_model(agent, env):
    env = apply_wrappers(env)
    # train your agent here
    hparams = HYPERPARAMS
    buffer = ReplayBuffer(capacity=hparams["capacity"])
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=hparams["lr"])
    agent.train_mode()

    eps_start = hparams["eps_start"]
    eps_end = hparams["eps_end"]
    eps_decay = hparams["eps_decay_steps"]

    obs, info = env.reset()
    total_reward = 0.0
    episode = 0
    last_loss = None

    for step in range(hparams["steps_total"]):
        frac = min(1.0, step / eps_decay)
        eps = eps_start + frac * (eps_end - eps_start)

        action = agent.act(obs, eps=eps)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            episode += 1
            obs, info = env.reset()
            total_reward = 0.0

        if len(buffer) >= hparams["min_buffer"]:
            for _ in range(hparams["updates_per_step"]):
                batch = buffer.sample(hparams["batch_size"])
                loss = compute_double_dqn_loss(agent, batch, hparams["gamma"])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.item())

        if step > 0 and step % hparams["target_update_interval"] == 0:
            agent.target.load_state_dict(agent.online.state_dict())

    env.close()
    agent.eval_mode()
    return agent