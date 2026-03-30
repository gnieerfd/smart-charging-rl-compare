import torch
import torch.nn as nn
import numpy as np
import random

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # Policy head (mean)
        self.mu = nn.Linear(hidden_dim, act_dim)
        # Log std as a parameter (per-action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.mu(h), self.log_std, self.value(h)

# -----------------------------------------------------------------------------
# SAC Networks & Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = torch.exp(log_std)
        return mu, std

class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2

# -----------------------------------------------------------------------------
# MODRL Networks
# -----------------------------------------------------------------------------
class MODRLActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        self.value_peak = nn.Linear(hidden_dim, 1)
        self.value_variance = nn.Linear(hidden_dim, 1)
        self.value_time = nn.Linear(hidden_dim, 1)
        self.value_combined = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        v_peak = self.value_peak(h)
        v_variance = self.value_variance(h)
        v_time = self.value_time(h)
        v_combined = self.value_combined(h)
        return mu, log_std, (v_peak, v_variance, v_time, v_combined)
