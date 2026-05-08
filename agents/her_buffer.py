import numpy as np
import torch
from typing import Callable


class HERReplayBuffer:
    """
    Episode-based replay buffer with HER (future strategy).
    Supports external/demo trajectory ingestion for Project 2+.
    """

    def __init__(
        self,
        max_episodes: int,
        max_episode_steps: int,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        compute_reward_fn: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
        her_k: int = 4,
        device: str = "cuda",
    ):
        self.max_episodes = max_episodes
        self.T = max_episode_steps
        self.compute_reward = compute_reward_fn
        self.her_k = her_k
        self.future_p = 1 - (1.0 / (1 + her_k))   # ≈ 0.8
        self.device = device

        # Pre-allocated. obs/achieved_goal have T+1 entries (start..end).
        self.obs            = np.zeros((max_episodes, self.T + 1, obs_dim),    dtype=np.float32)
        self.achieved_goal  = np.zeros((max_episodes, self.T + 1, goal_dim),   dtype=np.float32)
        self.desired_goal   = np.zeros((max_episodes, self.T,     goal_dim),   dtype=np.float32)
        self.actions        = np.zeros((max_episodes, self.T,     action_dim), dtype=np.float32)

        self.is_demo         = np.zeros(max_episodes, dtype=bool)
        self.episode_lengths = np.zeros(max_episodes, dtype=np.int32)
        self.n_episodes      = 0
        self.write_idx       = 0  # circular

    def store_episode(self, episode: dict, is_demo: bool = False):
        """
        episode keys: 'obs' (T+1, obs_dim), 'achieved_goal' (T+1, goal_dim),
                      'desired_goal' (T, goal_dim), 'actions' (T, action_dim)
        """
        idx = self.write_idx
        T = episode['actions'].shape[0]
        assert T <= self.T, f"Episode length {T} > max {self.T}"

        self.obs[idx, :T + 1]           = episode['obs']
        self.achieved_goal[idx, :T + 1] = episode['achieved_goal']
        self.desired_goal[idx, :T]      = episode['desired_goal']
        self.actions[idx, :T]           = episode['actions']
        self.is_demo[idx]               = is_demo
        self.episode_lengths[idx]       = T

        self.write_idx  = (self.write_idx + 1) % self.max_episodes
        self.n_episodes = min(self.n_episodes + 1, self.max_episodes)

    def sample(self, batch_size: int) -> dict:
        # Episode + timestep indices
        ep_idx     = np.random.randint(0, self.n_episodes, size=batch_size)
        ep_lengths = self.episode_lengths[ep_idx]
        t          = (np.random.uniform(size=batch_size) * ep_lengths).astype(np.int32)

        # HER relabel mask
        her_mask = np.random.uniform(size=batch_size) < self.future_p

        # Future timestep in [t+1, ep_length]
        future_offset = (np.random.uniform(size=batch_size) * (ep_lengths - t)).astype(np.int32) + 1
        future_t      = np.minimum(t + future_offset, ep_lengths)

        # Gather
        obs              = self.obs[ep_idx, t]
        next_obs         = self.obs[ep_idx, t + 1]
        achieved_next    = self.achieved_goal[ep_idx, t + 1]
        actions          = self.actions[ep_idx, t]
        original_desired = self.desired_goal[ep_idx, t]
        her_desired      = self.achieved_goal[ep_idx, future_t]

        # Relabel goals + recompute reward (vectorized, no env calls)
        desired = np.where(her_mask[:, None], her_desired, original_desired)
        rewards = self.compute_reward(achieved_next, desired, {})

        # Sparse-reward HER: dones typically 0; SAC handles via gamma + time limit
        dones = np.zeros(batch_size, dtype=np.float32)

        return {
            'obs':          torch.from_numpy(obs).float().to(self.device),
            'next_obs':     torch.from_numpy(next_obs).float().to(self.device),
            'actions':      torch.from_numpy(actions).float().to(self.device),
            'desired_goal': torch.from_numpy(desired).float().to(self.device),
            'achieved_next': torch.from_numpy(achieved_next).float().to(self.device),
            'rewards':      torch.from_numpy(rewards).float().to(self.device),
            'dones':        torch.from_numpy(dones).to(self.device),
        }

    def load_demonstrations(self, path: str):
        """Bulk-load demo episodes from disk. Format defined when Project 2 ships."""
        raise NotImplementedError("Implemented in Project 2 once teleop format is fixed.")

    def __len__(self):
        return self.n_episodes