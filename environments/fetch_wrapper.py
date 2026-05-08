"""Wrapper around Gymnasium-Robotics Fetch envs for HER training."""
import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers envs)
import numpy as np


class FetchEnvWrapper:
    """
    Wraps a Fetch env for HER training.
    - Exposes flat dimensions for network construction
    - Exposes compute_reward for the replay buffer
    - Provides collect_episode() returning the dict shape store_episode expects
    """

    def __init__(self, env_id: str = "FetchPush-v4", seed: int | None = None):
        self.env_id = env_id
        self.env = gym.make(env_id)

        sample_obs, _ = self.env.reset(seed=seed)
        self.obs_dim = sample_obs['observation'].shape[0]
        self.goal_dim = sample_obs['achieved_goal'].shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.max_episode_steps = self.env.spec.max_episode_steps

    def reset(self, seed: int | None = None):
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Passthrough to env's compute_reward. Used by replay buffer."""
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def collect_episode(self, policy_fn=None) -> dict:
        """
        Run one full episode. If policy_fn is None, uses random actions.
        policy_fn signature: policy_fn(obs, desired_goal) -> action (np.ndarray)
        Returns dict matching store_episode's expected format.
        """
        obs_dict, _ = self.reset()
        T = self.max_episode_steps

        obs_buf      = np.zeros((T + 1, self.obs_dim),    dtype=np.float32)
        achieved_buf = np.zeros((T + 1, self.goal_dim),   dtype=np.float32)
        desired_buf  = np.zeros((T,     self.goal_dim),   dtype=np.float32)
        action_buf   = np.zeros((T,     self.action_dim), dtype=np.float32)

        obs_buf[0]      = obs_dict['observation']
        achieved_buf[0] = obs_dict['achieved_goal']

        for t in range(T):
            if policy_fn is None:
                action = self.env.action_space.sample()
            else:
                action = policy_fn(obs_dict['observation'], obs_dict['desired_goal'])
                action = np.clip(action, self.action_low, self.action_high)

            desired_buf[t] = obs_dict['desired_goal']
            action_buf[t]  = action

            obs_dict, _, terminated, truncated, _ = self.env.step(action)

            obs_buf[t + 1]      = obs_dict['observation']
            achieved_buf[t + 1] = obs_dict['achieved_goal']

            if terminated or truncated:
                T_actual = t + 1
                return {
                    'obs':           obs_buf[:T_actual + 1].copy(),
                    'achieved_goal': achieved_buf[:T_actual + 1].copy(),
                    'desired_goal':  desired_buf[:T_actual].copy(),
                    'actions':       action_buf[:T_actual].copy(),
                }

        return {
            'obs':           obs_buf,
            'achieved_goal': achieved_buf,
            'desired_goal':  desired_buf,
            'actions':       action_buf,
        }

    def close(self):
        self.env.close()