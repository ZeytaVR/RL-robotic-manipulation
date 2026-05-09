"""SAC + HER agent for goal-conditioned manipulation."""
import torch
import torch.nn.functional as F
import numpy as np

from agents.networks import Actor, Critic
from agents.normalizer import Normalizer


class SACAgent:
    def __init__(
        self,
        obs_dim, goal_dim, action_dim,
        action_low, action_high,
        hidden_dim=256,
        gamma=0.98, tau=0.005,
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        target_entropy=None,
        device="cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        state_dim = obs_dim + goal_dim
        self.actor     = Actor(state_dim, action_dim, hidden_dim, action_low, action_high).to(device)
        self.q1        = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q2        = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt    = torch.optim.Adam(self.q1.parameters(),    lr=critic_lr)
        self.q2_opt    = torch.optim.Adam(self.q2.parameters(),    lr=critic_lr)

        # Auto-tuned entropy
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.tensor(np.log(0.2), device=device)  # FIXED alpha=0.2, no gradient
        self.alpha_opt = None  # unused; kept for clarity

        self.obs_normalizer  = Normalizer(obs_dim,  device=device)
        self.goal_normalizer = Normalizer(goal_dim, device=device)

        # For Q-target clipping (HER stability)
        self.q_clip_min = -1.0 / (1.0 - gamma)
        self.q_clip_max = 0.0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _state(self, obs, goal):
        obs_n  = self.obs_normalizer.normalize(obs)
        goal_n = self.goal_normalizer.normalize(goal)
        return torch.cat([obs_n, goal_n], dim=-1)

    def select_action(self, obs, goal, deterministic=False):
        obs  = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(self.device)
        goal = torch.from_numpy(np.asarray(goal)).float().unsqueeze(0).to(self.device)
        state = self._state(obs, goal)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, batch):
        obs      = batch['obs']
        next_obs = batch['next_obs']
        actions  = batch['actions']
        goal     = batch['desired_goal']
        rewards  = batch['rewards'].unsqueeze(-1)
        dones    = batch['dones'].unsqueeze(-1)

        state      = self._state(obs, goal)
        next_state = self._state(next_obs, goal)  # goal stays constant within episode

        # ---- Critic ----
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            tq1 = self.q1_target(next_state, next_action)
            tq2 = self.q2_target(next_state, next_action)
            target_v = torch.min(tq1, tq2) - self.alpha * next_log_pi
            target = rewards + self.gamma * (1 - dones) * target_v
            target = target.clamp(self.q_clip_min, self.q_clip_max)

        q1 = self.q1(state, actions)
        q2 = self.q2(state, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # ---- Actor ----
        new_action, log_pi = self.actor.sample(state)
        q_new = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ---- Target nets ----
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            'q1_loss':    q1_loss.item(),
            'q2_loss':    q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': 0.0,
            'alpha':      self.alpha.item(),
            'q1_mean':    q1.mean().item(),
            'log_pi_mean': log_pi.mean().item(),
        }

    def _soft_update(self, source, target):
        with torch.no_grad():
            for s, t in zip(source.parameters(), target.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)

    def update_normalizers(self, episode):
        """Call after each rollout to update obs/goal running stats."""
        self.obs_normalizer.update(episode['obs'])
        self.goal_normalizer.update(episode['desired_goal'])