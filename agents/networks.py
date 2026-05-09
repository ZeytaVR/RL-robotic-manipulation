"""Actor and Critic networks for SAC."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy."""
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim, action_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer('action_scale', torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer('action_bias',  torch.as_tensor(action_bias,  dtype=torch.float32))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # reparameterization
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        # log_prob with tanh correction
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Q(s, a) network."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)