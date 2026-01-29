import torch
import torch.nn as nn
import math
import torch.nn.utils as utils
from torch.distributions import Normal

# Encoder Nets

class LayerNormMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim) 
        )
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, state_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormMLP(state_dim, hidden),
            LayerNormMLP(hidden, hidden),
            LayerNormMLP(hidden, hidden),
            nn.Linear(hidden, z_dim)
        )
    def forward(self, s): return self.net(s)

class EnsembledValueFcuntion(nn.Module):
    def __init__(self, state_dim, z_dim, hidden):
        super().__init__()
        self.encoder1 = Encoder(state_dim, z_dim, hidden)
        self.encoder2 = Encoder(state_dim, z_dim, hidden)

    def forward(self, s, g):
        z_s1 = self.encoder1(s)
        z_g1 = self.encoder1(g)
        z_s2 = self.encoder2(s)
        z_g2 = self.encoder2(g)
        
        dist1 = torch.norm(z_s1 - z_g1, p=2, dim=-1)
        dist2 = torch.norm(z_s2 - z_g2, p=2, dim=-1)
        return -dist1, -dist2


class SkillPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim=512):
        super().__init__()
        # Input: State + Skill
        input_dim = state_dim + z_dim
        
        # Shared trunk
        self.trunk = nn.Sequential(
            LayerNormMLP(input_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim)
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Log std as a learnable parameter, initialized to 0
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s, z, temperature=1.0):
        x = torch.cat([s, z], dim=-1)
        x = self.trunk(x)
        mu = self.mean(x)
        
        # Clamp log_std for stability (as per HILP reference)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = log_std.exp().expand_as(mu) * temperature
        return Normal(mu, std)

    @torch.no_grad()
    def sample_actions(self, s, z, temperature, deterministic=False):
        """
        Use this method when interacting with the environment.
        """
        dist = self(s, z, temperature=temperature)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            
        # CRITICAL: HILP manually clips actions to [-1, 1]
        return torch.clamp(action, -1.0, 1.0)

class SkillCritic(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim=512):
        super().__init__()
        # Input: State + Action + Skill
        input_dim = state_dim + action_dim + z_dim
        
        self.net = nn.Sequential(
            LayerNormMLP(input_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, a, z):
        x = torch.cat([s, a, z], dim=-1)
        return self.net(x)

class SkillValueFunction(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_dim=512):
        super().__init__()
        # Input: State + Skill
        input_dim = state_dim + z_dim
        
        self.net = nn.Sequential(
            LayerNormMLP(input_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim),
            LayerNormMLP(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        return self.net(x)