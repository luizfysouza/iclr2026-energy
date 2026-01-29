import torch
import torch.nn as nn
import math
import torch.nn.utils as utils


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

# EBT Nets
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class EBTPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, nhead=4, num_layers=4, window_size=16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        
        # 1. Embeddings (Spectral Norm Stabilizes inputs)
        self.state_emb = utils.spectral_norm(nn.Linear(state_dim, d_model))
        self.action_emb = utils.spectral_norm(nn.Linear(action_dim, d_model))
        self.pos_encoder = SinusoidalPositionalEmbedding(d_model, max_len=window_size)
        
        # 2. Transformer Encoder
        # Note: Standard LayerNorm inside the Transformer is usually sufficient, 
        # but Spectral Norm on the feedforward blocks inside is also possible if desperate.
        # For now, standard TransformerEncoder is likely fine if input/output are controlled.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Energy Head (CRITICAL for Spectral Norm)
        # This prevents the final scalar from exploding.
        self.energy_head = utils.spectral_norm(nn.Linear(d_model, 1))

    def forward(self, state, action_chunk):
        # ... (Forward logic remains exactly the same) ...
        B, H, _ = action_chunk.shape
        state_token = self.state_emb(state).unsqueeze(1)
        action_tokens = self.action_emb(action_chunk)
        action_tokens = self.pos_encoder(action_tokens)
        seq = torch.cat([state_token, action_tokens], dim=1)
        out = self.transformer(seq)
        state_out = out[:, 0, :]
        
        # Output is (B, 1)
        return self.energy_head(state_out)

# IDM

class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        """
        Predicts action a_t given state s_t and next state s_{t+1}.
        Input: Concatenation of (s_t, s_{t+1}) -> Dimension 2 * state_dim
        Output: Action a_t -> Dimension action_dim
        """
        super().__init__()
        
        input_dim = state_dim * 2
        
        layers = []
        # Input Layer
        layers.append(LayerNormMLP(input_dim, hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(LayerNormMLP(hidden_dim, hidden_dim))
        
        # Output Layer
        # We use a raw Linear layer (no activation) to allow the model to 
        # predict the exact action magnitude required.
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state, next_state):
        # Concatenate s_t and s_{t+1} along the feature dimension
        x = torch.cat([state, next_state], dim=-1)
        return torch.tanh(self.net(x))  # Assuming actions are bounded between -1 and 1


# FDM

class ForwardDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        """
        Predicts next state s_{t+1} given state s_t and action a_t.
        """
        super().__init__()
        input_dim = state_dim + action_dim
        
        layers = []
        layers.append(LayerNormMLP(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(LayerNormMLP(hidden_dim, hidden_dim))
            
        # Output is delta_s (change in state) or next_s directly.
        # Predicting delta is usually easier for optimization.
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        # Often better to predict residual: s_next = s + delta
        delta_s = self.net(x)
        return state + delta_s