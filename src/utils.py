import torch
import numpy as np

# Encoder utils
class HindsightSampler:
    def __init__(self, dataset, device, indices=None):
        self.device = device
        
        # 1. Extract raw numpy data
        obs = dataset['observations']
        next_obs = dataset['next_observations']
        terminals = dataset['terminals'].flatten().astype(bool)
        if 'timeouts' in dataset:
            timeouts = dataset['timeouts'].flatten().astype(bool)
        else:
            timeouts = np.zeros_like(terminals)
        
        # 2. Handle subsetting
        if indices is not None:
            obs = obs[indices]
            next_obs = next_obs[indices]
            terminals = terminals[indices]
            
        self.num_samples = len(obs)

        # 3. Pre-calculate trajectory ends (CPU is fine for this one-time setup)
        traj_end_indices = np.zeros(self.num_samples, dtype=int)
        
        # Initialize with the last index to prevent wrapping to 0
        current_end = self.num_samples - 1 
        
        for i in range(self.num_samples - 1, -1, -1):
            # If this step is terminal OR timeout, it's the end of a trajectory
            if terminals[i] or timeouts[i]:
                current_end = i
            traj_end_indices[i] = current_end

        # Move to GPU
        self.obs_tensor = torch.from_numpy(obs).float().to(device)
        self.next_obs_tensor = torch.from_numpy(next_obs).float().to(device)
        self.terminals_tensor = torch.from_numpy(terminals).float().to(device) # Need float for math
        self.traj_end_indices = torch.from_numpy(traj_end_indices).long().to(device)

    def sample(self, batch_size):
        # 1. Generate random indices directly on GPU
        idxs = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        
        # 2. Logic for Future Goals
        # Mask: True if we should use future goal, False for random goal
        mask_future = torch.rand(batch_size, device=self.device) < 0.625
        
        # Initialize goal_idxs with random goals
        goal_idxs = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        
        # Apply Hindsight Logic (Vectorized on GPU)
        if mask_future.any():
            future_idxs = idxs[mask_future]
            ends = self.traj_end_indices[future_idxs]
            remaining_len = ends - future_idxs
            
            # Geometric distribution (p=0.1)
            # We create a float tensor to sample geometric, then cast to long
            geom_offsets = torch.empty_like(remaining_len, dtype=torch.float32).geometric_(0.1).long()
            
            # Clamp to remaining length
            actual_offsets = torch.minimum(geom_offsets, remaining_len)
            
            # Update goal indices
            goal_idxs[mask_future] = future_idxs + actual_offsets

        s = self.obs_tensor[idxs]
        s_next = self.next_obs_tensor[idxs]
        g = self.obs_tensor[goal_idxs]
        
        dones = self.terminals_tensor[idxs]
        
        return s, s_next, g, dones

def expectile_loss_fn(diff, adv, expectile):
    weight = torch.where(adv >= 0, expectile, 1 - expectile)
    return (weight * (diff**2)).mean()

# Helper for optimized soft update
@torch.compile(disable=not hasattr(torch, "compile"))
def soft_update_params(target_net, source_net, tau):
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            # In-place linear interpolation: target = target + tau * (source - target)
            target_param.data.lerp_(param.data, tau)


# EBT modules

class TrajectorySampler:
    """
    Samples consecutive chunks (windows) of actions from the OGBench dataset.
    Ensures chunks do not cross episode boundaries.
    """
    def __init__(self, dataset, window_size, device):
        self.device = device
        self.window_size = window_size
        
        # Extract data from OGBench dataset dict
        # OGBench structure usually: inputs (s), actions (a), terminals, etc.
        self.states = dataset['observations'] # Shape (N, obs_dim)
        self.actions = dataset['actions']     # Shape (N, act_dim)
        
        # Handle Episode boundaries
        # We assume 'terminals' or 'timeouts' indicate ends.
        # If not present, we assume one long episode or handle standard logic.
        terminals = dataset.get('terminals', np.zeros(len(self.actions)))
        timeouts = dataset.get('timeouts', np.zeros(len(self.actions)))
        done = np.logical_or(terminals, timeouts).astype(bool).flatten()
        
        # Calculate valid start indices
        # A start index i is valid if i + window_size <= next_done_index
        n_samples = len(self.actions)
        self.valid_indices = []
        
        # Fast vectorization for validity is hard with generic dones, using loop for safety on init
        # (This runs once at startup)
        current_idx = 0
        while current_idx < n_samples - window_size:
            # Check if there is a done flag in the window [current, current+window]
            window_dones = done[current_idx : current_idx + window_size]
            if not np.any(window_dones[:-1]): # Last step being done is okay
                self.valid_indices.append(current_idx)
            current_idx += 1
            
        self.valid_indices = np.array(self.valid_indices)
        print(f"TrajectorySampler: Found {len(self.valid_indices)} valid chunks of size {window_size}")

    def sample(self, batch_size):
        idxs = np.random.choice(self.valid_indices, size=batch_size)
        
        batch_s = []
        batch_a = []
        
        for i in idxs:
            # S is the start state of the window
            batch_s.append(self.states[i])
            # A is the sequence of H actions
            batch_a.append(self.actions[i : i + self.window_size])
            
        return (
            torch.tensor(np.stack(batch_s), dtype=torch.float32, device=self.device),
            torch.tensor(np.stack(batch_a), dtype=torch.float32, device=self.device)
        )

def rms_norm(x, eps=1e-8):
    """
    Root Mean Square Normalization.
    x: (B, H, D)
    """
    # Calculate RMS along the last dimension or flattening H*D?
    # Paper implies normalizing the action trajectory vector.
    # Usually applied over the full trajectory dimension or per action.
    # EBT Paper: "Action trajectories are normalized to [-1, 1]... we use RMSNorm"
    # We norm over the (H, D) dimensions combined per batch item.
    scale = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + eps
    return x / scale


# IDM

class TransitionSampler:
    """
    Simple sampler for IDM training. 
    Returns (s, a, s_next) tuples from the dataset.
    """
    def __init__(self, dataset, device):
        self.device = device
        
        self.obs = torch.from_numpy(dataset['observations']).float().to(device)
        self.actions = torch.from_numpy(dataset['actions']).float().to(device)
        self.next_obs = torch.from_numpy(dataset['next_observations']).float().to(device)
        
        self.num_samples = self.obs.shape[0]

    def sample(self, batch_size):
        idxs = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        
        s = self.obs[idxs]
        a = self.actions[idxs]
        s_next = self.next_obs[idxs]
        
        return s, a, s_next


class DynamicsDataManager:
    """
    Handles data loading and normalization statistics for BOTH IDM and FDM.
    """
    def __init__(self, dataset, device):
        self.device = device
        
        # 1. Load Raw Data
        obs = torch.from_numpy(dataset['observations']).float()
        actions = torch.from_numpy(dataset['actions']).float()
        next_obs = torch.from_numpy(dataset['next_observations']).float()
        
        # 2. Compute Statistics (One source of truth)
        self.stats = {
            'obs_mean': obs.mean(dim=0, keepdim=True).to(device),
            'obs_std':  obs.std(dim=0, keepdim=True).to(device) + 1e-6,
            'act_mean': actions.mean(dim=0, keepdim=True).to(device),
            'act_std':  actions.std(dim=0, keepdim=True).to(device) + 1e-6
        }

        # 3. Store Normalized Tensors
        self.obs_norm = (obs.to(device) - self.stats['obs_mean']) / self.stats['obs_std']
        self.next_obs_norm = (next_obs.to(device) - self.stats['obs_mean']) / self.stats['obs_std']
        self.act_norm = (actions.to(device) - self.stats['act_mean']) / self.stats['act_std']
        
        self.num_samples = self.obs_norm.shape[0]

    def sample(self, batch_size):
        idxs = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        return self.obs_norm[idxs], self.act_norm[idxs], self.next_obs_norm[idxs]

    def unnormalize_action(self, a_norm):
        return a_norm * self.stats['act_std'] + self.stats['act_mean']
    
    def unnormalize_state(self, s_norm):
        return s_norm * self.stats['obs_std'] + self.stats['obs_mean']

    def normalize_action(self, a):
        return (a - self.stats['act_mean']) / self.stats['act_std']
    
    def normalize_state(self, s):
        return (s - self.stats['obs_mean']) / self.stats['obs_std']
    
    def get_stats(self):
        return self.stats