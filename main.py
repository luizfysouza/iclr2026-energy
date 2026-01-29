import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ogbench
from tqdm import tqdm

# --- Import Models & Training Functions ---
from src.models.small_nets import Encoder, EBTPolicy, InverseDynamicsModel, ForwardDynamicsModel
from src.models.encoder import train_encoder
from src.models.ebt import train_ebt
from src.models.idm import train_idm
from src.models.fdm import train_fdm
from src.models.gas import GASGraph
# --- Import Utils ---
from src.utils import DynamicsDataManager
from src.utils import rms_norm


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_maze_background(ax, env):
    """
    Plots the maze background with high contrast (Walls=Black, Free=Light Gray).
    """
    try:
        # 1. Locate Maze Object
        if hasattr(env.unwrapped, 'maze'):
            maze_obj = env.unwrapped.maze
        elif hasattr(env.unwrapped, 'maze_map'):
            maze_obj = env.unwrapped
        else:
            return

        # 2. Extract Map
        if hasattr(maze_obj, 'maze_map'):
            maze_map = maze_obj.maze_map
        elif hasattr(env.unwrapped, 'maze_map'):
            maze_map = env.unwrapped.maze_map
        else:
            maze_map = maze_obj

        # Convert to numpy
        maze_map = np.array(maze_map, dtype=np.float32)
        
        # 3. Coordinate Conversion
        if hasattr(maze_obj, 'ij_to_xy'):
            converter = maze_obj
        else:
            converter = env.unwrapped

        H, W = maze_map.shape
        
        # --- FIX: Pass coordinates as a LIST or TUPLE ---
        # Error happened here because we passed (0, 0) as two args instead of one tuple
        c00 = np.array(converter.ij_to_xy([0, 0]))
        c_last = np.array(converter.ij_to_xy([H - 1, W - 1]))
        
        # Calculate Extent [xmin, xmax, ymin, ymax]
        step_x = (c_last[0] - c00[0]) / (H - 1)
        step_y = (c_last[1] - c00[1]) / (W - 1)
        
        pad_x = step_x / 2.0
        pad_y = step_y / 2.0
        
        extent = [c00[0] - pad_x, c_last[0] + pad_x, 
                  c00[1] - pad_y, c_last[1] + pad_y]

        # 4. Plot
        # Transpose (.T) to align [row, col] with [x, y]
        # vmin/vmax/cmap ensures Walls (1) are Black and Free (0) is White/Grey
        ax.imshow(maze_map, cmap="Greys", origin="lower", extent=extent, 
                  vmin=0, vmax=1, alpha=0.3, zorder=0)
        
        ax.set_aspect('equal')
        
    except Exception as e:
        print(f"Could not plot maze background: {e}")


def get_component(name, model_class, train_fn, config, base_config, env, train_ds, val_ds, extra_args={}):
    """
    Generic helper to Load OR Train a component based on config/existence.
    """
    print(f"\n--- Initializing {name.upper()} ---")
    device = torch.device(base_config['device'])
    out_dir = os.path.join(base_config['root'], config['output']['dir'], name)
    final_path = os.path.join(out_dir, f"{name}_final.pth")
    
    # 1. Initialize Model Architecture
    if name == 'encoder':
        cfg = config['encoder']
        state_dim = env.observation_space.shape[0]
        model = model_class(state_dim, cfg['z_dim'], cfg['hidden_dim']).to(device)

    # 2. Check Logic
    model_exists = os.path.exists(final_path)
    force_train = base_config.get('force_train', False)
    
    # Overrides
    if base_config.get(f'force_train_{name}', False):
        force_train = True

    if model_exists and not force_train:
        print(f"Loading {name} from {final_path}")
        checkpoint = torch.load(final_path, map_location=device)
        
        # Handle dict vs raw weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        else:
            weights = checkpoint
            
        # --- FIX: Strip '_orig_mod.' prefix if present ---
        new_weights = {}
        for k, v in weights.items():
            new_key = k.replace("_orig_mod.", "")
            new_weights[new_key] = v
            
        try:
            model.load_state_dict(new_weights)
        except RuntimeError as e:
            print(f"Error loading {name}: {e}")
            print("Trying strict=False load as fallback...")
            model.load_state_dict(new_weights, strict=False)
            
        model.eval()
        return model
    else:
        if force_train:
            print(f"Force Train active for {name}. Starting training...")
        else:
            print(f"{name} weights not found at {final_path}. Starting training...")
            

        return train_fn(env, train_ds, val_ds, config, base_config)
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/antmaze/navigate/antmaze_medium_navigate.yaml")
    parser.add_argument("--base_config", type=str, default="config/base.yaml")
    args = parser.parse_args()

    # 1. Load Configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    device = torch.device(base_config['device'])
    print(f"Main running on {device}")
    
    # 2. Init Env & Data
    print(f"Loading environment: {config['env']['name']}")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(config['env']['name'], base_config.get('download_ogbench_dir', None))
    
    # 3. Load/Train Components
    
    # A. Encoder
    encoder = get_component('encoder', Encoder, train_encoder, 
                            config, base_config, env, train_dataset, val_dataset)

                            
if __name__ == "__main__":
    main()