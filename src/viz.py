import matplotlib.pyplot as plt
import os
import numpy as np
import torch


# Encoder viz

def save_training_graphs(steps, train_loss, val_loss, v_mean, v_min, v_max, out_dir):
    print("Generating training graphs")
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label='Train Loss', marker='o', markersize=3, alpha=0.7)
    val_steps_mask = np.isfinite(val_loss)
    if np.any(val_steps_mask):
        plt.plot(np.array(steps)[val_steps_mask], 
                 np.array(val_loss)[val_steps_mask], 
                 label='Val Loss', linestyle='--', marker='x', color='orange')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, v_min, v_max, color='blue', alpha=0.15, label='V Min/Max Range')
    plt.plot(steps, v_mean, label='V Mean', color='crimson', linewidth=2)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.xlabel("Steps")
    plt.ylabel("V Value (Negative Latent Distance)")
    plt.title("Value Function Statistics")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "v_stats.png"))
    plt.close()

def evaluate_encoder(encoder, train_dataset, env, config, base_config):
    device = torch.device(config['env']['device'] if torch.cuda.is_available() else "cpu")
    print("Generating evaluation plots")
    encoder.eval()
    
    # Analysis Parameters
    PLOT_SAMPLES = 50000 
    START_TRAJECTORY_INDEX = 19
    START_STEP_INDEX = 0
    
    # --- LOGIC: Select a fixed start state from TRAJECTORY ---
    terminals = np.where(train_dataset['terminals'])[0]
    start_idx = 0 if START_TRAJECTORY_INDEX == 0 else terminals[START_TRAJECTORY_INDEX - 1] + 1
    
    if START_TRAJECTORY_INDEX < len(terminals):
        traj_end = terminals[START_TRAJECTORY_INDEX]
        max_steps = traj_end - start_idx
        actual_step_idx = min(START_STEP_INDEX, max_steps)
    else:
        actual_step_idx = START_STEP_INDEX

    s_start_np = train_dataset['observations'][start_idx + actual_step_idx]
    
    # 2. Sample Background States
    total_obs = len(train_dataset['observations'])
    num_samples = min(PLOT_SAMPLES, total_obs)
    indices = np.random.choice(total_obs, num_samples, replace=False)
    s_sampled_np = train_dataset['observations'][indices]

    # 3. Compute Distances
    print("Calculating distances")
    start_pos_xy = s_start_np[:2]
    sampled_pos_xy = s_sampled_np[:, :2]
    ref_xy = start_pos_xy
    xy_coords = sampled_pos_xy 
    
    # Physical
    dists_euclidean = np.linalg.norm(sampled_pos_xy - start_pos_xy, axis=1)

    # Latent
    all_states_np = np.vstack([s_start_np, s_sampled_np])
    latent_embeddings = []
    
    # Use inference_mode for evaluation too
    with torch.inference_mode():
        for i in range(0, len(all_states_np), 2048):
            batch_s = torch.tensor(all_states_np[i:i+2048], dtype=torch.float32, device=device)
            batch_z = encoder(batch_s)
            latent_embeddings.append(batch_z.cpu())
    
    latent_embeddings = torch.cat(latent_embeddings, dim=0)
    z_start = latent_embeddings[0]
    z_sampled = latent_embeddings[1:]
    
    dists_latent = torch.norm(z_sampled - z_start, p=2, dim=1).numpy()

    def plot_maze_background(ax, env):
        try:
            if hasattr(env.unwrapped, 'maze'):
                maze_obj = env.unwrapped.maze
            elif hasattr(env.unwrapped, 'maze_map'):
                maze_obj = env.unwrapped
            else:
                return

            if hasattr(maze_obj, 'maze_map'):
                maze_map = maze_obj.maze_map
            elif hasattr(env.unwrapped, 'maze_map'):
                maze_map = env.unwrapped.maze_map
            else:
                maze_map = maze_obj

            if hasattr(maze_obj, 'ij_to_xy'):
                converter = maze_obj
            else:
                converter = env.unwrapped

            H, W = maze_map.shape
            c00 = np.array(converter.ij_to_xy((0, 0)))
            c_last = np.array(converter.ij_to_xy((H - 1, W - 1)))
            extent = [c00[0] - 0.5, c_last[0] + 0.5, c00[1] - 0.5, c_last[1] + 0.5]

            # cmap="binary" for white/black maze
            ax.imshow(maze_map, cmap="binary", origin="lower", extent=extent)
        except Exception as e:
            print(f"Could not plot maze background: {e}")

    print("Generating plot")
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Ground Truth
    plot_maze_background(ax[0], env)
    sc1 = ax[0].scatter(xy_coords[:, 0], xy_coords[:, 1], c=dists_euclidean, cmap='viridis', s=25, alpha=0.8)
    ax[0].scatter(ref_xy[0], ref_xy[1], c='red', marker='*', s=300, edgecolors='black', linewidth=2, label='Ref State')
    ax[0].set_title("Ground Truth (Euclidean)", fontsize=16)
    ax[0].set_xlabel("X Position")
    ax[0].set_ylabel("Y Position")
    ax[0].legend()
    plt.colorbar(sc1, ax=ax[0], label='Euclidean Distance')

    # Plot 2: Latent Distance
    plot_maze_background(ax[1], env)
    sc2 = ax[1].scatter(xy_coords[:, 0], xy_coords[:, 1], c=dists_latent, cmap='viridis', s=25, alpha=0.8)
    ax[1].scatter(ref_xy[0], ref_xy[1], c='red', marker='*', s=300, edgecolors='black', linewidth=2, label='Ref State')
    ax[1].set_title("Learned Latent Distance", fontsize=16)
    ax[1].set_xlabel("X Position")
    ax[1].set_ylabel("Y Position")
    ax[1].legend()
    plt.colorbar(sc2, ax=ax[1], label='Latent Distance')

    plt.suptitle(f"PointMaze Distance Comparison (Env: {config['env']['name']})", fontsize=20)
    plt.tight_layout()
    path = base_config['root'] + '/' + config['output']['dir'] + '/encoder'
    os.makedirs(path, exist_ok=True)
    plot_path = os.path.join(path, "encoder_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")
    plt.show()


# EBT

def save_loss_graphs(steps, train_loss, val_loss, out_dir, moving_avg_window=20):
    """
    Saves Training and Validation loss curves with moving averages for both.
    
    Args:
        steps (list or np.array): List of step numbers.
        train_loss (list or np.array): List of training loss values.
        val_loss (list or np.array): List of validation loss values (can contain NaNs/None).
        out_dir (str): Directory to save the image.
        moving_avg_window (int): Window size for smoothing the losses. 
                                 Set to None or 1 to disable.
    """
    print("Generating loss graph...")
    
    # Ensure inputs are numpy arrays for masking/math
    steps = np.array(steps)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    plt.figure(figsize=(10, 6))

    # --- 1. TRAINING LOSS ---
    # Raw Data
    raw_alpha = 0.3 if (moving_avg_window and moving_avg_window > 1) else 0.7
    plt.plot(steps, train_loss, label='Train Loss (Raw)', 
             color='steelblue', marker='o', markersize=2, alpha=raw_alpha)

    # Moving Average
    if moving_avg_window and moving_avg_window > 1 and len(train_loss) >= moving_avg_window:
        kernel = np.ones(moving_avg_window) / moving_avg_window
        train_smooth = np.convolve(train_loss, kernel, mode='valid')
        smooth_steps = steps[moving_avg_window - 1:]
        plt.plot(smooth_steps, train_smooth, label=f'Train Loss (MA-{moving_avg_window})', 
                 color='navy', linewidth=2)

    # --- 2. VALIDATION LOSS ---
    # We must handle the case where val_loss is the same length as steps (padded with NaNs)
    # or a shorter list. Assuming logic where it matches 'steps' length with sparse entries.
    
    if val_loss is not None and len(val_loss) == len(steps):
        # Create a mask for valid (finite) validation entries
        val_mask = np.isfinite(val_loss)
        
        if np.any(val_mask):
            valid_steps = steps[val_mask]
            valid_val_loss = val_loss[val_mask]

            # Plot Raw Validation
            plt.plot(valid_steps, valid_val_loss, 
                     label='Val Loss (Raw)', linestyle='--', marker='x', 
                     markersize=4, color='orange', alpha=raw_alpha)

            # Plot Moving Average for Validation
            # We perform convolution ONLY on the valid entries
            if moving_avg_window and moving_avg_window > 1 and len(valid_val_loss) >= moving_avg_window:
                kernel = np.ones(moving_avg_window) / moving_avg_window
                val_smooth = np.convolve(valid_val_loss, kernel, mode='valid')
                
                # Align steps for the smoothed validation curve
                # We use the subset of steps that correspond to the valid validation points
                val_smooth_steps = valid_steps[moving_avg_window - 1:]
                
                plt.plot(val_smooth_steps, val_smooth, label=f'Val Loss (MA-{moving_avg_window})', 
                         color='darkorange', linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved to {save_path}")