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
from .small_nets import Encoder, EnsembledValueFcuntion
from ..utils import HindsightSampler, expectile_loss_fn, soft_update_params
from ..viz import save_training_graphs, evaluate_encoder


def train_encoder(env, train_dataset, val_dataset, config, base_config):
    device = torch.device(config['env']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Starting Encoder training on {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Unpack configurations
    enc_cfg = config['encoder']
    out_dir = base_config['root'] + '/' + config['output']['dir'] + '/encoder'
    
    os.makedirs(out_dir, exist_ok=True)
    
    state_dim = env.observation_space.shape[0]
    
    # Init samplers (Loads data to GPU)
    train_sampler = HindsightSampler(train_dataset, device)
    val_sampler = HindsightSampler(val_dataset, device)

    value_fn = EnsembledValueFcuntion(state_dim, enc_cfg['z_dim'], enc_cfg['hidden_dim']).to(device)
    target_value_fn = EnsembledValueFcuntion(state_dim, enc_cfg['z_dim'], enc_cfg['hidden_dim']).to(device)
    target_value_fn.load_state_dict(value_fn.state_dict())
    
    # Freeze target network parameters explicitly (good practice)
    for p in target_value_fn.parameters(): 
        p.requires_grad = False

    if hasattr(torch, 'compile'):
        print("Compiling value function with torch.compile...")
        # 'reduce-overhead' is aggressive but good for small models; if it fails, remove mode argument
        value_fn = torch.compile(value_fn)

    # Only available on CUDA
    use_fused = (device.type == 'cuda')
    w_decay = float(enc_cfg['weight_decay'])

    opt = optim.AdamW(value_fn.parameters(), lr=enc_cfg['lr'], 
                     weight_decay=w_decay, fused=use_fused)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=enc_cfg['steps'])
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Stats containers
    log_steps, log_train_loss, log_val_loss = [], [], []
    log_v_mean, log_v_min, log_v_max = [], [], []

    pbar = tqdm(range(1, enc_cfg['steps'] + 1), dynamic_ncols=True)
    running_loss = 0.0
    val_loss_display = float('nan')
    running_v_mean, running_v_min, running_v_max = 0.0, 0.0, 0.0

    gamma = enc_cfg['discount_factor']
    tau = enc_cfg['target_update_rate']
    expectile = enc_cfg['expectile']

    for step in pbar:
        value_fn.train()
        
    # Inside training loop...

        # 1. Sample (Your sampler is fine!)
        s, s_next, g, _ = train_sampler.sample(enc_cfg['batch_size']) 

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            with torch.inference_mode():
                # 2. Compute Target V
                next_v1_targ, next_v2_targ = target_value_fn(s_next, g)
                next_v_min = torch.min(next_v1_targ, next_v2_targ)
                
                # CORRECT HILP TARGET:
                # We model the distance function d(s, g).
                # Bellman: d(s, g) = 1 + d(s', g)
                # In value space (V = -d): V(s, g) = -1 + gamma * V(s', g)
                # We do NOT mask with done because the goal 'g' is artificial/relabeled.
                target_v = -1.0 + gamma * next_v_min

            # 3. Compute Current V
            current_v1, current_v2 = value_fn(s, g)
            
            # 4. Compute Loss (Corrected Expectile Logic)
            diff1 = target_v - current_v1
            diff2 = target_v - current_v2
            
            # Pass 'diff' as the second argument so the weight depends on 
            # whether we are over- or under-estimating the target.
            loss1 = expectile_loss_fn(diff1, diff1, expectile)
            loss2 = expectile_loss_fn(diff2, diff2, expectile)
            
            loss = loss1 + loss2


        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
            
        scheduler.step()

        # Optimized soft update
        soft_update_params(target_value_fn, value_fn, tau)

        running_loss += loss.item()
        
        # Logging stats
        with torch.no_grad():
            v_avg_batch = (current_v1.detach() + current_v2.detach()) * 0.5
            running_v_mean += v_avg_batch.mean().item()
            running_v_min += v_avg_batch.min().item()
            running_v_max += v_avg_batch.max().item()

        # --- Validation Step ---
        if step % enc_cfg['val_every'] == 0:
            value_fn.eval()
            with torch.inference_mode():
                s_val, s_next_val, g_val, _ = val_sampler.sample(enc_cfg['batch_size'] * 2) 
                
                nv1_val, nv2_val = target_value_fn(s_next_val, g_val)
                nv_min_val = torch.min(nv1_val, nv2_val)
                q_min_targ_val = nv_min_val.mul(gamma).sub_(1.0)
                
                q1_targ_val = nv1_val.mul(gamma).sub_(1.0)
                q2_targ_val = nv2_val.mul(gamma).sub_(1.0)
                
                v1_t_val, v2_t_val = target_value_fn(s_val, g_val)
                v_t_avg_val = (v1_t_val + v2_t_val) * 0.5
                adv_val = q_min_targ_val - v_t_avg_val
                
                curr_v1_val, curr_v2_val = value_fn(s_val, g_val)
                l1_val = expectile_loss_fn(q1_targ_val - curr_v1_val, adv_val, expectile)
                l2_val = expectile_loss_fn(q2_targ_val - curr_v2_val, adv_val, expectile)
                val_loss_display = (l1_val + l2_val).item()

        # --- Logging ---
        if step % enc_cfg['log_every'] == 0:
            avg_train_loss = running_loss / enc_cfg['log_every']
            avg_v_mean = running_v_mean / enc_cfg['log_every']
            avg_v_min = running_v_min / enc_cfg['log_every']
            avg_v_max = running_v_max / enc_cfg['log_every']
            
            log_steps.append(step)
            log_train_loss.append(avg_train_loss)
            log_val_loss.append(val_loss_display) 
            log_v_mean.append(avg_v_mean)
            log_v_min.append(avg_v_min)
            log_v_max.append(avg_v_max)
            
            pbar.set_description(f"Step {step}")
            pbar.set_postfix({
                'loss': f"{avg_train_loss:.4f}",
                'val': f"{val_loss_display:.4f}",
                'v_avg': f"{avg_v_mean:.2f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            running_loss = 0.0
            running_v_mean = 0.0
            running_v_min = 0.0
            running_v_max = 0.0
        
        if step % enc_cfg['save_every'] == 0:
            # Unwrap if compiled
            raw_model = value_fn.orig_mod if hasattr(value_fn, 'orig_mod') else value_fn
            torch.save(raw_model.encoder1.state_dict(), os.path.join(out_dir, f"encoder_{step}.pth"))

    # Final Save
    raw_model = value_fn.orig_mod if hasattr(value_fn, 'orig_mod') else value_fn
    final_path = os.path.join(out_dir, "encoder_final.pth")
    torch.save(raw_model.encoder1.state_dict(), final_path)
    print(f"Training complete. Model saved to {final_path}")
    
    save_training_graphs(log_steps, log_train_loss, log_val_loss, log_v_mean, log_v_min, log_v_max, out_dir)
    
    return raw_model.encoder1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    device = torch.device(base_config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Loading environment: {config['env']['name']}")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(config['env']['name'], base_config.get('download_ogbench_dir', None))
    
    state_dim = env.observation_space.shape[0]
    final_model_path = os.path.join(config['output']['dir'], "encoder_final.pth")
    
    enc_cfg = config['encoder']
    encoder = Encoder(state_dim, enc_cfg['z_dim'], enc_cfg['hidden_dim']).to(device)

    model_exists = os.path.exists(final_model_path)
    force_train = base_config.get('force_train', False)
    
    if model_exists and not force_train:
        print(f"Found existing model at {final_model_path}. Loading")
        encoder.load_state_dict(torch.load(final_model_path, map_location=device))
    else:
        if force_train and model_exists:
            print(f"Model exists but force_train is set to True in config. Starting fresh training")
        else:
            print(f"No existing model found. Starting training")
            
        encoder_state_dict = train_encoder(env, train_dataset, val_dataset, config, base_config)
        
        if isinstance(encoder_state_dict, Encoder):
             encoder = encoder_state_dict
        else:
             encoder.load_state_dict(encoder_state_dict)

    evaluate_encoder(encoder, train_dataset, env, config, base_config)

if __name__ == "__main__":
    main()