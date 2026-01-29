import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ogbench
from tqdm import tqdm
import copy

# Import your models and utils
# Adjust the relative imports depending on if running as script or module
try:
    from .small_nets import Encoder, SkillPolicy, SkillCritic, SkillValueFunction
    from ..utils import TransitionSampler, expectile_loss_fn, soft_update_params
except ImportError:
    # Fallback for running as standalone script from src/models/
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from src.models.small_nets import Encoder, SkillPolicy, SkillCritic, SkillValueFunction
    from src.utils import TransitionSampler, expectile_loss_fn, soft_update_params


def train_actor(env, train_dataset, val_dataset, config, base_config):
    device = torch.device(config['env']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Starting Actor training on {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Unpack configurations
    actor_cfg = config['actor']
    enc_cfg = config['encoder']
    
    # Paths
    root_dir = base_config['root'] + '/' + config['output']['dir']
    encoder_path = os.path.join(root_dir, 'encoder', 'encoder_final.pth')
    out_dir = os.path.join(root_dir, 'actor')
    
    os.makedirs(out_dir, exist_ok=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    z_dim = enc_cfg['z_dim'] # Match encoder Z dim

    # -------------------------
    # 1. Load Pretrained Encoder
    # -------------------------
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}. Train encoder first.")
        
    print(f"Loading frozen encoder from {encoder_path}")
    encoder = Encoder(state_dim, z_dim, enc_cfg['hidden_dim']).to(device)
    
    # Handle loading weights (stripping _orig_mod if compiled)
    chkpt = torch.load(encoder_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in chkpt.items()}
    encoder.load_state_dict(new_state_dict)
    
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # -------------------------
    # 2. Initialize Actor Models
    # -------------------------
    hidden_dim = actor_cfg['hidden_dim']
    
    policy = SkillPolicy(state_dim, action_dim, z_dim, hidden_dim).to(device)
    
    critic1 = SkillCritic(state_dim, action_dim, z_dim, hidden_dim).to(device)
    critic2 = SkillCritic(state_dim, action_dim, z_dim, hidden_dim).to(device)
    target_critic1 = copy.deepcopy(critic1).to(device)
    target_critic2 = copy.deepcopy(critic2).to(device)
    
    value_fn = SkillValueFunction(state_dim, z_dim, hidden_dim).to(device)

    # Freeze targets
    for p in target_critic1.parameters(): p.requires_grad = False
    for p in target_critic2.parameters(): p.requires_grad = False

    # -------------------------
    # 3. Optimizers
    # -------------------------
    lr = float(actor_cfg['lr'])
    policy_opt = optim.Adam(policy.parameters(), lr=lr)
    critic_opt = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=lr)
    value_opt = optim.Adam(value_fn.parameters(), lr=lr)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Sampler
    sampler = TransitionSampler(train_dataset, device)

    # Hyperparameters
    skill_discount = actor_cfg['skill_discount']
    skill_expectile = actor_cfg['skill_expectile']
    skill_temp = actor_cfg['skill_temperature']
    tau = actor_cfg['target_update_rate']
    batch_size = actor_cfg['batch_size']
    total_steps = actor_cfg['steps']

    # Logging containers
    log_steps = []
    log_policy_loss, log_critic_loss, log_value_loss = [], [], []

    pbar = tqdm(range(1, total_steps + 1), dynamic_ncols=True)
    
    # Accumulators for logging
    run_pl, run_cl, run_vl = 0.0, 0.0, 0.0

    for step in pbar:
        # Sample
        s, a, s_next = sampler.sample(batch_size)
        
        # Sample Skills
        skills = torch.randn(batch_size, z_dim, device=device)
        skills = F.normalize(skills, p=2, dim=-1)

        # --- A. Compute Intrinsic Rewards ---
        with torch.inference_mode():
            phi_s = encoder(s)
            phi_s_next = encoder(s_next)
            intrinsic_rewards = ((phi_s_next - phi_s) * skills).sum(dim=-1)

        # --- B. Update Critic ---
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            with torch.no_grad():
                next_v = value_fn(s_next, skills).squeeze(-1)
                target_q = intrinsic_rewards + skill_discount * next_v

            q1 = critic1(s, a, skills).squeeze(-1)
            q2 = critic2(s, a, skills).squeeze(-1)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        critic_opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(critic_loss).backward()
            scaler.step(critic_opt)
        else:
            critic_loss.backward()
            critic_opt.step()

        # --- C. Update Value (IQL) ---
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            with torch.no_grad():
                tq1 = target_critic1(s, a, skills).squeeze(-1)
                tq2 = target_critic2(s, a, skills).squeeze(-1)
                min_tq = torch.min(tq1, tq2)

            v = value_fn(s, skills).squeeze(-1)
            adv = min_tq - v
            value_loss = expectile_loss_fn(adv, adv, skill_expectile)

        value_opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(value_loss).backward()
            scaler.step(value_opt)
        else:
            value_loss.backward()
            value_opt.step()

        # --- D. Update Actor (AWR) ---
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            adv_actor = adv.detach()
            exp_adv = torch.exp(adv_actor * skill_temp)
            exp_adv = torch.clamp(exp_adv, max=100.0)

            dist = policy(s, skills)
            log_probs = dist.log_prob(a).sum(dim=-1)
            policy_loss = -(exp_adv * log_probs).mean()
        
        policy_opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(policy_loss).backward()
            scaler.step(policy_opt)
            scaler.update()
        else:
            policy_loss.backward()
            policy_opt.step()

        # --- E. Soft Updates ---
        soft_update_params(target_critic1, critic1, tau)
        soft_update_params(target_critic2, critic2, tau)

        # Update running stats
        run_pl += policy_loss.item()
        run_cl += critic_loss.item()
        run_vl += value_loss.item()

        # --- Logging ---
        if step % actor_cfg['log_every'] == 0:
            avg_pl = run_pl / actor_cfg['log_every']
            avg_cl = run_cl / actor_cfg['log_every']
            avg_vl = run_vl / actor_cfg['log_every']
            
            log_steps.append(step)
            log_policy_loss.append(avg_pl)
            log_critic_loss.append(avg_cl)
            log_value_loss.append(avg_vl)

            pbar.set_description(f"Step {step}")
            pbar.set_postfix({
                'ploss': f"{avg_pl:.4f}",
                'closs': f"{avg_cl:.4f}",
                'vloss': f"{avg_vl:.4f}"
            })
            run_pl, run_cl, run_vl = 0.0, 0.0, 0.0
        
        # --- Saving ---
        if step % actor_cfg['save_every'] == 0:
            path = os.path.join(out_dir, f"actor_{step}.pth")
            torch.save(policy.state_dict(), path)

    # Final Save
    final_path = os.path.join(out_dir, "actor_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Actor saved to {final_path}")
    
    # Save Graphs
    print("Generating training graphs...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(log_steps, log_policy_loss, label='Policy Loss', color='blue')
    plt.title("Policy Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(log_steps, log_critic_loss, label='Critic Loss', color='orange')
    plt.title("Critic Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(log_steps, log_value_loss, label='Value Loss', color='green')
    plt.title("Value Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actor_losses.png"))
    plt.close()
    
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config file")
    args = parser.parse_args()

    # 1. Load Configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    device = torch.device(base_config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Main running on {device}")
    
    # 2. Init Env & Data
    print(f"Loading environment: {config['env']['name']}")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config['env']['name'], 
        base_config.get('download_ogbench_dir', None)
    )
    
    # 3. Check for existing model
    final_model_path = os.path.join(base_config['root'], config['output']['dir'], "actor", "actor_final.pth")
    
    # Initialize Architecture (to load weights if they exist)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    z_dim = config['encoder']['z_dim']
    hidden_dim = config['actor']['hidden_dim']
    
    actor = SkillPolicy(state_dim, action_dim, z_dim, hidden_dim).to(device)

    model_exists = os.path.exists(final_model_path)
    force_train = base_config.get('force_train', False)
    
    # Override logic specific to actor if desired
    # if base_config.get('force_train_actor', False): force_train = True

    if model_exists and not force_train:
        print(f"Found existing actor at {final_model_path}. Loading...")
        actor.load_state_dict(torch.load(final_model_path, map_location=device))
        actor.eval()
    else:
        if force_train and model_exists:
            print(f"Model exists but force_train is True. Starting fresh training...")
        else:
            print(f"No existing actor found. Starting training...")
            
        actor_state_dict = train_actor(env, train_dataset, val_dataset, config, base_config)
        
        # Handle if function returned model vs state_dict
        if isinstance(actor_state_dict, nn.Module):
             actor = actor_state_dict
        else:
             actor.load_state_dict(actor_state_dict)

if __name__ == "__main__":
    main()