import numpy as np
import torch
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_time_minutes(start_str, end_str, n):
    start = datetime.strptime(start_str, "%H:%M")
    end   = datetime.strptime(end_str, "%H:%M")
    delta = int((end - start).total_seconds() // 60)
    times = []
    for _ in range(n):
        offset = random.randint(0, delta)
        t = start + timedelta(minutes=offset)
        times.append(t.hour * 60 + t.minute)
    return times

def run_episode(env, model=None, policy="ppo"):
    env.mode = policy
    obs, _ = env.reset()
    done = False
    total_loads = []
    power_logs = []  # <--- simpan per charger

    while not done:
        if policy == "RBC":
            action = np.ones(env.num_chargers, dtype=np.float32)  # dummy
            obs, _, done, _, _ = env.step(action)
        elif policy == "ppo":  # PPO
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, log_std, _ = model(obs_t)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
            action_env = torch.clamp(action, -1, 1).cpu().numpy()
            obs, _, done, _, _ = env.step(action_env)
        elif policy == "sac":  # SAC
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, std = model(obs_t)
                dist = torch.distributions.Normal(mu, std)
                action_raw = dist.sample()
                action_env = torch.clamp(action_raw, -1, 1).squeeze(0).cpu().numpy()
            obs, _, done, _, _ = env.step(action_env)
        elif policy == "modrl":  # MODRL
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, log_std, _ = model(obs_t)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
            action_env = torch.clamp(action, -1, 1).cpu().numpy()
            obs, _, done, _, _ = env.step(action_env)
        
        # Collect data for plotting
        total_loads.append(env.current_total_load)
        power_logs.append(env.charger_powers.copy())

    # Metrics
    finish_time = env.current_time
    peak_load = max(total_loads) if total_loads else 0
    total_energy = sum(total_loads) / 60.0  # kW -> kWh approx

    return power_logs, total_loads, finish_time, peak_load, total_energy

def compute_extra_metrics(unctrl_loads, ppo_loads=None, sac_loads=None, modrl_loads=None,
                          unctrl_finish_time=None, ppo_finish_time=None, 
                          sac_finish_time=None, modrl_finish_time=None):
    metrics = {}
    
    base_peak = np.max(unctrl_loads) if unctrl_loads is not None else 0
    
    if unctrl_loads is not None:
        metrics["Peak Load (RBC)"] = base_peak
        metrics["Variance (RBC)"] = np.var(unctrl_loads)
        metrics["Total Energy (RBC)"] = np.sum(unctrl_loads) / 60.0
        if unctrl_finish_time is not None:
            metrics["Finish Time (RBC)"] = unctrl_finish_time
    
    if ppo_loads is not None:
        metrics["Peak Load (PPO)"] = np.max(ppo_loads)
        if base_peak > 0:
            metrics["Peak Reduction (PPO) (%)"] = (base_peak - metrics["Peak Load (PPO)"]) / base_peak * 100
        metrics["Variance (PPO)"] = np.var(ppo_loads)
        metrics["Total Energy (PPO)"] = np.sum(ppo_loads) / 60.0
        if ppo_finish_time is not None:
            metrics["Finish Time (PPO)"] = ppo_finish_time
    
    if sac_loads is not None:
        metrics["Peak Load (SAC)"] = np.max(sac_loads)
        if base_peak > 0:
            metrics["Peak Reduction (SAC) (%)"] = (base_peak - metrics["Peak Load (SAC)"]) / base_peak * 100
        metrics["Variance (SAC)"] = np.var(sac_loads)
        metrics["Total Energy (SAC)"] = np.sum(sac_loads) / 60.0
        if sac_finish_time is not None:
            metrics["Finish Time (SAC)"] = sac_finish_time
    
    if modrl_loads is not None:
        metrics["Peak Load (MODRL)"] = np.max(modrl_loads)
        if base_peak > 0:
            metrics["Peak Reduction (MODRL) (%)"] = (base_peak - metrics["Peak Load (MODRL)"]) / base_peak * 100
        metrics["Variance (MODRL)"] = np.var(modrl_loads)
        metrics["Total Energy (MODRL)"] = np.sum(modrl_loads) / 60.0
        if modrl_finish_time is not None:
            metrics["Finish Time (MODRL)"] = modrl_finish_time
    
    return metrics

def compare_metrics(metrics_list):
    # Helper to aggregate metrics from multiple runs if needed
    pass
