import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import ActorCritic, SACActor, SACCritic, ReplayBuffer, MODRLActorCritic

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# PPO Training Loop
# -----------------------------------------------------------------------------
def ppo_train(env, epochs=1000, steps_per_epoch=2048, gamma=0.99,
              clip_ratio=0.2, lr=3e-5, train_pi_iters=80, minibatch_size=256,
              target_kl=0.03, vf_coef=0.5, ent_coef=0.01, hidden_dim=256):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_state = None
    best_reward = -float('inf')
    mean_rewards = []
    metrics_log = []
    
    env.mode = "ppo"

    for epoch in range(epochs):
        obs, _ = env.reset()
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        ep_rews = []

        # --- Rollout ---
        for t in range(steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, log_std, value = model(obs_t)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)

                action = dist.sample()  # unsquashed, real policy action
                logp = dist.log_prob(action).sum(-1)  # logprob of that action
            
            action_env = torch.clamp(action, -1, 1).cpu().numpy()  # clamp to Box

            next_obs, reward, done, _, _ = env.step(action_env)

            obs_buf.append(obs_t)
            act_buf.append(action)  # store the un-clamped action (already on device)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)

            obs = next_obs
            ep_rews.append(reward)

            if done:
                obs, _ = env.reset()

        # After rollout ends
        mean_reward = np.mean(ep_rews) if len(ep_rews) > 0 else 0.0
        mean_rewards.append(mean_reward)

        # --- Compute returns ---
        returns = []
        G = 0
        for r in reversed(rew_buf):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Stack
        obs_tensor = torch.stack(obs_buf)
        act_tensor = torch.stack(act_buf)
        val_tensor = torch.cat(val_buf).squeeze(-1)
        logp_tensor = torch.stack(logp_buf)

        # --- Compute returns and advantages with GAE ---
        values = val_tensor.detach().cpu().numpy()
        rewards = np.array(rew_buf, dtype=np.float32)

        adv = np.zeros_like(rewards)
        lastgaelam = 0
        lam = 0.95  # GAE lambda
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 0.0
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        adv = torch.tensor(adv, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # --- Multiple minibatch updates ---
        for _ in range(train_pi_iters):
            idx = np.random.randint(0, len(obs_tensor), minibatch_size)
            obs_mb = obs_tensor[idx]
            act_mb = act_tensor[idx]
            adv_mb = adv[idx]
            ret_mb = returns[idx]
            logp_old_mb = logp_tensor[idx]

            mu, log_std, values = model(obs_mb)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)

            logp_new = dist.log_prob(act_mb).sum(-1)
            ratio = torch.exp(logp_new - logp_old_mb)

            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_mb
            loss_pi = -(torch.min(ratio * adv_mb, clip_adv)).mean()
            loss_v = ((ret_mb - values.squeeze(-1)) ** 2).mean()
            entropy_bonus = ent_coef * dist.entropy().mean()

            loss = loss_pi + vf_coef * loss_v - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # KL early stopping
            approx_kl = (logp_new - logp_old_mb).mean().item()
            if approx_kl > target_kl:
                break

        with torch.no_grad():
            mu_all, log_std_all, values_all = model(obs_tensor)
            std_all = torch.exp(log_std_all)
            dist_all = torch.distributions.Normal(mu_all, std_all)

            logp_new_all = dist_all.log_prob(act_tensor).sum(-1)
            ratio_all = torch.exp(logp_new_all - logp_tensor)

            approx_kl_epoch = (logp_new_all - logp_tensor).mean().item()
            clip_fraction = ((ratio_all > (1 + clip_ratio)) | (ratio_all < (1 - clip_ratio))).float().mean().item()
            entropy = dist_all.entropy().mean().item()
            val_loss = ((returns - values_all.squeeze(-1)) ** 2).mean().item()

            y_pred = values_all.squeeze(-1).detach().cpu().numpy()
            y_true = returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

        metrics_log.append({
            "epoch": epoch + 1,
            "mean_reward": mean_reward,
            "kl": approx_kl_epoch,
            "clip_fraction": clip_fraction,
            "entropy": entropy,
            "value_loss": val_loss,
            "explained_var": explained_var
        })

        print(f"Epoch {epoch + 1}: reward={mean_reward:.2f}, "
              f"KL={approx_kl_epoch:.4f}, clip_frac={clip_fraction:.2f}, "
              f"entropy={entropy:.2f}, val_loss={val_loss:.2f}, EV={explained_var:.2f}")

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device) # Move back to device if needed
        print(f"PPO Training finished. Loaded best model with reward: {best_reward:.2f}")

    return model, mean_rewards, metrics_log


# -----------------------------------------------------------------------------
# SAC Training
# -----------------------------------------------------------------------------
def sac_train(env, epochs=1000, steps_per_epoch=512, batch_size=256, gamma=0.99,
              tau=0.005, alpha=0.2, lr=3e-4, start_steps=1000, replay_size=1000000, hidden_dim=256):
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SACActor(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    critic = SACCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    critic_target = SACCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    best_actor_state = None
    best_reward = -float('inf')
    replay_buffer = ReplayBuffer(replay_size)
    mean_rewards = []
    total_steps = 0
    
    env.mode = "sac"

    for epoch in range(epochs):
        obs, _ = env.reset()
        ep_rew = 0
        ep_len = 0

        for step in range(steps_per_epoch):
            total_steps += 1

            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    mu, std = actor(obs_t)
                    dist = torch.distributions.Normal(mu, std)
                    action_raw = dist.sample()
                    action_env = torch.clamp(action_raw, -1, 1).squeeze(0).cpu().numpy()
                    action = action_env

            next_obs, reward, done, _, _ = env.step(action)
            ep_rew += reward
            ep_len += 1

            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                obs, _ = env.reset()

            if len(replay_buffer) > batch_size and total_steps >= start_steps:
                batch = replay_buffer.sample(batch_size)
                obs_batch = torch.as_tensor(batch[0], dtype=torch.float32).to(device)
                act_batch = torch.as_tensor(batch[1], dtype=torch.float32).to(device)
                rew_batch = torch.as_tensor(batch[2], dtype=torch.float32).to(device)
                next_obs_batch = torch.as_tensor(batch[3], dtype=torch.float32).to(device)
                done_batch = torch.as_tensor(batch[4], dtype=torch.float32).to(device)

                with torch.no_grad():
                    mu_next, std_next = actor(next_obs_batch)
                    dist_next = torch.distributions.Normal(mu_next, std_next)
                    next_action_raw = dist_next.sample()
                    next_action = torch.clamp(next_action_raw, -1, 1)
                    next_log_prob = dist_next.log_prob(next_action_raw).sum(-1, keepdim=True)

                    q1_next, q2_next = critic_target(next_obs_batch, next_action)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                    target_q = rew_batch.unsqueeze(-1) + gamma * (1 - done_batch.unsqueeze(-1)) * q_next

                q1, q2 = critic(obs_batch, act_batch)
                critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                if total_steps % 2 == 0:
                    mu_pred, std_pred = actor(obs_batch)
                    dist_pred = torch.distributions.Normal(mu_pred, std_pred)
                    action_pred_raw = dist_pred.sample()
                    action_pred = torch.clamp(action_pred_raw, -1, 1)
                    log_prob = dist_pred.log_prob(action_pred_raw).sum(-1, keepdim=True)

                    q1_pred, q2_pred = critic(obs_batch, action_pred)
                    q_pred = torch.min(q1_pred, q2_pred)
                    actor_loss = (alpha * log_prob - q_pred).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        mean_rewards.append(ep_rew / max(ep_len, 1))
        print(f"SAC Epoch {epoch + 1}: mean_reward={mean_rewards[-1]:.2f}")

        # Save best model
        if mean_rewards[-1] > best_reward:
            best_reward = mean_rewards[-1]
            best_actor_state = {k: v.cpu() for k, v in actor.state_dict().items()}

    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)
        actor.to(device)
        print(f"SAC Training finished. Loaded best model with reward: {best_reward:.2f}")

    return actor, mean_rewards


# -----------------------------------------------------------------------------
# MODRL Training
# -----------------------------------------------------------------------------
def modrl_train(env, epochs=1000, steps_per_epoch=512, gamma=0.99,
                clip_ratio=0.2, lr=3e-5, train_pi_iters=80, minibatch_size=256,
                target_kl=0.03, vf_coef=0.5, ent_coef=0.01,
                weight_peak=0.4, weight_variance=0.4, weight_time=0.2, hidden_dim=256):
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = MODRLActorCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_model_state = None
    best_reward = -float('inf')
    env.mode = "modrl"

    mean_rewards = []
    mean_peak = []
    mean_variance = []
    mean_time = []

    for epoch in range(epochs):
        obs, _ = env.reset()
        obs_buf, act_buf, rew_buf = [], [], []
        peak_buf, var_buf, time_buf = [], [], []
        val_peak_buf, val_var_buf, val_time_buf, val_comb_buf = [], [], [], []
        logp_buf = []
        ep_rews = []
        ep_peaks = []
        ep_vars = []
        ep_times = []

        for t in range(steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                mu, log_std, (v_peak, v_variance, v_time, v_combined) = model(obs_t)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)

                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
            
            action_env = torch.clamp(action, -1, 1).cpu().numpy()

            next_obs, reward, done, _, info = env.step(action_env)

            obs_buf.append(obs_t)
            act_buf.append(action)
            rew_buf.append(reward)
            logp_buf.append(logp)
            
            val_peak_buf.append(v_peak)
            val_var_buf.append(v_variance)
            val_time_buf.append(v_time)
            val_comb_buf.append(v_combined)

            if "peak_load" in info:
                peak_buf.append(info["peak_load"])
                var_buf.append(info["load_variance"])
                time_buf.append(info["time_penalty"])
                ep_peaks.append(info["peak_load"])
                ep_vars.append(info["load_variance"])
                ep_times.append(-info["time_penalty"] * 10)

            obs = next_obs
            ep_rews.append(reward)

            if done:
                obs, _ = env.reset()

        mean_reward = np.mean(ep_rews) if len(ep_rews) > 0 else 0.0
        mean_rewards.append(mean_reward)
        
        if len(ep_peaks) > 0:
            mean_peak.append(np.mean(ep_peaks))
            mean_variance.append(np.mean(ep_vars))
            mean_time.append(np.mean(ep_times))

        obs_tensor = torch.stack(obs_buf)
        act_tensor = torch.stack(act_buf)
        logp_tensor = torch.stack(logp_buf)
        
        val_peak_tensor = torch.cat(val_peak_buf).squeeze(-1)
        val_var_tensor = torch.cat(val_var_buf).squeeze(-1)
        val_time_tensor = torch.cat(val_time_buf).squeeze(-1)
        val_comb_tensor = torch.cat(val_comb_buf).squeeze(-1)

        rewards = np.array(rew_buf, dtype=np.float32)
        peaks = np.array(peak_buf) if len(peak_buf) > 0 else np.zeros_like(rewards)
        variances = np.array(var_buf) if len(var_buf) > 0 else np.zeros_like(rewards)
        times = np.array(time_buf) if len(time_buf) > 0 else np.zeros_like(rewards)

        returns_peak = []
        returns_var = []
        returns_time = []
        returns_combined = []
        
        G_peak = G_var = G_time = G_comb = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val_peak = next_val_var = next_val_time = next_val_comb = 0.0
                next_non_terminal = 0.0
            else:
                next_val_peak = val_peak_tensor[i+1].item()
                next_val_var = val_var_tensor[i+1].item()
                next_val_time = val_time_tensor[i+1].item()
                next_val_comb = val_comb_tensor[i+1].item()
                next_non_terminal = 1.0

            reward_obj = -peaks[i] / 3000.0 * weight_peak
            reward_obj += -variances[i] / 500000.0 * weight_variance
            reward_obj += times[i] * weight_time
            reward_obj += rewards[i] * 0.5

            G_peak = reward_obj * weight_peak + gamma * G_peak * next_non_terminal
            G_var = reward_obj * weight_variance + gamma * G_var * next_non_terminal
            G_time = reward_obj * weight_time + gamma * G_time * next_non_terminal
            G_comb = reward_obj + gamma * G_comb * next_non_terminal

            returns_peak.insert(0, G_peak)
            returns_var.insert(0, G_var)
            returns_time.insert(0, G_time)
            returns_combined.insert(0, G_comb)

        returns_peak = torch.tensor(returns_peak, dtype=torch.float32).to(device)
        returns_var = torch.tensor(returns_var, dtype=torch.float32).to(device)
        returns_time = torch.tensor(returns_time, dtype=torch.float32).to(device)
        returns_combined = torch.tensor(returns_combined, dtype=torch.float32).to(device)

        returns_peak = (returns_peak - returns_peak.mean()) / (returns_peak.std() + 1e-8)
        returns_var = (returns_var - returns_var.mean()) / (returns_var.std() + 1e-8)
        returns_time = (returns_time - returns_time.mean()) / (returns_time.std() + 1e-8)
        returns_combined = (returns_combined - returns_combined.mean()) / (returns_combined.std() + 1e-8)

        for _ in range(train_pi_iters):
            idx = np.random.randint(0, len(obs_tensor), minibatch_size)
            obs_mb = obs_tensor[idx]
            act_mb = act_tensor[idx]
            logp_old_mb = logp_tensor[idx]
            ret_peak_mb = returns_peak[idx]
            ret_var_mb = returns_var[idx]
            ret_time_mb = returns_time[idx]
            ret_comb_mb = returns_combined[idx]
            val_peak_mb = val_peak_tensor[idx]
            val_var_mb = val_var_tensor[idx]
            val_time_mb = val_time_tensor[idx]
            val_comb_mb = val_comb_tensor[idx]

            mu, log_std, (v_peak, v_variance, v_time, v_combined) = model(obs_mb)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)

            logp_new = dist.log_prob(act_mb).sum(-1)
            ratio = torch.exp(logp_new - logp_old_mb)

            adv_peak = ret_peak_mb - val_peak_mb.detach()
            adv_var = ret_var_mb - val_var_mb.detach()
            adv_time = ret_time_mb - val_time_mb.detach()
            adv_comb = ret_comb_mb - val_comb_mb.detach()

            adv_weighted = (adv_peak * weight_peak + adv_var * weight_variance + 
                           adv_time * weight_time + adv_comb * 0.3)
            adv_weighted = (adv_weighted - adv_weighted.mean()) / (adv_weighted.std() + 1e-8)

            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_weighted
            loss_pi = -(torch.min(ratio * adv_weighted, clip_adv)).mean()

            loss_v_peak = ((ret_peak_mb - v_peak.squeeze(-1)) ** 2).mean()
            loss_v_var = ((ret_var_mb - v_variance.squeeze(-1)) ** 2).mean()
            loss_v_time = ((ret_time_mb - v_time.squeeze(-1)) ** 2).mean()
            loss_v_comb = ((ret_comb_mb - v_combined.squeeze(-1)) ** 2).mean()
            loss_v = (loss_v_peak * weight_peak + loss_v_var * weight_variance + 
                     loss_v_time * weight_time + loss_v_comb * 0.3)

            entropy_bonus = ent_coef * dist.entropy().mean()

            loss = loss_pi + vf_coef * loss_v - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            approx_kl = (logp_new - logp_old_mb).mean().item()
            if approx_kl > target_kl:
                break

        peak_str = f", peak={mean_peak[-1]:.1f}" if len(mean_peak) > 0 else ""
        var_str = f", var={mean_variance[-1]:.1f}" if len(mean_variance) > 0 else ""
        time_str = f", time={mean_time[-1]:.1f}" if len(mean_time) > 0 else ""
        print(f"MODRL Epoch {epoch + 1}: reward={mean_reward:.2f}{peak_str}{var_str}{time_str}")

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"MODRL Training finished. Loaded best model with reward: {best_reward:.2f}")

    return model, mean_rewards
