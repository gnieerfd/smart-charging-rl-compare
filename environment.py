import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

# -----------------------------------------------------------------------------
# Environment: Battery depot charging with PPO and Rule Based Control (RBC) modes
# -----------------------------------------------------------------------------
class SmartChargingEnvPPO(gym.Env):
    def __init__(self, bus_df, num_chargers=15):
        super().__init__()
        self.num_buses = len(bus_df)
        self.arrival = bus_df["arrival_minute"].to_numpy()
        self.soc_init = bus_df["soc_init"].to_numpy()
        self.capacity = bus_df["capacity"].to_numpy()
        self.energy = (self.soc_init * self.capacity).astype(np.float32)
        self.required = ((1.0 - self.soc_init) * self.capacity).astype(np.float32)
        self.status = np.zeros(self.num_buses, dtype=np.int8)  # 0=waiting,1=queued,2=done
        self.num_chargers = num_chargers
        self.max_power = 200  # kW per charger
        self.current_time = 0
        self.queue = []
        self.chargers = [None] * num_chargers
        self.cooldown = [0] * num_chargers

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_chargers * 4 + 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_chargers,), dtype=np.float32
        )

        # Log for plotting
        self.power_log = []
        self.charger_powers = np.zeros(num_chargers, dtype=np.float32) # Current power per charger
        self.current_total_load = 0.0 # Current total load

        # For PPO smoothing
        self.prev_action = np.zeros(num_chargers, dtype=np.float32)
        self.prev_power = np.zeros(num_chargers, dtype=np.float32)
        self.mode = "ppo" # Default mode

    def reset(self, seed=None, options=None, schedule_df=None):
        self.current_time = 0
        self.queue = []
        self.chargers = [None] * self.num_chargers
        self.cooldown = [0] * self.num_chargers
        
        if schedule_df is not None:
            # Update bus data from schedule
            self.num_buses = len(schedule_df)
            self.arrival = schedule_df["arrival_minute"].to_numpy()
            self.soc_init = schedule_df["soc_init"].to_numpy()
            self.capacity = schedule_df["capacity"].to_numpy()
            self.status = np.zeros(self.num_buses, dtype=np.int8)
            
        self.energy = (self.soc_init * self.capacity).astype(np.float32)
        self.status[:] = 0
        self.power_log = []  # <--- reset log
        self.prev_power[:] = 0.0
        self.charger_powers[:] = 0.0
        self.current_total_load = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for charger in self.chargers:
            if charger is None:
                # saat sedang cooldown, flag tetap 0 pada 'in-use'
                obs.extend([0, 0, 0, 0])
            else:
                bus_idx = charger
                # SOC saat ini (0–1)
                current_soc = self.energy[bus_idx] / self.capacity[bus_idx]
                # Remaining SOC ke penuh
                remaining_soc = 1.0 - current_soc

                progress = current_soc
                time_since = (self.current_time - self.arrival[bus_idx]) / 600  # tetap skala 0–1
                remaining = remaining_soc
                obs.extend([progress, 1, time_since, remaining])
        obs.append(len(self.queue) / self.num_buses)
        
        # Add total power normalized (approx max power = num_chargers * max_power)
        current_total_power = sum([p for p in self.power_log[-1]]) if self.power_log else 0
        max_possible_power = self.num_chargers * self.max_power
        obs.append(current_total_power / max_possible_power)
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        mode = self.mode
        # Advance time & cooldown
        self.current_time += 1
        for i in range(self.num_chargers):
            if self.cooldown[i] > 0:
                self.cooldown[i] -= 1

        # Add arrivals
        arrivals = np.where((self.status == 0) & (self.arrival == self.current_time))[0]
        for idx in arrivals:
            self.queue.append(idx)
            self.status[idx] = 1

        # Allocate free chargers
        for i in range(self.num_chargers):
            if self.chargers[i] is None and (mode == "RBC" or self.cooldown[i] == 0) and self.queue:
                bus_idx = self.queue.pop(0)
                self.chargers[i] = bus_idx

        total_power = 0
        reward = 0
        powers = []

        # Compute charging per bus
        for i, bus_idx in enumerate(self.chargers):
            if bus_idx is not None:
                current_soc = self.energy[bus_idx] / self.capacity[bus_idx]

                # Determine raw_power
                # Ubah profil daya Rule BAsed Control
                if mode == "RBC":
                    if current_soc < 0.8:
                        raw_power = self.max_power
                    else:
                        taper_factor = 1.0 - (current_soc - 0.8) / 0.2
                        taper_factor = max(0.25, taper_factor)
                        raw_power = self.max_power * taper_factor
                else:
                    # PPO, SAC, and MODRL logic
                    power = ((action[i] + 1) / 2) * self.max_power
                    if current_soc < 0.4:
                        raw_power = self.max_power
                    elif current_soc < 0.8:
                        base_power = ((action[i] + 1) / 2) * self.max_power
                        raw_power = max(150.0, base_power)
                    else:
                        taper_factor = 1.0 - 0.75 * (current_soc - 0.8) / 0.2
                        taper_factor = max(0.25, taper_factor)
                        raw_power = self.max_power * taper_factor

                # +Batas kenaikan dan penurunan daya
                max_ramp_up = 10.0  # kW per menit, sesuaikan jika mau lebih landai
                max_ramp_down = 10.0  # kW per menit, atur untuk turunan

                # Rate limiter for PPO, SAC, and MODRL
                if mode == "ppo" or mode == "sac" or mode == "modrl":
                    delta = raw_power - self.prev_power[i]
                    delta = np.clip(delta, -max_ramp_down, max_ramp_up)
                    power = self.prev_power[i] + delta
                    self.prev_power[i] = power
                else:
                    power = raw_power

                # Apply energy update
                energy = power / 90
                target_energy = self.capacity[bus_idx]
                prev_energy = self.energy[bus_idx]
                self.energy[bus_idx] = min(prev_energy + energy, target_energy)
                total_power += power
                powers.append(power)

                # Mark done & cooldown
                if self.energy[bus_idx] >= target_energy - 1e-6:
                    self.status[bus_idx] = 2
                    self.chargers[i] = None
                    if mode != "RBC":
                        self.cooldown[i] = np.random.randint(2, 6)

                # PPO, SAC, and MODRL reward shaping
                if mode == "ppo" or mode == "sac" or mode == "modrl":
                    delta_soc = (self.energy[bus_idx] - prev_energy) / self.capacity[bus_idx]
                    reward += delta_soc * 5.0  # bonus per % SOC naik (Reduced from 10.0)
            else:
                powers.append(0.0)
        self.power_log.append(powers)
        
        # Update state variables for external access
        self.charger_powers = np.array(powers, dtype=np.float32)
        self.current_total_load = total_power

        # Penalties & bonuses for PPO, SAC, and MODRL
        if mode == "ppo" or mode == "sac" or mode == "modrl":
            idle_penalty = sum([1 for i, c in enumerate(self.chargers)
                                if c is None and self.cooldown[i] == 0]) * -0.1
            queue_penalty = len(self.queue) * -0.05
            power_coef = -0.001 # Changed to negative to penalize high power
            reward += total_power * power_coef + idle_penalty * 0.1 + queue_penalty * 0.1

            # smoothness penalty
            smooth_penalty = -0.01 * np.sum((action - self.prev_action) ** 2)
            reward += smooth_penalty
            self.prev_action = action.copy()

        all_done = np.all(self.status == 2)
        time_cap = self.current_time >= 1000
        done = all_done or time_cap

        info = {}
        if mode == "modrl":
            info["peak_load"] = total_power
            info["load_variance"] = np.var([sum(p) for p in self.power_log[-10:]] + [total_power]) if len(self.power_log) > 0 else 0
            info["time_penalty"] = -0.1 * self.current_time / 1000.0

        return self._get_obs(), reward, done, False, info

class RealWorldSmartChargingEnv(SmartChargingEnvPPO):
    def __init__(self, base_schedule_df, num_chargers=15):
        super().__init__(base_schedule_df, num_chargers)
        self.base_schedule = base_schedule_df.copy()
        
    def reset(self, seed=None, options=None):
        # Data Augmentation
        current_schedule = self.base_schedule.copy()
        
        # Randomize arrival times
        # Original data has arrival=0 (relative). We shift to "night" (e.g. 21:00 = 1260 min)
        # Shift to random time between 21:00 and 23:00
        base_start_time = 0 # Relative to simulation start
        
        # Add random offset for each bus
        random_offsets = np.random.randint(0, 120, size=len(current_schedule)) # 0 to 2 hours window
        current_schedule['arrival_minute'] = base_start_time + random_offsets
        
        # Randomize SOC slightly (+/- 5%)
        soc_noise = np.random.uniform(-0.05, 0.05, size=len(current_schedule))
        current_schedule['soc_init'] = np.clip(current_schedule['soc_init'] + soc_noise, 0.05, 0.95)
        
        return super().reset(seed=seed, options=options, schedule_df=current_schedule)
