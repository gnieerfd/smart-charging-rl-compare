import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class Visualizer:
    def __init__(self, output_dir="real_world_data/output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set style for professional looking plots
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        self.colors = {
            "RBC": "#e74c3c",    # Red
            "ppo": "#3498db",    # Blue
            "sac": "#2ecc71",    # Green
            "modrl": "#9b59b6"   # Purple
        }

    def plot_training_dashboard(self, metrics_log, model_name="PPO"):
        """
        Plots a 2x2 dashboard of training metrics: Reward, Entropy, Value Loss, KL/Clip.
        metrics_log: List of dicts containing training metrics.
        """
        if not metrics_log:
            print("No metrics log provided for plotting.")
            return

        df = pd.DataFrame(metrics_log)
        epochs = df["epoch"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{model_name} Training Dashboard", fontsize=16)

        # 1. Mean Reward
        sns.lineplot(ax=axes[0, 0], x=epochs, y=df["mean_reward"], color=self.colors.get(model_name.lower(), "blue"), linewidth=2)
        axes[0, 0].set_title("Mean Reward per Epoch")
        axes[0, 0].set_ylabel("Reward")

        # 2. Entropy
        if "entropy" in df.columns:
            sns.lineplot(ax=axes[0, 1], x=epochs, y=df["entropy"], color="orange", linewidth=2)
            axes[0, 1].set_title("Policy Entropy (Exploration)")
            axes[0, 1].set_ylabel("Entropy")

        # 3. Value Loss
        if "value_loss" in df.columns:
            sns.lineplot(ax=axes[1, 0], x=epochs, y=df["value_loss"], color="red", linewidth=2)
            axes[1, 0].set_title("Value Function Loss")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_yscale("log")

        # 4. KL Divergence or Explained Variance
        if "kl" in df.columns:
            sns.lineplot(ax=axes[1, 1], x=epochs, y=df["kl"], color="green", linewidth=2)
            axes[1, 1].set_title("Approx. KL Divergence")
            axes[1, 1].set_ylabel("KL")
        elif "explained_var" in df.columns:
            sns.lineplot(ax=axes[1, 1], x=epochs, y=df["explained_var"], color="purple", linewidth=2)
            axes[1, 1].set_title("Explained Variance")
            axes[1, 1].set_ylabel("EV")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"training_dashboard_{model_name.lower()}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_comparative_training(self, rewards_dict):
        """
        Plots mean rewards for multiple models on one chart.
        rewards_dict: {"PPO": [r1, r2...], "SAC": [r1, r2...]}
        """
        plt.figure(figsize=(10, 6))
        for name, rewards in rewards_dict.items():
            plt.plot(rewards, label=name, linewidth=2, color=self.colors.get(name.lower(), None))
        
        plt.title("Training Progress Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, "training_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_load_profile_comparison(self, load_profiles_dict):
        """
        Plots the total load profile over time for different policies.
        load_profiles_dict: {"RBC": [l1, l2...], "PPO": [l1, l2...]}
        """
        plt.figure(figsize=(12, 6))
        
        for name, load in load_profiles_dict.items():
            if load is None: continue
            # Smooth lines slightly for better visualization if dense
            plt.plot(load, label=name, linewidth=2, alpha=0.8, color=self.colors.get(name.lower(), None))

        plt.title("Total Depot Load Profile Comparison (One Episode)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Total Power (kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, "load_profile_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_charger_heatmap(self, power_logs, policy_name):
        """
        Plots a heatmap of charger utilization over time.
        power_logs: np.array of shape (time_steps, num_chargers)
        """
        if power_logs is None or len(power_logs) == 0:
            return

        # Ensure power_logs is a numpy array
        power_logs = np.array(power_logs)

        plt.figure(figsize=(12, 8))
        # Transpose so Chargers are Y-axis, Time is X-axis
        sns.heatmap(power_logs.T, cmap="viridis", cbar_kws={'label': 'Power (kW)'}, vmin=0)
        
        plt.title(f"Charger Utilization Heatmap - {policy_name}")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Charger ID")
        
        save_path = os.path.join(self.output_dir, f"heatmap_{policy_name.lower()}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_benchmark_boxplots(self, results_df):
        """
        Plots boxplots for Peak Load and Variance to show stability across runs.
        results_df: DataFrame containing benchmark results with columns 'policy', 'peak_load', 'variance'
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Peak Load Boxplot
        sns.boxplot(data=results_df, x="policy", y="peak_load", ax=axes[0], hue="policy", palette=self.colors, legend=False)
        axes[0].set_title("Peak Load Distribution (Lower is Better)")
        axes[0].set_ylabel("Peak Load (kW)")
        
        # Variance Boxplot
        sns.boxplot(data=results_df, x="policy", y="variance", ax=axes[1], hue="policy", palette=self.colors, legend=False)
        axes[1].set_title("Load Variance Distribution (Lower is Better)")
        axes[1].set_ylabel("Variance")
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "benchmark_boxplots.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_load_distribution(self, load_profiles_dict):
        """
        Plots KDE (Kernel Density Estimate) of load distributions.
        """
        plt.figure(figsize=(10, 6))
        for name, load in load_profiles_dict.items():
            if load is None: continue
            sns.kdeplot(load, label=name, fill=True, alpha=0.3, linewidth=2, color=self.colors.get(name.lower(), None))
            
        plt.title("Load Distribution Density")
        plt.xlabel("Total Load (kW)")
        plt.ylabel("Density")
        plt.legend()
        
        save_path = os.path.join(self.output_dir, "load_distribution_density.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")
