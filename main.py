import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from environment import RealWorldSmartChargingEnv, SmartChargingEnvPPO
from algorithms import ppo_train, sac_train, modrl_train
from utils import run_episode, compute_extra_metrics
from visualization import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Smart Charging RL Training")
    
    # General Args
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")    
    # Hyperparameters
    parser.add_argument("--lr_ppo", type=float, default=3e-5, help="Learning rate for PPO")
    parser.add_argument("--lr_sac", type=float, default=3e-4, help="Learning rate for SAC")
    parser.add_argument("--lr_modrl", type=float, default=3e-5, help="Learning rate for MODRL")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for networks")
    parser.add_argument("--use_best_params", action="store_true", help="Load best params from Optuna if available")
    
    args = parser.parse_args()

    best_algo = None
    best_alpha = None
    best_ent_coef = None

    if args.use_best_params:
        best_params_path = os.path.join(args.output_dir, "best_params.txt")
        if os.path.exists(best_params_path):
            print(f"Loading best parameters from {best_params_path}...")
            with open(best_params_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if ":" in line:
                        key, val = line.split(":")
                        key = key.strip()
                        val = val.strip()
                        if key == "Best Algo":
                            best_algo = val.strip()
                        elif key == "algo":
                            best_algo = val.strip()
                        elif key == "lr": 
                            args.lr_ppo = args.lr_sac = args.lr_modrl = float(val)
                        elif key == "gamma": 
                            args.gamma = float(val)
                        elif key == "batch_size": 
                            args.batch_size = int(val)
                        elif key == "hidden_dim": 
                            args.hidden_dim = int(val)
                        elif key == "epochs": 
                            args.epochs = int(val)
                        elif key == "alpha":
                            best_alpha = float(val)
                        elif key == "ent_coef":
                            best_ent_coef = float(val)
            if best_algo:
                print(f"Best algorithm from params: {best_algo}")
        else:
            print("Best params file not found. Using default/CLI arguments.")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Real World Data
    schedule_path = "bus_schedule.csv"
    if os.path.exists(schedule_path):
        real_schedule_df = pd.read_csv(schedule_path)
        print(f"Loaded real schedule with {len(real_schedule_df)} buses.")
        
        # Ensure columns exist
        required_cols = ["arrival_minute", "soc_init", "capacity"]
        if not all(col in real_schedule_df.columns for col in required_cols):
            print(f"Error: CSV missing columns. Found: {real_schedule_df.columns}")
            exit(1)
    else:
        print("Real schedule not found! Please run prepare_real_data.py first.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/verified: {args.output_dir}")
    
    viz = Visualizer(output_dir=args.output_dir)
    
    ppo_model = None
    sac_model = None
    modrl_model = None
    ppo_mean_rewards = None
    sac_mean_rewards = None
    modrl_mean_rewards = None
    ppo_metrics_log = None
    
    train_all = not args.use_best_params or best_algo is None
    algo_to_train = best_algo if args.use_best_params and best_algo else None
    
    if train_all or algo_to_train == "ppo":
        print(f"\nTraining PPO with Real World Data (lr={args.lr_ppo}, gamma={args.gamma})...")
        env_ppo = RealWorldSmartChargingEnv(real_schedule_df)
        train_kwargs = {
            "env": env_ppo,
            "epochs": args.epochs,
            "steps_per_epoch": 512,
            "lr": args.lr_ppo,
            "gamma": args.gamma,
            "minibatch_size": args.batch_size,
            "hidden_dim": args.hidden_dim
        }
        if best_ent_coef is not None:
            train_kwargs["ent_coef"] = best_ent_coef
        ppo_model, ppo_mean_rewards, ppo_metrics_log = ppo_train(**train_kwargs)
        viz.plot_training_dashboard(ppo_metrics_log, model_name="PPO")
    
    if train_all or algo_to_train == "sac":
        print(f"\nTraining SAC with Real World Data (lr={args.lr_sac}, gamma={args.gamma})...")
        env_sac = RealWorldSmartChargingEnv(real_schedule_df)
        train_kwargs = {
            "env": env_sac,
            "epochs": args.epochs,
            "steps_per_epoch": 512,
            "lr": args.lr_sac,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim
        }
        if best_alpha is not None:
            train_kwargs["alpha"] = best_alpha
        sac_model, sac_mean_rewards = sac_train(**train_kwargs)
    
    if train_all or algo_to_train == "modrl":
        print(f"\nTraining MODRL with Real World Data (lr={args.lr_modrl}, gamma={args.gamma})...")
        env_modrl = RealWorldSmartChargingEnv(real_schedule_df)
        modrl_model, modrl_mean_rewards = modrl_train(
            env_modrl, 
            epochs=args.epochs, 
            steps_per_epoch=512,
            lr=args.lr_modrl,
            gamma=args.gamma,
            minibatch_size=args.batch_size,
            hidden_dim=args.hidden_dim
        )

    training_data = {}
    if ppo_mean_rewards is not None:
        training_data["PPO"] = ppo_mean_rewards
    if sac_mean_rewards is not None:
        training_data["SAC"] = sac_mean_rewards
    if modrl_mean_rewards is not None:
        training_data["MODRL"] = modrl_mean_rewards
    
    if len(training_data) > 1:
        viz.plot_comparative_training(training_data)

    print("\nRunning Benchmark on Real Data Scenarios...")
    n_runs = 10
    results = {"RBC": []}
    
    if ppo_model is not None:
        results["ppo"] = []
    if sac_model is not None:
        results["sac"] = []
    if modrl_model is not None:
        results["modrl"] = []

    for run in range(n_runs):
        run_seed = run + args.seed
        random.seed(run_seed)
        np.random.seed(run_seed)
        
        run_schedule = real_schedule_df.copy()
        base_start_time = 0
        random_offsets = np.random.randint(0, 120, size=len(run_schedule))
        run_schedule['arrival_minute'] = base_start_time + random_offsets
        soc_noise = np.random.uniform(-0.05, 0.05, size=len(run_schedule))
        run_schedule['soc_init'] = np.clip(run_schedule['soc_init'] + soc_noise, 0.05, 0.95)
        
        policies_to_test = ["RBC"]
        if ppo_model is not None:
            policies_to_test.append("ppo")
        if sac_model is not None:
            policies_to_test.append("sac")
        if modrl_model is not None:
            policies_to_test.append("modrl")
        
        for policy in policies_to_test:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            
            test_env = SmartChargingEnvPPO(run_schedule)
            
            if policy == "ppo":
                metrics = run_episode(test_env, model=ppo_model, policy="ppo")
            elif policy == "sac":
                metrics = run_episode(test_env, model=sac_model, policy="sac")
            elif policy == "modrl":
                metrics = run_episode(test_env, model=modrl_model, policy="modrl")
            else:
                metrics = run_episode(test_env, model=None, policy="RBC")
            results[policy].append(metrics)

    # Summary after benchmark
    df_list = []
    for policy, runs in results.items():
        for power_logs, total_loads, finish_time, peak_load, total_energy in runs:
            df_list.append({
                "policy": policy,
                "finish_time": finish_time,
                "peak_load": peak_load,
                "total_energy": total_energy,
                "variance": np.var(total_loads),
                "energy_delivered": sum(total_loads) / 60.0
            })
    df = pd.DataFrame(df_list)
    summary = df.groupby("policy").agg({
        "finish_time": ["mean", "std"],
        "peak_load": ["mean", "std"],
        "total_energy": ["mean", "std"],
        "variance": ["mean", "std"],
        "energy_delivered": ["mean", "std"]
    })
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== REAL WORLD DATA BENCHMARK SUMMARY ===")
    print(summary)
    
    summary_path = os.path.join(args.output_dir, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== REAL WORLD DATA BENCHMARK COMPARISON RESULTS ===\n\n")
        f.write(str(summary))
        f.write("\n\n")
    print(f"\nSaved: {summary_path}")

    # Plot Benchmark Boxplots
    viz.plot_benchmark_boxplots(df)

    plot_env = RealWorldSmartChargingEnv(real_schedule_df)
    
    unctrl_powers, unctrl_loads, unctrl_finish_time, _, _ = run_episode(plot_env, model=None, policy="RBC")
    
    load_profile_data = {"RBC": unctrl_loads}
    load_dist_data = {"RBC": unctrl_loads}
    
    ppo_powers = None
    ppo_loads = None
    ppo_finish_time = None
    sac_powers = None
    sac_loads = None
    sac_finish_time = None
    modrl_powers = None
    modrl_loads = None
    modrl_finish_time = None
    
    if ppo_model is not None:
        ppo_powers, ppo_loads, ppo_finish_time, _, _ = run_episode(plot_env, model=ppo_model, policy="ppo")
        load_profile_data["PPO"] = ppo_loads
        load_dist_data["PPO"] = ppo_loads
    
    if sac_model is not None:
        sac_powers, sac_loads, sac_finish_time, _, _ = run_episode(plot_env, model=sac_model, policy="sac")
        load_profile_data["SAC"] = sac_loads
        load_dist_data["SAC"] = sac_loads
    
    if modrl_model is not None:
        modrl_powers, modrl_loads, modrl_finish_time, _, _ = run_episode(plot_env, model=modrl_model, policy="modrl")
        load_profile_data["MODRL"] = modrl_loads
        load_dist_data["MODRL"] = modrl_loads

    viz.plot_load_profile_comparison(load_profile_data)
    viz.plot_load_distribution(load_dist_data)

    viz.plot_charger_heatmap(unctrl_powers, "RBC")
    if ppo_powers is not None:
        viz.plot_charger_heatmap(ppo_powers, "PPO")
    if sac_powers is not None:
        viz.plot_charger_heatmap(sac_powers, "SAC")
    if modrl_powers is not None:
        viz.plot_charger_heatmap(modrl_powers, "MODRL")
    
    metrics_path = os.path.join(args.output_dir, "metrics_comparison.txt")
    with open(metrics_path, "w") as f:
        f.write("=== DETAILED METRICS COMPARISON ===\n\n")
        metrics = compute_extra_metrics(
            unctrl_loads, ppo_loads, sac_loads, modrl_loads,
            unctrl_finish_time, ppo_finish_time, sac_finish_time, modrl_finish_time
        )
        
        algo_order = ["RBC", "PPO", "SAC", "MODRL"]
        metric_order = [
            "Peak Load",
            "Peak Reduction",
            "Variance",
            "Total Energy",
            "Finish Time"
        ]
        
        for algo in algo_order:
            algo_metrics = []
            for key, value in metrics.items():
                if f"({algo})" in key:
                    algo_metrics.append((key, value))
            
            if algo_metrics:
                f.write(f"--- {algo} ---\n")
                sorted_metrics = []
                for metric_type in metric_order:
                    for key, value in algo_metrics:
                        if metric_type.lower() in key.lower() and (key, value) not in sorted_metrics:
                            sorted_metrics.append((key, value))
                for key, value in sorted_metrics:
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
    print(f"\nSaved: {metrics_path}")

if __name__ == "__main__":
    main()
