import optuna
import pandas as pd
import os
import torch
import numpy as np
import random
from environment import RealWorldSmartChargingEnv
from algorithms import ppo_train, sac_train, modrl_train

def objective(trial):
    # 0. Select Algorithm to tune
    algo = trial.suggest_categorical("algo", ["ppo", "sac", "modrl"])

    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 5, 100, step=5)
    
    # 2. Setup Environment
    schedule_path = "real_world_data/bus_schedule.csv"
    if not os.path.exists(schedule_path):
        return -float('inf')
    
    real_schedule_df = pd.read_csv(schedule_path)
    env = RealWorldSmartChargingEnv(real_schedule_df)
    
    # 3. Train (Short run for tuning)
    try:
        if algo == "ppo":
            ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True)
            model, mean_rewards, _ = ppo_train(
                env, 
                epochs=epochs, 
                steps_per_epoch=512,
                lr=lr,
                gamma=gamma,
                minibatch_size=batch_size,
                hidden_dim=hidden_dim,
                ent_coef=ent_coef
            )
        elif algo == "sac":
            alpha = trial.suggest_float("alpha", 0.1, 0.5)
            model, mean_rewards = sac_train(
                env,
                epochs=epochs,
                steps_per_epoch=512,
                lr=lr,
                gamma=gamma,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                alpha=alpha
            )
        else: # modrl
            model, mean_rewards = modrl_train(
                env,
                epochs=epochs,
                steps_per_epoch=512,
                lr=lr,
                gamma=gamma,
                minibatch_size=batch_size,
                hidden_dim=hidden_dim
            )
        
        # Return the average of the last 3 epochs as the score
        score = np.mean(mean_rewards[-3:])
        return score
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -float('inf')

def main():
    print("Starting Optuna Study for RL Hyperparameter Tuning (PPO, SAC, MODRL)...")
    
    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30) # Increased trials for multiple algos

    print("\nStudy finished!")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params to a file
    best_params_path = "real_world_data/output/best_params.txt"
    os.makedirs("real_world_data/output", exist_ok=True)
    with open(best_params_path, "w") as f:
        f.write(f"Best Algo: {trial.params['algo']}\n")
        f.write(f"Best Value: {trial.value}\n")
        f.write("Best Params:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest parameters saved to: {best_params_path}")

if __name__ == "__main__":
    main()

