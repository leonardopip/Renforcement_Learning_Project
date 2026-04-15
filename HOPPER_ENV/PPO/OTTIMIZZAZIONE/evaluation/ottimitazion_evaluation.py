import os
import sys
import numpy as np
import csv
import torch
import glob
import gymnasium as gym
from multiprocessing import Pool, current_process
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
# We need to add the directory containing 'env' to sys.path
# Based on your image: Renforcement_Learning_Project/HOPPER_ENV/PPO/env
CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# Move up 2 levels: from 'evaluation' to 'OTTIMIZZAZIONE', then to 'PPO'
PPO_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", ".."))
sys.path.append(PPO_ROOT)

try:
    import env.custom_hopper
except ModuleNotFoundError:
    print(f"Critical: Could not find 'env' folder in {PPO_ROOT}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Using relative paths from the script location
MODELS_DIR  = os.path.join(CURRENT_SCRIPT_PATH, "..", "models")
ENV_ID_OPT  = "Hopper-MassFric-Uni-80-v1"
MODEL_PATTERN = os.path.join(MODELS_DIR, f"PPO_{ENV_ID_OPT}_s*-ottimizzato.zip")
MODEL_PATHS = glob.glob(MODEL_PATTERN)
RESULTS_DIR = os.path.join(PPO_ROOT, "results_v2")

N_EPISODES  = 50
N_PROCESSES = 4
OS_ENV_TAG  = f"OPTIMIZED_{ENV_ID_OPT}"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TARGET_IDS = [
    "CustomHopper-target-v0",
    "Hopper-Target-Mass-Easy-v0",
    "Hopper-Target-Mass-Medium-v0",
    "Hopper-Target-Mass-Hard-v0",
    "Hopper-Target-Fric-Easy-v0",
    "Hopper-Target-Fric-Medium-v0",
    "Hopper-Target-Fric-Hard-v0",
    "Hopper-Target-Both-Easy-v0",
    "Hopper-Target-Both-Medium-v0",
    "Hopper-Target-Both-Hard-v0",
]

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION KERNEL
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_target_batch(target_id):
    torch.set_num_threads(1)
    proc_name = current_process().name
    
    if not MODEL_PATHS:
        return None

    all_rewards = []
    all_lengths = []
    n_seeds_used = 0

    for model_path in MODEL_PATHS:
        try:
            model = PPO.load(model_path, device="cpu")
            eval_env = gym.make(target_id)
            ep_rewards, ep_lengths = evaluate_policy(
                model, 
                eval_env,
                n_eval_episodes=N_EPISODES,
                deterministic=True,
                return_episode_rewards=True
            )
            eval_env.close()
            all_rewards.extend(ep_rewards)
            all_lengths.extend(ep_lengths)
            n_seeds_used += 1
        except Exception as e:
            print(f"[{proc_name}] ✗ Error on {target_id} with {os.path.basename(model_path)}: {str(e)}")

    if not all_rewards:
        return None
        
    mean_r = np.mean(all_rewards)
    std_r  = np.std(all_rewards)
    mean_l = np.mean(all_lengths)

    print(f"[{proc_name}] ✓ {target_id}: Reward {mean_r:.2f} | EpLen: {mean_l:.1f}")
    return {
        "target": target_id, 
        "mean": round(float(mean_r), 2), 
        "std": round(float(std_r), 2),
        "ep_len": round(float(mean_l), 1),
        "n_seeds": n_seeds_used
    }

if __name__ == "__main__":
    print(f"Commencing Parallel Validation...")
    print(f"Policy: {ENV_ID_OPT} ({len(MODEL_PATHS)} seeds found)")
    print("─" * 70)

    with Pool(processes=N_PROCESSES) as pool:
        raw_results = pool.map(evaluate_target_batch, TARGET_IDS)

    valid_results = [res for res in raw_results if res is not None]

    if valid_results:
        n_seeds = valid_results[0]["n_seeds"]
        row = {"policy": OS_ENV_TAG, "n_seeds": n_seeds}
        for res in valid_results:
            row[f"{res['target']}_mean"] = res['mean']
            row[f"{res['target']}_std"]  = res['std']
            row[f"{res['target']}_ep_len"] = res['ep_len']

        fieldnames = ["policy", "n_seeds"] + [f"{t}_{s}" for t in TARGET_IDS for s in ["mean", "std", "ep_len"]]
        csv_path = os.path.join(RESULTS_DIR, "evaluation_matrix_OPTIMIZED.csv")
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([row])

        print(f"\n✓ Telemetry exported to: {csv_path}")