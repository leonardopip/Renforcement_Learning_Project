import os
import re
import csv
import sys
import glob
import numpy as np
import gymnasium as gym
import torch

from multiprocessing import Pool, current_process
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# ──────────────────────────────────────────────────────────────────────────────
# Path Resolution
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAC_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
HOPPER_ROOT = os.path.abspath(os.path.join(SAC_ROOT, ".."))
MODELS_DIR  = os.path.join(SAC_ROOT, "results", "models")
RESULTS_DIR = os.path.join(SAC_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, HOPPER_ROOT)
import env.custom_hopper

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Configuration
# ──────────────────────────────────────────────────────────────────────────────
N_EPISODES   = 50
N_PROCESSES  = 2   # Aumentato a 2 per valutare sia il source che il nuovo modello DR

# Valutiamo il modello source e il nuovo modello addestrato con Domain Randomization
ENV_IDS = [
    "CustomHopper-source-v0",
    "Hopper-MassFric-Uni-80-v0"
]

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

def get_model_paths(env_id: str) -> list[str]:
    """Trova tutti i path dei modelli salvati per un certo env_id."""
    if env_id == "CustomHopper-source-v0":
        model_dir = os.path.join(MODELS_DIR, "source")
    else:
        parts = env_id.split("-")
        tipo = f"{parts[1]}-{parts[2]}"
        perc = parts[3]
        model_dir = os.path.join(MODELS_DIR, tipo, perc)
        
    pattern = os.path.join(model_dir, f"SAC_{env_id}_s*.zip")
    return sorted(glob.glob(pattern))

def extract_seed(model_path: str) -> int | None:
    match = re.search(r"_s(\d+)\.zip$", os.path.basename(model_path))
    return int(match.group(1)) if match else None

def evaluate_policy_family(env_id: str) -> dict | None:
    torch.set_num_threads(1)
    proc_name = current_process().name

    model_paths = get_model_paths(env_id)
    if not model_paths:
        return None

    print(f"[{proc_name}] Inizio valutazione: {env_id} ({len(model_paths)} seed trovati)")

    row = {"policy": env_id, "n_seeds": 0, "train_seeds": "[]"}
    for tid in TARGET_IDS:
        row[f"{tid}_mean"] = np.nan
        row[f"{tid}_std"]  = np.nan
        row[f"{tid}_ep_len"] = np.nan

    seed_scores = {tid: [] for tid in TARGET_IDS}
    used_seeds = []

    for model_path in model_paths:
        seed = extract_seed(model_path)
        try:
            model = SAC.load(model_path, device="cpu")
        except Exception as e:
            print(f"[{proc_name}][ERROR] Impossibile caricare '{model_path}': {e}")
            continue

        used_seeds.append(seed)

        for target_id in TARGET_IDS:
            try:
                env = gym.make(target_id)
                ep_rewards, ep_lengths = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=N_EPISODES,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                env.close()
                seed_scores[target_id].append((float(np.mean(ep_rewards)), float(np.std(ep_rewards)), float(np.mean(ep_lengths))))
            except Exception as e:
                print(f"[{proc_name}][WARN] Fallimento su '{target_id}': {e}")
                seed_scores[target_id].append((np.nan, np.nan, np.nan))
                
    if not used_seeds:
        return None

    row["n_seeds"] = len(used_seeds)
    row["train_seeds"] = str(used_seeds)

    for target_id in TARGET_IDS:
        scores = seed_scores[target_id]
        means = np.array([s[0] for s in scores], dtype=float)
        stds  = np.array([s[1] for s in scores], dtype=float)
        lens  = np.array([s[2] for s in scores], dtype=float)
        if not np.all(np.isnan(means)):
            row[f"{target_id}_mean"] = round(float(np.nanmean(means)), 2)
            row[f"{target_id}_std"]  = round(float(np.nanmean(stds)), 2)
            row[f"{target_id}_ep_len"] = round(float(np.nanmean(lens)), 1)

    print(f"[{proc_name}] ✓ Completato: {env_id}")
    return row

def main() -> None:
    print(f"Policy SAC da testare: {len(ENV_IDS)}")
    print("═" * 60)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(evaluate_policy_family, ENV_IDS)

    results = [r for r in results if r is not None]

    if not results:
        print("[ERROR] Nessun modello trovato.")
        return

    # Salvataggio CSV
    fieldnames = ["policy", "n_seeds", "train_seeds"] + [f"{tid}_{stat}" for tid in TARGET_IDS for stat in ("mean", "std", "ep_len")]
    csv_path = os.path.join(RESULTS_DIR, "evaluation_matrix_SAC.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Matrice di valutazione SAC salvata in: {csv_path}")

if __name__ == "__main__":
    main()
