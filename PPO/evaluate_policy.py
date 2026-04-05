import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import env.custom_hopper
import csv
import torch
from multiprocessing import Pool, current_process

# ──────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────
MODELS_DIR  = "models"
RESULTS_DIR = "results"
N_EPISODES  = 50
BASE_SEED   = 42
N_SEEDS     = 5
N_PROCESSES = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

variations = [10, 20, 50, 80]

ENV_IDS = []
for percent in variations:
    ENV_IDS += [
        f"Hopper-Mass-Uni-{percent}-v0",
        f"Hopper-Mass-Gauss-{percent}-v0",
        f"Hopper-Fric-Uni-{percent}-v0",
        f"Hopper-Fric-Gauss-{percent}-v0",
        f"Hopper-MassFric-Uni-{percent}-v0",
        f"Hopper-MassFric-Gauss-{percent}-v0",
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

# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────
def get_model_path(env_id, seed):
    parts     = env_id.split("-")
    tipo      = f"{parts[1]}-{parts[2]}"
    perc      = parts[3]
    env_id_v1 = env_id.replace("-v0", "-v1")
    return os.path.join(MODELS_DIR, tipo, perc, f"PPO_{env_id_v1}_s{seed}.zip")


# ──────────────────────────────────────────────
# Worker: valuta una policy (env_id) su tutti i target
# ──────────────────────────────────────────────
def evaluate_policy_all_targets(env_id):
    torch.set_num_threads(1)
    proc_name = current_process().name
    print(f"[{proc_name}] Inizio: {env_id.replace('-v0', '-v1')}")

    seed_results = {t: [] for t in TARGET_IDS}
    n_found = 0

    for s in range(N_SEEDS):
        seed = BASE_SEED + s
        model_path = get_model_path(env_id, seed)

        if not os.path.exists(model_path):
            print(f"  ⚠ Seed {seed} non trovato: {model_path}")
            continue

        n_found += 1
        model = PPO.load(model_path, device="cpu")

        for target_id in TARGET_IDS:
            eval_env = gym.make(target_id)
            mean, _ = evaluate_policy(
                model, eval_env,
                n_eval_episodes=N_EPISODES,
                deterministic=True,
            )
            eval_env.close()
            seed_results[target_id].append(round(mean, 2))

    if n_found == 0:
        print(f"  ✗ Nessun modello trovato per {env_id}")
        return None

    row = {"policy": env_id.replace("-v0", "-v1"), "n_seeds": n_found}
    for target_id in TARGET_IDS:
        values = seed_results[target_id]
        row[f"{target_id}_mean"] = round(float(np.mean(values)), 2)
        row[f"{target_id}_std"]  = round(float(np.std(values)),  2)

    print(f"[{proc_name}] ✓ Completato: {env_id.replace('-v0', '-v1')}")
    return row


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Policy da testare:   {len(ENV_IDS)}")
    print(f"Seed per policy:     {N_SEEDS}")
    print(f"Target environments: {len(TARGET_IDS)}")
    print(f"Episodi per coppia:  {N_EPISODES}")
    print(f"Processi paralleli:  {N_PROCESSES}")
    print("═" * 60)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(evaluate_policy_all_targets, ENV_IDS)

    # Rimuovi None (modelli non trovati)
    results = [r for r in results if r is not None]

    # ── Salva CSV ─────────────────────────────
    fieldnames = ["policy", "n_seeds"] + [
        f"{t}_{s}" for t in TARGET_IDS for s in ["mean", "std"]
    ]
    csv_path = os.path.join(RESULTS_DIR, "evaluation_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Risultati salvati in: {csv_path}")

    # ── Migliore policy per ogni target ───────
    print("\n" + "═" * 60)
    print("MIGLIORE POLICY PER OGNI TARGET")
    print("═" * 60)
    for target_id in TARGET_IDS:
        col  = f"{target_id}_mean"
        best = max(results, key=lambda r: r.get(col, -np.inf))
        print(f"  {target_id:<40} → {best['policy']} ({best[col]})")