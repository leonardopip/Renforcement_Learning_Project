import sys
import os

# ──────────────────────────────────────────────
# Path setup: adatta i path alla struttura
# PPO/
# ├── models_pusher/
# ├── pusher_PPO.py
# └── test/
#     └── test_pusher_PPO.py  ← questo file
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PPO/
sys.path.insert(0, BASE_DIR)  # permette import di pusher_PPO.py

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import pusher_PPO  # registra gli ambienti custom
import csv
import torch
from multiprocessing import Pool, current_process

# ──────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────
MODELS_DIR  = os.path.join(BASE_DIR, "models_pusher")
RESULTS_DIR = os.path.join(BASE_DIR, "results_pusher")
N_EPISODES  = 50
BASE_SEED   = 42
N_SEEDS     = 5
N_PROCESSES = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

variations = [10, 20, 50, 80]

ENV_IDS = []
for percent in variations:
    ENV_IDS += [
        f"Pusher-Mass-Uni-{percent}-v0",
        f"Pusher-Mass-Gauss-{percent}-v0",
        f"Pusher-Fric-Uni-{percent}-v0",
        f"Pusher-Fric-Gauss-{percent}-v0",
        f"Pusher-MassFric-Uni-{percent}-v0",
        f"Pusher-MassFric-Gauss-{percent}-v0",
    ]

TARGET_IDS = [
    "CustomPusher-target-v0",
    "Pusher-Target-Mass-Easy-v0",
    "Pusher-Target-Mass-Medium-v0",
    "Pusher-Target-Mass-Hard-v0",
    "Pusher-Target-Fric-Easy-v0",
    "Pusher-Target-Fric-Medium-v0",
    "Pusher-Target-Fric-Hard-v0",
    "Pusher-Target-Both-Easy-v0",
    "Pusher-Target-Both-Medium-v0",
    "Pusher-Target-Both-Hard-v0",
]

# ──────────────────────────────────────────────
# Seed: identici a quelli usati nel training
# Stessa funzione e stesso BASE_SEED del training
# ──────────────────────────────────────────────
def generate_robust_seeds(base_seed, n):
    """Genera n seed deterministici ma non consecutivi, identici al training."""
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(low=1000, high=1000000, size=n)]

seeds = generate_robust_seeds(BASE_SEED, N_SEEDS)

# ──────────────────────────────────────────────
# Utility: costruisce il path del modello salvato
# Struttura: MODELS_DIR/{tipo}/{perc}/PPO_{env_id}_s{seed}.zip
# Deve essere identica al formato usato in training
# ──────────────────────────────────────────────
def get_model_path(env_id, seed):
    parts = env_id.split("-")         # es. ["Pusher", "Mass", "Uni", "10", "v0"]
    tipo  = f"{parts[1]}-{parts[2]}"  # es. "Mass-Uni"
    perc  = parts[3]                  # es. "10"
    return os.path.join(MODELS_DIR, tipo, perc, f"PPO_{env_id}_s{seed}.zip")


# ──────────────────────────────────────────────
# Worker: carica ogni modello e lo valuta su
# tutti i target environment
# ──────────────────────────────────────────────
def evaluate_policy_all_targets(env_id):
    torch.set_num_threads(1)  # evita overhead da parallelismo interno torch
    proc_name = current_process().name
    print(f"[{proc_name}] Inizio: {env_id}")

    # Struttura dati per raccogliere risultati per seed
    seed_results = {t: {"means": [], "stds_ep": []} for t in TARGET_IDS}
    n_found = 0

    for seed in seeds:
        model_path = get_model_path(env_id, seed)

        # Debug: verifica esistenza del file modello
        if not os.path.exists(model_path):
            print(f"  [{proc_name}] ⚠ Modello non trovato (seed={seed}): {model_path}")
            continue

        n_found += 1
        print(f"  [{proc_name}] Carico modello seed={seed}: {model_path}")

        try:
            model = PPO.load(model_path, device="cpu")
        except Exception as e:
            print(f"  [{proc_name}] ✗ Errore caricamento modello seed={seed}: {e}")
            continue

        for target_id in TARGET_IDS:
            try:
                eval_env = gym.make(target_id)
                eval_env.reset(seed=seed)  # seed fisso per riproducibilità

                mean, std_ep = evaluate_policy(
                    model, eval_env,
                    n_eval_episodes=N_EPISODES,
                    deterministic=True,
                )
                eval_env.close()

                seed_results[target_id]["means"].append(round(mean, 2))
                seed_results[target_id]["stds_ep"].append(round(std_ep, 2))

                print(f"  [{proc_name}] {target_id} | seed={seed} "
                      f"→ mean={mean:.2f}, std_ep={std_ep:.2f}")

            except Exception as e:
                print(f"  [{proc_name}] ✗ Errore valutazione su {target_id} "
                      f"(seed={seed}): {e}")

        # Libera memoria esplicitamente
        del model

    # Se nessun modello trovato, salta questa policy
    if n_found == 0:
        print(f"[{proc_name}] ✗ Nessun modello trovato per {env_id}, skip.")
        return None

    # Aggrega risultati tra seed:
    # - std_seed:    variabilità dovuta al training (diversi seed)
    # - std_ep_mean: variabilità media della policy in esecuzione
    row = {"policy": env_id, "n_seeds": n_found}
    for target_id in TARGET_IDS:
        means   = seed_results[target_id]["means"]
        stds_ep = seed_results[target_id]["stds_ep"]
        row[f"{target_id}_mean"]        = round(float(np.mean(means)),    2)
        row[f"{target_id}_std_seed"]    = round(float(np.std(means)),     2)
        row[f"{target_id}_std_ep_mean"] = round(float(np.mean(stds_ep)), 2)

    print(f"[{proc_name}] ✓ Completato: {env_id} ({n_found}/{N_SEEDS} seed trovati)")
    return row


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"BASE_DIR:            {BASE_DIR}")
    print(f"MODELS_DIR:          {MODELS_DIR}")
    print(f"RESULTS_DIR:         {RESULTS_DIR}")
    print(f"Seed usati:          {seeds}")
    print(f"Policy da testare:   {len(ENV_IDS)}")
    print(f"Target environments: {len(TARGET_IDS)}")
    print(f"Episodi per coppia:  {N_EPISODES}")
    print(f"Processi paralleli:  {N_PROCESSES}")
    print("═" * 60)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(evaluate_policy_all_targets, ENV_IDS)

    # Rimuovi eventuali None (policy senza modelli trovati)
    results = [r for r in results if r is not None]

    if not results:
        print("✗ Nessun risultato da salvare. Controlla i path dei modelli.")
        exit(1)

    # ── Salva CSV ─────────────────────────────
    fieldnames = ["policy", "n_seeds"] + [
        f"{t}_{s}"
        for t in TARGET_IDS
        for s in ["mean", "std_seed", "std_ep_mean"]
    ]

    csv_path = os.path.join(RESULTS_DIR, "evaluation_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Risultati salvati in: {csv_path}")

    # ── Migliore policy per ogni target ───────
    print("\n" + "═" * 60)
    print("MIGLIORE POLICY PER OGNI TARGET (per mean reward)")
    print("═" * 60)
    for target_id in TARGET_IDS:
        col  = f"{target_id}_mean"
        best = max(results, key=lambda r: r.get(col, -np.inf))
        print(f"  {target_id:<45} → {best['policy']:<35} "
              f"mean={best[col]:.2f}  std_seed={best[f'{target_id}_std_seed']:.2f}")