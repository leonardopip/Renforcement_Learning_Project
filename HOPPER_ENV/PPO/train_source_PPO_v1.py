import os
import torch
import gymnasium as gym
import numpy as np
import gc
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Pool, current_process

# Assicurati che l'import rifletta la struttura delle tue cartelle
import env.custom_hopper 

# --- CONFIGURAZIONE PERCORSI ---
# Modificato per rispecchiare la struttura HOPPER_ENV/PPO/results/
BASE_PATH = os.path.join("HOPPER_ENV", "PPO")
SAVE_DIR  = os.path.join(BASE_PATH, "results", "models")
LOG_DIR   = os.path.join(BASE_PATH, "results", "logs")

# --- CONFIGURAZIONE TRAINING ---
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

TIMESTEPS = 1_000_000
N_PROCESSES = 10
N_ENVS = 1  
BASE_SEED = 42
N_SEEDS = 5

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def generate_robust_seeds(base_seed, n):
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(low=1000, high=1000000, size=n)]

def train_one(args):
    env_id, seed = args
    proc_name = current_process().name

    torch.set_num_threads(1)

    # Parsing per sottocartelle (es. Mass-Uni/10)
    parts = env_id.split("-")
    tipo = f"{parts[1]}-{parts[2]}"
    perc = parts[3]

    # Percorsi specifici per questa variante
    specific_save_dir = os.path.join(SAVE_DIR, tipo, perc)
    specific_log_dir  = os.path.join(LOG_DIR, tipo, perc)
    os.makedirs(specific_save_dir, exist_ok=True)
    os.makedirs(specific_log_dir, exist_ok=True)

    print(f"[{proc_name}] Inizio: {env_id} (seed={seed})")

    try:
        env = make_vec_env(
            env_id,
            n_envs=N_ENVS,
            seed=seed,
            monitor_dir=specific_log_dir,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            device="cpu",
            tensorboard_log=specific_log_dir,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=f"PPO_{env_id}_s{seed}",
            progress_bar=False,
        )

        save_path = os.path.join(specific_save_dir, f"PPO_{env_id}_s{seed}")
        model.save(save_path)
        
        env.close()
        del model
        del env
        gc.collect()

        print(f"[{proc_name}] ✓ Completato: {env_id} (s={seed})")
        return f"{env_id}_s{seed}", True

    except Exception as e:
        print(f"[{proc_name}] ✗ Errore su {env_id} (seed={seed}): {e}")
        return f"{env_id}_s{seed}", False


if __name__ == "__main__":
    seeds = generate_robust_seeds(BASE_SEED, N_SEEDS)
    
    jobs = []
    for env_id in ENV_IDS:
        for s in seeds:
            jobs.append((env_id, s))

    print(f"Directory di salvataggio: {SAVE_DIR}")
    print(f"Totale training:          {len(jobs)}")
    print("─" * 50)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(train_one, jobs)

    print("\n" + "═" * 50)
    print("RIEPILOGO ESECUZIONE")
    print("═" * 50)
    
    n_ok = sum(ok for _, ok in results)
    for res_name, ok in results:
        if not ok:
            print(f"  ✗ FALLITO: {res_name}")
    
    print(f"\nCompletati con successo: {n_ok}/{len(results)}")