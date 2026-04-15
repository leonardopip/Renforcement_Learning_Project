import os
import sys
import time
import csv
import torch
import numpy as np
import gc
from multiprocessing import Pool, current_process
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# Aggiungiamo la root del progetto al PYTHONPATH per importare env
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import env.custom_hopper

# --- CONFIGURAZIONE ---
# Selezioniamo solo l'ambiente migliore usato anche nell'ottimizzazione PPO
ENV_IDS = [
    "Hopper-MassFric-Uni-80-v0"
]

TIMESTEPS = 1_000_000
N_PROCESSES = 2  # SAC è pesante, 2 processi simultanei per 2 seed
N_ENVS = 1
SAVE_DIR = os.path.join(SCRIPT_DIR, "results", "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "results", "logs")
BASE_SEED = 42
N_SEEDS = 2  # Usiamo 2 seed per contenere i tempi di calcolo

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def generate_robust_seeds(base_seed, n):
    """Genera n seed deterministici ma non consecutivi."""
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(low=1000, high=1000000, size=n)]

def train_one(args):
    env_id, seed = args
    proc_name = current_process().name

    torch.set_num_threads(1)
    start_time = time.time()

    # Parsing per creare le sottocartelle corrette (es. MassFric-Uni/80)
    parts = env_id.split("-")
    tipo = f"{parts[1]}-{parts[2]}"
    perc = parts[3]

    save_dir = os.path.join(SAVE_DIR, tipo, perc)
    log_dir = os.path.join(LOG_DIR, tipo, perc)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[{proc_name}] Inizio training: {env_id} (seed={seed})")

    try:
        env = make_vec_env(
            env_id,
            n_envs=N_ENVS,
            seed=seed,
            monitor_dir=log_dir,
        )

        model = SAC(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            device="cpu",
            tensorboard_log=log_dir,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=f"SAC_{env_id}_s{seed}",
            progress_bar=False,
        )

        save_path = os.path.join(save_dir, f"SAC_{env_id}_s{seed}")
        model.save(save_path)
        env.close()
        
        del model
        del env
        gc.collect()
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60.0
        print(f"[{proc_name}] ✓ Completato: {env_id} (s={seed}) in {elapsed_min:.2f} min")
        return {"env_id": env_id, "seed": seed, "success": True, "time_min": round(elapsed_min, 2)}

    except Exception as e:
        print(f"[{proc_name}] ✗ Errore su {env_id} (seed={seed}): {e}")
        return {"env_id": env_id, "seed": seed, "success": False, "time_min": 0.0}

if __name__ == "__main__":
    seeds = generate_robust_seeds(BASE_SEED, N_SEEDS)
    jobs = [(env_id, s) for env_id in ENV_IDS for s in seeds]

    print(f"Ambienti Domain Randomization: {len(ENV_IDS)}")
    print(f"Totale esecuzioni (jobs):      {len(jobs)}")
    print("─" * 50)

    # Avvia il Pool di processi
    start_total = time.time()
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(train_one, jobs)

    total_time_min = (time.time() - start_total) / 60.0
    print(f"\nTempo totale di esecuzione globale: {total_time_min:.2f} min")

    # --- SALVATAGGIO TEMPI IN CSV ---
    csv_path = os.path.join(SCRIPT_DIR, "results", "training_times_SAC_DR.csv")
    fieldnames = ["env_id", "seed", "success", "time_min"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"✓ Statistiche di tempo salvate in: {csv_path}")
