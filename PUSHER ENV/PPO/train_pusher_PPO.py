import os
import torch
import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Pool, current_process

# Import del file che registra gli ambienti custom del pusher
import pusher_PPO 

# --- CONFIGURAZIONE ---
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

TIMESTEPS = 1_000_000
N_PROCESSES = 10
N_ENVS = 1
SAVE_DIR = "models_pusher"
LOG_DIR = "tensorboard_logs_pusher"
BASE_SEED = 42
N_SEEDS = 5

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def generate_robust_seeds(base_seed, n):
    """Genera n seed deterministici ma non consecutivi."""
    rng = np.random.default_rng(base_seed)
    # Generiamo numeri grandi per evitare sovrapposizioni con counter interni
    return [int(s) for s in rng.integers(low=1000, high=1000000, size=n)]

def train_one(args):
    env_id, seed = args
    proc_name = current_process().name

    # Importante per multiprocessing su CPU
    torch.set_num_threads(1)

    # Parsing nome ambiente per cartelle
    parts = env_id.split("-")
    tipo = f"{parts[1]}-{parts[2]}"
    perc = parts[3]

    save_dir = os.path.join(SAVE_DIR, tipo, perc)
    log_dir = os.path.join(LOG_DIR, tipo, perc)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[{proc_name}] Inizio: {env_id} (seed={seed})")

    try:
        env = make_vec_env(
            env_id,
            n_envs=N_ENVS,
            seed=seed,
            monitor_dir=log_dir,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            device="cpu",
            tensorboard_log=log_dir,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            # Includere il seed nel nome TB permette di confrontare le curve
            tb_log_name=f"PPO_{env_id}_s{seed}",
            progress_bar=False,
        )

        save_path = os.path.join(save_dir, f"PPO_{env_id}_s{seed}")
        model.save(save_path)
        env.close()

        print(f"[{proc_name}] ✓ Completato: {env_id} (s={seed})")
        return f"{env_id}_s{seed}", True

    except Exception as e:
        print(f"[{proc_name}] ✗ Errore su {env_id} (seed={seed}): {e}")
        return f"{env_id}_s{seed}", False


if __name__ == "__main__":
    # Generiamo la lista di seed 'random ma fissi'
    seeds = generate_robust_seeds(BASE_SEED, N_SEEDS)
    
    jobs = []
    for env_id in ENV_IDS:
        for s in seeds:
            jobs.append((env_id, s))

    print(f"Ambienti unici:    {len(ENV_IDS)}")
    print(f"Seed per ambiente: {N_SEEDS} {seeds}")
    print(f"Totale training:   {len(jobs)}")
    print(f"Processi paralleli: {N_PROCESSES}")
    print("─" * 40)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(train_one, jobs)

    print("\n" + "═" * 40)
    print("RIEPILOGO")
    print("═" * 40)
    
    n_ok = sum(ok for _, ok in results)
    for res_name, ok in results:
        if not ok:
            print(f"  ✗ FALLITO: {res_name}")
    
    print(f"\nCompletati con successo: {n_ok}/{len(results)}")
    if n_ok < len(results):
        print(f"Alcuni job sono falliti. Controlla l'output sopra.")