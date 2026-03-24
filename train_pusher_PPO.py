import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Pool, current_process

# Import del file che registra gli ambienti custom del pusher
import pusher_PPO

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

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def train_one(args):
    env_id, seed = args
    proc_name = current_process().name

    torch.set_num_threads(1)

    parts = env_id.split("-")
    # Esempio: Pusher-Mass-Uni-10-v0
    # parts = ["Pusher", "Mass", "Uni", "10", "v0"]
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
            tb_log_name=f"PPO_{env_id}",
            progress_bar=False,
        )

        save_path = os.path.join(save_dir, f"PPO_{env_id}_s{seed}")
        model.save(save_path)
        env.close()

        print(f"[{proc_name}] ✓ Completato: {env_id} → {save_path}")
        return env_id, True

    except Exception as e:
        print(f"[{proc_name}] ✗ Errore su {env_id}: {e}")
        return env_id, False


if __name__ == "__main__":
    N_SEEDS = 5
    jobs = []

    for env_id in ENV_IDS:
        for s in range(N_SEEDS):
            jobs.append((env_id, BASE_SEED + s))

    print(f"Ambienti da trainare: {len(ENV_IDS)}")
    print(f"Processi paralleli:   {N_PROCESSES}")
    print(f"Timesteps per env:    {TIMESTEPS:,}")
    print(f"Device:               cpu")
    print("─" * 40)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(train_one, jobs)

    print("\n" + "═" * 40)
    print("RIEPILOGO")
    print("═" * 40)
    for env_id, ok in results:
        stato = "✓" if ok else "✗ FALLITO"
        print(f"  {stato}  {env_id}")

    n_ok = sum(ok for _, ok in results)
    n_fail = len(results) - n_ok
    print(f"\nCompletati: {n_ok}/{len(results)}")
    if n_fail > 0:
        print(f"Falliti:    {n_fail} (controlla i log sopra)")