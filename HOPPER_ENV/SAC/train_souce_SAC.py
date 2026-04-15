import os
import sys

# Aggiungiamo la root del progetto (HOPPER_ENV) al PYTHONPATH per importare env
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

import torch
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import env.custom_hopper

# --- CONFIGURAZIONE PERCORSI ---
BASE_PATH = os.path.join("HOPPER_ENV", "SAC")
SAVE_DIR  = os.path.join(BASE_PATH, "results", "models")
LOG_DIR   = os.path.join(BASE_PATH, "results", "logs")

# --- CONFIGURAZIONE TRAINING ---
# Corretto il nome del source basato sulle tue registrazioni in custom_hopper.py
ENV_ID = "CustomHopper-source-v0"
TIMESTEPS = 1_000_000
N_ENVS = 1
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print(f"Inizio training SAC su {ENV_ID} (seed={SEED})")
    print(f"Directory di salvataggio: {SAVE_DIR}")
    print("─" * 50)

    torch.set_num_threads(1)  # Per misurare le prestazioni su singolo thread in modo analogo al PPO

    # Sottocartelle per il modello base source
    specific_save_dir = os.path.join(SAVE_DIR, "source")
    specific_log_dir  = os.path.join(LOG_DIR, "source")
    os.makedirs(specific_save_dir, exist_ok=True)
    os.makedirs(specific_log_dir, exist_ok=True)

    try:
        env = make_vec_env(
            ENV_ID,
            n_envs=N_ENVS,
            seed=SEED,
            monitor_dir=specific_log_dir,
        )

        model = SAC(
            policy="MlpPolicy",
            env=env,
            seed=SEED,
            verbose=1,  # Tienilo a 1 per vedere gli FPS in tempo reale
            device="cpu",
            tensorboard_log=specific_log_dir,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=f"SAC_{ENV_ID}_s{SEED}",
            progress_bar=True,
        )

        save_path = os.path.join(specific_save_dir, f"SAC_{ENV_ID}_s{SEED}")
        model.save(save_path)
        
        env.close()
        print(f"\n✓ Completato: {ENV_ID} (s={SEED})")
        print(f"  Modello salvato in: {save_path}")

    except Exception as e:
        print(f"\n✗ Errore durante il training di {ENV_ID}: {e}")

if __name__ == "__main__":
    main()