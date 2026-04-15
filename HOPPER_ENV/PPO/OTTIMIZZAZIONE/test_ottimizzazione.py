from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
import env.custom_hopper  # noqa: E402

ENV_ID = "Hopper-MassFric-Uni-80-v0"
SEEDS = [42, 90161] # Addestriamo due modelli con seed diversi
TIMESTEPS = 3_000_000 # Usiamo 3M di step come nella v2

# Iperparametri ottimizzati che hai trovato con Optuna.
# Se hai parametri diversi per "MassFric-Uni-80", sostituiscili qui.
OPTIMIZED_PARAMS = {
    "learning_rate": 9.682609487894454e-05,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 15,
    "gamma": 0.9948646413801842,
    "gae_lambda": 0.956474414048344,
    "clip_range": 0.17298348295317015,
    "ent_coef": 0.0002674701807186936,
    "vf_coef": 0.7848396206682986,
    "max_grad_norm": 0.7245950693318323,
}

for seed in SEEDS:
    print(f"\n{'='*60}\nTraining per {ENV_ID} con seed {seed}\n{'='*60}")
    env = make_vec_env(ENV_ID, n_envs=12, seed=seed)
    save_path = os.path.join(SCRIPT_DIR, "models", f"PPO_{ENV_ID.replace('-v0', '-v1')}_s{seed}-ottimizzato.zip")
    model = PPO("MlpPolicy", env, seed=seed, verbose=1, device="cpu", **OPTIMIZED_PARAMS)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(save_path)
    print(f"✓ Modello salvato in: {save_path}")