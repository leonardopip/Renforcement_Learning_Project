import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from multiprocessing import Pool
import env.custom_hopper 

# SOLO 2 AMBIENTI: Il Re e lo Sfidante
BENCHMARK_ENVS = ["Hopper-v4", "Hopper-Mass-Gauss-10-v0"]

TIMESTEPS   = 1_000_000
N_PROCESSES = 2  # Un core/processo dedicato a testa, massima pulizia
SAVE_DIR    = "test_zoo_vs_standard"
LOG_DIR     = "logs_test_zoo"

os.makedirs(SAVE_DIR, exist_ok=True)

ZOO_PPO_PARAMS = {
    "n_steps": 512,
    "batch_size": 64, # Teniamo 64 per sicurezza
    "gamma": 0.999,
    "learning_rate": 9.8e-05,
    "ent_coef": 0.002,
    "clip_range": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.99,
    "max_grad_norm": 0.7,
    "vf_coef": 0.8,
    "policy_kwargs": dict(
        log_std_init=-2,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
}

def train_one(env_id):
    torch.set_num_threads(4) # Ora che abbiamo solo 2 processi, possiamo dare più thread a ognuno!
    print(f"Lancio: {env_id}")
    try:
        venv = make_vec_env(env_id, n_envs=1)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

        model = PPO("MlpPolicy", venv, verbose=0, tensorboard_log=LOG_DIR, **ZOO_PPO_PARAMS)
        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"COMPARE_{env_id}")
        
        model.save(os.path.join(SAVE_DIR, f"model_{env_id}"))
        venv.save(os.path.join(SAVE_DIR, f"stats_{env_id}.pkl"))
        return True
    except Exception as e:
        print(f"Errore: {e}")
        return False

if __name__ == "__main__":
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(train_one, BENCHMARK_ENVS)