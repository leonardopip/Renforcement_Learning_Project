import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import CustomHopper  # Assumendo sia registrato correttamente

def moving_average(values, window):
    """Media mobile robusta che tronca invece di fallire"""
    if len(values) < window:
        return values  # Troppo pochi dati
    n = len(values)
    n_windows = n // window
    return np.mean(values[:n_windows*window].reshape(1, n_windows, window), axis=2).reshape(-1)


def plot_results(log_folder, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " (Smoothed)")
    plt.grid(True)
    plt.show()

# Crea log dir
log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# 1. Crea env base con Monitor per logging
env = gym.make('CustomHopper-source-v0')
env = Monitor(env, log_dir)  # Logging attivato qui

# 2. Crea vec_env usando la funzione personalizzata (Monitor wrapper supportato)
vec_env = make_vec_env(
    'CustomHopper-source-v0', 
    n_envs=20,
    monitor_dir=log_dir  # Monitor applicato automaticamente a ogni sub-env [web:34]
)
spec = gym.spec('CustomHopper-source-v0')
print(spec)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=2000000)
model.save("PPO_CustomHopper_source_v0")

print("Training terminato!")
plot_results(log_dir)  # Passa lista di dir [web:23]
