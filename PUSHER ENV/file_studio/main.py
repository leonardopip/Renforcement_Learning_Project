import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from pusher import PusherEnv
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import numpy as np

def moving_average(values, window):
    """Media mobile robusta che tronca invece di fallire"""
    if len(values) < window:
        return values  # Troppo pochi dati
    n = len(values)
    n_windows = n // window
    return np.mean(values[:n_windows*window].reshape(1, n_windows, window), axis=2).reshape(-1)


def show_training_results(log_folder):
    results_plotter.plot_results([log_folder], 3e6, results_plotter.X_TIMESTEPS, "Learning Curve")
    plt.title(f"Addestramento Pusher")
    plt.grid(True)
    plt.savefig(f"{log_folder}/final_plot.png") # Salva il grafico prima di mostrarlo
    plt.show()

log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
# 1. Training

# Crea 20 ambienti paralleli, ognuno con il suo Monitor già configurato
vec_env = make_vec_env(
    'CustomPusher-vAttrito', 
    n_envs=20, 
    monitor_dir=log_dir
)
specifiche = gym.spec('CustomPusher-vAttrito')
print(specifiche)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu",tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=3000000)
model.save("ppo_pusher_Attrito")

show_training_results(log_dir)

print("terminato")
 