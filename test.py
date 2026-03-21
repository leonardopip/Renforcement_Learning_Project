import numpy as np
import argparse
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gymnasium as gym
from pusher import PusherEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

print("Testing...")
env = gym.make('CustomPusher-vAttrito',render_mode="human")

# --- TEST POLICY 1 (PPO) ---
model = PPO.load("ppo_pusher_Attrito_ent_std")
episode_rewards_1 = []

for ep in range(50):
    obs, _ = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
    episode_rewards_1.append(ep_reward)

# --- TEST POLICY 2 (Esempio: SAC) ---
# Carichiamo la seconda policy (assicurati che il file esista)
model_2 = PPO.load("ppo_pusher_senza_ostacoli") 
episode_rewards_2 = []

for ep in range(50):
    obs, _ = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0
    while not (done or truncated):
        action, _ = model_2.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
    episode_rewards_2.append(ep_reward)

env.close()

# --- GRAFICO DI CONFRONTO ---
data_to_plot = [episode_rewards_1, episode_rewards_2]
labels = ['PPO (Source)', 'PPO (target)']

plt.figure(figsize=(10, 6))

# Boxplot per vedere distribuzione, media e varianza
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.ylabel('Reward')
plt.title('Confronto Performance: PPO source vs PPO target')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Stampa delle medie finali
print(f"\nMedia Reward PPO: {np.mean(episode_rewards_1):.2f}")
print(f"Media Reward PPO parallelo: {np.mean(episode_rewards_2):.2f}")