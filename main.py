import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from pusher import PusherEnv

# 1. Training
vec_env = make_vec_env(
    'CustomPusher-vAttrito', 
    n_envs=23,    
    )
specifiche = gym.spec('CustomPusher-vAttrito')
print(specifiche)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=3000000)
model.save("ppo_pusher_Attrito")

print("terminato")
 