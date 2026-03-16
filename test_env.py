import gymnasium as gym
from env.custom_hopper import *

env = gym.make("CustomHopper-v0", render_mode="human")

obs, info = env.reset()

for i in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()