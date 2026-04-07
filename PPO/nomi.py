import gymnasium as gym
from env.custom_hopper import *

env = gym.make("CustomHopper-v0")

# forza inizializzazione completa
env.reset()

model = env.unwrapped.model

print("Lista body (id → nome):\n")

for i in range(len(model.body_mass)):
    name = model.body(i).name
    print(f"{i} → {name}")