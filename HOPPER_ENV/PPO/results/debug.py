import os
import sys
import glob
import gymnasium as gym
from stable_baselines3 import PPO

# Importa il modulo contenente i tuoi ambienti custom per registrarli in Gymnasium
# Aggiungiamo la cartella PPO al PYTHONPATH dinamicamente
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PPO_ROOT)

import env.custom_hopper

# Crea l'ambiente custom con il rendering attivo
# Dal nome del modello "uni_80", presumo tu voglia testare la variante Uniforme all'80%
env = gym.make("Hopper-Target-Mass-Hard-v0", render_mode="human")

# Cerca tutti i seed salvati per la variante Fric-Uni al 20% (in results_v2)
model_dir = os.path.join(PPO_ROOT, "results_v2", "models", "Fric-Uni", "20")
model_paths = sorted(glob.glob(os.path.join(model_dir, "PPO_Hopper-Fric-Uni-20-v0_s*.zip")))

if not model_paths:
    print(f"Nessun modello trovato in {model_dir}")
else:
    for path in model_paths:
        print(f"\n{'='*60}\nCaricamento modello da: {os.path.basename(path)}")
        model = PPO.load(path, device="cpu")
        
        steps = 0
        total_reward = 0.0
        obs, info = env.reset()
        
        # Testiamo 1 solo episodio per seed per vederli tutti rapidamente
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episodio terminato dopo {steps} step | Caduta: {terminated} | Reward Totale: {total_reward:.2f}")
                break

# È buona norma chiudere l'ambiente al termine della simulazione
env.close()
