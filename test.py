import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
from pusher import PusherEnv

import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
# 1. Carica l'ambiente
# Nota: assicurati che il nome dell'id sia corretto (es. "Pusher-v4")
env = gym.make("CustomPusher-vAttrito", render_mode="human")

# 2. Carica il modello salvato
model_path = "ppo_pusher_Attrito" # Non serve l'estensione .zip
model = PPO.load(model_path, env=env)

print(f"Modello {model_path} caricato. Inizio simulazione...")

# 3. Loop di valutazione e simulazione visiva
episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0
    
    while not (done or truncated):
        # Azione deterministica per il test
        action, _states = model.predict(obs, deterministic=True)
        
        # Esegui l'azione
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        
        # Il rendering 'human' non richiede env.render() esplicito in Gymnasium 
        # ma se non vedi nulla, decommenta la riga sotto:
        # env.render()
        
        # Opzionale: aggiungi un piccolissimo delay per goderti il movimento
        time.sleep(0.01) 

    print(f"Episodio: {ep + 1} | Punteggio totale: {score:.2f}")

env.close()

# 1. Esegui la valutazione raccogliendo i dati di ogni singolo episodio
# Supponiamo di valutare su 50 episodi per avere dati statisticamente significativi
rewards, lengths = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=50, 
    return_episode_rewards=True
)

# 2. Creazione dei grafici (Istogramma e Boxplot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Istogramma: mostra la frequenza dei punteggi ottenuti
ax1.hist(rewards, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(rewards), color='red', linestyle='dashed', linewidth=2, label=f'Media: {np.mean(rewards):.1f}')
ax1.set_title('Distribuzione dei Premi (Rewards)')
ax1.set_xlabel('Reward Totale per Episodio')
ax1.set_ylabel('Frequenza')
ax1.legend()

# Boxplot: evidenzia la varianza, i quartili e gli eventuali outlier (casi rari)
ax2.boxplot(rewards, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='lightgreen'),
            medianprops=dict(color='black'))
ax2.set_title('Varianza e Outlier')
ax2.set_ylabel('Reward Totale')
ax2.set_xticklabels(['Agente'])

plt.tight_layout()
plt.show()