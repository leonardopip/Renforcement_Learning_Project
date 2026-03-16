import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Training
vec_env = make_vec_env('Pusher-v5', n_envs=16)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model.learn(total_timesteps=900000)
model.save("ppo_pusher")

# 2. Reset e Caricamento
del model 
model = PPO.load("ppo_pusher")

# 3. VALUTAZIONE (Spostata PRIMA del ciclo infinito)
print("\nCalcolo dei risultati medi...")
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20)

print(f"--- Risultati ---")
print(f"Reward media: {mean_reward:.2f}")
print(f"Deviazione standard: {std_reward:.2f}")

# 4. TEST SINGOLO EPISODIO
print("\nDettaglio episodio singolo:")
obs = vec_env.reset()
total_reward = 0
for _ in range(100): 
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    total_reward += reward[0] 
    if done[0]:
        break
print(f"Reward finale episodio di test: {total_reward:.2f}")

# 5. VISUALIZZAZIONE (Solo alla fine metti il loop infinito se vuoi vedere il robot)
print("\nAvvio visualizzazione... (Premi Ctrl+C nel terminale per fermare)")
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human") # Decommenta se vuoi vedere la finestra grafica