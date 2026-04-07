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


# Crea 20 ambienti paralleli, ognuno con il suo Monitor già configurato
env =gym.make(
    'CustomPusher-vAttrito'
)
specifiche = gym.spec('CustomPusher-vAttrito')
print(specifiche)
timestesp_lear = 1000000
#1
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_ent_std_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_ent_std")
#2
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",ent_coef=0.1)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_ent_0-1_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_ent_0-1")
#3
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",ent_coef=0.01)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_ent_0-01_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_ent_0-01")
#4
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",ent_coef=0.001)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_ent_0-001_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_ent_0-001")
#5
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",ent_coef=0.1)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_ent_1_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_ent_1")
#6
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",batch_size=32)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_batch32_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_batch32")
#7
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",batch_size=64)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_batch64_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_batch64")
#8
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",batch_size=128)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_batch128_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_batch128")
#9
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",batch_size=256)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_batch256_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_batch256")
#10
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",batch_size=512)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_batch512_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_batch512")
#11
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",learning_rate=0.000003)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_lr3e-6_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_lr3e-6")
#12
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",learning_rate=0.00003)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_lr3e-5_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_lr3e-5")
#13
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",learning_rate=0.0003)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_lr3e-4_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_lr3e-4")
#14
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",learning_rate=0.003)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_lr3e-3_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_lr3e-3")
#15
model = PPO("MlpPolicy", env, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",sde_sample_freq=512)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_sampleFrequency512_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_sampleFrequency512")
#16
model = PPO("MlpPolicy", env,use_sde=True, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",sde_sample_freq=1024)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_sampleFrequency1024_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_sampleFrequency1024")
#17
model = PPO("MlpPolicy", env, use_sde=True,verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",sde_sample_freq=2048)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_sampleFrequency2048_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_sampleFrequency2048")
#18
model = PPO("MlpPolicy", env,use_sde=True, verbose=0, device="cpu",tensorboard_log="./tensorboard_logs/",sde_sample_freq=4096)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_sampleFrequency4096_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_sampleFrequency4096")
#19
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", gamma=0.8)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_gamma_0-8_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_gamma_0-8")
#20
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", gamma=0.9997)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_gamma_0-9997_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_gamma_0-9997")
#21
# --- TEST SU N_STEPS ---
# Quanta esperienza raccoglie prima di ogni aggiornamento (stabilità del gradiente)
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", n_steps=512)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_nsteps_512_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_nsteps_512")
#22
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", n_steps=1024)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_nsteps_1024_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_nsteps_1024")
#23
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", n_steps=2048)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_nsteps_2048_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_nsteps_2048")
#24
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", n_steps=4096)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_nsteps_4096_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_nsteps_4096")
#25
# --- TEST SU GAE_LAMBDA ---
# Gestisce il compromesso tra varianza e bias nelle stime di vantaggio
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", gae_lambda=0.9)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_gae_0-9_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_gae_0-9")
#26
model = PPO("MlpPolicy", env, verbose=0, device="cpu", tensorboard_log="./tensorboard_logs/", gae_lambda=1)
model.learn(total_timesteps=timestesp_lear, tb_log_name="PPO_SingleVec_gae_1_attrito_v1_1")
model.save("ppo_pusher_SingleEnv_Attrito_gae_1")



print("terminato")
 