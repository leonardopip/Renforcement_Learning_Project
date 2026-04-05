import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
import env.custom_hopper

# --- CONFIGURAZIONE ---
TARGET_ENV   = "Hopper-Mass-Gauss-10-v0"
TOTAL_STEPS  = 3_000_000
EVAL_FREQ    = 100_000      # Valuta ogni 100k step
EVAL_EPS     = 10           # Episodi per ogni valutazione
SAVE_DIR     = "final_model"
LOG_DIR      = "logs/final"
SEED         = 42

# --- BEST PARAMS DA OPTUNA ---
BEST_PARAMS = dict(
    learning_rate = 6.05e-5,
    n_steps       = 512,
    batch_size    = 64,
    ent_coef      = 0.000169,
    gae_lambda    = 0.944,
    n_epochs      = 15,
    clip_range    = 0.304,
)


class ProgressCallback(BaseCallback):
    """
    Stampa un riepilogo leggibile ogni EVAL_FREQ step:
    timestep corrente, reward medio sull'eval env, e segnala
    se il reward è ancora troppo basso a 1M step (early warning).
    """
    def __init__(self, eval_env, eval_freq=100_000, n_eval_episodes=10, warn_threshold=1000):
        super().__init__()
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_episodes= n_eval_episodes
        self.warn_threshold = warn_threshold
        self.best_reward    = -float("inf")
        self.warned         = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Sincronizza stats normalizzazione
            self.eval_env.obs_rms = self.training_env.obs_rms

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            tag = "🆕 BEST" if mean_reward > self.best_reward else ""
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward

            print(
                f"  [{self.n_calls:>8,} step] "
                f"reward: {mean_reward:7.1f} ± {std_reward:.1f}  "
                f"(best: {self.best_reward:.1f})  {tag}"
            )

            # Early warning a 1M step
            if self.n_calls >= 1_000_000 and mean_reward < self.warn_threshold and not self.warned:
                print(
                    f"\n  ⚠️  ATTENZIONE: reward {mean_reward:.1f} ancora sotto {self.warn_threshold} "
                    f"a 1M step.\n"
                    f"     Considera di interrompere e rifare la search con più timesteps.\n"
                )
                self.warned = True

        return True


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"🚀 Training finale su {TARGET_ENV}")
    print(f"   Timesteps totali : {TOTAL_STEPS:,}")
    print(f"   Eval ogni        : {EVAL_FREQ:,} step")
    print(f"   Parametri        : {BEST_PARAMS}\n")

    # --- ENV DI TRAINING ---
    venv = make_vec_env(TARGET_ENV, n_envs=4, seed=SEED)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    # --- ENV DI VALUTAZIONE (separato, non contamina le stats) ---
    eval_venv = make_vec_env(TARGET_ENV, n_envs=1, seed=SEED + 1)
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=True,
        norm_reward=False,  # reward grezzo per interpretazione corretta
        training=False,
    )
    eval_venv.obs_rms = venv.obs_rms

    # --- MODELLO ---
    model = PPO(
        "MlpPolicy",
        venv,
        **BEST_PARAMS,
        policy_kwargs=dict(log_std_init=-2, activation_fn=nn.ReLU),
        tensorboard_log=LOG_DIR,
        seed=SEED,
        verbose=0,
        device="cpu",
    )

    # --- CALLBACKS ---

    # 1) Salva il miglior modello in assoluto
    eval_callback = EvalCallback(
        eval_venv,
        best_model_save_path=f"{SAVE_DIR}/best",
        log_path=f"{LOG_DIR}/eval",
        eval_freq=EVAL_FREQ // 4,   # n_envs=4, quindi dividi per 4
        n_eval_episodes=EVAL_EPS,
        deterministic=True,
        verbose=0,
    )

    # 2) Checkpoint periodico (ogni 500k step) — utile per analisi post-hoc
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // 4,     # diviso n_envs
        save_path=f"{SAVE_DIR}/checkpoints",
        name_prefix="ppo_hopper",
        verbose=0,
    )

    # 3) Progress report leggibile + early warning
    progress_callback = ProgressCallback(
        eval_env=eval_venv,
        eval_freq=EVAL_FREQ // 4,
        n_eval_episodes=EVAL_EPS,
        warn_threshold=1000,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback, progress_callback])

    # --- TRAINING ---
    print("📈 Inizio training...\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True)

    # --- SALVATAGGIO FINALE ---
    model.save(f"{SAVE_DIR}/final_policy")
    venv.save(f"{SAVE_DIR}/final_vecnormalize.pkl")
    print(f"\n✅ Training completato!")
    print(f"   Best model  → {SAVE_DIR}/best/best_model.zip")
    print(f"   Final model → {SAVE_DIR}/final_policy.zip")
    print(f"   VecNormalize → {SAVE_DIR}/final_vecnormalize.pkl")

    venv.close()
    eval_venv.close()


if __name__ == "__main__":
    main()