import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import env.custom_hopper

# --- CONFIGURAZIONE ---
TARGET_ENV      = "Hopper-Mass-Gauss-10-v0"
N_TRIALS        = 20
EVAL_EPISODES   = 10
TRAIN_TIMESTEPS = 700_000   # Aumentati per valutazione più affidabile
TIMEOUT         = 3600 * 6  # 6 ore: multi-seed costa di più
DB_PATH         = "sqlite:///optuna_multiseed.db"

# I seed usati per ogni trial — media su 3 per ridurre la varianza
SEEDS = [42, 123]


class TrialEvalCallback(BaseCallback):
    """
    Callback per il pruning intermedio di Optuna.
    Il pruning avviene solo sul primo seed per non scartare
    trial buoni che potrebbero essere sfortunati su un seed intermedio.
    """
    def __init__(self, eval_env, trial, seed_idx, eval_freq=100_000, n_eval_episodes=5):
        super().__init__()
        self.eval_env        = eval_env
        self.trial           = trial
        self.seed_idx        = seed_idx
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        # Ogni seed usa un range di step diverso per non sovrascrivere i report
        self.eval_step       = seed_idx * 100
        self.is_pruned       = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = self.training_env.obs_rms

            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            self.trial.report(mean_reward, step=self.eval_step)
            self.eval_step += 1

            # Pota solo sul primo seed
            if self.seed_idx == 0 and self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def make_eval_env(training_venv, seed):
    eval_venv = make_vec_env(TARGET_ENV, n_envs=1, seed=seed)
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=True,
        norm_reward=False,
        training=False,
    )
    eval_venv.obs_rms = training_venv.obs_rms
    return eval_venv


def train_single_seed(trial, params, seed):
    """Esegue un training completo su un singolo seed e restituisce il reward medio."""
    venv = make_vec_env(TARGET_ENV, n_envs=4, seed=seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    eval_venv = make_eval_env(venv, seed=seed + 999)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        gae_lambda=params["gae_lambda"],
        n_epochs=params["n_epochs"],
        clip_range=params["clip_range"],
        policy_kwargs=dict(log_std_init=-2, activation_fn=nn.ReLU),
        seed=seed,
        verbose=0,
        device="cpu",
    )

    seed_idx = SEEDS.index(seed)
    callback = TrialEvalCallback(
        eval_env=eval_venv,
        trial=trial,
        seed_idx=seed_idx,
        eval_freq=100_000 // 4,   # diviso n_envs
        n_eval_episodes=5,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=callback)

    if callback.is_pruned:
        venv.close()
        eval_venv.close()
        return None  # Segnala pruning al chiamante

    # Valutazione finale su questo seed
    eval_venv.obs_rms = venv.obs_rms
    mean_reward, _ = evaluate_policy(
        model, eval_venv, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    venv.close()
    eval_venv.close()
    return mean_reward


def objective(trial):
    # --- SPAZIO DI RICERCA ---
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_steps":       trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 1.0),
        "n_epochs":      trial.suggest_int("n_epochs", 5, 20),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
    }

    rewards = []
    for seed in SEEDS:
        reward = train_single_seed(trial, params, seed)

        if reward is None:
            # Pruned sul primo seed: scarta il trial
            raise optuna.exceptions.TrialPruned()

        rewards.append(reward)
        print(f"    seed={seed} → reward={reward:.1f}")

    mean  = float(np.mean(rewards))
    std   = float(np.std(rewards))
    print(f"    → media={mean:.1f} ± std={std:.1f}")

    # Penalizza leggermente l'alta varianza tra seed:
    # parametri stabili valgono più di parametri instabili con stesso reward medio
    score = mean - 0.1 * std
    return score


if __name__ == "__main__":
    print(f"🧬 Ottimizzazione multi-seed per {TARGET_ENV}")
    print(f"   Seed: {SEEDS}")
    print(f"   Trial: {N_TRIALS} | Timesteps/seed: {TRAIN_TIMESTEPS:,} | DB: {DB_PATH}\n")

    storage = RDBStorage(url=DB_PATH)

    study = optuna.create_study(
        study_name=f"ppo_multiseed_{TARGET_ENV}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    print("\n🏆 OTTIMIZZAZIONE COMPLETATA!")
    print(f"   Trial completati : {len(study.trials)}")
    print(f"   Trial pruned     : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"   Miglior Score    : {study.best_value:.2f}  (mean - 0.1*std)")
    print("\n   Migliori Parametri:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")