import numpy as np
import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import env.custom_hopper

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────────

ENV_ID          = "Hopper-Mass-Gauss-10-v0"
N_ENVS          = 4
TRAIN_TIMESTEPS = 1_000_000   # allineato al training standard
EVAL_EPISODES   = 10
SEEDS           = [42, 123]

N_TRIALS        = 15          # bilanciato con il costo per trial
TIMEOUT         = 3600 * 8    # 8 ore — si ferma alla prima condizione
DB_PATH         = "sqlite:///optuna_ppo.db"

# ─────────────────────────────────────────────────────────────────────────────


def sample_params(trial):
    """
    Spazio di ricerca degli iperparametri PPO.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs":      trial.suggest_int("n_epochs", 5, 20),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 1.0),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
        "vf_coef":       trial.suggest_float("vf_coef", 0.3, 1.0),
    }


def train_and_eval(params, seed):
    """
    Addestra PPO su un singolo seed e restituisce il reward medio finale.
    """
    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=seed)
    eval_env  = make_vec_env(ENV_ID, n_envs=1, seed=seed + 999)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        ent_coef=params["ent_coef"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        vf_coef=params["vf_coef"],
        policy_kwargs=dict(log_std_init=-2, activation_fn=nn.ReLU),
        seed=seed,
        verbose=0,
        device="cpu",
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    train_env.close()
    eval_env.close()
    return mean_reward


def objective(trial):
    """
    Funzione obiettivo: media del reward su tutti i seed.
    """
    params = sample_params(trial)

    rewards = []
    for seed in SEEDS:
        reward = train_and_eval(params, seed)
        rewards.append(reward)
        print(f"  [trial {trial.number}] seed={seed} → reward={reward:.1f}")

    mean_reward = float(np.mean(rewards))
    print(f"  [trial {trial.number}] → media={mean_reward:.1f}\n")
    return mean_reward


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"ppo_{ENV_ID}",
        direction="maximize",
        storage=DB_PATH,
        load_if_exists=True,
    )

    print(f"Ottimizzazione PPO su {ENV_ID}")
    print(f"Seed: {SEEDS} | Trial: {N_TRIALS} | Timestep/seed: {TRAIN_TIMESTEPS:,}")
    print(f"Timeout: 8h | Costo stimato: ~{N_TRIALS * len(SEEDS) * TRAIN_TIMESTEPS / 1_000_000:.0f}M timestep totali\n")

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("─" * 50)
    print(f"Trial completati : {len(completed)}")
    print(f"Trial pruned     : {len(pruned)}")
    print(f"Miglior reward   : {study.best_value:.2f}")
    print("\nMigliori parametri:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")