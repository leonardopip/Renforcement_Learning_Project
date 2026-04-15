import numpy as np
import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import env.custom_hopper

# ─── CONFIGURAZIONE ───────────────────────────────────────────────────────────

SOURCE_ENV      = "Hopper-Mass-Gauss-10-v0"
TRAIN_TIMESTEPS = 1_000_000
EVAL_EPISODES   = 50
SEEDS           = [42]

DB_PATH         = "sqlite:///optuna_ppo.db"
MODELS_DIR      = "models/Mass-Gauss/10"

TARGET_ENVS = [
    "Hopper-Target-Mass-Easy-v0",
    "Hopper-Target-Mass-Medium-v0",
    "Hopper-Target-Mass-Hard-v0",
    "Hopper-Target-Fric-Easy-v0",
    "Hopper-Target-Fric-Medium-v0",
    "Hopper-Target-Fric-Hard-v0",
    "Hopper-Target-Both-Easy-v0",
    "Hopper-Target-Both-Medium-v0",
    "Hopper-Target-Both-Hard-v0",
]

# ─────────────────────────────────────────────────────────────────────────────


def load_best_params():
    """Carica i migliori parametri trovati dall'ottimizzazione Optuna."""
    study = optuna.load_study(
        study_name=f"ppo_{SOURCE_ENV}",
        storage=DB_PATH,
    )
    print(f"Miglior trial: #{study.best_trial.number} | reward={study.best_value:.2f}\n")
    return study.best_params


def load_default_model(seed):
    """Carica il modello default già trainato in precedenza."""
    path = f"{MODELS_DIR}/PPO_Hopper-Mass-Gauss-10-v1_s{seed}.zip"
    model = PPO.load(path, device="cpu")
    print(f"  [default] caricato da {path}")
    return model


def train_optimized_model(params, seed):
    """Traina la policy ottimizzata con i parametri di Optuna."""
    print(f"  [ottimizzata] training seed={seed}...")
    train_env = make_vec_env(SOURCE_ENV, n_envs=4, seed=seed)

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
    train_env.close()
    return model


def eval_on_targets(model, seed):
    """Valuta il modello su tutti i target environments."""
    results = {}
    for target_env_id in TARGET_ENVS:
        eval_env = make_vec_env(target_env_id, n_envs=1, seed=seed + 999)
        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
        )
        eval_env.close()
        results[target_env_id] = mean_reward
    return results


def compare():
    best_params = load_best_params()

    # { env_id: [reward_seed1, ...] }
    results_opt = {env: [] for env in TARGET_ENVS}
    results_def = {env: [] for env in TARGET_ENVS}

    for seed in SEEDS:
        print(f"\n{'─'*55}")
        print(f"SEED {seed}")
        print(f"{'─'*55}")

        # Policy default — carica modello già trainato
        model_def = load_default_model(seed)
        rewards_def = eval_on_targets(model_def, seed)
        for env_id, r in rewards_def.items():
            results_def[env_id].append(r)
            print(f"  [default    ] {env_id:<40} → {r:.1f}")

        print()

        # Policy ottimizzata — traina da zero
        model_opt = train_optimized_model(best_params, seed)
        rewards_opt = eval_on_targets(model_opt, seed)
        for env_id, r in rewards_opt.items():
            results_opt[env_id].append(r)
            print(f"  [ottimizzata] {env_id:<40} → {r:.1f}")

    # ─── RIEPILOGO FINALE ─────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"{'RIEPILOGO FINALE':^72}")
    print(f"{'═'*72}")
    print(f"  {'Target Environment':<40} {'Ottimizzata':>12} {'Default':>12} {'Delta':>8}")
    print(f"{'─'*72}")

    deltas = []
    for env_id in TARGET_ENVS:
        mean_opt = np.mean(results_opt[env_id])
        mean_def = np.mean(results_def[env_id])
        delta    = mean_opt - mean_def
        deltas.append(delta)
        marker = "↑" if delta > 0 else "↓"
        print(f"  {env_id:<40} {mean_opt:>12.1f} {mean_def:>12.1f} {delta:>+7.1f} {marker}")

    print(f"{'─'*72}")
    mean_all_opt = np.mean([np.mean(v) for v in results_opt.values()])
    mean_all_def = np.mean([np.mean(v) for v in results_def.values()])
    delta_mean   = mean_all_opt - mean_all_def
    print(f"  {'Media su tutti i target':<40} {mean_all_opt:>12.1f} {mean_all_def:>12.1f} {delta_mean:>+7.1f}")
    print(f"{'═'*72}")


if __name__ == "__main__":
    print(f"Confronto policy ottimizzata vs default")
    print(f"Source: {SOURCE_ENV} | Timesteps: {TRAIN_TIMESTEPS:,}")
    print(f"Seed: {SEEDS} | Target envs: {len(TARGET_ENVS)} | Episodi eval: {EVAL_EPISODES}\n")
    compare()