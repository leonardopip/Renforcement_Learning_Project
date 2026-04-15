import os
import torch
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from multiprocessing import Pool, current_process
import gymnasium as gym
import env.custom_hopper

# ──────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────
TRAIN_ENVS  = [
    "Hopper-Mass-Gauss-20-v0",
    "Hopper-Fric-Uni-50-v0",
]
TIMESTEPS   = 500_000
N_ENVS      = 1
N_PROCESSES = 6
BASE_SEED   = 42
RESULTS_DIR = "results"
LOG_DIR     = "hyperparam_logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

TARGET_IDS = [
    "CustomHopper-target-v0",
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

# ──────────────────────────────────────────────
# Iperparametri da testare (OFAT)
# ──────────────────────────────────────────────
DEFAULT_PARAMS = {
    "learning_rate": 3e-4,
    "batch_size":    64,
    "n_steps":       2048,
    "gae_lambda":    0.95,
    "ent_coef":      0.0,
}

HYPERPARAM_VARIANTS = [
    ("default",    {}),
    ("lr_3e-6",    {"learning_rate": 3e-6}),
    ("lr_3e-5",    {"learning_rate": 3e-5}),
    ("lr_3e-3",    {"learning_rate": 3e-3}),
    ("batch_32",   {"batch_size": 32}),
    ("batch_128",  {"batch_size": 128}),
    ("batch_256",  {"batch_size": 256}),
    ("batch_512",  {"batch_size": 512}),
    ("nsteps_512", {"n_steps": 512}),
    ("nsteps_1024",{"n_steps": 1024}),
    ("nsteps_4096",{"n_steps": 4096}),
    ("gae_0-9",    {"gae_lambda": 0.9}),
    ("gae_1-0",    {"gae_lambda": 1.0}),
    ("ent_0-001",  {"ent_coef": 0.001}),
    ("ent_0-01",   {"ent_coef": 0.01}),
    ("ent_0-1",    {"ent_coef": 0.1}),
]


# ──────────────────────────────────────────────
# Funzione worker
# ──────────────────────────────────────────────
def train_and_evaluate(args):
    train_env, run_name, override_params, seed = args
    proc_name = current_process().name
    torch.set_num_threads(1)

    params = {**DEFAULT_PARAMS, **override_params}
    env_tag = train_env.replace("Hopper-", "").replace("-v0", "")
    full_run_name = f"{env_tag}_{run_name}"

    print(f"[{proc_name}] Inizio: {full_run_name} | {override_params if override_params else 'default'}")

    try:
        env = make_vec_env(train_env, n_envs=N_ENVS, seed=seed)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=0,
            device="cpu",
            tensorboard_log=LOG_DIR,
            **params,
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=f"PPO_{full_run_name}",
            progress_bar=False,
        )
        env.close()

        # ── Evaluation su tutti i target ──────
        target_results = {}
        for target_id in TARGET_IDS:
            eval_env = gym.make(target_id)
            mean, std = evaluate_policy(
                model, eval_env,
                n_eval_episodes=20,
                deterministic=True,
            )
            eval_env.close()
            target_results[target_id] = (round(mean, 2), round(std, 2))

        avg_reward = round(np.mean([v[0] for v in target_results.values()]), 2)
        print(f"[{proc_name}] ✓ {full_run_name} → avg={avg_reward}")

        return {
            "train_env":  train_env,
            "run_name":   full_run_name,
            "avg_reward": avg_reward,
            **params,
            **{f"{t}_mean": target_results[t][0] for t in TARGET_IDS},
            **{f"{t}_std":  target_results[t][1] for t in TARGET_IDS},
        }

    except Exception as e:
        print(f"[{proc_name}] ✗ {full_run_name} → Errore: {e}")
        return {
            "train_env":  train_env,
            "run_name":   full_run_name,
            "avg_reward": -999,
            **params,
            **{f"{t}_mean": "N/A" for t in TARGET_IDS},
            **{f"{t}_std":  "N/A" for t in TARGET_IDS},
        }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    jobs = []
    job_idx = 0
    for train_env in TRAIN_ENVS:
        for run_name, override in HYPERPARAM_VARIANTS:
            jobs.append((train_env, run_name, override, BASE_SEED + job_idx))
            job_idx += 1

    print(f"Ambienti:            {TRAIN_ENVS}")
    print(f"Varianti per env:    {len(HYPERPARAM_VARIANTS)}")
    print(f"Training totali:     {len(jobs)}")
    print(f"Processi paralleli:  {N_PROCESSES}")
    print(f"Timesteps per run:   {TIMESTEPS:,}")
    print("─" * 50)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(train_and_evaluate, jobs)

    # ── Salva CSV ─────────────────────────────
    fieldnames = (
        ["train_env", "run_name", "avg_reward"] +
        list(DEFAULT_PARAMS.keys()) +
        [f"{t}_mean" for t in TARGET_IDS] +
        [f"{t}_std"  for t in TARGET_IDS]
    )
    csv_path = os.path.join(RESULTS_DIR, "hyperparam_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Risultati salvati in: {csv_path}")

    # ── Riepilogo per ambiente ─────────────────
    print("\n" + "═" * 50)
    for train_env in TRAIN_ENVS:
        env_results = [r for r in results if r["train_env"] == train_env and r["avg_reward"] != -999]
        env_results.sort(key=lambda x: x["avg_reward"], reverse=True)

        print(f"\n{train_env}")
        print(f"  {'Run':<30} {'Avg Reward':>12}")
        print("  " + "─" * 44)
        for r in env_results:
            marker = " ← MIGLIORE" if r == env_results[0] else ""
            print(f"  {r['run_name']:<30} {r['avg_reward']:>12}{marker}")