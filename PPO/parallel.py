import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from env.custom_hopper import *

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


def make_env(env_id, rank, seed=0, log_dir=None):
    """
    Factory per creare un env separato.
    Necessaria per SubprocVecEnv / DummyVecEnv.
    """
    def _init():
        env = gym.make(env_id)

        if log_dir is not None:
            monitor_file = os.path.join(log_dir, f"env_{rank}", "monitor.csv")
            os.makedirs(os.path.dirname(monitor_file), exist_ok=True)
            env = Monitor(env, monitor_file)

        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def create_model(args, env):
    if args.algo == 'ppo':
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            clip_range=args.clip,
            verbose=1,
            seed=args.seed,
        )

    elif args.algo == 'sac':
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
            seed=args.seed,
        )

    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")

    return model


def load_model(args, env):
    if args.algo == 'ppo':
        model = PPO.load(args.test, env=env)
    elif args.algo == 'sac':
        model = SAC.load(args.test, env=env)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model


def moving_average(values, window):
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def load_all_monitor_files(log_folder):
    """
    Legge tutti i file monitor.csv presenti in log_folder e sottocartelle.
    Funziona sia con env singolo sia con env paralleli.
    """
    monitor_files = glob.glob(os.path.join(log_folder, "**", "*.monitor.csv"), recursive=True)
    monitor_files += glob.glob(os.path.join(log_folder, "**", "monitor.csv"), recursive=True)

    if len(monitor_files) == 0:
        raise FileNotFoundError(f"Nessun file monitor trovato in {log_folder}")

    dfs = []
    for file in monitor_files:
        try:
            df = pd.read_csv(file, skiprows=1)
            if {"r", "l", "t"}.issubset(df.columns):
                df["source_file"] = file
                dfs.append(df)
        except Exception as e:
            print(f"Warning: impossibile leggere {file}: {e}")

    if len(dfs) == 0:
        raise ValueError("Nessun monitor valido trovato con colonne r, l, t.")

    data = pd.concat(dfs, ignore_index=True)
    return data


def plot_results(log_folder, title="Learning Curve", window=50):
    """
    Plot compatibile con env singolo e env paralleli.
    x = timestep cumulativi approssimati
    y = reward episodico smussato
    """
    try:
        data = load_all_monitor_files(log_folder)
    except Exception as e:
        print(f"Impossibile creare il grafico: {e}")
        return

    data = data.sort_values("t").reset_index(drop=True)

    x = data["l"].cumsum().to_numpy()
    y = data["r"].to_numpy()

    if len(y) == 0:
        print("Nessun episodio trovato nei monitor.")
        return

    y_smooth = moving_average(y, window=window)
    x_smooth = x[len(x) - len(y_smooth):]

    plt.figure(figsize=(10, 5))
    plt.plot(x_smooth, y_smooth, label=f"Reward (moving avg, window={window})")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_env_info(env_id):
    """
    Crea un env temporaneo singolo solo per stampare info.
    """
    tmp_env = gym.make(env_id)

    print("State space:", tmp_env.observation_space)
    print("Action space:", tmp_env.action_space)

    masses = tmp_env.unwrapped.get_parameters()
    names = tmp_env.unwrapped.body_names

    print("\nDynamics parameters (nome -> massa):")
    for name, mass in zip(names, masses):
        print(f"{name}: {mass}")

    tmp_env.close()


def build_train_env(args, log_dir):
    """
    Costruisce l'env di training.
    PPO supporta parallelizzazione.
    SAC qui resta singolo per semplicità.
    """
    if args.algo == "ppo" and args.n_envs > 1:
        env_fns = [
            make_env(args.env, i, seed=args.seed, log_dir=log_dir)
            for i in range(args.n_envs)
        ]

        if args.vec_env == "subproc":
            env = SubprocVecEnv(env_fns)
        elif args.vec_env == "dummy":
            env = DummyVecEnv(env_fns)
        else:
            raise ValueError(f"VecEnv type not supported: {args.vec_env}")

        env = VecMonitor(env, log_dir)

        print(f"\nUsing vectorized PPO training with {args.n_envs} envs ({args.vec_env})")
        print(
            f"Effective rollout size per PPO update: "
            f"{args.n_steps} x {args.n_envs} = {args.n_steps * args.n_envs}"
        )
        return env

    env = gym.make(args.env)
    monitor_file = os.path.join(log_dir, "single_env_monitor.csv")
    env = Monitor(env, monitor_file)
    print("\nUsing single environment training")
    return env


def build_test_env(args, log_dir):
    """
    Env di test singolo.
    """
    env = gym.make(args.env)
    monitor_file = os.path.join(log_dir, "test_monitor.csv")
    env = Monitor(env, monitor_file)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--algo", default="ppo", type=str, help="RL Algo [ppo, sac]")
    parser.add_argument("--lr", default=0.0003, type=float, help="Learning rate")
    parser.add_argument("--n_steps", default=2048, type=int, help="Number of steps for PPO")
    parser.add_argument("--batch_size", default=64, type=int, help="Mini-batch size")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--clip", default=0.2, type=float, help="PPO clipping range")
    parser.add_argument("--test_episodes", default=50, type=int, help="Episodes for evaluation")

    # Parallel PPO
    parser.add_argument("--n_envs", default=4, type=int, help="Number of parallel envs for PPO")
    parser.add_argument("--vec_env", default="subproc", type=str, help="VecEnv type [subproc, dummy]")

    args = parser.parse_args()

    set_seed(args.seed)

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    print_env_info(args.env)

    if args.test is None:
        env = build_train_env(args, log_dir)

        try:
            model = create_model(args, env)
            model.learn(total_timesteps=args.total_timesteps)

            if args.env == "CustomHopper-source-v0":
                save_name = f"{args.algo}_source"
            else:
                save_name = f"{args.algo}_target"

            model.save(save_name)
            print(f"\nModel saved to: {save_name}.zip")

            plot_results(
                log_dir,
                title=f"{args.algo.upper()} Training Curve",
                window=50
            )

        except KeyboardInterrupt:
            print("Interrupted!")

        finally:
            env.close()

    else:
        print("\nTesting...")
        env = build_test_env(args, log_dir)
        model = load_model(args, env)

        episode_rewards = []
        episode_lengths = []
        all_actions = []

        for ep in range(args.test_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            ep_length = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                all_actions.append(action)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            print(f"Episode {ep + 1:02d} | Reward: {ep_reward:.2f} | Length: {ep_length}")

        all_actions = np.array(all_actions)

        print("\n--- Test summary ---")
        print(f"Mean reward        : {np.mean(episode_rewards):.2f}")
        print(f"Std reward         : {np.std(episode_rewards):.2f}")
        print(f"Min reward         : {np.min(episode_rewards):.2f}")
        print(f"Max reward         : {np.max(episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
        print(f"Min episode length : {np.min(episode_lengths)}")
        print(f"Max episode length : {np.max(episode_lengths)}")

        print("\n--- Action statistics ---")
        print(f"Action mean: {all_actions.mean(axis=0)}")
        print(f"Action std : {all_actions.std(axis=0)}")
        print(f"Action min : {all_actions.min(axis=0)}")
        print(f"Action max : {all_actions.max(axis=0)}")

        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.test_episodes,
            deterministic=True
        )

        print("\n--- SB3 evaluate_policy ---")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        env.close()


if __name__ == "__main__":
    main()