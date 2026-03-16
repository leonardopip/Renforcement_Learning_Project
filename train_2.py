import numpy as np
import argparse
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gymnasium as gym
from env.custom_hopper import *

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def create_model(args, env):
    if args.algo == 'ppo':
        model = PPO(policy="MlpPolicy",env=env,learning_rate=args.lr,n_steps=args.n_steps,batch_size=args.batch_size,gamma=args.gamma,clip_range=args.clip,verbose=1,seed=args.seed)
    
    elif args.algo == 'sac':
        model = SAC(policy="MlpPolicy",env=env,learning_rate=args.lr,batch_size=args.batch_size,gamma=args.gamma,verbose=1,seed=args.seed)
    
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model

def load_model(args, env):
    
    if args.algo == 'ppo':
        model = PPO.load(args.test, env=env)

    elif args.algo == 'sac':
        model = SAC.load(args.test, env= env)    
    
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model

def moving_average(values, window):
    
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="The total number of samples to train on")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='number of steps')
    parser.add_argument('--batch_size', default=64, type=int, help='Dimension minibatch')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--clip', default=0.2, type=float, help='clipping PPO')
    parser.add_argument('--test_episodes', default=50, type=int, help='# episodes for test evaluations')
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.unwrapped.get_parameters())  # masses of each link of the Hopper

    """
        TODO:

            - train a policy with stable-baselines3 on the source Hopper env
            - test the policy with stable-baselines3 on <source,target> Hopper envs (hint: see the evaluate_policy method of stable-baselines3)
    """

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(env, log_dir)

    
    if args.test is None:
        try:
            model = create_model(args, env)
            
            model.learn(total_timesteps=args.total_timesteps)
            if args.env == "CustomHopper-source-v0":
                 save_name = f"{args.algo}_source"
                 model.save(save_name)
                 print(f"Model saved to: {save_name}.zip")
            else:
                 save_name = f"{args.algo}_target"
                 model.save(save_name)
                 print(f"Model saved to: {save_name}.zip")    

            plot_results(log_dir)
        
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        print("Testing...")
        model = load_model(args, env)
        
        mean_reward, std_reward = evaluate_policy(model,env,n_eval_episodes=args.test_episodes,deterministic=True)

        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")
        


    env.close()    



if __name__ == '__main__':
    main()