from powergym.env_register import make_env, remove_parallel_dss
import numpy as np
import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

from stable_baselines3 import PPO
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                         help='random seed')

    args = parser.parse_args()
    return args

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__=='__main__':
    args = parse_arguments()
    seeding(args.seed)
    worker_idx=None

    algo = 'PPO'
    env_name='13Bus'
    # Create training log dir of the DRL
    log_dir = "PowerFlow/results/" + algo + "/"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_name, worker_idx)
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num, env.bat_num))
    print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
    print('-' * 80)

    # assuming we only train on one load profile for now
    # load_profile_idx=0
    # obs = env.reset(load_profile_idx=load_profile_idx)

    model = PPO("MlpPolicy",
                env,
                # learning_rate=linear_schedule(1e-3),
                verbose=1)

    model.learn(total_timesteps=1000000)
    model.save(log_dir + "\\" + algo + "model")

    print(env.reward_recorder)
    print(len(env.reward_recorder))
    np.save('E:\powergym/training_reward_{}.npy'.format(algo), env.reward_recorder)

