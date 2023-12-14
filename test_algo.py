from powergym.env_register import make_env
import numpy as np
import argparse
import random
import itertools
import sys, os
import multiprocessing as mp
import matplotlib.pyplot as plt

from stable_baselines3 import PPO,A2C
import os
import matplotlib.colors as colors

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

    use_plot=True
    plot_trend=True  # to plot the trend of bus voltages and load profiles
    cwd = os.getcwd()


    algo = 'A2C'
    env_name = '13Bus'
    log_dir = "PowerFlow/results/" + algo + "/"

    ## plot the voltage
    if use_plot and not os.path.exists(os.path.join(cwd, algo+'_plots')):
        os.makedirs(os.path.join(cwd, algo+'_plots'))

    env = make_env(env_name, worker_idx)
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num, env.bat_num))
    print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
    print('-' * 80)

    # load the model
    model=A2C.load(log_dir+'\\'+algo+"model",env=env)
    episode_reward = 0.0
    load_profile_idx=0
    #reset the environment

    for j in range(1):
        obs = env.reset(load_profile_idx=j)
        for i in range(env.horizon):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                print("Load_profile: {}: Episode is: {}".format(j,episode_reward))
                episode_reward = 0

            if use_plot:
                # node_bound argument to decide to plot max or min node voltage for nodes with more than 1 phase
                fig, _ = env.plot_graph()
                fig.tight_layout(pad=0.1)
                fig.savefig(os.path.join(cwd, algo+'_plots', 'node_voltage_' + str(i) + '.png'))
                plt.close()

        voltage_mat=env.voltage_mat
        busName_list=env.bus_name_list

    if plot_trend==True:
        voltage_array=np.array(voltage_mat)
        timestep=np.arange(voltage_array.shape[0])

        # timestep=np.array([timestep]).T
        # voltage_array=np.hstack((timestep,voltage_array))
        # voltage_dataframe=pd.DataFrame(voltage_array,index=['timestep', busName_list])
        color_list=[]
        for i,(name,hsv) in enumerate(colors.cnames.items()):
            if (i+1)%8==0:
                color_list.append(name)
            if len(color_list)==16:
                break
        color_list[7]='r'
        color_list[8]='darkorange'
        color_list[15]='green'
        plt.figure(dpi=300, figsize=(12, 8))
        for i in range(voltage_array.shape[1]):
            plt.plot(timestep,voltage_array[:,i],label=busName_list[i],c=color_list[i],linewidth=2.0)
        plt.legend(fontsize=15,loc="upper right")
        # plt.title('The change of voltage under RL control (PPO)',fontsize=20)
        plt.ylabel('Voltage Magnitude (p.u.)',fontsize=20)
        plt.xlabel('Time step from 00:00 AM to 23:00 PM',fontsize=20)
        plt.tight_layout()
        plt.savefig('voltage_trend_A2C.png')
        plt.show()
        print(1)



