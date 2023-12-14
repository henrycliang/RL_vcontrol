# RL_vcontrol

This is a course project of 2023 fall course CUHK IERG6130.

# Preparation
This project focuses on daily voltage control with reinforcement learning. Please see the [reference](https://github.com/siemens/powergym) of this project, also see the [toolkit](https://stable-baselines3.readthedocs.io/en/master/) to prepare this project.

# Simulation Result
In this project, we use both [PPO](https://arxiv.org/abs/1707.06347) and A2C to learn near-optimal control laws. 

The learning curve of PPO and A2C during training:

<img width="500"  src=training_reward.png>


During testing, the change of all bus voltages controlled by PPO policy in a day is:

<img width="500"  src=systems\13Bus\voltage_trend_PPO.png>

The the change of all bus voltages controlled by A2C policy in a day is:

<img width="500"  src=systems\13Bus\voltage_trend_A2C.png>

The result of random control is:

<img width="500"  src=systems\13Bus\voltage_trend_random.png>