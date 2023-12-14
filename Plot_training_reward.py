import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import pandas as pd
import seaborn as sns

algo_name='PPO'

training_reward=np.load('training_reward_'+algo_name+'.npy')

timestep=np.arange(len(training_reward))
curve_reward_PPO=np.vstack((timestep,training_reward))
curve_reward_PPO=pd.DataFrame(curve_reward_PPO,index=["Traing episode","Episode reward"])
curve_reward_PPO=curve_reward_PPO.T
curve_reward_PPO['Algo'] = 'PPO'

algo_name='A2C'

training_reward_A2C=np.load('training_reward_'+algo_name+'.npy')
timestep=np.arange(len(training_reward_A2C))
curve_reward_A2C=np.vstack((timestep,training_reward_A2C))
curve_reward_A2C=pd.DataFrame(curve_reward_A2C,index=["Training episode","Episode reward"])
curve_reward_A2C=curve_reward_A2C.T
curve_reward_A2C['Algo']='A2C'

curve_reward=pd.concat([curve_reward_PPO[:len(curve_reward_A2C)],curve_reward_A2C])
# curve_reward = curve_reward.reset_index()

plt.figure(dpi=300,figsize=(12,6))
sns.set("notebook",font_scale=1.2)
sns.set_style("whitegrid")
ax = sns.lineplot(
    data=curve_reward,
    x="Training episode",
    y="Episode reward",
    hue='Algo',
)

plt.tight_layout()
# ax.set_title("Learning curve")
plt.savefig("training_reward.png")
plt.show()