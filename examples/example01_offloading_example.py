# example01_offloading_example.py

import sys
import os
import numpy as np
import random
import yaml
import sys
import time
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler
from airfogsim.data_manager import DataManager

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

# 1. Load the configuration file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. Create the environment
env = AirFogSimEnv(config, interactive_mode='graphic')
# env = AirFogSimEnv(config, interactive_mode=None)

# 初始化DataManager，传入SUMO网络文件
sumo_net_file = "./sumo_wujiaochang/osm.net.xml" 
data_manager = DataManager(env, sumo_net_file, update_interval=10)

# 3. Get algorithm module
algorithm_module = BaseAlgorithmModule()
algorithm_module.initialize(env)
RewardScheduler.setModel(env, 'REWARD', '1/task_delay')
accumulated_reward = 0
np.random.seed(0)
random.seed(0)
v2u_rate = [0]
v2i_rate = [0]
u2i_rate = [0]
for i in range(10):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"entity_info_{timestamp}_iteration_{i}.txt"

    step_count = 0  # 初始化步骤计数器

    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()

        accumulated_reward += algorithm_module.getRewardByTask(env)        #累计奖励
        task_num = TaskScheduler.getDoneTaskNum(env)  # 已完成的任务数
        out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(env)  # 超时任务数
        succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)  # 计算任务成功率
        
        step_count += 1
        data_manager.update_data(timestamp, i, step_count)  # 每10步更新一次数据


        env.render()
        v2u_rate.append(env.getChannelAvgRate('V2U'))
        v2i_rate.append(env.getChannelAvgRate('V2I'))
        u2i_rate.append(env.getChannelAvgRate('U2I'))

        # ‘\r'让下面一行打印一直打印在同一行
        print(f'Simulation time: {env.simulation_time:.2f}, 已完成任务数: {task_num:.2f}, 超时任务数: {out_of_ddl_task_num}, Ratio: {succ_ratio:.2f}, ACC_Reward: {succ_ratio*accumulated_reward/max(1,task_num):.2f} V2U: {v2u_rate[-1]:.2f}, V2I: {v2i_rate[-1]:.2f}, U2I: {u2i_rate[-1]:.2f}', end='\r')
    # 每次重置环境时强制更新一次数据
    data_manager.update_data(force_update=True)
    print()
    env.reset()
env.close()


# plt绘制
import matplotlib.pyplot as plt
plt.plot(v2u_rate[1:],label='V2U')
plt.plot(v2i_rate[1:],label='V2I')
plt.plot(u2i_rate[1:],label='U2I')
plt.legend()
plt.savefig('rate.png',dpi=300)

# 进行存储数据和可视化
data_manager.save_to_json()
data_manager.plot_results()
print('Simulation done!')