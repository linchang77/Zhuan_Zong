# example01_offloading_example.py

import sys
import os
import numpy as np
import random
import yaml
import sys
import time
import datetime

SEED = 42  # 你可以用任何你喜欢的整数

random.seed(SEED)
np.random.seed(SEED)

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
for i in range(1):
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
    # data_manager.update_data(force_update=True)
    # print()
    # env.reset()
env.close()

# 在仿真结束后，生成交通流密度相关的数据和可视化
print("\n生成交通流密度数据和可视化...")

# 1. 计算全局的车辆和UAV交通流密度热力图
print("正在生成全局交通流密度热力图...")
data_manager.compute_traffic_density_map(grid_size=100)

# 2. 找出车辆最多的交叉路口
print("正在查找最拥堵的交叉路口...")
congested_id, vehicle_density, uav_density = data_manager.find_most_congested_intersection()
print(f"最拥堵的交叉路口ID: {congested_id}, 车辆密度: {vehicle_density:.2f}/km², UAV密度: {uav_density:.2f}/km²")

# 3. 保存每个交叉路口的交通流密度数据到JSON
print("正在保存交叉路口交通流密度数据...")
data_manager.save_intersection_density_to_json(output_file="results/intersection_density.json")

# 4. 生成交通流热力图
print("正在生成交通流热力图...")
data_manager.generate_traffic_heatmap(time_window=60, output_file="results/traffic_heatmap.png")


# plt绘制
import matplotlib.pyplot as plt
plt.plot(v2u_rate[1:],label='V2U')
plt.plot(v2i_rate[1:],label='V2I')
plt.plot(u2i_rate[1:],label='U2I')
plt.legend()
plt.savefig('rate.png',dpi=300)

# 进行存储数据和可视化
data_manager.save_to_json()
print('Simulation done!')