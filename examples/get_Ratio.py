# example01_offloading_example.py

import sys
import os
import numpy as np
import random
import yaml
import sys
import time
import json
import datetime

SEED = 42  # 你可以用任何你喜欢的整数
def load_config(path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            return config

for index in range(10):
    print(f"开始第{index}次模拟")
    random.seed(SEED)
    np.random.seed(SEED)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    dir_name = os.path.dirname(__file__)

    from airfogsim import AirFogSimEnv, BaseAlgorithmModule
    from airfogsim.scheduler import RewardScheduler, TaskScheduler

    # 1. Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    # 每次模拟将config中的配置进行调整
    if index<10:
         config['task']['task_max_cpu'] = index*3/10.0+0.5
         config['task']['task_min_cpu'] = index*3/10.0+0.5
    elif index<20:
         config['task']['task_max_size'] = (index-10)*3/10.0+0.5
         config['task']['task_min_size'] = (index-10)*3/10.0+0.5
    else:
        config['task']['task_min_deadline'] = (index-20)*3/10.0+0.5
        config['task']['task_max_deadline'] = (index-20)*3/10.0+0.5
    # 2. Create the environment
    env = AirFogSimEnv(config)
    # env = AirFogSimEnv(config, interactive_mode=None)

    # 初始化DataManager，传入SUMO网络文件
    sumo_net_file = "./sumo_wujiaochang/osm.net.xml" 

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

        
    # 开始模拟
    for i in range(1):
        step_count = 0  # 初始化步骤计数器

        while not env.isDone():
            algorithm_module.scheduleStep(env)
            env.step()
            step_count += 1
            env.render()
            print(f'Simulation time: {env.simulation_time:.2f}', end='\r')
            
    print("模拟结束")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用时间戳来生成文件名
        # 在json文件中写入config文件中的内容
    directory = "info"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/succ_ratios.json"

    # 运行结束之后计算完成率
    task_num = TaskScheduler.getDoneTaskNum(env)  # 已完成的任务数
    out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(env)  # 超时任务数
    succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)  # 计算任务成功率
    task_config_info = {
            f"task_config{index+1}": {
                "task_size": config['task'].get('task_max_size'),
                "task_cpu": config['task'].get('task_max_cpu'),
                "task_deadline": config['task'].get('task_max_deadline'),
                "succ_ratio": succ_ratio
            }
        }
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(task_config_info)

    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

    env.close()
    print('Simulation done!')