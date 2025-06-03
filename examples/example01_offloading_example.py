# example01_offloading_example.py

import sys
import os
import numpy as np
import random
import yaml
import time
import json
import datetime
import matplotlib.pyplot as plt

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler, ComputationScheduler
from airfogsim.data_manager import DataManager
from live_plot import MultiLivePlot

# 平均速率计算工具函数（基于 EntityScheduler）
def get_vehicle_avg_speed(env):
    from airfogsim.scheduler import EntityScheduler
    vehicle_nodes = EntityScheduler.getFogNodesByType(env, 'vehicle')
    vehicle_list = [v.to_dict() for v in vehicle_nodes]
    speeds = [v['speed'] for v in vehicle_list if 'speed' in v]
    return sum(speeds) / len(speeds) if speeds else 0.0

def get_uav_avg_speed(env):
    from airfogsim.scheduler import EntityScheduler
    uav_nodes = EntityScheduler.getFogNodesByType(env, 'uav')
    uav_list = [u.to_dict() for u in uav_nodes]
    speeds = [u['speed'] for u in uav_list if 'speed' in u]
    return sum(speeds) / len(speeds) if speeds else 0.0

# 加载配置
def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
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
# data_manager = DataManager(env, sumo_net_file, update_interval=10)

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

# 数据提取动态图
# plot_titles = ["succ_ratio", "avg_speed_car", "avg_speed_uav", "v2u_rate", "v2i_rate", "u2i_rate", "avg_running_compute_delay"]
# plotter = MultiLivePlot(plot_titles, ncols=3)

# 创建主输出目录
main_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
main_output_dir = os.path.join("info", main_timestamp)
os.makedirs(main_output_dir, exist_ok=True)

for i in range(1):
    # 本轮输出信息
    round_dir = os.path.join(main_output_dir, f"round_{i}")
    os.makedirs(round_dir, exist_ok=True)

    mode1_file = os.path.join(round_dir, "mode1_info.json")
    mode2_file = os.path.join(round_dir, "mode2_info.json")

    # 初始化信息结构
    task_config_info = {
        "task_config": {
            "task_min_size": config['task'].get('task_min_size'),
            "task_max_size": config['task'].get('task_max_size'),
            "task_min_cpu": config['task'].get('task_min_cpu'),
            "task_max_cpu": config['task'].get('task_max_cpu'),
            "task_min_deadline": config['task'].get('task_min_deadline'),
            "task_max_deadline": config['task'].get('task_max_deadline')
        }
    }
    mode1_info = dict(task_config_info)
    mode1_info["succ_ratio_list"] = []
    mode2_info = {"info_list": []}

    step_count = 0  # 初始化步骤计数器

    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()

        accumulated_reward += algorithm_module.getRewardByTask(env)        #累计奖励
        task_num = TaskScheduler.getDoneTaskNum(env)  # 已完成的任务数
        out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(env)  # 超时任务数
        succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)  # 计算任务成功率

        step_count += 1
        # data_manager.update_data(main_timestamp, i, step_count)  # 每10步更新一次数据

        env.render()
        v2u_rate.append(env.getChannelAvgRate('V2U'))
        v2i_rate.append(env.getChannelAvgRate('V2I'))
        u2i_rate.append(env.getChannelAvgRate('U2I'))

        # mode1: 每步记录
        mode1_info["succ_ratio_list"].append({
            "step": step_count,
            "succ_ratio": round(succ_ratio, 4)
        })
        with open(mode1_file, 'w') as f1:
            json.dump(mode1_info, f1, indent=4)

        # mode2: 每5步记录
        if step_count % 5 == 0:
            avg_speed_car = get_vehicle_avg_speed(env)
            avg_speed_uav = get_uav_avg_speed(env)

             # === 任务状态统计 ===
            status_counts = {
                "waiting": len(TaskScheduler.getAllToOffloadTasks(env)),
                "offloading": len(TaskScheduler.getAllOffloadingTaskInfos(env)),
                "computing": len(TaskScheduler.getAllComputingTaskInfos(env)),
                "done": TaskScheduler.getDoneTaskNum(env),
                "failed": TaskScheduler.getOutOfDDLTasks(env)
            }

            # === 正在执行的任务详情（每个实体） ===
            running_tasks_info = []
            compute_delays = []
            all_ids, type_list = EntityScheduler.getAllNodeIdsWithType(env)
            for node_id, node_type in zip(all_ids, type_list):
                total_remain_cpu, cpu, compute_delay = ComputationScheduler.getComputeStatusByNodeId(env, node_id)
                if total_remain_cpu > 0:
                    entity_task_info = {
                        "node_id": node_id,
                        "node_type": node_type,
                        "remaining_cpu": round(total_remain_cpu, 2),
                        "cpu": round(cpu, 2),
                        "compute_delay": round(compute_delay, 4)
                    }
                    running_tasks_info.append(entity_task_info)
                    compute_delays.append(compute_delay)
            avg_running_compute_delay = (
                round(sum(compute_delays) / len(compute_delays), 4) if compute_delays else 0.0
            )

            mode2_info["info_list"].append({
                "step": step_count,
                "succ_ratio": round(succ_ratio, 4),
                "avg_speed_car": round(avg_speed_car, 2),
                "avg_speed_uav": round(avg_speed_uav, 2),
                "v2u_rate": round(v2u_rate[-1], 2),
                "v2i_rate": round(v2i_rate[-1], 2),
                "u2i_rate": round(u2i_rate[-1], 2),
                "task_status": status_counts,
                "running_tasks": running_tasks_info,
                "avg_running_compute_delay": avg_running_compute_delay
            })

            # plotter.update(step_count, [
            #     succ_ratio,
            #     avg_speed_car,
            #     avg_speed_uav,
            #     v2u_rate[-1],
            #     v2i_rate[-1],
            #     u2i_rate[-1],
            #     avg_running_compute_delay
            # ])

        with open(mode2_file, 'w') as f2:
            json.dump(mode2_info, f2, indent=4)


        # ‘\r'让下面一行打印一直打印在同一行
        print(f'Simulation time: {env.simulation_time:.2f}, 已完成任务数: {task_num:.2f}, 超时任务数: {out_of_ddl_task_num}, Ratio: {succ_ratio:.2f}, ACC_Reward: {succ_ratio*accumulated_reward/max(1,task_num):.2f} V2U: {v2u_rate[-1]:.2f}, V2I: {v2i_rate[-1]:.2f}, U2I: {u2i_rate[-1]:.2f}', end='\r')
    # 每次重置环境时强制更新一次数据
    # data_manager.update_data(force_update=True)
    # print()
    # env.reset()
env.close()
# 数据提取动态图
# plotter.save_gif('info/multi_plot.gif', duration=300)  # 每帧 300ms，可根据需要调整
# print('GIF saved at info/multi_plot.gif')

# # 在仿真结束后，生成交通流密度相关的数据和可视化
# print("\n生成交通流密度数据和可视化...")

# # 1. 计算全局的车辆和UAV交通流密度热力图
# print("正在生成全局交通流密度热力图...")
# data_manager.compute_traffic_density_map(grid_size=100)

# # 2. 找出车辆最多的交叉路口
# print("正在查找最拥堵的交叉路口...")
# congested_id, vehicle_density, uav_density = data_manager.find_most_congested_intersection()
# print(f"最拥堵的交叉路口ID: {congested_id}, 车辆密度: {vehicle_density:.2f}/km², UAV密度: {uav_density:.2f}/km²")

# # 3. 保存每个交叉路口的交通流密度数据到JSON
# print("正在保存交叉路口交通流密度数据...")
# data_manager.save_intersection_density_to_json(output_file="results/intersection_density.json")

# # 4. 生成交通流热力图
# print("正在生成交通流热力图...")
# data_manager.generate_traffic_heatmap(time_window=60, output_file="results/traffic_heatmap.png")


# # plt绘制
# import matplotlib.pyplot as plt
# plt.plot(v2u_rate[1:],label='V2U')
# plt.plot(v2i_rate[1:],label='V2I')
# plt.plot(u2i_rate[1:],label='U2I')
# plt.legend()
# plt.savefig('rate.png',dpi=300)

# # 进行存储数据和可视化
# data_manager.save_to_json()
print('Simulation done!')