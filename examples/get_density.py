import sys
import os
import numpy as np
import random
import yaml
import time
import json
import datetime

SEED = 42  # 你可以用任何你喜欢的整数

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

for index in range(1):
    print(f"开始第{index}次模拟")
    random.seed(SEED)
    np.random.seed(SEED)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    dir_name = os.path.dirname(__file__)

    from airfogsim import AirFogSimEnv, BaseAlgorithmModule
    from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler
    from airfogsim.data_manager import DataManager  # 导入DataManager
    # 1. Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)

    # 2. Create the environment
    env = AirFogSimEnv(config, interactive_mode='graphic')

    # 初始化DataManager，传入SUMO网络文件
    sumo_net_file = "./sumo_wujiaochang/osm.net.xml"
    data_manager = DataManager(env, sumo_net_file=sumo_net_file)  # 初始化DataManager

    # 3. Get algorithm module
    algorithm_module = BaseAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, 'REWARD', '1/task_delay')
    accumulated_reward = 0
    np.random.seed(0)
    random.seed(0)
    
    # 初始化通道速率列表
    v2u_rate = [0]
    v2i_rate = [0]

    # 📝 初始化特征日志
    feature_log = []
    
    # 定义感兴趣的区域范围
    region_x = [1250, 1500]  # x坐标范围
    region_y = [1400, 1600]  # y坐标范围

    # 开始模拟
    for i in range(1):
        step_count = 0

        while not env.isDone():
            algorithm_module.scheduleStep(env)
            env.step()
            step_count += 1
            env.render()
            
            # 获取通道平均速率
            v2u_rate.append(env.getChannelAvgRate('V2U'))
            v2i_rate.append(env.getChannelAvgRate('V2I'))
            
            print(f'模拟时间: {env.simulation_time:.2f}, V2U: {v2u_rate[-1]:.2f}, V2I: {v2i_rate[-1]:.2f}', end='\r')

            # 每10步获取一次车辆密度
            if step_count % 10 == 0:
                # 获取指定区域的车辆密度
                _, vehicle_density = data_manager.update_traffic_density(
                    x_min=region_x[0],
                    x_max=region_x[1],
                    y_min=region_y[0],
                    y_max=region_y[1]
                )
                # 单位转化为平方公里
                vehicle_density = vehicle_density * 1e6
                
                # 获取区域内车辆的平均速度
                # 使用EntityScheduler获取所有车辆节点
                vehicles_nodes = EntityScheduler.getFogNodesByType(env, 'vehicle')
                
                # 将节点对象转换为字典并筛选在指定区域内的车辆
                vehicles_in_region = []
                for node in vehicles_nodes:
                    vehicle_dict = node.to_dict()
                    if (region_x[0] <= vehicle_dict['position_x'] <= region_x[1] and
                        region_y[0] <= vehicle_dict['position_y'] <= region_y[1]):
                        vehicles_in_region.append(vehicle_dict)
                
                # 计算平均速度
                total_speed = 0
                if vehicles_in_region:
                    for vehicle in vehicles_in_region:
                        total_speed += vehicle['speed']
                    avg_speed = total_speed / len(vehicles_in_region)
                else:
                    avg_speed = 0
                
                # ✨ 保存本步特征
                feature_log.append({
                    "step": step_count,
                    "simulation_time": env.simulation_time,
                    "vehicle_density": vehicle_density,
                    "vehicle_count": len(vehicles_in_region),
                    "vehicle_avg_speed": avg_speed,
                    "v2u_rate": v2u_rate[-1],
                    "v2i_rate": v2i_rate[-1]
                })
                
                print(f"\n第 {step_count} 步:")
                print(f"车辆密度: {vehicle_density:.2f}/km²")
                print(f"区域内车辆数量: {len(vehicles_in_region)}")
                print(f"车辆平均速度: {avg_speed:.2f} m/s")
                print(f"V2U通道速率: {v2u_rate[-1]:.2f}")
                print(f"V2I通道速率: {v2i_rate[-1]:.2f}")

    print("\n模拟结束")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ✅ 保存特征日志到文件
    directory = "info"
    if not os.path.exists(directory):
        os.makedirs(directory)

    feature_file_path = f"{directory}/traffic_density_{timestamp}.json"
    with open(feature_file_path, "w") as f:
        json.dump(feature_log, f, indent=2)
    
    print(f"密度数据已保存至: {feature_file_path}")

    # 结束环境
    env.close()
    '''
    # 绘制通道速率图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(v2u_rate[1:], label='V2U')
    plt.plot(v2i_rate[1:], label='V2I')
    plt.title('通道平均速率')
    plt.xlabel('模拟步数')
    plt.ylabel('速率')
    plt.legend()
    plt.grid(True)
    rate_file_path = f"{directory}/channel_rate_{timestamp}.png"
    plt.savefig(rate_file_path, dpi=300)
    plt.close()
    
    print(f"通道速率图已保存至: {rate_file_path}")
    '''
    
    print('模拟完成!')
