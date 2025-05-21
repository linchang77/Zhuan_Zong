import sys
import os
import numpy as np
import random
import yaml
import time
import json
import datetime
import matplotlib.pyplot as plt

SEED = 42

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

def find_max_density_grid(density_map, grid_info):
    """
    找出车辆密度最大的网格
    
    Args:
        density_map: 车辆密度矩阵
        grid_info: 网格信息(x_min, x_max, y_min, y_max, grid_size等)
    
    Returns:
        max_density: 最大密度值
        max_grid_coords: 密度最大的网格坐标 (x_idx, y_idx)
        max_real_coords: 密度最大的网格实际坐标 (x_center, y_center)
    """
    # 找出密度最大的网格
    y_idx, x_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
    max_density = density_map[y_idx][x_idx]
    
    # 将网格索引转换为实际坐标
    grid_size = grid_info["grid_size"]
    x_min = grid_info["x_min"]
    y_min = grid_info["y_min"]
    
    # 计算网格中心点的实际坐标
    x_center = x_min + (x_idx + 0.5) * grid_size
    y_center = y_min + (y_idx + 0.5) * grid_size
    
    return max_density, (x_idx, y_idx), (x_center, y_center)

def visualize_max_density_grid(density_map, max_grid_coords, grid_info, output_file="max_density_grid.png"):
    """
    可视化车辆密度热力图，并标记密度最大的网格
    """
    plt.figure(figsize=(10, 8))
    x_min, x_max = grid_info["x_min"], grid_info["x_max"]
    y_min, y_max = grid_info["y_min"], grid_info["y_max"]
    
    # 使用通用字体设置，避免字体依赖问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 显示密度热力图
    im = plt.imshow(density_map, cmap="Reds", origin='lower',
                   extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(im, label="Vehicle Density (/km²)")
    
    # 标记最大密度的网格
    x_idx, y_idx = max_grid_coords
    grid_size = grid_info["grid_size"]
    x_center = x_min + (x_idx + 0.5) * grid_size
    y_center = y_min + (y_idx + 0.5) * grid_size
    
    # 在最大密度网格上画红色方框
    x_grid_min = x_min + x_idx * grid_size
    y_grid_min = y_min + y_idx * grid_size
    plt.plot([x_grid_min, x_grid_min + grid_size, x_grid_min + grid_size, x_grid_min, x_grid_min],
             [y_grid_min, y_grid_min, y_grid_min + grid_size, y_grid_min + grid_size, y_grid_min],
             'r-', linewidth=2)
    
    plt.title("Vehicle Density Heatmap (Max Density Grid Marked)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"热力图已保存至 {output_file}")

# 主函数
if __name__ == "__main__":
    print("开始寻找车辆密度最大网格")
    random.seed(SEED)
    np.random.seed(SEED)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    dir_name = os.path.dirname(__file__)

    from airfogsim import AirFogSimEnv, BaseAlgorithmModule
    from airfogsim.scheduler import RewardScheduler
    from airfogsim.data_manager import DataManager

    # 1. 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)

    # 2. 创建环境
    env = AirFogSimEnv(config)

    # 初始化SUMO网络文件路径
    sumo_net_file = "./sumo_wujiaochang/osm.net.xml"

    # 3. 初始化DataManager，用于获取密度数据
    data_manager = DataManager(env, sumo_net_file=sumo_net_file, output_dir="results")

    # 4. 获取算法模块
    algorithm_module = BaseAlgorithmModule()
    algorithm_module.initialize(env)
    RewardScheduler.setModel(env, 'REWARD', '1/task_delay')
    
    # 设置随机种子
    np.random.seed(0)
    random.seed(0)

    # 记录网格密度分析结果
    density_analysis = {
        "simulation_info": {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "seed": SEED
        },
        "max_density_grids": []
    }

    # 开始模拟
    print("开始模拟...")
    step_count = 0
    
    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()
        step_count += 1
        env.render()
        print(f'模拟时间: {env.simulation_time:.2f}', end='\r')
        
        # 每隔一定步数计算一次交通密度热力图
        if step_count % 10 == 0:
            print(f"\n第 {step_count} 步: 计算交通密度热力图...")
            
            # 计算交通密度热力图
            data_manager.compute_traffic_density_map(grid_size=100)
            
            # 从保存的JSON文件中读取密度数据
            try:
                with open(os.path.join("results", "traffic_density_data.json"), "r") as f:
                    density_data = json.load(f)
                
                # 将车辆密度矩阵转换为NumPy数组
                vehicle_density_map = np.array(density_data["vehicle_density"])
                
                # 找出密度最大的网格
                max_density, max_grid_coords, max_real_coords = find_max_density_grid(
                    vehicle_density_map, density_data["grid_info"])
                
                # 可视化密度最大的网格
                output_file = f"results/max_density_grid_step_{step_count}.png"
                visualize_max_density_grid(
                    vehicle_density_map, max_grid_coords, 
                    density_data["grid_info"], output_file)
                
                # 记录结果
                density_analysis["max_density_grids"].append({
                    "step": step_count,
                    "simulation_time": env.simulation_time,
                    "max_density": float(max_density),
                    "grid_coords": max_grid_coords,
                    "real_coords": max_real_coords
                })
                
                print(f"第 {step_count} 步: 最大车辆密度 {max_density:.2f}/km², "
                      f"位置: 网格坐标({max_grid_coords[0]}, {max_grid_coords[1]}), "
                      f"实际坐标({max_real_coords[0]:.1f}, {max_real_coords[1]:.1f})")
            
            except Exception as e:
                print(f"无法读取或处理密度数据: {e}")
    
    print("\n模拟结束")
    
    # 保存分析结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/max_density_analysis_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(density_analysis, f, indent=2)
    
    print(f"密度分析结果已保存至 {result_file}")
    
    # 找出所有时间步中密度最大的网格
    if density_analysis["max_density_grids"]:
        max_step = max(density_analysis["max_density_grids"], 
                       key=lambda x: x["max_density"])
        
        print("\n==== 所有时间步中密度最大的网格 ====")
        print(f"时间步: {max_step['step']}")
        print(f"模拟时间: {max_step['simulation_time']:.2f}")
        print(f"最大密度: {max_step['max_density']:.2f}/km²")
        print(f"网格坐标: ({max_step['grid_coords'][0]}, {max_step['grid_coords'][1]})")
        print(f"实际坐标: ({max_step['real_coords'][0]:.1f}, {max_step['real_coords'][1]:.1f})")
    
    # 结束环境
    env.close()
    print('模拟完成!')