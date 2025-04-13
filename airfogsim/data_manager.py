import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
import datetime
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler, ComputationScheduler
from sumolib import net  # SUMO 交通拓扑解析库
import json
import os

class DataManager:
    def __init__(self, env, sumo_net_file=None, output_dir="results", update_interval=10):
        self.env = env  # 绑定仿真环境

        crossroads = EntityScheduler.getAllCrossroadPositions(self.env)
        print(f"所有交叉路口坐标: {crossroads}")
        
        # 确保输出目录存在
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 任务相关数据
        self.finish_task_num = 0                 # 已完成任务数量
        self.out_of_ddl_task_num = 0             # 超时任务数量

        self.succ_ratio_list = []                # 任务完成率列表
        
        # 交通流密度
        self.uav_density_list = []               # UAV 交通流密度
        self.vehicle_density_list = []           # 车辆交通流密度

        # SUMO 交通拓扑
        self.sumo_net = net.readNet(sumo_net_file) if sumo_net_file else None
        self.traffic_topology = {}

        # 输出文件路径
        self.output_file = os.path.join(self.output_dir, "simulation_data.json")

        # 数据存储
        self.simulation_data = []  # 存储每个时间步的数据

        # 数据更新间隔和计数器
        self.update_interval = update_interval
        self.step_counter = 0

        # 预提取交通拓扑
        self.extract_sumo_traffic_topology()


    def update_traffic_density(self, x_min=0, x_max=500, y_min=0, y_max=500):
        """
        计算 UAV 和 车辆的交通流密度
        :param x_min, x_max, y_min, y_max: 指定的区域范围
        :return: (uav_density, vehicle_density)
        """
        # 使用 EntityScheduler 获取实体信息
        # 获取所有 UAV 节点信息
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        # 获取所有车辆节点信息
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
        
        # 将节点对象转换为字典
        uavs = [node.to_dict() for node in uavs_nodes]
        vehicles = [node.to_dict() for node in vehicles_nodes]
        
        # 统计单位面积内的数量
        uav_count = sum(1 for uav in uavs if x_min <= uav['position_x'] <= x_max and y_min <= uav['position_y'] <= y_max)
        vehicle_count = sum(1 for vehicle in vehicles if x_min <= vehicle['position_x'] <= x_max and y_min <= vehicle['position_y'] <= y_max)
                
        area = (x_max - x_min) * (y_max - y_min)
        uav_density = uav_count / area if area > 0 else 0
        vehicle_density = vehicle_count / area if area > 0 else 0

        self.uav_density_list.append(uav_density)
        self.vehicle_density_list.append(vehicle_density)

        return uav_density, vehicle_density

    def update_task_completion_rate(self):
        """
        计算任务完成率 = 任务完成数 / (任务完成数 + 超时任务数)
        """
        # 获取当前任务数据
        task_num = TaskScheduler.getDoneTaskNum(self.env)
        out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(self.env)
        succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)

        # 记录数据
        self.finish_task_num = task_num                           # 已完成任务数量
        self.out_of_ddl_task_num = out_of_ddl_task_num            # 超时任务数量
        self.succ_ratio_list.append(succ_ratio)

        return succ_ratio

    def extract_sumo_traffic_topology(self):
        """
        提取 SUMO 交通拓扑信息，包括：
        1. 道路连接性
        2. 交通信号
        3. 限速
        """
        if not self.sumo_net:
            print("SUMO 网络未加载，跳过交通拓扑提取")
            return
        
        self.traffic_topology = {
            "edges": [],
            "signals": [],
            "speed_limits": {}
        }

        for edge in self.sumo_net.getEdges():
            self.traffic_topology["edges"].append(edge.getID())
            lanes = edge.getLanes()
            if lanes:
                self.traffic_topology["speed_limits"][edge.getID()] = lanes[0].getSpeed()  # 记录第一条车道的限速
        
        for tls in self.sumo_net.getTrafficLights():
            self.traffic_topology["signals"].append(tls.getID())

        print("SUMO 交通拓扑信息提取完成")

    def update_entity_info(self, timestamp, iteration, step):
        '''
        获取车辆和无人机的位置、速度、正在使用的计算资源、剩余的计算资源
        下面是一个车辆/无人机的所有信息
        id: UAV_0
        position_x: 659.0568260908746
        position_y: 2255.549149629989
        position_z: 131.6287023569144
        speed: 26.888437030500963
        angle: 3.448296944257913
        acceleration: -268.8843703050096
        is_transmitting: False
        is_receiving: False
        revenue: 0
        ai_model_dict: {}
        token: None
        fog_profile: {'cpu': 3, 'memory': 1, 'storage': 1}
        task_profile: {'lambda': 0.5, 'dag_edge_prob': 0.3}
        phi: 0
        last_updated_time: 0
        node_type: 'U'
        '''
        
        # 使用时间戳来生成文件名
        directory = "info"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = f"{directory}/entity_info_{timestamp}_it{iteration}.txt"

        # # 获取所有节点的ID和类型
        all_ids, type_list = EntityScheduler.getAllNodeIdsWithType(self.env)
        if step % 10 == 0:
            with open(file_path, "a") as f:
                # 写入调用次数和每次的实体信息
                f.write(f"Entitied Update at Step {step}:\n")
                
                for node_id, node_type in zip(all_ids, type_list):
                    node_info = EntityScheduler.getNodeInfoById(self.env, node_id)

                    # 调用 getComputeDelayByNodeId 获取计算资源信息
                    compute_info = ComputationScheduler.getComputeInfoByNodeId(self.env, node_id)
                    
                    f.write(f"Node ID: {node_info['id']}, Type: {node_type}, ")
                    f.write(f"Position (x, y, z): ({node_info['position_x']}, {node_info['position_y']}, {node_info['position_z']}), ")
                    f.write(f"Speed: {node_info['speed']}, Angle: {node_info['angle']}, Acceleration: {node_info['acceleration']}, ")
                    f.write(f"Compute Delay: {compute_info['compute_delay']:.4f}, ")
                    f.write(f"Used CPU: {compute_info['used_cpu']:.2f}, Remaining CPU: {compute_info['remaining_cpu']:.2f}\n")
                
                # 添加分隔行以便区分不同调用
                f.write("-" * 50 + "\n")

    def get_task_status_counts(self, timestamp, iteration, step):
        """
        获取所有任务，并计算不同状态任务的个数
        """
        # 使用时间戳来生成文件名
        directory = "info"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = f"{directory}/entity_info_{timestamp}_it{iteration}.txt"

        # 统计任务状态
        status_counts = {
            "等待卸载": len(TaskScheduler.getAllToOffloadTasks(self.env)),  # 等待卸载
            "正在卸载": len(TaskScheduler.getAllOffloadingTaskInfos(self.env)),  # 正在卸载
            "计算中": len(TaskScheduler.getAllComputingTaskInfos(self.env)),  # 计算中
            "已完成": TaskScheduler.getDoneTaskNum(self.env),  # 已完成
            "失败": TaskScheduler.getOutOfDDLTasks(self.env)  # 失败（超时）
        }

        # 输出统计结果
        if step % 10 == 0:
            with open(file_path, "a") as f:
                f.write(f"Task Status Update at Step {step}:\n")
                for status, count in status_counts.items():
                    f.write(f"{status}: {count}\n")
                f.write("-" * 50 + "\n")
        return status_counts

    def compute_traffic_density_map(self, grid_size=100):
        """
        计算全局的车辆和 UAV 交通流密度，并生成热力图
        :param grid_size: 每个小网格的边长（单位：m），默认 100m x 100m
        """
        # 获取所有 UAV 和 车辆
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
    
        # 获取所有节点的坐标
        all_nodes = uavs_nodes + vehicles_nodes
        if not all_nodes:
            print("没有车辆或 UAV 数据！")
            return
    
        positions_x = [node.to_dict()['position_x'] for node in all_nodes]
        positions_y = [node.to_dict()['position_y'] for node in all_nodes]
    
        # 确定全局坐标范围
        x_min, x_max = min(positions_x), max(positions_x)
        y_min, y_max = min(positions_y), max(positions_y)
    
        # 计算网格数
        x_bins = max(1, int((x_max - x_min) / grid_size))
        y_bins = max(1, int((y_max - y_min) / grid_size))
    
        print(f"坐标范围: X({x_min:.1f}-{x_max:.1f}), Y({y_min:.1f}-{y_max:.1f})")
        print(f"网格数量: {x_bins}x{y_bins}")
        print(f"UAV数量: {len(uavs_nodes)}, 车辆数量: {len(vehicles_nodes)}")
    
        # 初始化密度矩阵
        uav_density_map = np.zeros((y_bins, x_bins))
        vehicle_density_map = np.zeros((y_bins, x_bins))
    
        # 计算每个网格的节点数量
        for node in uavs_nodes:
            pos = node.to_dict()
            x_idx = min(x_bins-1, max(0, int((pos['position_x'] - x_min) / grid_size)))
            y_idx = min(y_bins-1, max(0, int((pos['position_y'] - y_min) / grid_size)))
            uav_density_map[y_idx, x_idx] += 1
    
        for node in vehicles_nodes:
            pos = node.to_dict()
            x_idx = min(x_bins-1, max(0, int((pos['position_x'] - x_min) / grid_size)))
            y_idx = min(y_bins-1, max(0, int((pos['position_y'] - y_min) / grid_size)))
            vehicle_density_map[y_idx, x_idx] += 1
    
        # 计算每平方公里米的密度
        area_km2 = (grid_size * grid_size) / 1e6
        uav_density_map = uav_density_map / area_km2
        vehicle_density_map = vehicle_density_map / area_km2
    
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
        # 画出热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 设置中文字体，解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # UAV 热力图
        im1 = axes[0].imshow(uav_density_map, cmap="Blues", origin='lower', 
                             extent=[x_min, x_max, y_min, y_max])
        axes[0].set_title("UAV Traffic Density (per km²)")
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")
        fig.colorbar(im1, ax=axes[0], label="UAV Density (/km²)")
        
        # 车辆热力图
        im2 = axes[1].imshow(vehicle_density_map, cmap="Reds", origin='lower', 
                             extent=[x_min, x_max, y_min, y_max])
        axes[1].set_title("Vehicle Traffic Density (per km²)")
        axes[1].set_xlabel("X Coordinate")
        axes[1].set_ylabel("Y Coordinate")
        fig.colorbar(im2, ax=axes[1], label="Vehicle Density (/km²)")
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "traffic_density_map.png"), dpi=300)
        plt.close()
        
        print(f"全局交通流密度热力图已保存至 {os.path.join(self.output_dir, 'traffic_density_map.png')}")
        
        # 保存密度数据到 JSON
        density_data = {
            "grid_info": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "grid_size": grid_size,
                "x_bins": x_bins,
                "y_bins": y_bins
            },
            "uav_density": uav_density_map.tolist(),
            "vehicle_density": vehicle_density_map.tolist()
        }
        
        with open(os.path.join(self.output_dir, "traffic_density_data.json"), "w") as f:
            json.dump(density_data, f, indent=4)
        
        print(f"密度数据已保存至 {os.path.join(self.output_dir, 'traffic_density_data.json')}")


    def find_most_congested_intersection(self):
        """
        找出车辆最多的交叉路口
        :return: (intersection_id, vehicle_density, uav_density)
        """
        # 获取所有交叉路口
        intersections = EntityScheduler.getAllCrossroadPositions(self.env)
        
        if not intersections:
            print("没有找到交叉路口数据！")
            return None, 0, 0
        
        # 检查 intersections 的类型并进行适当处理
        if isinstance(intersections, list):
            # 如果是列表，将其转换为字典格式 {index: (x, y)}
            intersections_dict = {i: pos for i, pos in enumerate(intersections)}
            print(f"交叉路口数据为列表格式，已转换为字典。共 {len(intersections_dict)} 个交叉路口")
        else:
            # 已经是字典格式
            intersections_dict = intersections
            print(f"交叉路口数据为字典格式。共 {len(intersections_dict)} 个交叉路口")
        
        # 获取所有 UAV 和车辆节点
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
        
        # 记录每个交叉路口的密度
        intersection_densities = {}
        region_size = 500  # 500m x 500m 区域
        
        for intersection_id, (x_center, y_center) in intersections_dict.items():
            # 计算区域范围
            x_min, x_max = x_center - region_size/2, x_center + region_size/2
            y_min, y_max = y_center - region_size/2, y_center + region_size/2
            
            # 统计区域内的节点数量
            uav_count = sum(1 for node in uavs_nodes 
                          if x_min <= node.to_dict()['position_x'] <= x_max and 
                          y_min <= node.to_dict()['position_y'] <= y_max)
            
            vehicle_count = sum(1 for node in vehicles_nodes 
                            if x_min <= node.to_dict()['position_x'] <= x_max and 
                            y_min <= node.to_dict()['position_y'] <= y_max)
            
            # 计算密度（每平方公里）
            area_km2 = (region_size * region_size) / 1e6
            uav_density = uav_count / area_km2 if area_km2 > 0 else 0
            vehicle_density = vehicle_count / area_km2 if area_km2 > 0 else 0
            
            intersection_densities[intersection_id] = {
                "vehicle_density": vehicle_density,
                "uav_density": uav_density,
                "position": (x_center, y_center)
            }
        
        # 找出车辆密度最高的交叉路口
        if not intersection_densities:
            return None, 0, 0
        
        most_congested_id = max(intersection_densities.keys(), 
                               key=lambda k: intersection_densities[k]["vehicle_density"])
        
        vehicle_density = intersection_densities[most_congested_id]["vehicle_density"]
        uav_density = intersection_densities[most_congested_id]["uav_density"]
        position = intersection_densities[most_congested_id]["position"]
        
        print(f"最拥堵的交叉路口ID: {most_congested_id}, 位置: ({position[0]:.1f}, {position[1]:.1f})")
        print(f"车辆密度: {vehicle_density:.2f}/km², UAV密度: {uav_density:.2f}/km²")
        
        return most_congested_id, vehicle_density, uav_density

    def save_intersection_density_to_json(self, output_file="intersection_density.json", region_size=500):
        """
        保存每个交叉路口的交通流密度数据到JSON
        :param output_file: 输出文件路径
        :param region_size: 计算密度的区域大小（单位：米），默认 500m x 500m
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取交叉路口坐标
        intersections = EntityScheduler.getAllCrossroadPositions(self.env)
        
        if not intersections:
            print("没有找到交叉路口数据！")
            return
        
        # 检查 intersections 的类型并进行适当处理
        if isinstance(intersections, list):
            # 如果是列表，将其转换为字典格式 {index: (x, y)}
            intersections_dict = {i: pos for i, pos in enumerate(intersections)}
        else:
            # 已经是字典格式
            intersections_dict = intersections
        
        print(f"处理 {len(intersections_dict)} 个交叉路口的密度数据...")
        
        # 获取所有 UAV 和车辆节点
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')

        density_data = {}

        for intersection_id, (x_center, y_center) in intersections_dict.items():
            x_min, x_max = x_center - region_size / 2, x_center + region_size / 2
            y_min, y_max = y_center - region_size / 2, y_center + region_size / 2
        
            # 统计区域内的节点数量
            uav_count = sum(1 for node in uavs_nodes 
                          if x_min <= node.to_dict()['position_x'] <= x_max and 
                          y_min <= node.to_dict()['position_y'] <= y_max)
            
            vehicle_count = sum(1 for node in vehicles_nodes 
                            if x_min <= node.to_dict()['position_x'] <= x_max and 
                            y_min <= node.to_dict()['position_y'] <= y_max)
        
            # 计算密度（每平方公里）
            area_km2 = (region_size * region_size) / 1e6
            uav_density = uav_count / area_km2 if area_km2 > 0 else 0
            vehicle_density = vehicle_count / area_km2 if area_km2 > 0 else 0
        
            # 存储数据
            density_data[str(intersection_id)] = {
                "position": {"x": float(x_center), "y": float(y_center)},
                "vehicle_density": float(vehicle_density),
                "uav_density": float(uav_density),
                "vehicle_count": vehicle_count,
                "uav_count": uav_count,
                "area_km2": float(area_km2)
            }
        
        # 将结果保存为 JSON 文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(density_data, f, indent=4, ensure_ascii=False)
        
        print(f"交叉路口交通流密度数据已保存至 {output_file}！")
    
    def generate_traffic_heatmap(self, time_window=60, output_file="traffic_heatmap.png"):
        """
        生成 UAV 和车辆的交通流热力图，统计一定时间窗口内的平均值
        :param time_window: 统计的时间窗口（秒），默认 60 秒
        :param output_file: 生成的热力图文件名
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
        # 获取所有交叉路口
        intersections = EntityScheduler.getAllCrossroadPositions(self.env)
        if not intersections:
            print("没有找到交叉路口数据！")
            return
    
        # 记录每个交叉路口的密度数据
        density_data = {id: {"vehicle_density": 0, "uav_density": 0} for id in intersections}
    
        # 获取所有 UAV 和车辆节点
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
    
        # 计算每个交叉路口的密度
        for intersection_id, (x, y) in intersections.items():
            x_min, x_max = x - 250, x + 250  # 500m 区域
            y_min, y_max = y - 250, y + 250
            
            # 统计区域内的节点数量
            uav_count = sum(1 for node in uavs_nodes 
                          if x_min <= node.to_dict()['position_x'] <= x_max and 
                          y_min <= node.to_dict()['position_y'] <= y_max)
            
            vehicle_count = sum(1 for node in vehicles_nodes 
                            if x_min <= node.to_dict()['position_x'] <= x_max and 
                            y_min <= node.to_dict()['position_y'] <= y_max)
    
            area_km2 = (500 * 500) / 1e6  # 转换平方公里
            uav_density = uav_count / area_km2 if area_km2 > 0 else 0
            vehicle_density = vehicle_count / area_km2 if area_km2 > 0 else 0
    
            density_data[intersection_id]["vehicle_density"] = vehicle_density
            density_data[intersection_id]["uav_density"] = uav_density
    
        # 生成热力图数据
        x_vals = []
        y_vals = []
        vehicle_densities = []
        uav_densities = []
    
        for intersection_id, (x, y) in intersections.items():
            x_vals.append(x)
            y_vals.append(y)
            vehicle_densities.append(density_data[intersection_id]["vehicle_density"])
            uav_densities.append(density_data[intersection_id]["uav_density"])
    
        # 生成车辆热力图
        plt.figure(figsize=(10, 8))
        
        # 修复：使用scatter返回的对象作为colorbar的mappable
        sc_vehicle = plt.scatter(x_vals, y_vals, c=vehicle_densities, cmap='Reds', 
                        s=[max(50, d*10) for d in vehicle_densities], alpha=0.7)
        
        # 使用scatter返回的对象创建colorbar
        plt.colorbar(sc_vehicle, label="Vehicle Density (/km²)")
        
        plt.title("Vehicle Traffic Density Heatmap")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        vehicle_output = output_file.replace('.png', '_vehicle.png')
        plt.savefig(vehicle_output, dpi=300)
        plt.close()
    
        # 生成 UAV 热力图
        plt.figure(figsize=(10, 8))
        
        # 修复：使用scatter返回的对象作为colorbar的mappable
        sc_uav = plt.scatter(x_vals, y_vals, c=uav_densities, cmap='Blues', 
                        s=[max(50, d*10) for d in uav_densities], alpha=0.7)
        
        # 使用scatter返回的对象创建colorbar
        plt.colorbar(sc_uav, label="UAV Density (/km²)")
        
        plt.title("UAV Traffic Density Heatmap")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        uav_output = output_file.replace('.png', '_uav.png')
        plt.savefig(uav_output, dpi=300)
        plt.close()
    
        print(f"热力图已生成: {vehicle_output}, {uav_output}")
    
        # 也可以保存数据到 JSON
        json_output = output_file.replace('.png', '_data.json')
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump({
                "intersections": {id: {"x": float(x_vals[i]), "y": float(y_vals[i])} 
                                for i, id in enumerate(list(intersections.keys())[:len(x_vals)])},
                "vehicle_density": {id: float(vehicle_densities[i]) 
                                for i, id in enumerate(list(intersections.keys())[:len(vehicle_densities)])},
                "uav_density": {id: float(uav_densities[i]) 
                              for i, id in enumerate(list(intersections.keys())[:len(uav_densities)])}
            }, f, indent=4, ensure_ascii=False)
    
        print(f"热力图数据已保存至 {json_output}")

    def update_data(self, timestamp=None, iteration=None, step=None, force_update=False):
        """
        在每个仿真时间步调用，提取数据并存储
        :param timestamp: 时间戳，用于文件名（可选）
        :param iteration: 迭代次数，用于文件名（可选）
        :param step: 当前仿真步数（可选）
        :param force_update: 是否强制更新，不考虑计数器
        """

        # 暂时用不到
        # # **更新实体信息（车辆 & UAV）**
        # if timestamp and iteration is not None and step is not None:
        #     self.update_entity_info(timestamp, iteration, step)
        #     self.get_task_status_counts(timestamp, iteration, step)

        # 计数器更新
        self.step_counter += 1

        # 控制更新频率，只有到达指定间隔才更新，或者 `force_update=True` 时强制更新
        if self.step_counter % self.update_interval != 0 and not force_update:
            return

        # **更新任务完成率和交通流密度**
        task_completion_rate = self.update_task_completion_rate()
        uav_density, vehicle_density = self.update_traffic_density()

        # 记录当前时间步的数据
        self.simulation_data.append({
            "simulation_time": self.env.simulation_time,
            "task_completion_rate": task_completion_rate,
            "uav_density": uav_density,
            "vehicle_density": vehicle_density
        })


    def save_to_json(self):
        """ 存储数据到 JSON 文件 """
        output_data = {
            "global_metrics": self.simulation_data,
            "context_information": self.traffic_topology
        }

        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"数据已保存至 {self.output_file}")

        
    def plot_results(self, save_filename="simulation_results1.png"):
        """
        可视化任务完成率和交通流密度
        """
        save_path = os.path.join(self.output_dir, save_filename)
        plt.figure(figsize=(10, 5))

        # 任务完成率
        plt.subplot(1, 2, 1)
        plt.plot(self.succ_ratio_list, label="Task Completion Rate", color='blue')
        plt.xlabel("Time Step")
        plt.ylabel("Completion Rate")
        plt.legend()
        plt.title("Task Completion Rate over Time")

        # 交通流密度
        plt.subplot(1, 2, 2)
        plt.plot(self.uav_density_list, label="UAV Density", color='red')
        plt.plot(self.vehicle_density_list, label="Vehicle Density", color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Traffic Density over Time")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"可视化结果已保存至 {save_path}")

