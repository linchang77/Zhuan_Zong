import matplotlib.pyplot as plt
import time
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
        self.output_file = os.path.join(self.output_dir, "simulation_data1.json")

        # 数据存储
        self.simulation_data = []  # 存储每个时间步的数据

        # 数据更新间隔和计数器
        self.update_interval = update_interval
        self.step_counter = 0

        # 预提取交通拓扑
        self.extract_sumo_traffic_topology()


    def update_traffic_density(self, x_center=500, y_center=500, region_size=500):
        """
        计算固定区域（如交叉路口附近）的 UAV 和 车辆交通流密度
        :param x_center: 区域中心的 X 坐标
        :param y_center: 区域中心的 Y 坐标
        :param region_size: 计算密度的区域大小（单位：米），默认 500m x 500m
        :return: (uav_density, vehicle_density)
        """
        # 计算区域范围（固定在某个交叉路口）
        x_min, x_max = x_center - region_size / 2, x_center + region_size / 2
        y_min, y_max = y_center - region_size / 2, y_center + region_size / 2

        # 获取 UAV 和 车辆的节点信息
        uavs_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')

        # 统计区域内的 UAV 和 车辆数量
        uav_count = sum(1 for uav in uavs_nodes if x_min <= uav.to_dict()['position_x'] <= x_max and
                                                        y_min <= uav.to_dict()['position_y'] <= y_max)

        vehicle_count = sum(1 for vehicle in vehicles_nodes if x_min <= vehicle.to_dict()['position_x'] <= x_max and
                                                                y_min <= vehicle.to_dict()['position_y'] <= y_max)

        # 计算密度（单位：每平方公里）
        area_km2 = (region_size * region_size) / 1e6  # 500m × 500m = 0.25 km²
        uav_density = uav_count / area_km2 if area_km2 > 0 else 0
        vehicle_density = vehicle_count / area_km2 if area_km2 > 0 else 0

        # 记录数据
        self.uav_density_list.append(uav_density)
        self.vehicle_density_list.append(vehicle_density)

        # 打印调试信息
        print(f"交叉路口固定区域: X({x_min:.1f} - {x_max:.1f}), Y({y_min:.1f} - {y_max:.1f})，面积: {area_km2:.4f} km²")
        print(f"UAV 数量: {uav_count}, 车辆数量: {vehicle_count}")
        print(f"UAV 密度: {uav_density:.2f}/km², 车辆密度: {vehicle_density:.2f}/km²")

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
        file_path = f"info/entity_info_{timestamp}_it{iteration}.txt"

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
        file_path = f"info/entity_info_{timestamp}_it{iteration}.txt"

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

    def update_data(self, timestamp=None, iteration=None, step=None, force_update=False):
        """
        在每个仿真时间步调用，提取数据并存储
        :param timestamp: 时间戳，用于文件名（可选）
        :param iteration: 迭代次数，用于文件名（可选）
        :param step: 当前仿真步数（可选）
        :param force_update: 是否强制更新，不考虑计数器
        """
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

        # **更新实体信息（车辆 & UAV）**
        if timestamp and iteration is not None and step is not None:
            self.update_entity_info(timestamp, iteration, step)
            self.get_task_status_counts(timestamp, iteration, step)


    # def update_data(self, timestamp, iteration, step):
    #     """
    #     在每个仿真时间步调用，提取数据并存储
    #     """
    #     # self.update_task_completion_rate()
    #     # self.update_traffic_density()
    #     self.update_entity_info(timestamp, iteration, step)
    #     self.get_task_status_counts(timestamp, iteration, step)

    # def update_data(self, force_update=False):
    #     """
    #     在每个仿真时间步调用，提取数据并存储
    #     :param force_update: 是否强制更新，不考虑计数器
    #     """
    #     # 增加计数器
    #     self.step_counter += 1
        
    #     # 如果没到更新间隔且不是强制更新，则直接返回
    #     if self.step_counter % self.update_interval != 0 and not force_update:
    #         return
            
    #     task_completion_rate = self.update_task_completion_rate()
    #     uav_density, vehicle_density = self.update_traffic_density()

    #     # 记录当前时间步的数据
    #     self.simulation_data.append({
    #         "simulation_time": self.env.simulation_time,
    #         "task_completion_rate": task_completion_rate,
    #         "uav_density": uav_density,
    #         "vehicle_density": vehicle_density
    #     })

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

