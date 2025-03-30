import matplotlib.pyplot as plt
import time
import datetime
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler

class DataManager:
    def __init__(self, env):
        """
        初始化 DataManager

        Args:
            env (AirFogSimEnv): 仿真环境对象
        """
        self.env = env  # 绑定仿真环境
        
        # 任务相关数据
        self.finish_task_num = 0                 # 已完成任务数量
        self.out_of_ddl_task_num = 0             # 超时任务数量
        self.succ_ratio_list = []                # 任务完成率列表
        
        # 交通流密度
        self.uav_density_list = []
        self.vehicle_density_list = []

        # 车辆和无人机信息
        self.entity_info = []  # 用于存储每秒的车辆和无人机数据



    # def update_traffic_density(self, x_min=0, x_max=1000, y_min=0, y_max=1000):
    #     """
    #     计算 UAV 和 车辆的交通流密度
    #     :param x_min, x_max, y_min, y_max: 指定的区域范围
    #     :return: (uav_density, vehicle_density)
    #     """
    #     entity_scheduler = self.env.entity_scheduler
    #     uavs = entity_scheduler.getUAVs()
    #     vehicles = entity_scheduler.getVehicles()
        
    #     # 统计单位面积内的数量
    #     uav_count = sum(1 for uav in uavs.values() if x_min <= uav['x'] <= x_max and y_min <= uav['y'] <= y_max)
    #     vehicle_count = sum(1 for vehicle in vehicles.values() if x_min <= vehicle['x'] <= x_max and y_min <= vehicle['y'] <= y_max)
        
    #     area = (x_max - x_min) * (y_max - y_min)
    #     uav_density = uav_count / area if area > 0 else 0
    #     vehicle_density = vehicle_count / area if area > 0 else 0

    #     self.uav_density_list.append(uav_density)
    #     self.vehicle_density_list.append(vehicle_density)

    #     return uav_density, vehicle_density
    
    # def update_task_completion_rate(self):
    #     """
    #     计算任务完成率 = 任务完成数 / (任务完成数 + 超时任务数)
    #     """
    #     # 获取当前任务数据
    #     task_num = TaskScheduler.getDoneTaskNum(self.env)
    #     out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(self.env)
    #     succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)

    #     # 记录数据
    #     self.finish_task_num = task_num                           # 已完成任务数量
    #     self.out_of_ddl_task_num = out_of_ddl_task_num            # 超时任务数量
    #     self.succ_ratio_list.append(succ_ratio)

    #     return 
    
    def update_entity_info(self, timestamp, iteration, step):
        '''
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

        # 获取所有节点的ID和类型
        all_ids, type_list = EntityScheduler.getAllNodeIdsWithType(self.env)

        # 统计调用次数
        call_count = getattr(self, "call_count", 0) + 1
        self.call_count = call_count  # 更新调用次数

        if step % 10 == 0:
            with open(file_path, "a") as f:
                # 写入调用次数和每次的实体信息
                f.write(f"Update Call #{call_count}:\n")
                
                for node_id, node_type in zip(all_ids, type_list):
                    node_info = EntityScheduler.getNodeInfoById(self.env, node_id)
                    f.write(f"Node ID: {node_info['id']}, Type: {node_type}, ")
                    f.write(f"Position (x, y, z): ({node_info['position_x']}, {node_info['position_y']}, {node_info['position_z']}), ")
                    f.write(f"Speed: {node_info['speed']}, Angle: {node_info['angle']}, Acceleration: {node_info['acceleration']}\n")
                
                # 添加分隔行以便区分不同调用
                f.write("-" * 50 + "\n")



    def update_data(self, timestamp, iteration, step):
        """
        在每个仿真时间步调用，提取数据并存储
        """
        # self.update_task_completion_rate()
        # self.update_traffic_density()
        self.update_entity_info(timestamp, iteration, step)
        

