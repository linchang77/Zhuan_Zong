import matplotlib.pyplot as plt
from airfogsim.scheduler import RewardScheduler, TaskScheduler

class DataManager:
    def __init__(self, env):
        self.env = env  # 绑定仿真环境
        
        # 任务相关数据
        self.finish_task_num = 0                 # 已完成任务数量
        self.out_of_ddl_task_num = 0             # 超时任务数量
        self.succ_ratio_list = []                # 任务完成率列表
        
        # 交通流密度
        self.uav_density_list = []
        self.vehicle_density_list = []



    def update_traffic_density(self, x_min=0, x_max=1000, y_min=0, y_max=1000):
        """
        计算 UAV 和 车辆的交通流密度
        :param x_min, x_max, y_min, y_max: 指定的区域范围
        :return: (uav_density, vehicle_density)
        """
        entity_scheduler = self.env.entity_scheduler
        uavs = entity_scheduler.getUAVs()
        vehicles = entity_scheduler.getVehicles()
        
        # 统计单位面积内的数量
        uav_count = sum(1 for uav in uavs.values() if x_min <= uav['x'] <= x_max and y_min <= uav['y'] <= y_max)
        vehicle_count = sum(1 for vehicle in vehicles.values() if x_min <= vehicle['x'] <= x_max and y_min <= vehicle['y'] <= y_max)
        
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


    def update_data(self):
        """
        在每个仿真时间步调用，提取数据并存储
        """
        self.update_task_completion_rate()

        self.update_traffic_density()
        

