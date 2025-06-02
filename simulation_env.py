import sys
import os
import numpy as np
import random
import yaml
import json
import datetime
from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler
from airfogsim.data_manager import DataManager

class SimulationEnvironment:
    """
    封装自 examples/get_density.py 的模拟环境类
    提供初始化和运行模拟的方法
    """
    
    def __init__(self, config_path, update_callback=None):
        """
        初始化模拟环境
        
        参数:
            config_path: 配置文件路径
            update_callback: 回调函数，用于更新UI显示，接收字符串参数
        """
        self.config_path = config_path
        self.update_callback = update_callback
        self.SEED = 42
        self.env = None
        self.data_manager = None
        self.algorithm_module = None
        self.feature_log = []
        self.v2u_rate = [0]
        self.v2i_rate = [0]
        # 定义感兴趣的区域范围
        self.region_x = [1250, 1500]  # x坐标范围
        self.region_y = [1400, 1600]  # y坐标范围
    
    def _send_update(self, message):
        """发送更新消息到UI"""
        if self.update_callback:
            self.update_callback(message)
        else:
            print(message)
    
    def load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    
    def initialize(self):
        """初始化模拟环境"""
        self._send_update("开始模拟...")
        
        # 设置随机种子
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        
        # 加载配置文件
        config = self.load_config()
        
        # 创建环境
        self._send_update("创建模拟环境...")
        self.env = AirFogSimEnv(config, interactive_mode='graphic')
        
        # 初始化DataManager
        sumo_net_file = "./sumo_wujiaochang/osm.net.xml"
        self.data_manager = DataManager(self.env, sumo_net_file=sumo_net_file)
        
        # 获取算法模块
        self.algorithm_module = BaseAlgorithmModule()
        self.algorithm_module.initialize(self.env)
        RewardScheduler.setModel(self.env, 'REWARD', '1/task_delay')
        
        # 重置随机种子
        np.random.seed(0)
        random.seed(0)
        
        # 初始化通道速率列表
        self.v2u_rate = [0]
        self.v2i_rate = [0]
        
        # 初始化特征日志
        self.feature_log = []
        
        return self.env
    
    def run_simulation(self):
        """运行模拟"""
        if not self.env:
            self._send_update("错误：环境未初始化")
            return False
        
        self._send_update("开始执行模拟步骤...")
        step_count = 0
        
        try:
            while not self.env.isDone():
                self.algorithm_module.scheduleStep(self.env)
                self.env.step()
                step_count += 1
                self.env.render()
                
                # 获取通道平均速率
                self.v2u_rate.append(self.env.getChannelAvgRate('V2U'))
                self.v2i_rate.append(self.env.getChannelAvgRate('V2I'))
                
                status_msg = f'模拟时间: {self.env.simulation_time:.2f}, V2U: {self.v2u_rate[-1]:.2f}, V2I: {self.v2i_rate[-1]:.2f}'
                self._send_update(status_msg)
                
                # 每10步获取一次车辆密度
                if step_count % 10 == 0:
                    # 获取指定区域的车辆密度
                    _, vehicle_density = self.data_manager.update_traffic_density(
                        x_min=self.region_x[0],
                        x_max=self.region_x[1],
                        y_min=self.region_y[0],
                        y_max=self.region_y[1]
                    )
                    # 单位转化为平方公里
                    vehicle_density = vehicle_density * 1e6
                    
                    # 获取区域内车辆的平均速度
                    # 使用EntityScheduler获取所有车辆节点
                    vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
                    
                    # 将节点对象转换为字典并筛选在指定区域内的车辆
                    vehicles_in_region = []
                    for node in vehicles_nodes:
                        vehicle_dict = node.to_dict()
                        if (self.region_x[0] <= vehicle_dict['position_x'] <= self.region_x[1] and
                            self.region_y[0] <= vehicle_dict['position_y'] <= self.region_y[1]):
                            vehicles_in_region.append(vehicle_dict)
                    
                    # 计算平均速度
                    total_speed = 0
                    if vehicles_in_region:
                        for vehicle in vehicles_in_region:
                            total_speed += vehicle['speed']
                        avg_speed = total_speed / len(vehicles_in_region)
                    else:
                        avg_speed = 0
                    
                    # 保存本步特征
                    self.feature_log.append({
                        "step": step_count,
                        "simulation_time": self.env.simulation_time,
                        "vehicle_density": vehicle_density,
                        "vehicle_count": len(vehicles_in_region),
                        "vehicle_avg_speed": avg_speed,
                        "v2u_rate": self.v2u_rate[-1],
                        "v2i_rate": self.v2i_rate[-1]
                    })
                    
                    density_msg = f"\n第 {step_count} 步:\n"
                    density_msg += f"车辆密度: {vehicle_density:.2f}/km²\n"
                    density_msg += f"区域内车辆数量: {len(vehicles_in_region)}\n"
                    density_msg += f"车辆平均速度: {avg_speed:.2f} m/s\n"
                    density_msg += f"V2U通道速率: {self.v2u_rate[-1]:.2f}\n"
                    density_msg += f"V2I通道速率: {self.v2i_rate[-1]:.2f}"
                    self._send_update(density_msg)
            
            self._send_update("\n模拟结束")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存特征日志到文件
            directory = "info"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            feature_file_path = f"{directory}/traffic_density_{timestamp}.json"
            with open(feature_file_path, "w") as f:
                json.dump(self.feature_log, f, indent=2)
            
            self._send_update(f"密度数据已保存至: {feature_file_path}")
            
            # 结束环境
            self.env.close()
            self._send_update('模拟完成!')
            return True
            
        except Exception as e:
            self._send_update(f"模拟过程中出错: {str(e)}")
            if self.env:
                try:
                    self.env.close()
                except:
                    pass
            return False