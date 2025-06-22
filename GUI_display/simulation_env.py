import sys
import os
import numpy as np
import random
import yaml
import json
import datetime
import matplotlib.pyplot as plt
from airfogsim import AirFogSimEnv, BaseAlgorithmModule
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler, ComputationScheduler
from airfogsim.data_manager import DataManager
from GUI_display.live_plot import MultiLivePlot

class SimulationEnvironment:
    """
    封装自 examples/get_density.py 的模拟环境类
    提供初始化和运行模拟的方法
    """
    
    def __init__(self, config_path, update_callback=None, mode="预测交通流密度"):
        """
        初始化模拟环境
        
        参数:
            config_path: 配置文件路径
            update_callback: 回调函数，用于更新UI显示，接收字符串参数
            mode: 模拟模式，默认为"预测交通流密度"
        """
        self.config_path = config_path
        self.update_callback = update_callback
        self.mode = mode
        self.SEED = 42
        self.env = None
        self.data_manager = None
        self.algorithm_module = None
        self.feature_log = []
        self.current_vehicle_density = 0  # 添加vehicle_density类变量
        self.current_vehicle_avg_speed = 0  # 添加vehicle_avg_speed类变量
        self.v2u_rate = [0]
        self.v2i_rate = [0]
        self.u2i_rate = [0]
        # 定义感兴趣的区域范围
        self.region_x = [1250, 1500]  # x坐标范围
        self.region_y = [1400, 1600]  # y坐标范围
        # 数据记录相关变量
        self.accumulated_reward = 0
        self.step_count = 0
        
        # 初始化 plotter
        if mode == "预测交通流密度":
            self.plotter = MultiLivePlot(mode=3)
        elif mode == "预测任务成功率":
            self.plotter = MultiLivePlot(mode=1)
        else:
            self.plotter = MultiLivePlot(mode=2) # 预测平均计算时延
        
        # 创建主输出目录
        self.main_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_output_dir = os.path.join("info", self.main_timestamp)
        os.makedirs(self.main_output_dir, exist_ok=True)
        
        # 创建轮次目录
        self.round_dir = os.path.join(self.main_output_dir, "round_0")
        os.makedirs(self.round_dir, exist_ok=True)
        
        # 初始化模式文件路径
        self.mode1_file = os.path.join(self.round_dir, "mode1_info.json")
        self.mode2_file = os.path.join(self.round_dir, "mode2_info.json")
        
        # 初始化任务配置信息结构
        self.task_config_info = {"task_config": {}}
        self.mode1_info = dict(self.task_config_info)
        self.mode1_info["succ_ratio_list"] = []
        self.mode2_info = {"info_list": []}
    
    def _send_update(self, message):
        """发送更新消息到UI"""
        if self.update_callback:
            self.update_callback(message)
        else:
            print(message)
    
    def get_vehicle_avg_speed(self):
        """获取所有车辆的平均速度"""
        vehicle_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
        vehicle_list = [v.to_dict() for v in vehicle_nodes]
        speeds = [v['speed'] for v in vehicle_list if 'speed' in v]
        return sum(speeds) / len(speeds) if speeds else 0.0
    
    def get_uav_avg_speed(self):
        """获取所有无人机的平均速度"""
        uav_nodes = EntityScheduler.getFogNodesByType(self.env, 'uav')
        uav_list = [u.to_dict() for u in uav_nodes]
        speeds = [u['speed'] for u in uav_list if 'speed' in u]
        return sum(speeds) / len(speeds) if speeds else 0.0
    
    def calculate_vehicle_density(self, region_x, region_y):
        """计算指定区域内的车辆密度"""
        # 获取所有车辆节点
        vehicles_nodes = EntityScheduler.getFogNodesByType(self.env, 'vehicle')
        
        # 将节点对象转换为字典并筛选在指定区域内的车辆
        vehicles_in_region = []
        for node in vehicles_nodes:
            vehicle_dict = node.to_dict()
            if (region_x[0] <= vehicle_dict['position_x'] <= region_x[1] and
                region_y[0] <= vehicle_dict['position_y'] <= region_y[1]):
                vehicles_in_region.append(vehicle_dict)
        
        # 计算车辆数量
        vehicle_count = len(vehicles_in_region)
        
        # 计算区域面积（假设单位是米，转换为平方公里）
        area_km2 = (region_x[1] - region_x[0]) * (region_y[1] - region_y[0]) / 1e6
        
        # 计算车辆密度
        vehicle_density = vehicle_count / area_km2 if area_km2 > 0 else 0
        
        return vehicle_density, vehicle_count, vehicles_in_region
    
    def load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as file:
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
        self.step_count = 0
        self.accumulated_reward = 0
        
        # 创建主输出目录
        self.main_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_output_dir = os.path.join("info", self.main_timestamp)
        os.makedirs(self.main_output_dir, exist_ok=True)
        
        # 创建轮次目录
        self.round_dir = os.path.join(self.main_output_dir, "round_0")
        os.makedirs(self.round_dir, exist_ok=True)
        
        # 初始化模式文件路径
        self.mode1_file = os.path.join(self.round_dir, "mode1_info.json")
        self.mode2_file = os.path.join(self.round_dir, "mode2_info.json")
        
        try:
            while not self.env.isDone():
                self.algorithm_module.scheduleStep(self.env)
                self.env.step()
                
                self.accumulated_reward += self.algorithm_module.getRewardByTask(self.env)
                task_num = TaskScheduler.getDoneTaskNum(self.env)
                out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(self.env)
                succ_ratio = task_num / max(1, task_num + out_of_ddl_task_num)
                
                self.step_count += 1
                self.env.render()
                
                # 获取通道平均速率
                self.v2u_rate.append(self.env.getChannelAvgRate('V2U'))
                self.v2i_rate.append(self.env.getChannelAvgRate('V2I'))
                self.u2i_rate.append(self.env.getChannelAvgRate('U2I'))
                
                # mode1: 每步记录
                self.mode1_info["succ_ratio_list"].append({
                    "step": self.step_count,
                    "succ_ratio": round(succ_ratio, 4)
                })
                with open(self.mode1_file, 'w') as f1:
                    json.dump(self.mode1_info, f1, indent=4)
                
                # mode2: 每5步记录
                if self.step_count % 5 == 0:
                    avg_speed_car = self.get_vehicle_avg_speed()
                    avg_speed_uav = self.get_uav_avg_speed()
                    
                    # 任务状态统计
                    status_counts = {
                        "waiting": len(TaskScheduler.getAllToOffloadTasks(self.env)),
                        "offloading": len(TaskScheduler.getAllOffloadingTaskInfos(self.env)),
                        "computing": len(TaskScheduler.getAllComputingTaskInfos(self.env)),
                        "done": TaskScheduler.getDoneTaskNum(self.env),
                        "failed": TaskScheduler.getOutOfDDLTasks(self.env)
                    }
                    
                    # 正在执行的任务详情
                    running_tasks_info = []
                    compute_delays = []
                    all_ids, type_list = EntityScheduler.getAllNodeIdsWithType(self.env)
                    for node_id, node_type in zip(all_ids, type_list):
                        total_remain_cpu, cpu, compute_delay = ComputationScheduler.getComputeStatusByNodeId(self.env, node_id)
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
                    
                    self.mode2_info["info_list"].append({
                        "step": self.step_count,
                        "succ_ratio": round(succ_ratio, 4),
                        "avg_speed_car": round(avg_speed_car, 2),
                        "avg_speed_uav": round(avg_speed_uav, 2),
                        "v2u_rate": round(self.v2u_rate[-1], 2),
                        "v2i_rate": round(self.v2i_rate[-1], 2),
                        "u2i_rate": round(self.u2i_rate[-1], 2),
                        "task_status": status_counts,
                        "running_tasks": running_tasks_info,
                        "avg_running_compute_delay": avg_running_compute_delay
                    })
                    
                    # 计算区域内的车辆密度
                    # 计算区域内的车辆密度和平均速度
                    vehicle_density, vehicle_count, vehicles_in_region = self.calculate_vehicle_density(
                        self.region_x,
                        self.region_y
                    )

                    # 计算区域内车辆平均速度
                    total_speed = 0
                    if vehicles_in_region:
                        for vehicle in vehicles_in_region:
                            total_speed += vehicle['speed']
                        vehicle_avg_speed = total_speed / len(vehicles_in_region)
                    else:
                        vehicle_avg_speed = 0

                    # 保存车辆密度和速度信息到类变量
                    self.current_vehicle_density = vehicle_density
                    self.current_vehicle_avg_speed = vehicle_avg_speed
                    
                    # 更新可视化
                    self.plotter.update(self.step_count, {
                        "succ_ratio": succ_ratio,
                        "avg_speed_car": avg_speed_car,
                        "avg_speed_uav": avg_speed_uav,
                        "v2u_rate": self.v2u_rate[-1],
                        "v2i_rate": self.v2i_rate[-1],
                        "u2i_rate": self.u2i_rate[-1],
                        "avg_running_compute_delay": avg_running_compute_delay,
                        "task_status": status_counts,
                        "vehicle_density": vehicle_density,
                        "vehicle_avg_speed": vehicle_avg_speed,
                    })
                    
                    # 捕获当前帧用于生成gif
                    # self.plotter.capture_frame()
                    
                    with open(self.mode2_file, 'w') as f2:
                        json.dump(self.mode2_info, f2, indent=4)
                
                # 更新状态消息
                status_msg = (f'Simulation time: {self.env.simulation_time:.2f}, '
                            f'已完成任务数: {task_num:.2f}, 超时任务数: {out_of_ddl_task_num}, '
                            f'Ratio: {succ_ratio:.2f}, ACC_Reward: {succ_ratio*self.accumulated_reward/max(1,task_num):.2f} '
                            f'V2U: {self.v2u_rate[-1]:.2f}, V2I: {self.v2i_rate[-1]:.2f}, '
                            f'U2I: {self.u2i_rate[-1]:.2f},vehicle_density: {self.current_vehicle_density:.4f}')
                self._send_update(status_msg)
            self.env.close()
            self._send_update('模拟完成!')
            
            # 保存gif图，根据模式选择文件名
            # if self.plotter.frames:
            #     if self.mode == "预测交通流密度":
            #         gif_filename = os.path.join(self.round_dir, 'mode3_traffic_density.gif')
            #     elif self.mode == "预测任务成功率":
            #         gif_filename = os.path.join(self.round_dir, 'mode1_success_rate.gif')
            #     else:
            #         gif_filename = os.path.join(self.round_dir, 'mode2_delay.gif')
                
            #     self.plotter.save_gif(gif_filename, duration=500)
            #     self._send_update(f'gif图已保存为: {gif_filename}')
            #     self._send_update(f'总共生成了 {len(self.plotter.frames)} 帧')
            # else:
            #     self._send_update('没有生成gif帧')
            
            return True
            
        except Exception as e:
            self._send_update(f"模拟过程中出错: {str(e)}")
            if self.env:
                try:
                    self.env.close()
                except:
                    pass
            return False