import sys
import os
import yaml
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QGroupBox, 
                            QFormLayout, QMessageBox, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'examples/config.yaml')

class SimulationThread(QThread):
    """模拟线程，用于在后台运行模拟"""
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def run(self):
        try:
            # 设置环境变量以确保可以导入airfogsim
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(current_dir)
            
            # 导入必要的模块
            import random
            import numpy as np
            import json
            import datetime
            from airfogsim import AirFogSimEnv, BaseAlgorithmModule
            from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler
            from airfogsim.data_manager import DataManager
            
            SEED = 42
            
            def load_config(path):
                with open(path, 'r') as file:
                    config = yaml.safe_load(file)
                    return config
            
            self.update_signal.emit("开始模拟...")
            random.seed(SEED)
            np.random.seed(SEED)
            
            # 加载配置文件
            config = load_config(CONFIG_PATH)
            
            # 创建环境
            self.update_signal.emit("创建模拟环境...")
            env = AirFogSimEnv(config, interactive_mode='graphic')
            
            # 初始化DataManager
            sumo_net_file = "./sumo_wujiaochang/osm.net.xml"
            data_manager = DataManager(env, sumo_net_file=sumo_net_file)
            
            # 获取算法模块
            algorithm_module = BaseAlgorithmModule()
            algorithm_module.initialize(env)
            RewardScheduler.setModel(env, 'REWARD', '1/task_delay')
            accumulated_reward = 0
            np.random.seed(0)
            random.seed(0)
            
            # 初始化通道速率列表
            v2u_rate = [0]
            v2i_rate = [0]
            
            # 初始化特征日志
            feature_log = []
            
            # 定义感兴趣的区域范围
            region_x = [1250, 1500]  # x坐标范围
            region_y = [1400, 1600]  # y坐标范围
            
            # 开始模拟
            self.update_signal.emit("开始执行模拟步骤...")
            step_count = 0
            
            while not env.isDone():
                algorithm_module.scheduleStep(env)
                env.step()
                step_count += 1
                env.render()
                
                # 获取通道平均速率
                v2u_rate.append(env.getChannelAvgRate('V2U'))
                v2i_rate.append(env.getChannelAvgRate('V2I'))
                
                status_msg = f'模拟时间: {env.simulation_time:.2f}, V2U: {v2u_rate[-1]:.2f}, V2I: {v2i_rate[-1]:.2f}'
                self.update_signal.emit(status_msg)
                
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
                    
                    # 保存本步特征
                    feature_log.append({
                        "step": step_count,
                        "simulation_time": env.simulation_time,
                        "vehicle_density": vehicle_density,
                        "vehicle_count": len(vehicles_in_region),
                        "vehicle_avg_speed": avg_speed,
                        "v2u_rate": v2u_rate[-1],
                        "v2i_rate": v2i_rate[-1]
                    })
                    
                    density_msg = f"\n第 {step_count} 步:\n"
                    density_msg += f"车辆密度: {vehicle_density:.2f}/km²\n"
                    density_msg += f"区域内车辆数量: {len(vehicles_in_region)}\n"
                    density_msg += f"车辆平均速度: {avg_speed:.2f} m/s\n"
                    density_msg += f"V2U通道速率: {v2u_rate[-1]:.2f}\n"
                    density_msg += f"V2I通道速率: {v2i_rate[-1]:.2f}"
                    self.update_signal.emit(density_msg)
            
            self.update_signal.emit("\n模拟结束")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存特征日志到文件
            directory = "info"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            feature_file_path = f"{directory}/traffic_density_{timestamp}.json"
            with open(feature_file_path, "w") as f:
                json.dump(feature_log, f, indent=2)
            
            self.update_signal.emit(f"密度数据已保存至: {feature_file_path}")
            
            # 结束环境
            env.close()
            self.update_signal.emit('模拟完成!')
            self.finished_signal.emit()
            
        except Exception as e:
            self.update_signal.emit(f"模拟过程中出错: {str(e)}")
            self.finished_signal.emit()


class ConfigDialog(QWidget):
    """配置对话框，用于修改配置文件"""
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.config = self.load_config()
        self.init_ui()
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载配置文件: {str(e)}")
            return {}
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("修改配置")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # 创建表单布局
        form_layout = QFormLayout()
        
        # 模拟时间
        self.max_simulation_time_edit = QLineEdit(str(self.config.get('simulation', {}).get('max_simulation_time', '')))
        form_layout.addRow("模拟时间 (max_simulation_time):", self.max_simulation_time_edit)
        
        # 车辆数量
        self.vehicle_count_edit = QLineEdit(str(self.config.get('simulation', {}).get('vehicle_count', '')))
        form_layout.addRow("车辆数量 (vehicle_count):", self.vehicle_count_edit)
        
        # 车辆CPU算力
        self.vehicle_cpu_edit = QLineEdit(str(self.config.get('fog_profile', {}).get('vehicle', {}).get('cpu', '')))
        form_layout.addRow("车辆CPU算力:", self.vehicle_cpu_edit)
        
        # 无人机CPU算力
        self.uav_cpu_edit = QLineEdit(str(self.config.get('fog_profile', {}).get('uav', {}).get('cpu', '')))
        form_layout.addRow("无人机CPU算力:", self.uav_cpu_edit)
        
        # RSU算力
        self.rsu_cpu_edit = QLineEdit(str(self.config.get('fog_profile', {}).get('rsu', {}).get('cpu', '')))
        form_layout.addRow("RSU算力:", self.rsu_cpu_edit)
        
        # 云端算力
        self.cloud_cpu_edit = QLineEdit(str(self.config.get('fog_profile', {}).get('cloud', {}).get('cpu', '')))
        form_layout.addRow("云端算力:", self.cloud_cpu_edit)
        
        # 添加表单到布局
        layout.addLayout(form_layout)
        
        # 添加按钮
        button_layout = QHBoxLayout()
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save_config)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.close)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 更新配置
            if 'simulation' not in self.config:
                self.config['simulation'] = {}
            
            # 更新模拟时间
            try:
                self.config['simulation']['max_simulation_time'] = float(self.max_simulation_time_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "模拟时间必须是数字")
                return
            
            # 更新车辆数量
            try:
                self.config['simulation']['vehicle_count'] = int(self.vehicle_count_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "车辆数量必须是整数")
                return
            
            # 更新车辆CPU算力
            if 'fog_profile' not in self.config:
                self.config['fog_profile'] = {}
            if 'vehicle' not in self.config['fog_profile']:
                self.config['fog_profile']['vehicle'] = {}
            
            try:
                self.config['fog_profile']['vehicle']['cpu'] = float(self.vehicle_cpu_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "车辆CPU算力必须是数字")
                return
            
            # 更新无人机CPU算力
            if 'uav' not in self.config['fog_profile']:
                self.config['fog_profile']['uav'] = {}
            
            try:
                self.config['fog_profile']['uav']['cpu'] = float(self.uav_cpu_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "无人机CPU算力必须是数字")
                return
            
            # 更新RSU算力
            if 'rsu' not in self.config['fog_profile']:
                self.config['fog_profile']['rsu'] = {}
            
            try:
                self.config['fog_profile']['rsu']['cpu'] = float(self.rsu_cpu_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "RSU算力必须是数字")
                return
            
            # 更新云端算力
            if 'cloud' not in self.config['fog_profile']:
                self.config['fog_profile']['cloud'] = {}
            
            try:
                self.config['fog_profile']['cloud']['cpu'] = float(self.cloud_cpu_edit.text())
            except ValueError:
                QMessageBox.warning(self, "警告", "云端算力必须是数字")
                return
            
            # 保存到文件
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            
            QMessageBox.information(self, "成功", "配置已保存")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置时出错: {str(e)}")


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.simulation_thread = None
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AirFogSim GUI")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 添加按钮
        button_layout = QHBoxLayout()
        
        # 修改配置按钮
        config_button = QPushButton("修改配置")
        config_button.clicked.connect(self.open_config_dialog)
        button_layout.addWidget(config_button)
        
        # 运行按钮
        self.run_button = QPushButton("运行模拟")
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)
        
        main_layout.addLayout(button_layout)
        
        # 添加日志显示区域
        log_group = QGroupBox("模拟日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.log_text)
        scroll_area.setWidgetResizable(True)
        
        log_layout.addWidget(scroll_area)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        central_widget.setLayout(main_layout)
    
    def open_config_dialog(self):
        """打开配置对话框"""
        self.config_dialog = ConfigDialog(CONFIG_PATH)
        self.config_dialog.show()
    
    def run_simulation(self):
        """运行模拟"""
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "警告", "模拟已在运行中")
            return
        
        # 清空日志
        self.log_text.clear()
        self.log_text.append("准备开始模拟...")
        
        # 禁用运行按钮
        self.run_button.setEnabled(False)
        
        # 创建并启动模拟线程
        self.simulation_thread = SimulationThread()
        self.simulation_thread.update_signal.connect(self.update_log)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.start()
    
    def update_log(self, message):
        """更新日志显示"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def simulation_finished(self):
        """模拟完成时的处理"""
        self.run_button.setEnabled(True)
        self.log_text.append("\n模拟已完成")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())