import sys
import os
import yaml
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QGroupBox,
                            QFormLayout, QMessageBox, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from simulation_env import SimulationEnvironment

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'examples/config.yaml')

class SimulationThread(QThread):
    """模拟线程，用于在后台运行模拟"""
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def run(self):
        try:
            self.simulation_env = SimulationEnvironment(CONFIG_PATH, self.update_signal.emit)
            self.simulation_env.initialize()
            self.simulation_env.run_simulation()
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