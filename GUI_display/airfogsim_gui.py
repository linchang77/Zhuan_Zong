import sys
import os
import yaml
import subprocess
from drawpic.plot_utils import DrawPlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QGroupBox,
                            QFormLayout, QMessageBox, QTextEdit, QScrollArea, QComboBox,
                            QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QMovie
from GUI_display.simulation_env import SimulationEnvironment
from LLMsforSR.workflow import AirFogSimWorkflow

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'examples/config.yaml')

class SimulationThread(QThread):
    """模拟线程，用于在后台运行模拟"""
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # (success, mode)
    
    def __init__(self, mode="预测交通流密度", model="claude-sonnet-4-20250514"):
        super().__init__()
        self.mode = mode
        self.model = model
        self.simulation_env = None
        self.workflow = None
    
    def run(self):
        try:
            # 第一步：运行模拟环境
            self.update_signal.emit("开始运行模拟环境...")
            self.simulation_env = SimulationEnvironment(CONFIG_PATH, self.update_signal.emit, self.mode)
            self.simulation_env.initialize()
            sim_result = self.simulation_env.run_simulation()
            
            # 第二步：如果模拟成功，运行LLM工作流
            if sim_result:
                self.update_signal.emit("\n模拟环境运行完成，开始运行LLM工作流...\n")
                self.workflow = AirFogSimWorkflow(self.mode, self.model, self.update_signal.emit)
                results, directory_path = self.workflow.run()
                self.update_signal.emit(f"\nLLM工作流执行完成，结果已保存到: {directory_path}")
                self.finished_signal.emit(True, self.mode)
            else:
                self.update_signal.emit("\n模拟环境运行失败，终止执行")
                self.finished_signal.emit(False, "")
        except Exception as e:
            error_msg = f"执行过程中出错: {str(e)}"
            self.update_signal.emit(error_msg)
            print(error_msg)  # 同时在控制台打印错误信息以便调试
            self.finished_signal.emit()
        finally:
            # 确保环境正确关闭
            if self.simulation_env:
                try:
                    self.simulation_env.env.close()
                except:
                    pass


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
            with open(self.config_path, 'r', encoding='utf-8') as file:
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
            
            '''假装保存配置到文件
            # 保存到文件
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            '''
            QMessageBox.information(self, "成功", "配置已保存")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置时出错: {str(e)}")


class GifViewDialog(QDialog):
    """GIF动画查看对话框"""
    def __init__(self, gif_path, mode_name, parent=None):
        super().__init__(parent)
        self.gif_path = gif_path
        self.mode_name = mode_name
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"{self.mode_name} - 模拟数据动图")
        self.setModal(True)
        
        # 创建主布局
        layout = QVBoxLayout()
        
        # 检查gif文件是否存在
        if not os.path.exists(self.gif_path):
            # 如果gif文件不存在，显示提示信息
            no_gif_label = QLabel("暂无此模式的模拟数据动图")
            no_gif_label.setAlignment(Qt.AlignCenter)
            no_gif_label.setStyleSheet("font-size: 16px; color: #666; padding: 50px;")
            layout.addWidget(no_gif_label)
        else:
            # 创建标签显示gif
            self.gif_label = QLabel()
            self.gif_label.setAlignment(Qt.AlignCenter)
            
            # 创建QMovie来播放gif
            self.movie = QMovie(self.gif_path)
            self.gif_label.setMovie(self.movie)
            
            # 开始播放
            self.movie.start()
            
            layout.addWidget(self.gif_label)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        
        if os.path.exists(self.gif_path):
            # 暂停/播放按钮
            self.play_pause_button = QPushButton("暂停")
            self.play_pause_button.clicked.connect(self.toggle_play_pause)
            button_layout.addWidget(self.play_pause_button)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # 设置窗口大小
        self.resize(800, 600)
    
    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        if hasattr(self, 'movie'):
            if self.movie.state() == QMovie.Running:
                self.movie.setPaused(True)
                self.play_pause_button.setText("播放")
            else:
                self.movie.setPaused(False)
                self.play_pause_button.setText("暂停")


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
        
        # 创建选项布局
        options_layout = QHBoxLayout()
        
        # 添加模式选择
        mode_layout = QHBoxLayout()
        mode_label = QLabel("模拟模式:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("预测交通流密度")
        self.mode_combo.addItem("预测任务成功率")
        self.mode_combo.addItem("预测计算时延")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        
        # 添加模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("推理模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItem("claude-sonnet-4-20250514")
        self.model_combo.addItem("qwen-max-latest")
        self.model_combo.addItem("gpt-4.1-2025-04-14")
        self.model_combo.addItem("gpt-4o")
        self.model_combo.addItem("deepseek-v3-250324")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # 将选项添加到布局
        options_layout.addLayout(mode_layout)
        options_layout.addLayout(model_layout)
        options_layout.addStretch()
        main_layout.addLayout(options_layout)
        
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
        
        # 查看上次模拟动图按钮
        view_gif_button = QPushButton("查看可视化指标动图")
        view_gif_button.clicked.connect(self.view_simulation_gif)
        button_layout.addWidget(view_gif_button)
        
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
    
    def view_simulation_gif(self):
        """查看模拟数据动图"""
        # 获取当前选择的模式
        selected_mode = self.mode_combo.currentText()
        
        # 根据模式确定对应的gif文件路径
        gif_mapping = {
            "预测交通流密度": "mode3_traffic_density.gif",
            "预测任务成功率": "mode1_success_rate.gif",
            "预测计算时延": "mode2_delay.gif"
        }
        
        gif_filename = gif_mapping.get(selected_mode)
        if not gif_filename:
            QMessageBox.warning(self, "警告", "未知的模拟模式")
            return
        
        # 构建gif文件的完整路径
        gif_path = os.path.join(os.path.dirname(__file__), gif_filename)
        
        # 创建并显示gif查看对话框
        self.gif_dialog = GifViewDialog(gif_path, selected_mode, self)
        self.gif_dialog.show()
    
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
        
        # 获取选定的模式
        selected_mode = self.mode_combo.currentText()
        
        # 检查是否选择了模式
        if not selected_mode:
            QMessageBox.warning(self, "警告", "请先选择模拟模式")
            self.run_button.setEnabled(True)
            return
            
        # 获取选定的模型
        selected_model = self.model_combo.currentText()
        
        # 创建并启动模拟线程
        self.simulation_thread = SimulationThread(mode=selected_mode, model=selected_model)
        self.simulation_thread.update_signal.connect(self.update_log)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.start()
        
    def draw_plot(self, mode):
        """在主线程中显示绘图结果"""
        self.update_log("\n开始绘制曲线图...")
        plotter = DrawPlot()
        
        # 获取对应的图片保存路径
        if mode == "预测交通流密度":
            img_path = plotter.draw_density()
        elif mode == "预测任务成功率":
            img_path = plotter.draw_success_rate()
        elif mode == "预测计算时延":
            img_path = plotter.draw_compute_delay()
            
        # 创建显示弹窗
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{mode}曲线图")
        dialog.setModal(True)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 创建标签显示图片
        label = QLabel()
        pixmap = QPixmap(img_path)
        # 等比例缩放图片
        scaled_pixmap = pixmap.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        layout.addWidget(label)
        
        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        dialog.setLayout(layout)
        
        # 适应图片大小调整弹窗
        dialog.adjustSize()
        dialog.show()
        
        self.update_log("曲线图绘制完成")
    
    def update_log(self, message):
        """更新日志显示"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def simulation_finished(self, success, mode):
        """模拟和工作流完成时的处理"""
        self.run_button.setEnabled(True)
        if success:
            self.draw_plot(mode)
        self.log_text.append("\n任务全部完成")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())