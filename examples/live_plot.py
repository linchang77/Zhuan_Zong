import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

class MultiLivePlot:
    def __init__(self, mode=1):
        self.mode = mode
        if self.mode == 1:
            self.layout = (3, 3)
        elif self.mode == 2:
            self.layout = (3, 3)
        elif self.mode == 3:
            self.layout = (2, 2)    
        self.fig, self.axes = plt.subplots(*self.layout, figsize=(15, 10))
        self.axes = self.axes.flatten()
        self.lines = []
        self.data = {}
        self.steps = []
        self.frames = []
        
        # 根据模式初始化子图
        if mode == 1:
            titles = [
                "Success Ratio", 
                "Avg Speed (Car)", 
                "Avg Speed (UAV)", 
                "V2U Rate", 
                "V2I Rate", 
                "U2I Rate", 
                "Avg Compute Delay"
            ]
            for i in range(7):
                line, = self.axes[i].plot([], [])
                self.axes[i].set_title(titles[i])
                self.lines.append(line)
            # 剩余2个子图留空
            for i in range(7, len(self.axes)):
                self.axes[i].axis('off')
        
        elif mode == 2:
            titles = [
                "V2U Rate", 
                "V2I Rate", 
                "U2I Rate", 
                "Waiting Task Number", 
                "Offloading Task Number",
                "Computing Task Number",
                "Done Task Number",
                "Failed Task Number",
                "Avg Compute Delay"
            ]
            for i in range(9):
                line, = self.axes[i].plot([], [])
                self.axes[i].set_title(titles[i])
                self.lines.append(line)
        
        elif mode == 3:
            titles = [
                "Avg Speed (Car)", 
                "V2U Rate", 
                "V2I Rate", 
                "Vehicle Density"
            ]
            for i in range(4):
                line, = self.axes[i].plot([], [])
                self.axes[i].set_title(titles[i])
                self.lines.append(line)
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def update(self, step, data):
        self.steps.append(step)
        
        if self.mode == 1:
            metrics = [
                data["succ_ratio"],
                data["avg_speed_car"],
                data["avg_speed_uav"],
                data["v2u_rate"],
                data["v2i_rate"],
                data["u2i_rate"],
                data["avg_running_compute_delay"]
            ]
            for i in range(7):
                # 确保 self.data[i] 是列表
                if f"metric_{i}" not in self.data:
                    self.data[f"metric_{i}"] = []
                self.data[f"metric_{i}"].append(metrics[i])
                self.lines[i].set_data(self.steps, self.data[f"metric_{i}"])
                self.axes[i].relim()
                self.axes[i].autoscale_view()
        
        elif self.mode == 2:
            metrics = [
                data["v2u_rate"],
                data["v2i_rate"],
                data["u2i_rate"],
                data["avg_running_compute_delay"]
            ]
            # 任务状态
            task_status = data["task_status"]
            
            # 更新通信速率和计算延迟
            for i in range(3):
                if f"v2u_{i}" not in self.data:
                    self.data[f"v2u_{i}"] = []
                self.data[f"v2u_{i}"].append(metrics[i])
                self.lines[i].set_data(self.steps, self.data[f"v2u_{i}"])
                self.axes[i].relim()
                self.axes[i].autoscale_view()
            
            # 更新计算延迟
            if "compute_delay" not in self.data:
                self.data["compute_delay"] = []
            self.data["compute_delay"].append(metrics[3])
            self.lines[8].set_data(self.steps, self.data["compute_delay"])
            self.axes[8].relim()
            self.axes[8].autoscale_view()
            
            # 更新任务状态
            for i, status in enumerate(["waiting", "offloading", "computing", "done", "failed"]):
                if f"task_{status}" not in self.data:
                    self.data[f"task_{status}"] = []
                self.data[f"task_{status}"].append(task_status.get(status, 0))
                self.lines[3 + i].set_data(self.steps, self.data[f"task_{status}"])
                self.axes[3 + i].relim()
                self.axes[3 + i].autoscale_view()
        
        elif self.mode == 3:
            vehicle_avg_speed = data["vehicle_avg_speed"]
            v2u_rate = data["v2u_rate"]
            v2i_rate = data["v2i_rate"]
            vehicle_density = data["vehicle_density"]
            
            # 更新车辆平均速度
            if "vehicle_avg_speed" not in self.data:
                self.data["vehicle_avg_speed"] = []
            self.data["vehicle_avg_speed"].append(vehicle_avg_speed)
            self.lines[0].set_data(self.steps, self.data["vehicle_avg_speed"])
            self.axes[0].relim()
            self.axes[0].autoscale_view()
            
            # 更新V2U速率
            if "v2u_rate" not in self.data:
                self.data["v2u_rate"] = []
            self.data["v2u_rate"].append(v2u_rate)
            self.lines[1].set_data(self.steps, self.data["v2u_rate"])
            self.axes[1].relim()
            self.axes[1].autoscale_view()
            
            # 更新V2I速率
            if "v2i_rate" not in self.data:
                self.data["v2i_rate"] = []
            self.data["v2i_rate"].append(v2i_rate)
            self.lines[2].set_data(self.steps, self.data["v2i_rate"])
            self.axes[2].relim()
            self.axes[2].autoscale_view()
            
            # 更新车辆密度
            if "vehicle_density" not in self.data:
                self.data["vehicle_density"] = []
            self.data["vehicle_density"].append(vehicle_density)
            self.lines[3].set_data(self.steps, self.data["vehicle_density"])
            self.axes[3].relim()
            self.axes[3].autoscale_view()
            
    
    def save_gif(self, filename='multi_plot.gif', duration=300):
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )
