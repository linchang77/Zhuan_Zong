import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class DrawPlot:
    def __init__(self):
        # 绘图中文支持设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.base_path = r'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\data'
        
    def __init__(self):
        # 绘图中文支持设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.base_path = r'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\data'
        self.save_path = r'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\output'
        
        # 确保保存目录存在
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
    def _setup_plot(self, title, xlabel, ylabel):
        """设置绘图的基本属性"""
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        
    def _add_explanation(self, explanation):
        """添加变量说明图例"""
        plt.gcf().text(0.75, 0.15, explanation, fontsize=10,
                      bbox=dict(facecolor='white', edgecolor='black'))
        
    def _finalize_plot(self, filename):
        """完成绘图设置并保存"""
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(self.save_path, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
        
    def draw_density(self):
        """绘制车辆密度预测图"""
        # 读取数据
        data = pd.read_csv(f'{self.base_path}/Airfogsim_next_step_density.csv')
        
        # 定义公式映射
        formulas = {
            '方程1: x1 + sqrt(x2 * x3)':
                lambda x: x[:, 0] + np.sqrt(x[:, 1] * x[:, 2]),
            '方程2: 0.9861*x1 + 0.0810*sqrt(x3)':
                lambda x: 0.9861 * x[:, 0] + 0.0810 * np.sqrt(x[:, 2]),
            '方程3: 0.9837*x1 + 0.2202*x3':
                lambda x: 0.9837 * x[:, 0] + 0.2202 * x[:, 2],
            '方程4: x1 * (0.9403 + 0.3329 * log(1 + x3))':
                lambda x: x[:, 0] * (0.9403 + 0.3329 * np.log(1 + x[:, 2])),
            '方程5: 0.9475 * x1 / (1 - 0.2662 * x3)':
                lambda x: 0.9475 * x[:, 0] / (1 - 0.2662 * x[:, 2]),
            '方程6: 0.9641 * x1 * (1 + 0.0126 * x2) / (1 + 0.0008 * x2**2)':
                lambda x: 0.9641 * x[:, 0] * (1 + 0.0126 * x[:, 1]) / (1 + 0.0008 * x[:, 1]**2),
            '方程7: 0.9316 * x1 * (1 + 0.0415 * sqrt(x2)) / (1 + 0.0005 * x2**2)':
                lambda x: 0.9316 * x[:, 0] * (1 + 0.0415 * np.sqrt(x[:, 1])) / (1 + 0.0005 * x[:, 1]**2),
            '方程8: x1 * (0.9209 + 0.1904 * sqrt(x3)) / (1 + 0.0000 * x2**3)':
                lambda x: x[:, 0] * (0.9209 + 0.1904 * np.sqrt(x[:, 2])) / (1 + 0.00000729 * x[:, 1]**3),
            '方程9: 0.9998 * x1 * (x2 -10.0621)/(x2 -10.0608)':
                lambda x: 0.9998 * x[:, 0] * (x[:, 1] - 10.0621) / (x[:, 1] - 10.0608),
            '方程10: x1 * (1 + 0.0012 * x3 / (sqrt(x2) -3.2543))':
                lambda x: x[:, 0] * (1 + 0.0012 * x[:, 2] / (np.sqrt(x[:, 1]) - 3.2543)),
        }
        
        # 构建x和真实y
        x = data[['pre_vehicle_density', 'vehicle_avg_speed', 'v2u_rate', 'v2i_rate']].to_numpy(copy=True)
        y = data['vehicle_density'].values
        
        # 计算预测值和MSE
        mse_results = []
        for name, f in formulas.items():
            try:
                y_pred = []
                for i in range(len(x)):
                    y_pred.append(f(x[i].reshape(1, -1)))
                y_pred = np.array(y_pred)
                mse = mean_squared_error(y, y_pred)
                mse_results.append((name, y_pred, mse))
            except Exception as e:
                print(f"公式出错: {name}，错误信息: {e}")
                continue
        
        # 选取MSE最低的公式
        top = sorted(mse_results, key=lambda x: x[2])[:1]
        
        # 绘图
        self._setup_plot('车辆密度预测', '时间(秒)', '车辆密度（辆/平方公里）')
        plt.plot(range(len(y)), y, 'rx', label='真实值', markersize=4)
        for name, y_pred, mse in top:
            plt.plot(range(len(y)), y_pred, '--', label=f'{name} (MSE={mse:.6f})')
            
        # 添加变量说明
        explanation = (
            "变量说明：\n"
            "x1: pre_vehicle_density\n"
            "x2: vehicle_avg_speed\n"
            "x3: v2u_rate\n"
            "x4: v2i_rate"
        )
        self._add_explanation(explanation)
        return self._finalize_plot('density.png')

    def draw_success_rate(self):
        """绘制成功率预测图"""
        # 读取数据
        data = pd.read_csv(f'{self.base_path}/Airfogsim_success_rate.csv')
        
        # 定义公式映射
        formulas = {
            '公式1: 0.7997 + (-0.0548)*sqrt(x1)': lambda x: 0.7997 + (-0.0548) * np.sqrt(x[:, 0]),
            '公式2: 0.7780 + (-0.9266)*x3': lambda x: 0.7780 + (-0.9266) * x[:, 2],
            '公式3: 1.1729 - 0.1061*x1 + 0.0000*exp(x1) + 0.0046*x1**2': lambda x: 1.1729 - 0.1061 * x[:, 0] + 0.0000 * np.exp(x[:, 0]) + 0.0046 * x[:, 0]**2,
            '公式4: 0.7095 - 0.0494*x1*x4': lambda x: 0.7095 - 0.0494 * x[:, 0] * x[:, 3],
            '公式5: 0.9837 - 0.0341*x1**2 + 0.0019*x1**3 + 0.1061*x1': lambda x: 0.9837 - 0.0341 * x[:, 0]**2 + 0.0019 * x[:, 0]**3 + 0.1061 * x[:, 0],
        }
        
        # 构建x和真实y
        x = data[['avg_speed_car','avg_speed_uav','v2u_rate','v2i_rate','u2i_rate','avg_running_compute_delay','pre_ratio']].to_numpy(copy=True)
        y = data['succ_ratio'].values
        
        # 计算预测值和MSE
        mse_results = []
        for name, f in formulas.items():
            try:
                y_pred = []
                for i in range(len(x)):
                    y_pred.append(f(x[i].reshape(1, -1)))
                    if i < len(x) - 1:
                        x[i+1][6] = f(x[i].reshape(1, -1))
                y_pred = np.array(y_pred)
                mse = mean_squared_error(y, y_pred)
                mse_results.append((name, y_pred, mse))
            except Exception as e:
                print(f"公式出错: {name}，错误信息: {e}")
                continue
                
        # 选取MSE最低的前3个公式
        top3 = sorted(mse_results, key=lambda x: x[2])[:3]
        
        # 绘图
        self._setup_plot('成功率递推关系预测', '时间(秒)', '成功率')
        plt.plot(range(len(y)), y, 'rx', label='真实值', markersize=4)
        for name, y_pred, mse in top3:
            plt.plot(range(len(y)), y_pred, '--', label=f'{name} (MSE={mse:.6f})')
            
        # 添加变量说明
        explanation = (
            "变量说明：\n"
            "x1: avg_speed_car\n"
            "x2: avg_speed_uav\n"
            "x3: v2u_rate\n"
            "x4: v2i_rate\n"
            "x5: u2i_rate\n"
            "x6: avg_running_compute_delay\n"
            "x7: pre_succ_ratio"
        )
        self._add_explanation(explanation)
        return self._finalize_plot('success_rate.png')

    def draw_compute_delay(self):
        """绘制计算时延图"""
        # 读取数据
        data = pd.read_csv(f'{self.base_path}/Airfogsim_cd.csv')
        
        # 构建x和真实y
        x = data[['v2u_rate','v2i_rate','u2i_rate', 'task_waiting', 'task_offload', 'task_computing', 'task_done', 'task_failed']].to_numpy(copy=True)
        y = data['avg_running_compute_delay'].values
        
        # 绘图
        self._setup_plot('平均计算时延', '时间(秒)', '平均计算时延（秒）')
        plt.plot(range(len(y)), y, 'r-', label='真实值', linewidth=2)
        
        # 添加变量说明
        explanation = (
            "变量说明：\n"
            "x1: v2u_rate\n"
            "x2: v2i_rate\n"
            "x3: u2i_rate\n"
            "x4: task_waiting\n"
            "x5: task_offload\n"
            "x6: task_computing\n"
            "x7: task_done\n"
            "x8: task_failed"
        )
        self._add_explanation(explanation)
        return self._finalize_plot('compute_delay.png')