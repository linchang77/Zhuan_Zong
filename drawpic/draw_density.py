import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 绘图中文支持设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv(rf'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\data\Airfogsim_next_step_density.csv')

# 定义公式映射（提取自SciPy反馈）
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
    '方程11: 1.0091 * x1 * (1 - 0.0192 * x3 / (1 - 0.0637 * x2))':
        lambda x: 1.0091 * x[:, 0] * (1 - 0.0192 * x[:, 2] / (1 - 0.0637 * x[:, 1])),
    '方程12: 0.9984 * x1 - 0.0029 * x3 / (1 - 0.4623 * sqrt(x2))':
        lambda x: 0.9984 * x[:, 0] - 0.0029 * x[:, 2] / (1 - 0.4623 * np.sqrt(x[:, 1])),
    '方程13: 0.9979 * x1 + 0.0073 * x2 * x3 / (x2 - 5.9184)':
        lambda x: 0.9979 * x[:, 0] + 0.0073 * x[:, 1] * x[:, 2] / (x[:, 1] - 5.9184),
}

# 构建x（x1~x4）和真实y
x = data[['pre_vehicle_density', 'vehicle_avg_speed', 'v2u_rate', 'v2i_rate']].to_numpy(copy=True)
y = data['vehicle_density'].values

# 对所有公式进行预测和MSE计算
mse_results = []
for name, f in formulas.items():
    try:
        y_pred = []
        for i in range(len(x)):
            y_pred.append(f(x[i].reshape(1, -1)))
            # if i < len(x) - 1 :#and i > len(x)//2:
            #     x[i+1][0] = f(x[i].reshape(1, -1)) 
        y_pred = np.array(y_pred)
        mse = mean_squared_error(y, y_pred)
        mse_results.append((name, y_pred, mse))
    except Exception as e:
        print(f"公式出错: {name}，错误信息: {e}")
        continue

# 选取MSE最低的前6个公式进行绘图
top6 = sorted(mse_results, key=lambda x: x[2])[:1]

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(range(len(y)), y, 'rx', label='真实值', markersize=4)
for name, y_pred, mse in top6:
    plt.plot(range(len(y)), y_pred, '--', label=f'{name} (MSE={mse:.6f})')

plt.title('车辆密度预测')
plt.xlabel('时间(秒)')
plt.ylabel('车辆密度（辆/平方公里）')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# 添加变量说明图例
explanation = (
    "变量说明：\n"
    "x1: pre_vehicle_density\n"
    "x2: vehicle_avg_speed\n"
    "x3: v2u_rate\n"
    "x4: v2i_rate"
)
plt.gcf().text(0.75, 0.15, explanation, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))


plt.tight_layout()
# plt.savefig('density.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
