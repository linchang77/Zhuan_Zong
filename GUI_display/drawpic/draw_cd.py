import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 绘图中文支持设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv(rf'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\data\Airfogsim_cd.csv')

# 定义公式映射（提取自SciPy反馈）
formulas = {
    '公式1: (5.5785*x5 + 0.0999*x6)**0.5':
        lambda x: np.sqrt(5.5785 * x[:, 4] + 0.0999 * x[:, 5]),
    '公式2: sqrt(-53.6586*x5 + 0.0999*x6)':
        lambda x: np.sqrt(-53.6586 * x[:, 4] + 0.0999 * x[:, 5]),
    '公式3: 0.3160 * (x5 + x6)**0.5':
        lambda x: 0.3160 * np.sqrt(x[:, 4] + x[:, 5]),
    '公式4: 730.0702 * log(1 + x5) + 0.2455 * x6**0.5 + 0.0004 * x7':
        lambda x: 730.0702 * np.log(1 + x[:, 4]) + 0.2455 * np.sqrt(x[:, 5]) + 0.0004 * x[:, 6],
    '公式5: 0.1444 * (x5 + x6)**0.5 + 0.0892 * log(1 + x7)':
        lambda x: 0.1444 * np.sqrt(x[:, 4] + x[:, 5]) + 0.0892 * np.log(1 + x[:, 6]),
    '公式6: 0.1444 * sqrt(x5 + x6) + 0.0892 * log(1 + x7)':
        lambda x: 0.1444 * np.sqrt(x[:, 4] + x[:, 5]) + 0.0892 * np.log(1 + x[:, 6]),
    '公式7: sqrt(331474.8438 * x5 + 0.4627 * log(1 + x6) + 0.0000 * x7)':
        lambda x: np.sqrt(331474.8438 * x[:, 4] + 0.4627 * np.log(1 + x[:, 5]) + 0.0 * x[:, 6]),
    '公式8: (2.3536 * x5 + 0.4627 * log(1 + x6))**0.5':
        lambda x: np.sqrt(2.3536 * x[:, 4] + 0.4627 * np.log(1 + x[:, 5])),
    '公式9: sqrt(-0.4727 * x5 + 0.4627 * log(1 + x6))':
        lambda x: np.sqrt(-0.4727 * x[:, 4] + 0.4627 * np.log(1 + x[:, 5])),
    '公式10: (x5 + 0.4627 * log(1 + x6))**0.5':
        lambda x: np.sqrt(x[:, 4] + 0.4627 * np.log(1 + x[:, 5])),
}


# 构建x（x1~x8）和真实y
x = data[['v2u_rate','v2i_rate','u2i_rate', 'task_waiting', 'task_offload', 'task_computing', 'task_done', 'task_failed']].to_numpy(copy=True)
y = data['avg_running_compute_delay'].values

# 对所有公式进行预测和MSE计算
mse_results = []
for name, f in formulas.items():
    try:
        y_pred = []
        for i in range(len(x)):
            y_pred.append(f(x[i].reshape(1, -1)))
            if i < len(x) - 1 :#and i > len(x)//2:
                x[i+1][7] = f(x[i].reshape(1, -1)) 
        y_pred = np.array(y_pred)
        mse = mean_squared_error(y, y_pred)
        mse_results.append((name, y_pred, mse))
    except Exception as e:
        print(f"公式出错: {name}，错误信息: {e}")
        continue

# 选取MSE最低的前6个公式进行绘图
top6 = sorted(mse_results, key=lambda x: x[2])[:3]

# 绘图
plt.figure(figsize=(12, 8))
# 绘图
plt.figure(figsize=(12, 8))
plt.plot(range(len(y)), y, 'r-', label='真实值', linewidth=2)  # 改这里

# plt.plot(range(len(y)), y, 'rx', label='真实值', markersize=4)
# for name, y_pred, mse in top6:
#     plt.plot(range(len(y)), y_pred, '--', label=f'{name} (MSE={mse:.6f})')

plt.title('平均计算时延')
plt.xlabel('时间(秒)')
plt.ylabel('平均计算时延（秒)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# 添加变量说明图例
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
plt.gcf().text(0.75, 0.15, explanation, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()
# plt.savefig(rf'D:\Desktop\平均计算负载公式\gt.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
