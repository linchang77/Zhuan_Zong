import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 绘图中文支持设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv(rf'C:\Users\linchang\Desktop\Zhuan_Zong\drawpic\data\Airfogsim_success_rate.csv')

# 定义公式映射（提取自SciPy反馈）
formulas = {
    '公式1: 0.7997 + (-0.0548)*sqrt(x1)': lambda x: 0.7997 + (-0.0548) * np.sqrt(x[:, 0]),
    '公式2: 0.7780 + (-0.9266)*x3': lambda x: 0.7780 + (-0.9266) * x[:, 2],
    '公式3: 1.1729 - 0.1061*x1 + 0.0000*exp(x1) + 0.0046*x1**2': lambda x: 1.1729 - 0.1061 * x[:, 0] + 0.0000 * np.exp(x[:, 0]) + 0.0046 * x[:, 0]**2,
    '公式4: 0.7095 - 0.0494*x1*x4': lambda x: 0.7095 - 0.0494 * x[:, 0] * x[:, 3],
    '公式5: 0.9837 - 0.0341*x1**2 + 0.0019*x1**3 + 0.1061*x1': lambda x: 0.9837 - 0.0341 * x[:, 0]**2 + 0.0019 * x[:, 0]**3 + 0.1061 * x[:, 0],
    '公式6: 0.0080*sqrt(x5) + 0.9905*x7': lambda x: 0.0080 * np.sqrt(x[:, 4]) + 0.9905 * x[:, 6],
    '公式7: 0.9773*x7 + 0.0025*sqrt(x2) + 0.0096*x3': lambda x: 0.9773 * x[:, 6] + 0.0025 * np.sqrt(x[:, 1]) + 0.0096 * x[:, 2],
    '公式8: 0.0153 + 0.0032*x3 + 0.0001*x5 + 0.9725*x7': lambda x: 0.0153 + 0.0032 * x[:, 2] + 0.0001 * x[:, 4] + 0.9725 * x[:, 6],
    '公式9: 0.0206 + (-0.0000)*x1**2 + 0.9724*x7': lambda x: 0.0206 + (-0.0000) * x[:, 0]**2 + 0.9724 * x[:, 6],
    '公式10: 0.0262 - 0.0009*x1 + 0.9705*x7': lambda x: 0.0262 - 0.0009 * x[:, 0] + 0.9705 * x[:, 6],
    '公式11: 0.0293 - 0.0009*x1 - 0.0007*sqrt(x2) + 0.9704*x7': lambda x: 0.0293 - 0.0009 * x[:, 0] - 0.0007 * np.sqrt(x[:, 1]) + 0.9704 * x[:, 6],
    '公式12: 0.0336 - 0.0049*sqrt(x1) + 0.9688*x7': lambda x: 0.0336 - 0.0049 * np.sqrt(x[:, 0]) + 0.9688 * x[:, 6],
    '公式13: 0.0447 - 0.0016*x1 + 0.9560*x7 - 0.0311*x4*x5': lambda x: 0.0447 - 0.0016 * x[:, 0] + 0.9560 * x[:, 6] - 0.0311 * x[:, 3] * x[:, 4],
}

# 构建x（x1~x7）和真实y
x = data[['avg_speed_car','avg_speed_uav','v2u_rate','v2i_rate','u2i_rate','avg_running_compute_delay','pre_ratio']].to_numpy(copy=True)
y = data['succ_ratio'].values

# 对所有公式进行预测和MSE计算
mse_results = []
for name, f in formulas.items():
    try:
        y_pred = []
        for i in range(len(x)):
            y_pred.append(f(x[i].reshape(1, -1)))
            if i < len(x) - 1 :#and i > len(x)//2:
                x[i+1][6] = f(x[i].reshape(1, -1)) 
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
plt.plot(range(len(y)), y, 'rx', label='真实值', markersize=4)
for name, y_pred, mse in top6:
    plt.plot(range(len(y)), y_pred, '--', label=f'{name} (MSE={mse:.6f})')

plt.title('成功率递推关系预测')
plt.xlabel('时间(秒)')
plt.ylabel('成功率')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# 添加变量说明图例
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
plt.gcf().text(0.75, 0.15, explanation, fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))


plt.tight_layout()
# plt.savefig(rf'D:\000wqy\threedown\zz\LLMsforSR\mid\fitted_scipy_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
