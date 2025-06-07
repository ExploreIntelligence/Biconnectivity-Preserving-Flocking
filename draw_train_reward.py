import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def read_data(file_path):
    """从文件中逐行读取数据"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

# 文件路径
file_path1 = './result/reward_cir.txt'  # 请替换为你的第一个文件路径
file_path2 = './result/reward_rec.txt'  # 请替换为你的第二个文件路径

# 读取数据
data1 = read_data(file_path1)
data2 = read_data(file_path2)

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图形大小

# 使用更美观的配色方案
colors = ['#44757A', '#DE6C4C']  # 蓝色和橙色
linestyles = ['-', '-']  # 实线和虚线

# 绘制两条曲线，使用不同的线型和透明度
ax.plot(data2, label=f'Rectangular obstacle environment', color=colors[1], linestyle=linestyles[1], linewidth=2, alpha=0.8)
ax.plot(data1, label=f'Circular obstacle environment', color=colors[0], linestyle=linestyles[0], linewidth=2, alpha=0.8)

# 添加图例
ax.legend(fontsize=18)

# 设置标题和标签
#ax.set_title('Reward Function Values', fontsize=20)
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward value')

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 设置x轴和y轴的范围
ax.set_xlim(0, len(data1) - 1)
ax.set_ylim(min(min(data1), min(data2)) * 0.9, max(max(data1), max(data2)) * 1.1)

# 保存图形
plt.savefig('fig_train_reward.jpg', dpi=600)  # 设置保存的分辨率

# 显示图形
plt.show()
