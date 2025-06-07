import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# 读取数据
min_distance_rec = []
with open(r'./result/max_distance_rec.txt', 'r') as file:
    for line in file:
        min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
min_distance_rec=min_distance_rec[:3000]
compare_min_distance_rec = []
with open(r'./result/compare_max_distance_rec.txt', 'r') as file:
    for line in file:
        compare_min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
compare_min_distance_rec=compare_min_distance_rec[:3000]
xiaorong_min_distance_rec = []
with open(r'./result/xiaorong_max_distance_rec.txt', 'r') as file:
    for line in file:
        xiaorong_min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
xiaorong_min_distance_rec=xiaorong_min_distance_rec[:3000]
# 创建图形和轴
#fig, ax1 = plt.subplots(figsize=(12, 8))  # 设置图形大小

fig, axs = plt.subplots(2, 1, figsize=(8, 6) )
# 使用更美观的配色方案
colors = [
    '#a02c2c',
    '#6a8078',
    '#4b57a2',
]
linestyles = ['-', '-']  # 实线和虚线
# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(min_distance_rec)) / 10
# 绘制曲线
axs[0].plot(x_values, min_distance_rec, color=colors[0], linestyle=linestyles[0], linewidth=1.5,alpha=1,label='BPF-FOA')

# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(compare_min_distance_rec)) / 10
# 绘制曲线
axs[0].plot(x_values, compare_min_distance_rec, color=colors[1], linestyle=linestyles[0], linewidth=1.5 ,label='CPF-OA')

# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(xiaorong_min_distance_rec)) / 10
# 绘制曲线
axs[0].plot(x_values, xiaorong_min_distance_rec, color=colors[2], linestyle=linestyles[0], linewidth=1.5,alpha=1,label='BPF-OA')
axs[0].set_xlim([0, 300])
axs[0].set_ylim([15, 68])
#axs[0].set_xlabel('Time (s)')
axs[0].set_xticks([])
axs[0].set_ylabel(r'$d_{\mathrm{max}}$ (m)')
# 添加网格线
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].legend(loc='upper right',fontsize=14,handlelength=1.2)


# 读取数据
min_distance_rec = []
with open(r'./result/max_distance_cir.txt', 'r') as file:
    for line in file:
        min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
min_distance_rec=min_distance_rec[:3000]
compare_min_distance_rec = []
with open(r'./result/compare_max_distance_cir.txt', 'r') as file:
    for line in file:
        compare_min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
compare_min_distance_rec=compare_min_distance_rec[:3000]
xiaorong_min_distance_rec = []
with open(r'./result/xiaorong_max_distance_cir.txt', 'r') as file:
    for line in file:
        xiaorong_min_distance_rec.append(float(line.strip()))  # 将字符串转换为浮点数
xiaorong_min_distance_rec=xiaorong_min_distance_rec[:3000]

colors = [
    '#a02c2c',
    '#6a8078',
    '#4b57a2',
]
# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(min_distance_rec)) / 10
# 绘制曲线
axs[1].plot(x_values, min_distance_rec, color=colors[0], linestyle=linestyles[0], linewidth=1.5,alpha=1,label='BPF-FOA')

# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(compare_min_distance_rec)) / 10
# 绘制曲线
axs[1].plot(x_values, compare_min_distance_rec, color=colors[1], linestyle=linestyles[0], linewidth=1.5 ,label='CPF-OA')

# 生成新的横坐标数组，值是原始索引值除以10
x_values = np.arange(len(xiaorong_min_distance_rec)) / 10
# 绘制曲线
axs[1].plot(x_values, xiaorong_min_distance_rec, color=colors[2], linestyle=linestyles[0], linewidth=1.5,alpha=1,label='BPF-OA')
axs[1].set_xlim([0, 300])
axs[1].set_ylim([15, 68])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel(r'$d_{\mathrm{max}}$ (m)')
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].legend(loc='upper right',fontsize=14,handlelength=1.2)

# 保存图形
plt.subplots_adjust(hspace=0.05)
plt.savefig('fig_change_farthest_distance.jpg', dpi=600)  # 设置保存的分辨率
plt.show()
