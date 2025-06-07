'''
画箱线图
https://geek-docs.com/matplotlib/matplotlib-ask-answer/matplotlib-boxplot-by-group_z1.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
T=99
# 读取数据
time_finish_low = []
with open(r'./result/200_low_time_finish_rec.txt', 'r') as file:
    for line in file:
        time_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_low = []
with open(r'./result/compare_200_low_time_finish_rec.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_low = []
with open(r'./result/xiaorong_200_low_time_finish_rec.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_low=time_finish_low[:T]+Compare_finishtime_finish_low[0:T]+Xiaorong_time_finish_low[0:T]

time_finish_med = []
with open(r'./result/200_med_time_finish_rec.txt', 'r') as file:#50_med_time_finish
    for line in file:
        time_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_med = []
with open(r'./result/compare_200_med_time_finish_rec.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_med = []
with open(r'./result/xiaorong_200_med_time_finish_rec.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_med=time_finish_med[:T]+Compare_finishtime_finish_med[0:T]+Xiaorong_time_finish_med[0:T]

time_finish_hig = []
with open(r'./result/200_hig_time_finish_rec.txt', 'r') as file:#50_hig_time_finish
    for line in file:
        time_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_hig = []
with open(r'./result/compare_200_hig_time_finish_rec.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_hig = []
with open(r'./result/xiaorong_200_hig_time_finish_rec.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_hig=time_finish_hig[:T]+Compare_finishtime_finish_hig[0:T]+Xiaorong_time_finish_hig[0:T]

# 创建示例数据
np.random.seed(42)
df = pd.DataFrame({
    'group': np.repeat(['Low density', 'Medium density', 'High density'], T),
    'var1': all_time_finish_low,
    'var2': all_time_finish_med,
    'var3': all_time_finish_hig,
})


# 创建图形和坐标轴
fig, axs = plt.subplots(2,1, figsize=(6, 8))  # 1行2列
#绘制多变量箱线图
positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]
variables = ['var1', 'var2', 'var3']
colors = ['#148088', '#DCCB8E', '#593205']

# 绘制第一个多变量箱线图
positions1 = [1, 2, 3]
variables1 = ['var1', 'var2', 'var3']
for i, var in enumerate(variables1):
    data = [df[df['group'] == g][var] for g in ['Low density', 'Medium density', 'High density']]
    # 创建箱线图
    bp = axs[0].boxplot(data, positions=positions[i * 3:(i + 1) * 3], patch_artist=True, widths=0.4,
                    boxprops=dict(color='black', linewidth=1),  # 设置箱体的颜色和边界颜色
                    medianprops=dict(color='black'),showfliers=False)  # 设置中位数线的颜色
    j = 0
    for patch in bp['boxes']:
        patch.set_facecolor(colors[j])
        patch.set_alpha(0.6)
        j += 1
# 添加图例

axs[0].plot([], [], color=colors[0], label='BPF-FOA')
axs[0].plot([], [], color=colors[1], label='CPF-OA')
axs[0].plot([], [], color=colors[2], label='BPF-OA')
axs[0].legend(loc='upper left', ncol=3,columnspacing =0.5,handletextpad=0.1,handlelength=1.2 )
# 设置第一个图的标题和标签
#axs[0].set_title('Rectangular obstacle environments')
axs[0].set_xlabel('Density of rectangular obstacles')
axs[0].set_ylabel('Task completion time (s)')
axs[0].set_xticks([2, 6, 10])
axs[0].set_xticklabels(['Low density', 'Medium density', 'High density'])

# 读取数据
time_finish_low = []
with open(r'./result/200_low_time_finish_cir.txt', 'r') as file:
    for line in file:
        time_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_low = []
with open(r'./result/compare_200_low_time_finish_cir.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_low = []
with open(r'./result/xiaorong_200_low_time_finish_cir.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_low.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_low=time_finish_low[:T]+Compare_finishtime_finish_low[0:T]+Xiaorong_time_finish_low[0:T]

time_finish_med = []
with open(r'./result/200_med_time_finish_cir.txt', 'r') as file:#50_med_time_finish
    for line in file:
        time_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_med = []
with open(r'./result/compare_200_med_time_finish_cir.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_med = []
with open(r'./result/xiaorong_200_med_time_finish_cir.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_med.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_med=time_finish_med[:T]+Compare_finishtime_finish_med[0:T]+Xiaorong_time_finish_med[0:T]

time_finish_hig = []
with open(r'./result/200_hig_time_finish_cir.txt', 'r') as file:#50_hig_time_finish
    for line in file:
        time_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
Compare_finishtime_finish_hig = []
with open(r'./result/compare_200_hig_time_finish_cir.txt', 'r') as file:
    for line in file:
        Compare_finishtime_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
Xiaorong_time_finish_hig = []
with open(r'./result/xiaorong_200_hig_time_finish_cir.txt', 'r') as file:
    for line in file:
        Xiaorong_time_finish_hig.append(float(line.strip())/10)  # 将字符串转换为浮点数
all_time_finish_hig=time_finish_hig[:T]+Compare_finishtime_finish_hig[0:T]+Xiaorong_time_finish_hig[0:T]

# 创建示例数据
np.random.seed(42)
df = pd.DataFrame({
    'group': np.repeat(['Low density', 'Medium density', 'High density'], T),
    'var1': all_time_finish_low,
    'var2': all_time_finish_med,
    'var3': all_time_finish_hig,
})

# 绘制第二个多变量箱线图
positions2 = [1, 2, 3]
variables2 = ['var1', 'var2', 'var3']
for i, var in enumerate(variables2):
    data = [df[df['group'] == g][var] for g in ['Low density', 'Medium density', 'High density']]
    # 创建箱线图
    bp = axs[1].boxplot(data, positions=positions[i * 3:(i + 1) * 3], patch_artist=True, widths=0.4,
                        boxprops=dict(color='black', linewidth=1),  # 设置箱体的颜色和边界颜色
                        medianprops=dict(color='black'),showfliers=False)  # 设置中位数线的颜色
    j = 0
    for patch in bp['boxes']:
        patch.set_facecolor(colors[j])
        patch.set_alpha(0.6)
        j += 1
axs[1].plot([], [], color=colors[0], label='BPF-FOA')
axs[1].plot([], [], color=colors[1], label='CPF-OA')
axs[1].plot([], [], color=colors[2], label='BPF-OA')
axs[1].legend(loc='upper left', ncol=3,columnspacing =0.5,handletextpad=0.1,handlelength=1.2 )

# 设置第二个图的标题和标签
#axs[1].set_title('Circular obstacle environments')
axs[1].set_xlabel('Density of circular obstacles')
axs[1].set_ylabel('Task completion time (s)')
axs[1].set_xticks([2, 6, 10])
axs[1].set_xticklabels(['Low density', 'Medium density', 'High density'])


# 添加网格线
for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.7)
plt.subplots_adjust(hspace=0.2 )
# 保存图形
plt.savefig('fig_task_completion_time.jpg', dpi=600)  # 设置保存的分辨率
# 显示图形
plt.show()

